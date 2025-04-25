import numpy as np
from scipy.stats import logistic, norm
from typing import Dict, List, Tuple
from scipy.optimize import minimize_scalar


class LinUCB_Pricing:
    def __init__(self, alpha: float = 1.0, F_dist: str = 'logistic', noise_B=1.0):
        """
        LinUCB algorithm for dynamic pricing without transfer learning

        Parameters:
        alpha: Exploration parameter
        F_dist: Noise distribution type ('logistic' or 'normal')
        """
        self.alpha = alpha
        self.F_dist = F_dist
        self.noise_B = noise_B
        self._init_F_functions()

    def _init_F_functions(self):
        """Initialize distribution functions"""
        if self.F_dist == 'logistic':
            self.F = logistic.cdf
            self.F_pdf = logistic.pdf
            self.F_deriv = lambda x: self.F(x) * (1 - self.F(x))
        elif self.F_dist == 'normal':
            self.F = norm.cdf
            self.F_pdf = norm.pdf
            self.F_deriv = norm.pdf
        else:
            raise ValueError("Supported distributions: 'logistic' or 'normal'")
        self.h = lambda u: self._solve_h(u)

    def fit(self,
            target_X: np.ndarray,
            target_p: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute LinUCB pricing algorithm

        Parameters:
        target_X: Target market features [n_samples, n_features]
        target_p: True market values (for simulation)

        Returns:
        {
            'prices': Price decisions,
            'betas': Parameter estimates,
            'ucbs': UCB values
        }
        """
        n_samples, d = target_X.shape
        prices = np.zeros(n_samples)
        betas = []
        ucbs = []

        # LinUCB initialization
        A = np.eye(d)  # Regularization matrix
        b = np.zeros(d)

        for t in range(n_samples):
            x_t = target_X[t]

            if t == 0:
                # Cold start - use random initialization
                beta = np.random.randn(d) * 0.1
            else:
                # Update parameters
                A += np.outer(x_t, x_t)
                b += x_t * prices[t - 1] * (target_p[t - 1] >= prices[t - 1])
                beta = np.linalg.solve(A, b)    # constant error, generalized bandits
                # real data, reward
                # cumulative regret --> avg regret

            # Compute UCB
            A_inv = np.linalg.inv(A)
            ucb = x_t @ beta + self.alpha * np.sqrt(x_t @ A_inv @ x_t)

            # Optimal pricing
            p_t = self._solve_h(ucb)

            # Store results
            prices[t] = p_t
            betas.append(beta.copy())
            ucbs.append(ucb)

        return {
            'prices': prices,
            'betas': betas,
            'ucbs': ucbs
        }

    def _solve_h(self, u):
        """Same pricing function as RMLP for fair comparison"""

        def phi(z):
            return z - (1 - self.F(z)) / (self.F_pdf(z) + 1e-10)

        res = minimize_scalar(
            lambda z: np.abs(phi(z) + u),
            bounds=(-self.noise_B, self.noise_B),
            method='bounded'
        )
        return max(u + res.x, 0)


class LinUCB_Online2Online:
    def __init__(self, alpha: float = 1.0, F_dist: str = 'logistic', W: float = 1.0, noise_B=1.0):
        """
        Online-to-Online Transfer LinUCB with two-stage estimation

        Parameters:
        alpha: Exploration parameter for UCB
        F_dist: Noise distribution type ('logistic' or 'normal')
        W: Bound for feature/parameter magnitudes
        """
        self.alpha = alpha
        self.F_dist = F_dist
        self.W = W
        self.noise_B = noise_B
        self._init_F_functions()

    def _init_F_functions(self):
        """Initialize distribution and pricing functions"""
        if self.F_dist == 'logistic':
            self.F = logistic.cdf
            self.F_pdf = logistic.pdf
        elif self.F_dist == 'normal':
            self.F = norm.cdf
            self.F_pdf = norm.pdf
        else:
            raise ValueError("Supported distributions: 'logistic' or 'normal'")

        self.h = lambda u: self._solve_h(u)

    def _solve_h(self, u):
        """Optimal pricing function"""

        def phi(z):
            return z - (1 - self.F(z)) / (self.F_pdf(z) + 1e-10)

        res = minimize_scalar(
            lambda z: np.abs(phi(z) + u),
            bounds=(-self.noise_B, self.noise_B),
            method='bounded'
        )
        return u + res.x

    def _linucb_estimate(self, X: np.ndarray, p: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate parameters using LinUCB-style ridge regression"""
        d = X.shape[1]
        A = np.eye(d)  # Regularization matrix
        b = np.zeros(d)

        for t in range(len(X)):
            x_t = X[t]
            A += np.outer(x_t, x_t)
            b += x_t * p[t] * y[t]

        return np.linalg.solve(A, b)

    def fit(self,
            source_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
            target_X: np.ndarray,
            target_p: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute two-stage online transfer learning with LinUCB

        Parameters:
        source_data: List of (prices, features, responses) tuples for source markets
        target_X: Target market features [n_samples, n_features]
        target_p: True market values (for simulation)

        Returns:
        {
            'prices': Price decisions,
            'betas': Parameter estimates,
            'ucbs': UCB values
        }
        """
        T = len(target_X)
        d = target_X.shape[1]
        prices = np.zeros(T)
        betas = []
        ucbs = []
        collected_data = []

        # Episode-based processing
        m = 1
        while True:
            prev_start_t = 2 ** (m - 2)-1 if m > 1 else 0
            prev_end_t = 2 ** (m - 1) -1
            start_t = 2 ** (m - 1)
            end_t = min(2 ** m - 1, T)

            # 1. Stage 1: Estimate β_ag from source markets (LinUCB)
            if m > 1:
                # Aggregate source data from previous episode
                X_source = np.concatenate([d[1][prev_start_t:prev_end_t + 1] for d in source_data])
                p_source = np.concatenate([d[0][prev_start_t:prev_end_t + 1] for d in source_data])
                y_source = np.concatenate([d[2][prev_start_t:prev_end_t + 1] for d in source_data])

                beta_ag = self._linucb_estimate(X_source, p_source, y_source)
            else:
                # Initial episode uses zero vector
                beta_ag = np.zeros(d)

            # 2. Stage 2: Estimate δ from target market (LinUCB)
            if m > 1:
                # Get target data from previous episode
                prev_data = collected_data[prev_start_t:prev_end_t + 1]
                X_target = np.array([d[1] for d in prev_data])
                p_target = np.array([d[0] for d in prev_data])
                y_target = np.array([d[2] for d in prev_data])

                # Estimate δ using LinUCB on adjusted responses
                # Create "pseudo-responses" for δ estimation
                pseudo_p = p_target - X_target @ beta_ag
                delta = self._linucb_estimate(X_target, pseudo_p, y_target)

                beta_m = beta_ag + delta
            else:
                # First episode uses β_ag only
                beta_m = beta_ag

            betas.append(beta_m.copy())

            # 3. Apply pricing with UCB exploration
            A = np.eye(d)  # Current covariance matrix
            for t in range(start_t, end_t + 1):
                if t >= T:
                    break

                x_t = target_X[t]
                A_inv = np.linalg.inv(A)

                # Compute UCB
                ucb = x_t @ beta_m + self.alpha * np.sqrt(x_t @ A_inv @ x_t)
                ucbs.append(ucb)

                # Set price
                p_t = self.h(ucb)
                prices[t] = p_t

                # Observe response (simulated)
                y_t = (target_p[t] >= p_t).astype(int)
                collected_data.append((p_t, x_t, y_t))

                # Update covariance matrix
                A += np.outer(x_t, x_t)

            m += 1
            if end_t >= T - 1:
                break

        return {
            'prices': prices,
            'betas': betas,
            'ucbs': ucbs
        }



class LinUCB_Offline2Online:
    def __init__(self, alpha: float = 1.0, F_dist: str = 'logistic', W: float = 1.0, noise_B=1.0):
        """
        Online-to-Online Transfer LinUCB with two-stage estimation

        Parameters:
        alpha: Exploration parameter for UCB
        F_dist: Noise distribution type ('logistic' or 'normal')
        W: Bound for feature/parameter magnitudes
        """
        self.alpha = alpha
        self.F_dist = F_dist
        self.W = W
        self.noise_B = noise_B
        self._init_F_functions()

    def _init_F_functions(self):
        """Initialize distribution and pricing functions"""
        if self.F_dist == 'logistic':
            self.F = logistic.cdf
            self.F_pdf = logistic.pdf
        elif self.F_dist == 'normal':
            self.F = norm.cdf
            self.F_pdf = norm.pdf
        else:
            raise ValueError("Supported distributions: 'logistic' or 'normal'")

        self.h = lambda u: self._solve_h(u)

    def _solve_h(self, u):
        """Optimal pricing function"""

        def phi(z):
            return z - (1 - self.F(z)) / (self.F_pdf(z) + 1e-10)

        res = minimize_scalar(
            lambda z: np.abs(phi(z) + u),
            bounds=(-self.noise_B, self.noise_B),
            method='bounded'
        )
        return u + res.x

    def _linucb_estimate(self, X: np.ndarray, p: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Estimate parameters using LinUCB-style ridge regression"""
        d = X.shape[1]
        A = np.eye(d)  # Regularization matrix
        b = np.zeros(d)

        for t in range(len(X)):
            x_t = X[t]
            A += np.outer(x_t, x_t)
            b += x_t * p[t] * y[t]

        return np.linalg.solve(A, b)

    def fit(self,
            source_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
            target_X: np.ndarray,
            target_p: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute two-stage online transfer learning with LinUCB

        Parameters:
        source_data: List of (prices, features, responses) tuples for source markets
        target_X: Target market features [n_samples, n_features]
        target_p: True market values (for simulation)

        Returns:
        {
            'prices': Price decisions,
            'betas': Parameter estimates,
            'ucbs': UCB values
        }
        """
        T = len(target_X)
        d = target_X.shape[1]
        prices = np.zeros(T)
        betas = []
        ucbs = []
        collected_data = []

        # 1. Stage 1: Estimate β_ag from source markets (LinUCB)
        X_source = np.concatenate([d[1] for d in source_data])
        p_source = np.concatenate([d[0] for d in source_data])
        y_source = np.concatenate([d[2] for d in source_data])

        beta_ag = self._linucb_estimate(X_source, p_source, y_source)

        # Episode-based processing
        m = 1
        while True:
            prev_start_t = 2 ** (m - 2)-1 if m > 1 else 0
            prev_end_t = 2 ** (m - 1) -1
            start_t = 2 ** (m - 1)
            end_t = min(2 ** m - 1, T)


            # 2. Stage 2: Estimate δ from target market (LinUCB)
            if m > 1:
                # Get target data from previous episode
                prev_data = collected_data[prev_start_t:prev_end_t + 1]
                X_target = np.array([d[1] for d in prev_data])
                p_target = np.array([d[0] for d in prev_data])
                y_target = np.array([d[2] for d in prev_data])

                # Estimate δ using LinUCB on adjusted responses
                # Create "pseudo-responses" for δ estimation
                pseudo_p = p_target - X_target @ beta_ag
                delta = self._linucb_estimate(X_target, pseudo_p, y_target)

                beta_m = beta_ag + delta
            else:
                # First episode uses β_ag only
                beta_m = beta_ag

            betas.append(beta_m.copy())

            # 3. Apply pricing with UCB exploration
            A = np.eye(d)  # Current covariance matrix
            for t in range(start_t, end_t + 1):
                if t >= T:
                    break

                x_t = target_X[t]
                A_inv = np.linalg.inv(A)

                # Compute UCB
                ucb = x_t @ beta_m + self.alpha * np.sqrt(x_t @ A_inv @ x_t)
                ucbs.append(ucb)

                # Set price
                p_t = self.h(ucb)
                prices[t] = p_t

                # Observe response (simulated)
                y_t = (target_p[t] >= p_t).astype(int)
                collected_data.append((p_t, x_t, y_t))

                # Update covariance matrix
                A += np.outer(x_t, x_t)

            m += 1
            if end_t >= T - 1:
                break

        return {
            'prices': prices,
            'betas': betas,
            'ucbs': ucbs
        }