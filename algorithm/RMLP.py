import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import logistic, norm
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings("ignore")


class RMLP:
    def __init__(self, F_dist='logistic', noise_B=1.0, W=1.0):
        """
        Offline-to-online cross-market pricing transfer algorithm

        Parameters:
        F_dist: Noise distribution type ('logistic' or 'normal')
        noise_B: Support boundary for noise [-B, B]
        W: Bound for feature/parameter magnitudes (for u_F calculation)
        """
        self.F_dist = F_dist
        self.noise_B = noise_B
        self.W = W
        self._init_F_functions()
        self.u_F = self._compute_uF()  # Precompute u_F constant

    def _init_F_functions(self):
        """Initialize distribution and pricing functions"""
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

    def _solve_h(self, u):
        def phi(z):
            return z - (1 - self.F(z)) / (self.F_pdf(z) + 1e-10)

        res = minimize_scalar(
            lambda z: np.abs(phi(z) + u),
            bounds=(-self.noise_B, self.noise_B),
            method='bounded'
        )
        return u + res.x

    def _compute_uF(self):
        """Compute u_F = max{ log'F(-2W), -log'(1-F(2W)) }"""
        x1, x2 = -2 * self.W, 2 * self.W
        if self.F_dist == 'logistic':
            term1 = 1 - self.F(x1)  # For logistic: log'(F(x)) = 1-F(x)
            term2 = self.F(x2)  # For logistic: -log'(1-F(x)) = F(x)
        else:
            term1 = self.F_deriv(x1) / (self.F(x1) + 1e-10)
            term2 = self.F_deriv(x2) / (1 - self.F(x2) + 1e-10)
        return max(term1, term2)

    def _get_lambda(self, t: int, d: int) -> float:
        """Compute λ = 4u_F * sqrt(log(d)/t)"""
        return 4 * self.u_F * np.sqrt(np.log(d) / t)

    def fit(self,
            target_X: np.ndarray,
            target_p: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Execute offline-to-online pricing transfer

        Parameters:
        target_X: Target market features [n_samples, n_features]
        n0: Number of transfer learning episodes

        Returns:
        {
            'prices': Price decisions,
            'betas': Parameter estimates,
            'lambdas': Lambda values used
        }
        """

        # Initialize storage
        T = len(target_X)
        prices = np.zeros(T)
        betas = []
        lambdas = []
        collected_data = []

        # 1. Initial pricing (m=1)
        d = target_X.shape[1]
        collected_data.append((0, target_X[0], 0))

        # 2. Episode-based processing
        m = 2
        while True:
            prev_start_t = 2 ** (m - 2)
            prev_end_t = 2 ** (m - 1) - 1
            start_t = 2 ** (m - 1)
            end_t = min(2 ** m - 1, T)

            prev_data = collected_data[prev_start_t-1:prev_end_t]
            beta_init = np.zeros(d)

            res = minimize(
                fun=self._negative_log_likelihood,
                x0=beta_init,
                args=(
                    np.array([d[1] for d in prev_data]),
                    np.array([d[0] for d in prev_data]),
                    np.array([d[2] for d in prev_data])),
                method='L-BFGS-B',
                bounds=[(-self.W ,self.W)] * d,
                options={'maxiter': 1000}
            )

            if not res.success:
                raise ValueError(f"Optimization failed: {res.message}")

            beta_m = res.x

            betas.append(beta_m)

            # 3. Apply pricing
            for t in range(start_t-1, end_t):
                prices[t] = self.h(np.dot(target_X[t], beta_m))
                y_t = (target_p[t] >= self.h(np.dot(target_X[t], beta_m))).astype(int)
                collected_data.append((prices[t], target_X[t], y_t))

            if end_t == T:
                break
            m += 1

        return {
            'prices': prices,
            'betas': betas,
            'lambdas': lambdas
        }

    def _negative_log_likelihood(self, beta, X, p, y):
        """Negative loglikelihood (Eq.10)"""
        utility = p - X @ beta
        prob1 = 1 - self.F(utility)  # P(y=1)
        prob0 = self.F(utility)  # P(y=0)

        epsilon = 1e-10
        loss = -np.sum(
            (y == 1) * np.log(np.maximum(prob1, epsilon)) +
            (y == 0) * np.log(np.maximum(prob0, epsilon))
        )
        return loss / len(y)

