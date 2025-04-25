import numpy as np
from scipy.stats import logistic
from typing import List, Tuple, Dict


class MarketSimulator:
    @staticmethod
    def generate_beta(d: int, W: float, sparsity: float) -> np.ndarray:
        """Generate sparse beta vector with ||beta||_1 <= W"""
        beta = np.zeros(d)
        non_zero = max(1, int(d * sparsity))
        beta[:non_zero] = np.random.randn(non_zero)
        beta = np.abs(beta * W / (np.sum(np.abs(beta)) + 1e-10))
        return beta

    @staticmethod
    def generate_market_data(config: dict) -> Tuple:
        """Generate data for a single market with guaranteed class balance"""
        np.random.seed(config['seed'])
        d = config['d']
        T = config['T']

        # Generate features with bounded support (||x||_∞ ≤ 1)
        X = np.random.uniform(0.3, 1, size=(T, d))

        # Generate beta based on configuration
        if config['scenario'] == 'identical':
            beta = config['base_beta']
        elif config['scenario'] == 'sparse_difference':
            delta = MarketSimulator.generate_beta(d, config['delta_W'], config['delta_sparsity'])
            beta = config['base_beta'] + delta

        if config.get('F_dist', 'logistic') == 'logistic':
            z = np.random.logistic(scale=1, size=T)
        else:
            z = np.random.normal(scale=1, size=T)

        X_beta = X @ beta
        z_min, z_max = max(-np.min(X_beta), -1), min(np.max(X_beta), 1)
        z = np.clip(z, z_min, z_max)
        v = X_beta + z

        perturbation = v * (np.random.rand(*v.shape) - 0.5)
        p = v + perturbation

        # Generate responses with guaranteed class balance
        max_attempts = 10
        for _ in range(max_attempts):
            y = (v >= p).astype(int)
            if len(np.unique(y)) >= 2:  # Ensure we have both classes
                return X, p, y, v, beta
            p = p * 0.8  # Adjust prices if we get all zeros or ones

        return X, p, y, v, beta



def compute_regret(algo_p: np.ndarray, target_X: np.ndarray,
                   true_beta: np.ndarray, model) -> np.ndarray:
    """Calculate cumulative regret with proper linear utility calculation"""
    # Calculate optimal prices using true beta
    utility = target_X @ true_beta  # Linear utility calculation
    opt_p = np.array([model.h(u) for u in utility])  # Apply pricing function to utility

    # Calculate revenue probabilities (clipped for numerical stability)
    eps = 1e-10
    opt_prob = np.clip(1 - model.F(opt_p - utility), eps, 1 - eps)
    algo_prob = np.clip(1 - model.F(algo_p - utility), eps, 1 - eps)

    # Calculate revenues
    opt_rev = opt_p * opt_prob
    algo_rev = algo_p * algo_prob

    regret = np.minimum(opt_rev - algo_rev, 2)

    return np.cumsum(regret)