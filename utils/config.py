from utils.utils import MarketSimulator

F_dist = 'logistic'  # 'normal'
W = 2
# Configuration Setup

CONFIGURATIONS_idt_10 = {
    'identical_1': {
        'd': 10, 'K': 1, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
    'identical_2': {
        'd': 10, 'K': 4, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
    'identical_3': {
        'd': 10, 'K': 7, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 350,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
    'identical_4': {
        'd': 10, 'K': 10, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
}

CONFIGURATIONS_idt_15 = {
    'identical_1': {
        'd': 15, 'K': 1, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
    'identical_2': {
        'd': 15, 'K': 4, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
    'identical_3': {
        'd': 15, 'K': 7, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 350,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
    'identical_4': {
        'd': 15, 'K': 10, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
}

CONFIGURATIONS_idt_20 = {
    'identical_1': {
        'd': 20, 'K': 1, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
    'identical_2': {
        'd': 20, 'K': 4, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
    'identical_3': {
        'd': 20, 'K': 7, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 350,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
    'identical_4': {
        'd': 20, 'K': 10, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'identical',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.6),
        'description': 'All markets share identical beta'
    },
}

CONFIGURATIONS_sparse_10 = {
    'sparse_1': {
        'd': 10, 'K': 1, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s1-sparse)'
    },
    'sparse_2': {
        'd': 10, 'K': 5, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
    'sparse_3': {
        'd': 10, 'K': 10, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 350,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
    'sparse_4': {
        'd': 10, 'K': 15, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(10, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
}

CONFIGURATIONS_sparse_15 = {
    'sparse_1': {
        'd': 15, 'K': 1, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
    'sparse_2': {
        'd': 15, 'K': 5, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
    'sparse_3': {
        'd': 15, 'K': 10, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 350,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
    'sparse_4': {
        'd': 15, 'K': 15, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(15, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
}

CONFIGURATIONS_sparse_20 = {
    'sparse_1': {
        'd': 20, 'K': 1, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 50,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
    'sparse_2': {
        'd': 20, 'K': 5, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 200,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
    'sparse_3': {
        'd': 20, 'K': 10, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 350,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
    'sparse_4': {
        'd': 20, 'K': 15, 'T':2000, 'seed': 42, 'F_dist': F_dist,
        'off_T': 500,  # Source market data in Offline2Online setting
        'scenario': 'sparse_difference',
        'base_beta': MarketSimulator.generate_beta(20, W=W, sparsity=0.6),
        'delta_W': 0.3,  # ||β⁰ - βᵏ||₁ ≤ 0.3
        'delta_sparsity': 0.2,  # At most 20% non-zero differences
        'description': 'Markets have sparse differences in beta (s₂-sparse)'
    },
}

# config_dict = {'cfg_idt_10': CONFIGURATIONS_idt_10,
#                'cfg_idt_15': CONFIGURATIONS_idt_15, 'cfg_idt_20':CONFIGURATIONS_idt_20,
#                'cfg_sparse_10': CONFIGURATIONS_sparse_10,
#                'cfg_sparse_15': CONFIGURATIONS_sparse_15, 'cfg_sparse_20': CONFIGURATIONS_sparse_20}

config_dict = {'cfg_sparse_10': CONFIGURATIONS_sparse_10,
               'cfg_sparse_15': CONFIGURATIONS_sparse_15, 'cfg_sparse_20': CONFIGURATIONS_sparse_20}