import matplotlib.pyplot as plt
from gpytorch.metrics import mean_absolute_error
from tqdm import tqdm
from utils.utils import *
from utils.config import *
from algorithm.on2on import Online2Online
from algorithm.off2on import Offline2Online
from algorithm.RMLP import RMLP
from algorithm.LinUCB import LinUCB_Pricing, LinUCB_Online2Online, LinUCB_Offline2Online



def run_comparison(config_list: dict, name: str, n_sim: int = 1):
    """Compare algorithms across different market scenarios"""
    # Initialize results storage for all algorithms
    # algorithms = ['rmlp', 'offline', 'online', 'linucb', 'linucb_online', 'linucb_offline']
    algorithms = ['rmlp', 'offline', 'online']

    for j in range(50):
        results = {}
        for key, config in config_list.items():
            one_result = {alg: {'regret': [], 'beta_error': []} for alg in algorithms}
            print(key)
            for _ in tqdm(range(n_sim), desc=config['description']):
                config['seed'] = j

                # Generate target market (always uses current time period)
                target_X, _, _, target_v, true_beta = MarketSimulator.generate_market_data(config)

                # Generate source markets
                offline_source_data = []
                online_source_data = []
                off_config = config.copy()
                if 'off_T' in config:
                    off_config['T'] = config['off_T']
                X_k, p_k, y_k, _, _ = MarketSimulator.generate_market_data(off_config)
                offline_source_data.append((p_k, X_k, y_k))
                for k in range(config['K']):
                    config['seed'] = config['seed']+(k+1)
                    X_k, p_k, y_k, _ , _ = MarketSimulator.generate_market_data(config)
                    online_source_data.append((p_k, X_k, y_k))

                # Initialize all models
                models = {
                    'offline': Offline2Online(F_dist=F_dist, W=W),
                    'online': Online2Online(F_dist=F_dist, W=W),
                    'rmlp': RMLP(F_dist=F_dist, W=W),
                    # 'linucb': LinUCB_Pricing(alpha=1.0, F_dist=F_dist),
                    # 'linucb_online': LinUCB_Online2Online(alpha=1.0, F_dist=F_dist),
                    # 'linucb_offline': LinUCB_Offline2Online(alpha=1.0, F_dist=F_dist)
                }

                # Run all algorithms
                for alg_name, model in models.items():
                    if alg_name == 'offline':
                        alg_results = model.fit(offline_source_data, target_X, target_v)
                    elif alg_name == 'online':
                        alg_results = model.fit(online_source_data, target_X, target_v)
                    elif alg_name == 'linucb_offline':
                        alg_results = model.fit(offline_source_data, target_X, target_v)
                    elif alg_name == 'linucb_online':
                        alg_results = model.fit(online_source_data, target_X, target_v)
                    else:  # rmlp and basic linucb
                        alg_results = model.fit(target_X, target_v)

                    regret = compute_regret(alg_results['prices'], target_X, true_beta, model)
                    one_result[alg_name]['regret'].append(regret)
            results[key] = one_result

        # Compute statistics
        stats = {}
        for key in config_list.keys():
            stats[key] = {}
            for alg in algorithms:
                regret_data = results[key][alg]['regret']  # shape: (n_runs, n_periods)

                mean_regret = np.mean(regret_data, axis=0)
                std_regret = np.std(regret_data, axis=0)

                n_periods = config['T']
                periods = np.arange(1, n_periods + 1)  # 1, 2, ..., n_periods
                avg_regret = mean_regret / periods  # 累积平均 regret
                avg_std = std_regret / periods

                stats[key][alg] = {
                    'mean_regret': mean_regret,  # 每个 period 的平均 regret
                    'std_regret': std_regret,  # 每个 period 的 regret 标准差
                    'avg_regret': avg_regret,  # 每个 period 的累积平均 regret
                    'avg_std': avg_std,
                }

        plot_and_save_results(stats, name=name, kk=j)

def plot_and_save_results(results: dict, name: str, kk:int):
    """Visualize and save comparison results"""
    plt.figure(figsize=(8, 6))

    alg_styles = {
        'offline': ('blue', '-', 'Offline to Online'),
        'online': ('green', '-', 'Online to Online'),
        'rmlp': ('#38ACEC', '--', 'RMLP'),
        'linucb': ('#FFD801', '--', 'LinUCB'),
        'linucb_online': ('#F70D1A', '-', 'LinUCB Online'),
        'linucb_offline': ('#F87217', '-', 'LinUCB Offline')
    }

    # pair_list = [('rmlp', 'online'), ('rmlp', 'offline'), ('linucb', 'linucb_online'), ('linucb', 'linucb_offline')]
    pair_list = [('rmlp', 'online'), ('rmlp', 'offline')]
    key_list = list(results.keys())

    for pair in pair_list:
        base, trans = pair

        mean_base = results[key_list[0]][base]['mean_regret']
        std_base = results[key_list[0]][base]['std_regret']
        plt.plot(mean_base, label=f'{alg_styles[base][2]}',
                 color=alg_styles[base][0], linestyle=alg_styles[base][1])
        plt.fill_between(range(len(mean_base)), mean_base - std_base, mean_base + std_base,
                         alpha=0.1, color=alg_styles[base][0])
        plt.text(len(mean_base), mean_base[-1], f'{base}')
        for i, one_key in enumerate(key_list):
            mean = results[one_key][trans]['mean_regret']
            std = results[one_key][trans]['std_regret']
            plt.plot(mean, label=f'{alg_styles[trans][2]}',
                     color=alg_styles[trans][0], linestyle=alg_styles[trans][1])
            plt.fill_between(range(len(mean)), mean - std, mean + std,
                             alpha=0.1, color=alg_styles[trans][0])
            if trans in ['offline', 'linucb_offline']:
                plt.text(len(mean), mean[-1], f'$n_0={50+150*i}$')
            if trans in ['online', 'linucb_online']:
                plt.text(len(mean), mean[-1], f'$M={1+3*i}$')

        plt.xlabel('Time Period')
        plt.ylabel('Cumulative Regret')
        plt.xlim(-50, len(mean) * 1.25)
        plt.grid(True)

        # Save plot
        filename = f"fig/cumulative_{name}_{base}_{trans}_{kk}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {filename}")

        # mean_base = results[key_list[0]][base]['avg_regret']
        # std_base = results[key_list[0]][base]['avg_std']
        # plt.plot(mean_base, label=f'{alg_styles[base][2]}',
        #          color=alg_styles[base][0], linestyle=alg_styles[base][1])
        # plt.fill_between(range(len(mean_base)), mean_base - std_base, mean_base + std_base,
        #                  alpha=0.1, color=alg_styles[base][0])
        # plt.text(len(mean_base), mean_base[-1], f'{base}')
        # for one_key in key_list:
        #     mean = results[one_key][trans]['avg_regret']
        #     std = results[one_key][trans]['avg_std']
        #     plt.plot(mean, label=f'{alg_styles[trans][2]}',
        #              color=alg_styles[trans][0], linestyle=alg_styles[trans][1])
        #     plt.fill_between(range(len(mean)), mean - std, mean + std,
        #                      alpha=0.1, color=alg_styles[trans][0])
        #     if trans in ['offline', 'linucb_offline']:
        #         plt.text(len(mean), mean[-1], f'$n_0={50+150*i}$')
        #     if trans in ['online', 'linucb_online']:
        #         plt.text(len(mean), mean[-1], f'$M={1+3*i}$')
        #
        # plt.xlabel('Time Period')
        # plt.ylabel('Average Regret')
        # plt.xlim(-50,len(mean) * 1.25)
        # plt.grid(True)
        #
        # # Save plot
        # filename = f"fig/average_{name}_{base}_{trans}.png"
        # plt.savefig(filename, dpi=300, bbox_inches='tight')
        # plt.close()
        # print(f"Saved plot to {filename}")



if __name__ == "__main__":
    all_results = {}
    for key, config in config_dict.items():
        print(f"\nRunning {key} ...")
        run_comparison(config, name=key)
