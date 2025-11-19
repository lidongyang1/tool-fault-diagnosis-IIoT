"""
mixed_integer_scheduler.py
混合整数规划问题求解器（含多策略对比版）
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time


class TaskScheduler:
    def __init__(self, I=20, N_device=100, N_edge=500):
        """初始化系统参数"""
        self.I = I  # 总任务数
        self.N_device = N_device  # 设备端总资源
        self.N_edge = N_edge  # 边缘端总资源

        # 随机生成任务参数（固定随机种子保证可重复性）
        np.random.seed(42)
        self.n_device = np.random.randint(1, 5, I)  # 各任务设备资源需求
        self.n_edge = np.random.randint(2, 6, I)  # 各任务边缘资源需求
        self.t_device = np.random.uniform(0.5, 2.0, I)  # 设备执行时延
        self.t_edge = np.random.uniform(1.0, 3.0, I)  # 边缘执行时延
        self.t_cloud = np.random.uniform(2.0, 5.0, I)  # 云端执行时延
        self.T_max = np.random.uniform(3.0, 6.0, I)  # 各任务最大允许时延

    def evaluate(self, solution, alpha=1e6):
        epsilon, xi = solution[:, 0], solution[:, 1]
        total_latency = 0.0
        R_device = 0
        R_edge = 0
        time_violations = 0
        fallback_count = 0
        task_details = []

        for i in range(self.I):
            task_info = {
                'task_id': i,
                'location': None,
                'latency': 0,
                'T_max': self.T_max[i],
                'violation': False,
                'fallback': False  # 新增：标记是否因资源不足回退
            }

            # 实时资源检查分配逻辑
            if epsilon[i] == 1:  # 尝试设备执行
                if R_device + self.n_device[i] <= self.N_device:
                    latency = self.t_device[i]
                    R_device += self.n_device[i]
                    task_info['location'] = 'Device'
                else:  # 设备资源不足回退云端
                    latency = self.t_cloud[i]
                    task_info['location'] = 'Cloud (Fallback)'
                    task_info['fallback'] = True
                    fallback_count += 1
            else:
                if xi[i] == 1:  # 尝试边缘执行
                    if R_edge + self.n_edge[i] <= self.N_edge:
                        latency = self.t_edge[i]
                        R_edge += self.n_edge[i]
                        task_info['location'] = 'Edge'
                    else:  # 边缘资源不足回退云端
                        latency = self.t_cloud[i]
                        task_info['location'] = 'Cloud (Fallback)'
                        task_info['fallback'] = True
                        fallback_count += 1
                else:  # 正常云端执行
                    latency = self.t_cloud[i]
                    task_info['location'] = 'Cloud'

            # 时延违规检查
            if latency > self.T_max[i]:
                time_violations += 1
                task_info['violation'] = True

            total_latency += latency
            task_details.append(task_info)

        # 更新后资源违规始终为0（已通过回退机制保证）
        penalty = alpha * (time_violations + fallback_count * 0.5)  # 新增回退惩罚项

        return {
            'total_latency': total_latency,
            'resource_usage': (R_device, R_edge),
            'time_violations': time_violations,
            'resource_violations': 0,  # 资源违规已通过回退机制消除
            'fallback_count': fallback_count,  # 新增关键指标
            'task_details': task_details,
            'fitness': total_latency + penalty
        }

    # 新增基准策略生成方法
    def generate_baseline_solution(self, strategy):
        """生成基准策略的解决方案"""
        solution = np.zeros((self.I, 2), dtype=int)

        if strategy == 'cloud-only':
            solution[:, 0] = 0  # 非设备执行
            solution[:, 1] = 0  # 强制云端

        elif strategy == 'edge-device':
            # 边端协同：优先设备，资源不足则边缘
            device_resources = self.N_device
            for i in range(self.I):
                if self.n_device[i] <= device_resources:
                    solution[i, 0] = 1
                    device_resources -= self.n_device[i]
                else:
                    solution[i, 0] = 0
                    solution[i, 1] = 1  # 分配到边缘
        elif strategy == 'cloud-device':
            # 边端协同：优先设备，资源不足则边缘
            device_resources = self.N_device
            for i in range(self.I):
                if self.n_device[i] <= device_resources:
                    solution[i, 0] = 1
                    device_resources -= self.n_device[i]
                else:
                    solution[i, 0] = 0
                    solution[i, 1] = 0  # 分配到边缘

        elif strategy == 'cloud-edge':
            # 云边协同：优先边缘，资源不足则云端
            edge_resources = self.N_edge
            for i in range(self.I):
                if self.n_edge[i] <= edge_resources:
                    solution[i, 0] = 0
                    solution[i, 1] = 1
                    edge_resources -= self.n_edge[i]
                else:
                    solution[i, 0] = 0
                    solution[i, 1] = 0
        return solution

    def genetic_algorithm(self, pop_size=100, max_gen=200):
        """自适应遗传算法主程序"""
        # 初始化种群 (I x 2矩阵)
        population = np.random.randint(0, 2, (pop_size, self.I, 2))
        best_solution = None
        best_fitness = float('inf')
        convergence = []

        for gen in range(max_gen):
            # 评估种群
            evaluations = [self.evaluate(ind) for ind in population]
            fitness_values = [e['fitness'] for e in evaluations]

            # 更新最优解
            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx] < best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_solution = population[current_best_idx].copy()
                convergence.append(best_fitness)

            # 锦标赛选择
            new_population = []
            for _ in range(pop_size):
                candidates = population[np.random.choice(pop_size, 3, replace=False)]
                winner = candidates[np.argmin([self.evaluate(c)['fitness'] for c in candidates])]
                new_population.append(winner.copy())
            new_population = np.array(new_population)

            # 两点交叉（自适应交叉率）
            crossover_rate = 0.8 - 0.3 * (gen / max_gen)  # 动态调整
            for i in range(0, pop_size, 2):
                if np.random.rand() < crossover_rate and i + 1 < pop_size:
                    cross_points = sorted(np.random.choice(self.I, 2))
                    # 交换中间段
                    temp = new_population[i, cross_points[0]:cross_points[1], :].copy()
                    new_population[i, cross_points[0]:cross_points[1], :] = \
                        new_population[i + 1, cross_points[0]:cross_points[1], :]
                    new_population[i + 1, cross_points[0]:cross_points[1], :] = temp

            # 自适应变异
            mutation_rate = 0.1 * (1 - gen / max_gen)  # 逐渐降低
            for i in range(pop_size):
                for j in range(self.I):
                    if np.random.rand() < mutation_rate:
                        # 随机选择变异位
                        bit = np.random.randint(0, 2)
                        new_population[i, j, bit] = 1 - new_population[i, j, bit]

            population = new_population

        # 最终评估
        final_eval = self.evaluate(best_solution)
        return best_solution, final_eval, convergence

    # 新增基准测试方法
    def benchmark_strategies(self, strategies):
        """运行所有基准策略对比"""
        results = {}
        for strategy in strategies:
            start_time = time()
            solution = self.generate_baseline_solution(strategy)
            eval_result = self.evaluate(solution)
            results[strategy] = {
                'evaluation': eval_result,
                'time': time() - start_time
            }
        return results


def run_scaling_experiment(param_name, param_values, default_I=30, default_N_device=150, default_N_edge=80):
    """
    运行参数扩展实验
    :param param_name: 要变化的参数名称 ('I', 'N_device', 'N_edge')
    :param param_values: 参数取值列表
    :return: 实验结果字典 {param_value: {strategy: metrics}}
    """
    results = {}
    strategies = ['Proposed GA', 'cloud-only', 'edge-device', 'cloud-device','cloud-edge']

    for value in param_values:
        # 根据参数设置实例化调度器
        if param_name == 'I':
            scheduler = TaskScheduler(I=value, N_device=default_N_device, N_edge=default_N_edge)
        elif param_name == 'N_device':
            scheduler = TaskScheduler(I=default_I, N_device=value, N_edge=default_N_edge)
        elif param_name == 'N_edge':
            scheduler = TaskScheduler(I=default_I, N_device=default_N_device, N_edge=value)
        else:
            raise ValueError("Invalid parameter name")

        # 运行遗传算法
        print(f"\nRunning {param_name}={value}...")
        start_time = time()
        best_solution, evaluation, _ = scheduler.genetic_algorithm(pop_size=80, max_gen=150)
        ga_time = time() - start_time

        # 运行基准策略
        baseline_results = scheduler.benchmark_strategies([
                                                           'cloud-only', 'edge-device','cloud-device',
                                                           'cloud-edge'])

        # 收集结果
        results[value] = {
            'proposed CDEC': {
                'total_latency': evaluation['total_latency'],
                'time_violations': evaluation['time_violations'],
                'resource_violations': evaluation['resource_violations']
            }
        }
        for strat in baseline_results:
            results[value][strat] = {
                'total_latency': baseline_results[strat]['evaluation']['total_latency'],
                'time_violations': baseline_results[strat]['evaluation']['time_violations'],
                'resource_violations': baseline_results[strat]['evaluation']['resource_violations']
            }

    return results


def plot_latency_comparison(all_results):
    """绘制时延对比曲线（独立图例+坐标控制）"""
    strategies = [
                  'cloud-only', 'edge-device', 'cloud-device','cloud-edge','proposed CDEC']
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'd', 'v', '*']
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]

    plt.figure()
    # plt.subplots_adjust(wspace=0.3)  # 调整子图间距

    param_config = [
        ('I', 'Task Numbers', [40, 50, 60,70,80,90])
    ]

    for idx, (param, title, values) in enumerate(param_config, 1):
        ax = plt.subplot(1, 1, idx)
        results = all_results[param]
        max_latency = 0

        # 绘制所有策略曲线
        for s_idx, strategy in enumerate(strategies):
            x = list(results.keys())
            y = [results[val][strategy]['total_latency'] for val in x]
            max_latency = max(max_latency, max(y))

            ax.plot(x, y,
                    color=colors[s_idx],
                    linestyle=line_styles[s_idx],
                    marker=markers[s_idx],
                    linewidth=2,
                    markersize=8,
                    label=strategy)

        ax.set_xlabel(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Total Latency (s)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(values)

        # 特殊坐标控制：设备/边缘资源图从0开始
        if param in ['N_device', 'N_edge']:
            ax.set_ylim(bottom=10, top=max_latency * 1.1)
        else:
            ax.set_ylim(auto=True)

        # 智能图例布局
        legend = ax.legend(
            loc='upper left' if param == 'I' else 'lower left',  #
            frameon=True,
            framealpha=0.9,
            edgecolor='#FFFFFF',
            ncol=2,
            fontsize=10,
            prop={'weight': 'bold'},
            borderpad=0.8,
            handlelength=2,
            handletextpad=0.5
        )
    # 如需进一步美化可添加以下设置
    plt.rcParams.update({
        'font.size': 12,  # 统一字体大小
        'axes.titlesize': 14,  # 子图标题字号
        'axes.labelsize': 12,  # 坐标轴标签字号
        'legend.fontsize': 10,  # 图例字号
        'figure.dpi': 150  # 输出分辨率
    })
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 运行不同参数的扩展实验
    param_config = [
        ('I', 'Task Quantity', [40, 50, 60,70,80,90])
    ]

    all_results = {}
    for param, label, values in param_config:
        print(f"\n=== Running {param} scaling experiment ===")
        results = run_scaling_experiment(param, values)
        all_results[param] = results

    # 新增结果输出
    target_I = 80
    print("\n=== 任务数量80时各方法性能对比 ===")
    print(f"{'策略':<15} | 总时延(s) | 时延违规 | 资源违规")
    print("-" * 45)

    # 获取目标数据
    target_data = all_results['I'][target_I]
    cdec_latency = target_data['proposed CDEC']['total_latency']

    # 打印详细数据
    for strategy in ['proposed CDEC', 'cloud-only', 'edge-device', 'cloud-device', 'cloud-edge']:
        data = target_data[strategy]
        print(
            f"{strategy:<15} | {data['total_latency']:8.2f} | {data['time_violations']:8d} | {data['resource_violations']:8d}")

    # 计算优化百分比
    print("\n=== 时延优化百分比 ===")
    for strategy in ['cloud-only', 'edge-device', 'cloud-device', 'cloud-edge']:
        base_latency = target_data[strategy]['total_latency']
        reduction = (base_latency - cdec_latency) / base_latency * 100
        print(f"相比 {strategy:<12}: {reduction:.1f}% 时延降低")

    plot_latency_comparison(all_results)