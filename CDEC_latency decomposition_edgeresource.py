import numpy as np
import matplotlib.pyplot as plt
from time import time


class TaskScheduler:
    def __init__(self, I=30, N_device=30, N_edge=100):
        """初始化系统参数"""

        self.I = I
        self.N_device = N_device
        self.N_edge = N_edge

        # 任务参数
        self.n_device = np.random.randint(1, 5, I)
        self.n_edge = np.random.randint(2, 6, I)

        self.t_device = np.random.uniform(0.5, 2.0, I)
        self.t_edge = np.random.uniform(1.0, 3.0, I)
        self.t_cloud = np.random.uniform(2.0, 5.0, I)

        self.T_max = np.random.uniform(3.0, 6.0, I)

    def evaluate(self, solution, alpha=1e6):

        total_latency = 0.0
        R_device = 0
        R_edge = 0
        time_violations = 0

        task_details = []

        for i in range(self.I):

            task_info = {
                'task_id': i,
                'location': None,
                'latency': 0
            }

            if solution[i, 0] == 1:

                latency = self.t_device[i]
                R_device += self.n_device[i]
                task_info['location'] = 'Device'

            else:

                if solution[i, 1] == 1:

                    latency = self.t_edge[i]
                    R_edge += self.n_edge[i]
                    task_info['location'] = 'Edge'

                else:

                    latency = self.t_cloud[i]
                    task_info['location'] = 'Cloud'

            task_info['latency'] = latency

            if latency > self.T_max[i]:
                time_violations += 1

            total_latency += latency
            task_details.append(task_info)

        penalty = alpha * (
                max(0, R_device - self.N_device)
                + max(0, R_edge - self.N_edge)
                + time_violations
        )

        return {
            'total_latency': total_latency,
            'task_details': task_details,
            'fitness': total_latency + penalty
        }

    def genetic_algorithm(self, pop_size=80, max_gen=150):

        population = np.random.randint(0, 2, (pop_size, self.I, 2))

        best_solution = None
        best_fitness = float('inf')

        for gen in range(max_gen):

            fitness_values = []

            for ind in population:
                fitness_values.append(self.evaluate(ind)['fitness'])

            best_idx = np.argmin(fitness_values)

            if fitness_values[best_idx] < best_fitness:

                best_fitness = fitness_values[best_idx]
                best_solution = population[best_idx].copy()

            new_population = []

            for _ in range(pop_size):

                candidates = population[np.random.choice(pop_size, 3, replace=False)]

                winner = candidates[np.argmin(
                    [self.evaluate(c)['fitness'] for c in candidates]
                )]

                new_population.append(winner.copy())

            new_population = np.array(new_population)

            # crossover
            crossover_rate = 0.8 - 0.3 * (gen / max_gen)

            for i in range(0, pop_size, 2):

                if np.random.rand() < crossover_rate and i + 1 < pop_size:

                    cp1, cp2 = sorted(np.random.choice(self.I, 2))

                    temp = new_population[i, cp1:cp2, :].copy()

                    new_population[i, cp1:cp2, :] = new_population[i + 1, cp1:cp2, :]
                    new_population[i + 1, cp1:cp2, :] = temp

            # mutation
            mutation_rate = 0.1 * (1 - gen / max_gen)

            for i in range(pop_size):

                for j in range(self.I):

                    if np.random.rand() < mutation_rate:

                        bit = np.random.randint(0, 2)
                        new_population[i, j, bit] = 1 - new_population[i, j, bit]

            population = new_population

        final_eval = self.evaluate(best_solution)

        return best_solution, final_eval


# =====================================================
# Latency decomposition vs Edge resources
# =====================================================

def latency_vs_edge_resources(edge_resources, I=30, N_device=30):

    device_lat, edge_lat, cloud_lat = [], [], []

    print("\nEdgeRes | Device Latency | Edge Latency | Cloud Latency")
    print("-----------------------------------------------------------")

    for edge_res in edge_resources:

        scheduler = TaskScheduler(I=I, N_device=N_device, N_edge=edge_res)

        best_solution, evaluation = scheduler.genetic_algorithm()

        task_details = evaluation['task_details']

        device_sum = sum(t['latency'] for t in task_details if t['location'] == 'Device')
        edge_sum = sum(t['latency'] for t in task_details if t['location'] == 'Edge')
        cloud_sum = sum(t['latency'] for t in task_details if t['location'] == 'Cloud')

        # 平均任务时延（SCI更常用）
        device_avg = device_sum / I
        edge_avg = edge_sum / I
        cloud_avg = cloud_sum / I

        device_lat.append(device_avg)
        edge_lat.append(edge_avg)
        cloud_lat.append(cloud_avg)

        print(f"{edge_res:<7} | {device_avg:<15.2f} | {edge_avg:<12.2f} | {cloud_avg:<12.2f}")

    # ========================
    # 画图
    # ========================

    x = np.arange(len(edge_resources))
    width = 0.25

    plt.figure()

    plt.bar(x - width, device_lat, width, label='Device', color='#1f77b4')
    plt.bar(x, edge_lat, width, label='Edge', color='#FF4500')
    plt.bar(x + width, cloud_lat, width, label='Cloud', color='#2ca02c')

    plt.xticks(x, edge_resources)

    plt.xlabel("Edge Resources", fontsize=12, fontweight='bold')
    plt.ylabel("Latency (s)", fontsize=12, fontweight='bold')

    plt.title("Latency Decomposition vs Edge Resources", fontsize=12, fontweight='bold')

    plt.legend()

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    plt.show()


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    np.random.seed(42)   # 只设置一次随机种子

    edge_resources = [20, 40, 60, 80, 100]

    latency_vs_edge_resources(edge_resources)