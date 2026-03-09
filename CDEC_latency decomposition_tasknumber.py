import numpy as np
import matplotlib.pyplot as plt
from time import time

class TaskScheduler:
    def __init__(self, I=30, N_device=30, N_edge=100):
        """初始化系统参数"""
        self.I = I  # 总任务数
        self.N_device = N_device  # 设备端总资源
        self.N_edge = N_edge  # 边缘端总资源

        # 随机生成任务参数（固定随机种子保证可重复性）
        np.random.seed(42)
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
            task_info = {'task_id': i, 'location': None, 'latency': 0, 'T_max': self.T_max[i], 'violation': False}
            if solution[i,0] == 1:  # Device
                latency = self.t_device[i]
                R_device += self.n_device[i]
                task_info['location'] = 'Device'
            else:
                if solution[i,1] == 1:  # Edge
                    latency = self.t_edge[i]
                    R_edge += self.n_edge[i]
                    task_info['location'] = 'Edge'
                else:
                    latency = self.t_cloud[i]
                    task_info['location'] = 'Cloud'

            task_info['latency'] = latency
            if latency > self.T_max[i]:
                task_info['violation'] = True
                time_violations += 1
            total_latency += latency
            task_details.append(task_info)

        resource_violations = 0
        res_vio_details = []
        if R_device > self.N_device:
            resource_violations += 1
            res_vio_details.append(f"Device超限({R_device}/{self.N_device})")
        if R_edge > self.N_edge:
            resource_violations += 1
            res_vio_details.append(f"Edge超限({R_edge}/{self.N_edge})")

        penalty = alpha * (max(0,R_device-self.N_device) + max(0,R_edge-self.N_edge) + time_violations)

        return {
            'total_latency': total_latency,
            'resource_usage': (R_device, R_edge),
            'time_violations': time_violations,
            'resource_violations': resource_violations,
            'task_details': task_details,
            'res_vio_details': res_vio_details,
            'fitness': total_latency + penalty
        }

    def generate_baseline_solution(self, strategy):
        solution = np.zeros((self.I,2),dtype=int)
        if strategy=='edge-only':
            solution[:,0]=0
            solution[:,1]=1
        elif strategy=='cloud-only':
            solution[:,0]=0
            solution[:,1]=0
        elif strategy=='edge-device':
            device_resources = self.N_device
            for i in range(self.I):
                if self.n_device[i]<=device_resources:
                    solution[i,0]=1
                    device_resources-=self.n_device[i]
                else:
                    solution[i,0]=0
                    solution[i,1]=1
        elif strategy=='cloud-edge':
            edge_resources=self.N_edge
            for i in range(self.I):
                if self.n_edge[i]<=edge_resources:
                    solution[i,0]=0
                    solution[i,1]=1
                    edge_resources-=self.n_edge[i]
                else:
                    solution[i,0]=0
                    solution[i,1]=0
        return solution

    def genetic_algorithm(self, pop_size=80, max_gen=150):
        population = np.random.randint(0,2,(pop_size,self.I,2))
        best_solution = None
        best_fitness = float('inf')
        convergence = []

        for gen in range(max_gen):
            evaluations = [self.evaluate(ind) for ind in population]
            fitness_values = [e['fitness'] for e in evaluations]

            current_best_idx = np.argmin(fitness_values)
            if fitness_values[current_best_idx]<best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_solution = population[current_best_idx].copy()
                convergence.append(best_fitness)

            new_population=[]
            for _ in range(pop_size):
                candidates = population[np.random.choice(pop_size,3,replace=False)]
                winner = candidates[np.argmin([self.evaluate(c)['fitness'] for c in candidates])]
                new_population.append(winner.copy())
            new_population=np.array(new_population)

            crossover_rate=0.8-0.3*(gen/max_gen)
            for i in range(0,pop_size,2):
                if np.random.rand()<crossover_rate and i+1<pop_size:
                    cross_points=sorted(np.random.choice(self.I,2))
                    temp=new_population[i,cross_points[0]:cross_points[1],:].copy()
                    new_population[i,cross_points[0]:cross_points[1],:]=new_population[i+1,cross_points[0]:cross_points[1],:]
                    new_population[i+1,cross_points[0]:cross_points[1],:]=temp

            mutation_rate=0.1*(1-gen/max_gen)
            for i in range(pop_size):
                for j in range(self.I):
                    if np.random.rand()<mutation_rate:
                        bit=np.random.randint(0,2)
                        new_population[i,j,bit]=1-new_population[i,j,bit]
            population=new_population

        final_eval=self.evaluate(best_solution)
        return best_solution, final_eval, convergence

    def benchmark_strategies(self, strategies):
        results={}
        for strat in strategies:
            start_time = time()
            solution=self.generate_baseline_solution(strat)
            eval_result=self.evaluate(solution)
            results[strat]={'evaluation':eval_result,'time':time()-start_time}
        return results



# 假设 TaskScheduler 类已经定义，包含 evaluate 和 genetic_algorithm 方法

def latency_decomposition_vs_tasknum(task_nums, N_device=30, N_edge=100):
    """
    对 CDEC 算法，在不同任务数下分析 Device/Edge/Cloud 延迟贡献（独立柱状图）
    """
    device_lat_total, edge_lat_total, cloud_lat_total = [], [], []

    for I in task_nums:
        scheduler = TaskScheduler(I=I, N_device=N_device, N_edge=N_edge)
        best_solution, evaluation, _ = scheduler.genetic_algorithm(pop_size=80, max_gen=150)
        task_details = evaluation['task_details']

        device_sum = sum(t['latency'] for t in task_details if t['location']=='Device')
        edge_sum   = sum(t['latency'] for t in task_details if t['location']=='Edge')
        cloud_sum  = sum(t['latency'] for t in task_details if t['location']=='Cloud')

        device_lat_total.append(device_sum)
        edge_lat_total.append(edge_sum)
        cloud_lat_total.append(cloud_sum)
        # ===== 打印结果 =====
        print(f"{I:<8} | {device_sum:<15.2f} | {edge_sum:<15.2f} | {cloud_sum:<15.2f}")

    # 横坐标位置
    x = np.arange(len(task_nums))
    bar_width = 0.25

    plt.figure()
    plt.bar(x - bar_width, device_lat_total, width=bar_width, color='#1f77b4', label='Device')
    plt.bar(x, edge_lat_total, width=bar_width, color='#FF4500', label='Edge')
    plt.bar(x + bar_width, cloud_lat_total, width=bar_width, color='#2ca02c', label='Cloud')

    plt.xticks(x, [str(n) for n in task_nums])
    plt.xlabel("Number of Tasks", fontsize=12, fontweight='bold')
    plt.ylabel("Latency (s)", fontsize=12, fontweight='bold')
    plt.title("CDEC Latency Decomposition vs Task Numbers", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ---------- 主程序 ----------
if __name__ == "__main__":
    task_nums = [10, 20, 30, 40, 50]  # 不同任务数实验
    latency_decomposition_vs_tasknum(task_nums)