import numpy as np
import matplotlib.pyplot as plt
from time import time

class TaskScheduler:
    def __init__(self, I=30, N_device=30, N_edge=100):

        self.I = I
        self.N_device = N_device
        self.N_edge = N_edge

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

            task_info = {'task_id': i,'location': None,'latency': 0}

            if solution[i,0] == 1:
                latency = self.t_device[i]
                R_device += self.n_device[i]
                task_info['location'] = 'Device'

            else:
                if solution[i,1] == 1:
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

        penalty = alpha*(max(0,R_device-self.N_device)+max(0,R_edge-self.N_edge)+time_violations)

        return {
            'total_latency': total_latency,
            'task_details': task_details,
            'fitness': total_latency + penalty
        }

    def genetic_algorithm(self, pop_size=80, max_gen=150):

        population = np.random.randint(0,2,(pop_size,self.I,2))

        best_solution=None
        best_fitness=float('inf')

        for gen in range(max_gen):

            evaluations=[self.evaluate(ind) for ind in population]
            fitness_values=[e['fitness'] for e in evaluations]

            best_idx=np.argmin(fitness_values)

            if fitness_values[best_idx] < best_fitness:
                best_fitness=fitness_values[best_idx]
                best_solution=population[best_idx].copy()

            new_population=[]

            for _ in range(pop_size):

                candidates=population[np.random.choice(pop_size,3,replace=False)]
                winner=candidates[np.argmin([self.evaluate(c)['fitness'] for c in candidates])]
                new_population.append(winner.copy())

            population=np.array(new_population)

        final_eval=self.evaluate(best_solution)

        return best_solution, final_eval


# ===============================
# Device Resource Analysis
# ===============================

def latency_vs_device_resource(device_resources, I=30, N_edge=100):

    device_lat_total=[]
    edge_lat_total=[]
    cloud_lat_total=[]

    print("\nDeviceRes | Device Latency | Edge Latency | Cloud Latency")
    print("------------------------------------------------------------")

    for dev_res in device_resources:

        scheduler=TaskScheduler(I=I, N_device=dev_res, N_edge=N_edge)

        best_solution, evaluation = scheduler.genetic_algorithm()

        task_details=evaluation['task_details']

        device_sum=sum(t['latency'] for t in task_details if t['location']=='Device')
        edge_sum=sum(t['latency'] for t in task_details if t['location']=='Edge')
        cloud_sum=sum(t['latency'] for t in task_details if t['location']=='Cloud')

        device_lat_total.append(device_sum)
        edge_lat_total.append(edge_sum)
        cloud_lat_total.append(cloud_sum)

        print(f"{dev_res:<10} | {device_sum:<14.2f} | {edge_sum:<12.2f} | {cloud_sum:<12.2f}")

    x=np.arange(len(device_resources))
    bar_width=0.25

    plt.figure()

    plt.bar(x-bar_width,device_lat_total,color='#1f77b4',width=bar_width,label='Device')
    plt.bar(x,edge_lat_total,color='#FF4500',width=bar_width,label='Edge')
    plt.bar(x+bar_width,cloud_lat_total,color='#2ca02c',width=bar_width,label='Cloud')

    plt.xticks(x,[str(n) for n in device_resources])

    plt.xlabel("Device Resources", fontsize=12, fontweight='bold')
    plt.ylabel("Latency (s)", fontsize=12, fontweight='bold')

    plt.title("Latency Decomposition vs Device Resources", fontsize=12, fontweight='bold')

    plt.legend()

    plt.grid(True,linestyle='--',alpha=0.5)

    plt.tight_layout()

    plt.show()


# ===============================
# Main
# ===============================

if __name__=="__main__":

    device_resources=[10,20,30,40,50]

    latency_vs_device_resource(device_resources)