"""
mixed_integer_scheduler_tf2_dqn.py
混合整数规划求解器（含多策略对比版 + HEFT + TF2 DQN DRL）
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import random

# -------------------------------
# TensorFlow2 DQN
# -------------------------------
class DQN:
    def __init__(self, state_dim=7, action_dim=3, lr=0.001, gamma=0.9, epsilon=0.2):
        self.model = tf.keras.Sequential([
            layers.Dense(64, activation="relu", input_shape=(state_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dense(action_dim)
        ])
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def predict(self, state):
        state = np.array(state).reshape(1, -1)
        q_values = self.model(state, training=False)
        return q_values.numpy()[0]

    def train(self, state, action, reward, next_state):
        state = np.array(state).reshape(1, -1)
        next_state = np.array(next_state).reshape(1, -1)
        with tf.GradientTape() as tape:
            q_values = self.model(state, training=True)
            next_q = self.model(next_state, training=False)
            target = q_values.numpy()
            target[0][action] = reward + self.gamma * np.max(next_q.numpy())
            loss = tf.keras.losses.MSE(target, q_values)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


class DRLAgent:
    def __init__(self, state_dim=7, action_dim=3):
        self.dqn = DQN(state_dim=state_dim, action_dim=action_dim)

    def choose_action(self, state):
        if random.random() < self.dqn.epsilon:
            return random.randint(0, 2)
        q_values = self.dqn.predict(state)
        return np.argmax(q_values)

    def train(self, state, action, reward, next_state):
        self.dqn.train(state, action, reward, next_state)

# -------------------------------
# Task Scheduler
# -------------------------------
class TaskScheduler:
    def __init__(self, I=20, N_device=100, N_edge=500):
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

    # -------------------------------
    # 评价函数
    # -------------------------------
    def evaluate(self, solution, alpha=1e6):
        epsilon, xi = solution[:, 0], solution[:, 1]
        total_latency = 0
        R_device = 0
        R_edge = 0
        time_violations = 0
        fallback_count = 0
        for i in range(self.I):
            if epsilon[i] == 1:
                if R_device + self.n_device[i] <= self.N_device:
                    latency = self.t_device[i]
                    R_device += self.n_device[i]
                else:
                    latency = self.t_cloud[i]
                    fallback_count += 1
            else:
                if xi[i] == 1:
                    if R_edge + self.n_edge[i] <= self.N_edge:
                        latency = self.t_edge[i]
                        R_edge += self.n_edge[i]
                    else:
                        latency = self.t_cloud[i]
                        fallback_count += 1
                else:
                    latency = self.t_cloud[i]
            if latency > self.T_max[i]:
                time_violations += 1
            total_latency += latency
        penalty = alpha * (time_violations + fallback_count * 0.5)
        return {
            'total_latency': total_latency,
            'time_violations': time_violations,
            'resource_violations': 0,
            'fallback_count': fallback_count,
            'fitness': total_latency + penalty
        }

    # -------------------------------
    # baseline策略
    # -------------------------------
    def generate_baseline_solution(self, strategy):
        solution = np.zeros((self.I, 2), dtype=int)
        if strategy == 'cloud-only':
            solution[:, :] = [0, 0]
        elif strategy == 'edge-device':
            device_resources = self.N_device
            for i in range(self.I):
                if self.n_device[i] <= device_resources:
                    solution[i] = [1, 0]
                    device_resources -= self.n_device[i]
                else:
                    solution[i] = [0, 1]
        elif strategy == 'cloud-device':
            device_resources = self.N_device
            for i in range(self.I):
                if self.n_device[i] <= device_resources:
                    solution[i] = [1, 0]
                    device_resources -= self.n_device[i]
                else:
                    solution[i] = [0, 0]
        elif strategy == 'cloud-edge':
            edge_resources = self.N_edge
            for i in range(self.I):
                if self.n_edge[i] <= edge_resources:
                    solution[i] = [0, 1]
                    edge_resources -= self.n_edge[i]
                else:
                    solution[i] = [0, 0]
        return solution

    # -------------------------------
    # HEFT算法
    # -------------------------------
    def generate_heft_solution(self):
        solution = np.zeros((self.I, 2), dtype=int)
        avg_time = (self.t_device + self.t_edge + self.t_cloud) / 3
        order = np.argsort(avg_time)
        R_device = 0
        R_edge = 0
        for i in order:
            options = [('device', self.t_device[i]),
                       ('edge', self.t_edge[i]),
                       ('cloud', self.t_cloud[i])]
            options.sort(key=lambda x: x[1])
            for loc, _ in options:
                if loc == 'device' and R_device + self.n_device[i] <= self.N_device:
                    solution[i] = [1, 0]
                    R_device += self.n_device[i]
                    break
                elif loc == 'edge' and R_edge + self.n_edge[i] <= self.N_edge:
                    solution[i] = [0, 1]
                    R_edge += self.n_edge[i]
                    break
                elif loc == 'cloud':
                    solution[i] = [0, 0]
                    break
        return solution

    # -------------------------------
    # TensorFlow2 DQN DRL
    # -------------------------------
    def generate_drl_solution(self, episodes=200):
        agent = DRLAgent()
        # 训练阶段
        for ep in range(episodes):
            R_device = 0
            R_edge = 0
            for i in range(self.I):
                state = [
                    R_device/self.N_device,
                    R_edge/self.N_edge,
                    self.n_device[i]/5,
                    self.n_edge[i]/5,
                    self.t_device[i]/5,
                    self.t_edge[i]/5,
                    self.t_cloud[i]/5
                ]
                action = agent.choose_action(state)
                if action == 1 and R_device + self.n_device[i] <= self.N_device:
                    latency = self.t_device[i]
                    R_device += self.n_device[i]
                elif action == 2 and R_edge + self.n_edge[i] <= self.N_edge:
                    latency = self.t_edge[i]
                    R_edge += self.n_edge[i]
                else:
                    latency = self.t_cloud[i]
                reward = -latency
                agent.train(state, action, reward, state)
        # 推理阶段
        solution = np.zeros((self.I,2),dtype=int)
        R_device = 0
        R_edge = 0
        for i in range(self.I):
            state = [
                R_device/self.N_device,
                R_edge/self.N_edge,
                self.n_device[i]/5,
                self.n_edge[i]/5,
                self.t_device[i]/5,
                self.t_edge[i]/5,
                self.t_cloud[i]/5
            ]
            action = np.argmax(agent.dqn.predict(state))
            if action == 1 and R_device + self.n_device[i] <= self.N_device:
                solution[i] = [1,0]
                R_device += self.n_device[i]
            elif action == 2 and R_edge + self.n_edge[i] <= self.N_edge:
                solution[i] = [0,1]
                R_edge += self.n_edge[i]
            else:
                solution[i] = [0,0]
        return solution

    # -------------------------------
    # 遗传算法 CEDC
    # -------------------------------
    def genetic_algorithm(self, pop_size=80, max_gen=150):
        population = np.random.randint(0, 2, (pop_size, self.I, 2))
        best_solution = None
        best_fitness = float('inf')
        for gen in range(max_gen):
            evaluations = [self.evaluate(ind) for ind in population]
            fitness_values = [e['fitness'] for e in evaluations]
            idx = np.argmin(fitness_values)
            if fitness_values[idx] < best_fitness:
                best_fitness = fitness_values[idx]
                best_solution = population[idx].copy()
            new_population = []
            for _ in range(pop_size):
                candidates = population[np.random.choice(pop_size,3,replace=False)]
                winner = candidates[np.argmin([self.evaluate(c)['fitness'] for c in candidates])]
                new_population.append(winner.copy())
            new_population = np.array(new_population)
            crossover_rate = 0.8 - 0.3*(gen/max_gen)
            for i in range(0,pop_size,2):
                if np.random.rand() < crossover_rate and i+1<pop_size:
                    p1,p2 = sorted(np.random.choice(self.I,2))
                    temp = new_population[i,p1:p2].copy()
                    new_population[i,p1:p2] = new_population[i+1,p1:p2]
                    new_population[i+1,p1:p2] = temp
            mutation_rate = 0.1*(1-gen/max_gen)
            for i in range(pop_size):
                for j in range(self.I):
                    if np.random.rand()<mutation_rate:
                        bit = np.random.randint(0,2)
                        new_population[i,j,bit] ^=1
            population = new_population
        final_eval = self.evaluate(best_solution)
        return best_solution, final_eval

# -------------------------------
# 扩展实验
# -------------------------------
def run_scaling_experiment(param_name, param_values, default_I=110, default_N_device=150, default_N_edge=80):
    results = {}
    for value in param_values:
        if param_name == 'N_device':
            scheduler = TaskScheduler(I=default_I, N_device=value, N_edge=default_N_edge)
        elif param_name == 'I':
            scheduler = TaskScheduler(I=value, N_device=default_N_device, N_edge=default_N_edge)
        print("Running", param_name, "=", value)
        ga_solution, ga_eval = scheduler.genetic_algorithm()
        heft_eval = scheduler.evaluate(scheduler.generate_heft_solution())
        drl_eval = scheduler.evaluate(scheduler.generate_drl_solution())
        cloud_eval = scheduler.evaluate(scheduler.generate_baseline_solution('cloud-only'))
        edge_dev_eval = scheduler.evaluate(scheduler.generate_baseline_solution('edge-device'))
        cloud_dev_eval = scheduler.evaluate(scheduler.generate_baseline_solution('cloud-device'))
        cloud_edge_eval = scheduler.evaluate(scheduler.generate_baseline_solution('cloud-edge'))
        results[value] = {
            'CEDC': {'total_latency': ga_eval['total_latency']},
            'HEFT': {'total_latency': heft_eval['total_latency']},
            'DRL': {'total_latency': drl_eval['total_latency']},
            'cloud-only': {'total_latency': cloud_eval['total_latency']},
            'edge-device': {'total_latency': edge_dev_eval['total_latency']},
            'cloud-device': {'total_latency': cloud_dev_eval['total_latency']},
            'cloud-edge': {'total_latency': cloud_edge_eval['total_latency']}
        }
    return results

# -------------------------------
# 绘图
# -------------------------------
def plot_latency_comparison(all_results):
    strategies = ['cloud-only','edge-device','cloud-device','cloud-edge','HEFT','DRL','CEDC']
    colors = plt.cm.tab10.colors
    markers = ['o','s','^','d','v','*','P']
    plt.figure()
    results = list(all_results.values())[0]
    x = list(results.keys())
    for i, strategy in enumerate(strategies):
        y = [results[val][strategy]['total_latency'] for val in x]
        plt.plot(x,y,marker=markers[i],color=colors[i % len(colors)],linewidth=2,label=strategy)
    plt.xlabel("Device Resources", fontsize=12, fontweight='bold')
    plt.ylabel("Total Latency (s)", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(230,420)
    plt.legend(ncol=3, fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.show()

# -------------------------------
# 主程序
# -------------------------------
if __name__ == "__main__":
    param_config = [('N_device', 'Device Resources', [20, 30, 40, 50, 60, 70])]
    all_results = {}
    for param, label, values in param_config:
        print("\n=== Running", param, "scaling experiment ===")
        results = run_scaling_experiment(param, values)
        all_results[param] = results
    # all_results = {'N_device': {20: {'CEDC': {'total_latency': 305.2700775453727}, 'HEFT': {'total_latency': 348.72124040660714}, 'DRL': {'total_latency': 328.823461422191}, 'cloud-only': {'total_latency': 380.5676312213168}, 'edge-device': {'total_latency': 337.6958681969496}, 'cloud-device': {'total_latency': 366.92307268520165}, 'cloud-edge': {'total_latency': 355.74668050272084}}, 30: {'CEDC': {'total_latency': 292.84448165315086}, 'HEFT': {'total_latency': 343.27351078266776}, 'DRL': {'total_latency': 322.59601332008435}, 'cloud-only': {'total_latency': 380.5676312213168}, 'edge-device': {'total_latency': 324.3343433274703}, 'cloud-device': {'total_latency': 356.14801676491464}, 'cloud-edge': {'total_latency': 355.74668050272084}}, 40: {'CEDC': {'total_latency': 270.11013871598686}, 'HEFT': {'total_latency': 334.0854665888696}, 'DRL': {'total_latency': 303.7262118072069}, 'cloud-only': {'total_latency': 380.5676312213168}, 'edge-device': {'total_latency': 316.6473554170681}, 'cloud-device': {'total_latency': 351.74567648110536}, 'cloud-edge': {'total_latency': 355.74668050272084}}, 50: {'CEDC': {'total_latency': 267.52866608525795}, 'HEFT': {'total_latency': 324.9589272439008}, 'DRL': {'total_latency': 303.7699751442528}, 'cloud-only': {'total_latency': 380.5676312213168}, 'edge-device': {'total_latency': 317.75835336349655}, 'cloud-device': {'total_latency': 349.2522823345924}, 'cloud-edge': {'total_latency': 355.74668050272084}}, 60: {'CEDC': {'total_latency': 246.1783385420415}, 'HEFT': {'total_latency': 317.11579245381375}, 'DRL': {'total_latency': 293.15426719663253}, 'cloud-only': {'total_latency': 380.5676312213168}, 'edge-device': {'total_latency': 310.9667926603501}, 'cloud-device': {'total_latency': 337.9078138143574}, 'cloud-edge': {'total_latency': 355.74668050272084}}, 70: {'CEDC': {'total_latency': 241.13401631228626}, 'HEFT': {'total_latency': 308.84022413133414}, 'DRL': {'total_latency': 286.9441012672285}, 'cloud-only': {'total_latency': 380.5676312213168}, 'edge-device': {'total_latency': 307.1609765201643}, 'cloud-device': {'total_latency': 332.83180876002973}, 'cloud-edge': {'total_latency': 355.74668050272084}}}}
    print(all_results)
    plot_latency_comparison(all_results)