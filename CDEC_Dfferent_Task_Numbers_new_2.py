"""
mixed_integer_scheduler_tf_drl.py
Mixed-integer scheduling simulator
with GA (CEDC), HEFT, and TensorFlow2 DQN-based DRL
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
    # Evaluation
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
    # Baselines
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
    # HEFT
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
        # Training
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
        # Inference
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
    # GA (CEDC)
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
# Scaling experiment
# -------------------------------
def run_scaling_experiment(task_values):
    results = {}
    for value in task_values:
        scheduler = TaskScheduler(I=value, N_device=150, N_edge=80)
        print("Running tasks =", value)
        ga_solution, ga_eval = scheduler.genetic_algorithm()
        heft_eval = scheduler.evaluate(scheduler.generate_heft_solution())
        drl_eval = scheduler.evaluate(scheduler.generate_drl_solution())
        cloud_eval = scheduler.evaluate(scheduler.generate_baseline_solution('cloud-only'))
        edge_dev_eval = scheduler.evaluate(scheduler.generate_baseline_solution('edge-device'))
        cloud_dev_eval = scheduler.evaluate(scheduler.generate_baseline_solution('cloud-device'))
        cloud_edge_eval = scheduler.evaluate(scheduler.generate_baseline_solution('cloud-edge'))
        results[value] = {
            'CEDC': ga_eval['total_latency'],
            'HEFT': heft_eval['total_latency'],
            'DRL': drl_eval['total_latency'],
            'cloud-only': cloud_eval['total_latency'],
            'edge-device': edge_dev_eval['total_latency'],
            'cloud-device': cloud_dev_eval['total_latency'],
            'cloud-edge': cloud_edge_eval['total_latency']
        }
    return results

# -------------------------------
# Plot
# -------------------------------
def plot_results(results):
    strategies = ["cloud-only","edge-device","cloud-device","cloud-edge","HEFT","DRL","CEDC"]
    colors = plt.cm.tab10.colors
    markers = ["o","s","^","d","v","*","P"]
    x = list(results.keys())
    plt.figure()
    for i, strategy in enumerate(strategies):
        y = [results[val][strategy] for val in x]
        plt.plot(x,y,marker=markers[i],color=colors[i],linewidth=2,markersize=7,label=strategy)
    plt.xlabel("Task Numbers", fontsize=12, fontweight="bold")
    plt.ylabel("Total Latency (s)", fontsize=12, fontweight="bold")
    plt.grid(True, alpha=0.3)
    ymin = min([results[val][s] for val in x for s in strategies])
    ymax = max([results[val][s] for val in x for s in strategies])
    plt.ylim(ymin*0.95, ymax*1.05)
    plt.legend(ncol=3, fontsize=10, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.show()

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    task_range = [60,70,80,90,100,110]
    results = run_scaling_experiment(task_range)
    # results = {60: {'CEDC': 70.78701333999474, 'HEFT': 70.78701333999474, 'DRL': 73.33104359201577, 'cloud-only': 210.86526745213797, 'edge-device': 76.01692061487458, 'cloud-device': 81.91470998063724, 'cloud-edge': 172.97004921319055}, 70: {'CEDC': 83.62412573777085, 'HEFT': 86.31776584355754, 'DRL': 84.8555342936697, 'cloud-only': 261.9077257882553, 'edge-device': 96.77759100499557, 'cloud-device': 128.28951752181376, 'cloud-edge': 223.99859268485665}, 80: {'CEDC': 98.15895155631117, 'HEFT': 100.08604067401599, 'DRL': 98.47144251559358, 'cloud-only': 288.0606065578287, 'edge-device': 119.1944086054441, 'cloud-device': 152.84024688140627, 'cloud-edge': 242.80689004559684}, 90: {'CEDC': 117.80145574249762, 'HEFT': 136.56870348547176, 'DRL': 134.00719430792512, 'cloud-only': 309.07967584349745, 'edge-device': 157.91281415078276, 'cloud-device': 183.87761894657868, 'cloud-edge': 277.3013239485678}, 100: {'CEDC': 142.55789817000232, 'HEFT': 185.66389899344887, 'DRL': 169.32998632764628, 'cloud-only': 347.3446822415104, 'edge-device': 204.2624938087917, 'cloud-device': 229.5506677942115, 'cloud-edge': 315.76806008681933}, 110: {'CEDC': 168.39551030606117, 'HEFT': 231.46495506331175, 'DRL': 203.90320145194195, 'cloud-only': 380.5676312213168, 'edge-device': 223.49684797828698, 'cloud-device': 265.60346843651496, 'cloud-edge': 355.74668050272084}}
    print(results)
    plot_results(results)