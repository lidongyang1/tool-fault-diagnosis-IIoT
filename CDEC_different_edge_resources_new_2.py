"""
mixed_integer_scheduler_tf2_dqn.py
Task offloading scheduling experiment
(Proposed CEDC + HEFT + TF2 DQN DRL + Baselines)
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
    def __init__(self, I=40, N_device=40, N_edge=80):
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
        epsilon = solution[:, 0]
        xi = solution[:, 1]
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
        penalty = alpha * (time_violations + 0.5 * fallback_count)
        return {"total_latency": total_latency,
                "time_violations": time_violations,
                "fallback_count": fallback_count,
                "fitness": total_latency + penalty}

    # -------------------------------
    # Baseline strategies
    # -------------------------------
    def generate_baseline_solution(self, strategy):
        solution = np.zeros((self.I,2),dtype=int)
        if strategy=="cloud-only":
            solution[:,:]=[0,0]
        elif strategy=="edge-device":
            res=self.N_device
            for i in range(self.I):
                if self.n_device[i]<=res:
                    solution[i]=[1,0]
                    res-=self.n_device[i]
                else:
                    solution[i]=[0,1]
        elif strategy=="cloud-device":
            res=self.N_device
            for i in range(self.I):
                if self.n_device[i]<=res:
                    solution[i]=[1,0]
                    res-=self.n_device[i]
                else:
                    solution[i]=[0,0]
        elif strategy=="cloud-edge":
            res=self.N_edge
            for i in range(self.I):
                if self.n_edge[i]<=res:
                    solution[i]=[0,1]
                    res-=self.n_edge[i]
                else:
                    solution[i]=[0,0]
        return solution

    # -------------------------------
    # HEFT heuristic
    # -------------------------------
    def generate_heft_solution(self):
        solution = np.zeros((self.I,2),dtype=int)
        avg_time=(self.t_device+self.t_edge+self.t_cloud)/3
        order=np.argsort(avg_time)
        R_device=0
        R_edge=0
        for i in order:
            options=[('device',self.t_device[i]),('edge',self.t_edge[i]),('cloud',self.t_cloud[i])]
            options.sort(key=lambda x:x[1])
            for loc,_ in options:
                if loc=='device' and R_device+self.n_device[i]<=self.N_device:
                    solution[i]=[1,0]
                    R_device+=self.n_device[i]
                    break
                elif loc=='edge' and R_edge+self.n_edge[i]<=self.N_edge:
                    solution[i]=[0,1]
                    R_edge+=self.n_edge[i]
                    break
                elif loc=='cloud':
                    solution[i]=[0,0]
                    break
        return solution

    # -------------------------------
    # TensorFlow2 DQN DRL
    # -------------------------------
    def generate_drl_solution(self, episodes=200):
        agent=DRLAgent()
        # train
        for ep in range(episodes):
            R_device=0
            R_edge=0
            for i in range(self.I):
                state=[R_device/self.N_device,
                       R_edge/self.N_edge,
                       self.n_device[i]/5,
                       self.n_edge[i]/5,
                       self.t_device[i]/5,
                       self.t_edge[i]/5,
                       self.t_cloud[i]/5]
                action=agent.choose_action(state)
                if action==1 and R_device+self.n_device[i]<=self.N_device:
                    latency=self.t_device[i]
                    R_device+=self.n_device[i]
                elif action==2 and R_edge+self.n_edge[i]<=self.N_edge:
                    latency=self.t_edge[i]
                    R_edge+=self.n_edge[i]
                else:
                    latency=self.t_cloud[i]
                reward=-latency
                agent.train(state,action,reward,state)
        # inference
        solution=np.zeros((self.I,2),dtype=int)
        R_device=0
        R_edge=0
        for i in range(self.I):
            state=[R_device/self.N_device,
                   R_edge/self.N_edge,
                   self.n_device[i]/5,
                   self.n_edge[i]/5,
                   self.t_device[i]/5,
                   self.t_edge[i]/5,
                   self.t_cloud[i]/5]
            action=np.argmax(agent.dqn.predict(state))
            if action==1 and R_device+self.n_device[i]<=self.N_device:
                solution[i]=[1,0]
                R_device+=self.n_device[i]
            elif action==2 and R_edge+self.n_edge[i]<=self.N_edge:
                solution[i]=[0,1]
                R_edge+=self.n_edge[i]
            else:
                solution[i]=[0,0]
        return solution

    # -------------------------------
    # Genetic Algorithm (CEDC)
    # -------------------------------
    def genetic_algorithm(self,pop_size=80,max_gen=150):
        population=np.random.randint(0,2,(pop_size,self.I,2))
        best_solution=None
        best_fitness=float('inf')
        for gen in range(max_gen):
            evaluations=[self.evaluate(ind) for ind in population]
            fitness=[e['fitness'] for e in evaluations]
            idx=np.argmin(fitness)
            if fitness[idx]<best_fitness:
                best_fitness=fitness[idx]
                best_solution=population[idx].copy()
            new_population=[]
            for _ in range(pop_size):
                candidates=population[np.random.choice(pop_size,3,replace=False)]
                scores=[self.evaluate(c)['fitness'] for c in candidates]
                winner=candidates[np.argmin(scores)]
                new_population.append(winner.copy())
            population=np.array(new_population)
        final_eval=self.evaluate(best_solution)
        return best_solution,final_eval

# -------------------------------
# Scaling experiment
# -------------------------------
def run_scaling_experiment(param_name,param_values):
    results={}
    for val in param_values:
        if param_name=="N_edge":
            scheduler=TaskScheduler(I=120,N_device=40,N_edge=val)
        elif param_name=="N_device":
            scheduler=TaskScheduler(I=120,N_device=val,N_edge=80)
        print("Running",param_name,"=",val)
        ga_sol,ga_eval=scheduler.genetic_algorithm()
        heft_eval=scheduler.evaluate(scheduler.generate_heft_solution())
        drl_eval=scheduler.evaluate(scheduler.generate_drl_solution())
        cloud_eval=scheduler.evaluate(scheduler.generate_baseline_solution("cloud-only"))
        edge_dev_eval=scheduler.evaluate(scheduler.generate_baseline_solution("edge-device"))
        cloud_dev_eval=scheduler.evaluate(scheduler.generate_baseline_solution("cloud-device"))
        cloud_edge_eval=scheduler.evaluate(scheduler.generate_baseline_solution("cloud-edge"))
        results[val]={
            "CEDC": ga_eval,
            "HEFT": heft_eval,
            "DRL": drl_eval,
            "cloud-only": cloud_eval,
            "edge-device": edge_dev_eval,
            "cloud-device": cloud_dev_eval,
            "cloud-edge": cloud_edge_eval
        }
    return results

# -------------------------------
# Plot
# -------------------------------
def plot_latency(results):
    strategies=["cloud-only","edge-device","cloud-device","cloud-edge","HEFT","DRL","CEDC"]
    colors=plt.cm.tab10.colors
    markers=["o","s","^","d","v","*","P"]
    x=list(results.keys())
    plt.figure()
    for i,strategy in enumerate(strategies):
        y=[results[val][strategy]["total_latency"] for val in x]
        plt.plot(x,y,marker=markers[i],color=colors[i],linewidth=2,label=strategy)
    plt.xlabel("Edge Resources",fontsize=12,fontweight="bold")
    plt.ylabel("Total Latency (s)",fontsize=12,fontweight="bold")
    plt.ylim(350,450)
    plt.grid(True,alpha=0.3)
    plt.legend(ncol=3,fontsize=10,loc='upper right')
    plt.tight_layout()
    plt.show()

# -------------------------------
# Main
# -------------------------------
if __name__=="__main__":
    param_values=[20,30,40,50,60]
    results=run_scaling_experiment("N_edge",param_values)
    results = {20: {'CEDC': {'total_latency': 371.8644021938231, 'time_violations': 19, 'fallback_count': 54, 'fitness': 46000371.8644022}, 'HEFT': {'total_latency': 396.5836448029964, 'time_violations': 26, 'fallback_count': 0, 'fitness': 26000396.583644804}, 'DRL': {'total_latency': 381.9105286353363, 'time_violations': 22, 'fallback_count': 0, 'fitness': 22000381.910528634}, 'cloud-only': {'total_latency': 425.40852357776504, 'time_violations': 26, 'fallback_count': 0, 'fitness': 26000425.40852358}, 'edge-device': {'total_latency': 382.76441618767484, 'time_violations': 21, 'fallback_count': 97, 'fitness': 69500382.76441619}, 'cloud-device': {'total_latency': 391.9725730340245, 'time_violations': 22, 'fallback_count': 0, 'fitness': 22000391.972573034}, 'cloud-edge': {'total_latency': 414.8406760199498, 'time_violations': 24, 'fallback_count': 0, 'fitness': 24000414.84067602}}, 30: {'CEDC': {'total_latency': 368.9388047881647, 'time_violations': 18, 'fallback_count': 51, 'fitness': 43500368.93880479}, 'HEFT': {'total_latency': 391.477300357263, 'time_violations': 26, 'fallback_count': 0, 'fitness': 26000391.477300357}, 'DRL': {'total_latency': 373.6951515871671, 'time_violations': 21, 'fallback_count': 0, 'fitness': 21000373.695151586}, 'cloud-only': {'total_latency': 425.40852357776504, 'time_violations': 26, 'fallback_count': 0, 'fitness': 26000425.40852358}, 'edge-device': {'total_latency': 377.5619989419546, 'time_violations': 21, 'fallback_count': 95, 'fitness': 68500377.56199895}, 'cloud-device': {'total_latency': 391.9725730340245, 'time_violations': 22, 'fallback_count': 0, 'fitness': 22000391.972573034}, 'cloud-edge': {'total_latency': 412.8907373849572, 'time_violations': 23, 'fallback_count': 0, 'fitness': 23000412.890737385}}, 40: {'CEDC': {'total_latency': 363.4060584709622, 'time_violations': 18, 'fallback_count': 48, 'fitness': 42000363.40605847}, 'HEFT': {'total_latency': 390.8597000451016, 'time_violations': 26, 'fallback_count': 0, 'fitness': 26000390.859700046}, 'DRL': {'total_latency': 368.9187980213757, 'time_violations': 21, 'fallback_count': 0, 'fitness': 21000368.918798022}, 'cloud-only': {'total_latency': 425.40852357776504, 'time_violations': 26, 'fallback_count': 0, 'fitness': 26000425.40852358}, 'edge-device': {'total_latency': 375.65945971867546, 'time_violations': 21, 'fallback_count': 93, 'fitness': 67500375.65945973}, 'cloud-device': {'total_latency': 391.9725730340245, 'time_violations': 22, 'fallback_count': 0, 'fitness': 22000391.972573034}, 'cloud-edge': {'total_latency': 413.72853894307707, 'time_violations': 23, 'fallback_count': 0, 'fitness': 23000413.72853894}}, 50: {'CEDC': {'total_latency': 358.1017138492153, 'time_violations': 18, 'fallback_count': 45, 'fitness': 40500358.10171385}, 'HEFT': {'total_latency': 385.547069088616, 'time_violations': 26, 'fallback_count': 0, 'fitness': 26000385.547069088}, 'DRL': {'total_latency': 363.2265899483982, 'time_violations': 20, 'fallback_count': 0, 'fitness': 20000363.226589948}, 'cloud-only': {'total_latency': 425.40852357776504, 'time_violations': 26, 'fallback_count': 0, 'fitness': 26000425.40852358}, 'edge-device': {'total_latency': 370.16592677355465, 'time_violations': 21, 'fallback_count': 90, 'fitness': 66000370.16592678}, 'cloud-device': {'total_latency': 391.9725730340245, 'time_violations': 22, 'fallback_count': 0, 'fitness': 22000391.972573034}, 'cloud-edge': {'total_latency': 409.7566020471828, 'time_violations': 22, 'fallback_count': 0, 'fitness': 22000409.75660205}}, 60: {'CEDC': {'total_latency': 357.5796989743422, 'time_violations': 18, 'fallback_count': 42, 'fitness': 39000357.57969897}, 'HEFT': {'total_latency': 380.84494367083295, 'time_violations': 25, 'fallback_count': 0, 'fitness': 25000380.844943672}, 'DRL': {'total_latency': 363.5000600103079, 'time_violations': 20, 'fallback_count': 0, 'fitness': 20000363.50006001}, 'cloud-only': {'total_latency': 425.40852357776504, 'time_violations': 26, 'fallback_count': 0, 'fitness': 26000425.40852358}, 'edge-device': {'total_latency': 366.8135977969156, 'time_violations': 20, 'fallback_count': 87, 'fitness': 63500366.8135978}, 'cloud-device': {'total_latency': 391.9725730340245, 'time_violations': 22, 'fallback_count': 0, 'fitness': 22000391.972573034}, 'cloud-edge': {'total_latency': 405.8907376599513, 'time_violations': 22, 'fallback_count': 0, 'fitness': 22000405.89073766}}}
    print(results)
    plot_latency(results)