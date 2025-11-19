# User-Centric Joint UAV Deployment and Content Caching Optimization Approach in Multi-UAVs Assisted Networks
## Abstract
This work investigates the edge caching strategy for Multi-UAVs assisted networks, where unmanned aerial vehicle (UAV) can be deployed as aerial base station to assist content delivery for reducing the network latency. In doing so, there
are two challenges: 1) How to find the optimal deployment locations for UAVs to provide the best network services to ground users; 2) Which contents should be cached at these UAVs to meet user-specific demands. To address these, a user-centric joint UAV deployment and content caching optimization
strategy is developed. In this strategy, we first propose a novel content and social-aware user preference learning (CS-UPL) method to predict user’s dynamic content demand by jointly exploiting the content correlation among different contents and
the influence of users’ social relationships. In detail, the content-aware features learning module is developed to capture high-order similarity representation among different contents. Meanwhile, an attention-based social-aware features learning module is
introduced to learn the social influence weights across users with varying social relationships. Furthermore, a cross-domain adaptive feature fusion module is developed to effectively integrate the content and social-aware features to improve the accuracy
of user content preference prediction. Based on the predicted user content preference, a joint UAV deployment and content caching optimization problem is formulated to minimize the average content access latency of all GUs. Since the formulated
problem is non-convex and difficult to be solved, we propose a user-centric joint optimization approach (UC-JOA) to obtain the near-optimal solutions. Simulation results show that the proposed CS-UPL method achieves higher prediction accuracy of user
content preference than the baseline methods and the proposed UC-JOA outperform the existing UAV development and content caching strategies in terms of the average content transmission  latency and caching hit ratio.

## Requirements
stellargraph == 1.2.1  
tensorflow-gpu == 2.1.0  
pandas = 1.3.4  
numpy == 1.19.5  
matplotlib == 3.5.0  
## Dataset
We uploaded the processed dataset to：https://pan.baidu.com/s/1ggosjgBudOOlie97KTGMJw 
Extraction Code：afy5

## Pleasae cite the work if you would like to use it

[1] Dongyang Li, Haixia Zhang, Tiantian Li, Hui Ding and Dongfeng Yuan, "Community Detection and Attention-Weighted Federated Learning Based Proactive Edge Caching for D2D-Assisted Wireless Networks," in IEEE Transactions on Wireless Communications, vol. 22, no. 11, pp. 7287-7303, Nov. 2023, doi: 10.1109/TWC.2023.3249756.

[2] Dongyang Li, Haixia Zhang, Dongfeng Yuan and Minggao Zhang, "Learning-Based Hierarchical Edge Caching for Cloud-Aided Heterogeneous Networks," in IEEE Transactions on Wireless Communications, vol. 22, no. 3, pp. 1648-1663, March 2023, doi: 10.1109/TWC.2022.3206236. 

[3] Dongyang Li, Haixia Zhang, Hui Ding, Tiantian Li, Daojun Liang and Dongfeng Yuan, "User Preference Learning-based Proactive Edge Caching for D2D-Assisted Wireless Networks," in IEEE Internet of Things Journal, vol. 10, no. 13, pp. 11922-11937, 1 July1, 2023, doi: 10.1109/JIOT.2023.3244621.
