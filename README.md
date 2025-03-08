# Resilient-Leader-Follower-CBF-QP-Controller

This repository contains the implementation of Strong $r$-robustness maintenance Control Barrier Function-based Quadratic Programming (CBF-QP) Controller. This guarantees the multi-robot network to maintain the sufficient number of connections among robots such a way that the robots can achieve leader-follower consensus despite the presence of misbehaving agents. Contrary to the previous works that assume robots can maintain predetermined graph topologies with known robustness properties, our CBF directly addresses robustness without imposing any fixed topology. This allows the robots to flexiblely form reconfigurable communication networks with desired levels of strong $r$-robustness while navigating in spatially restrictive environments. The algorithm also includes inter-agent and obstacle collision avoidances encoded as additional CBFs. We present three different scenarios, whose videos are shown below. For more information, please refer to our paper.

# Hardware Experiment
Results for hardware experiments can be seen [here](https://youtu.be/QhoByZszucg).


# Dependencies
This repository requires the following libraries: **numpy**, **matplotlib**, **jax**, and **jaxlib**.

For solving the CBF-QP, we need the following libraries: **cvxpy** and **gurobi**

# Dynamics Model
We have adopted double integrator dynamics model to simulate robots' motions. 

# How to Run
In this repository, we have three different simulations: `spread_out_sim`, `complex_sim1`, and `complex_sim2`.
1) `spread_out_sim` shows the simulation of robots spreading out in an open space.
2) `complex_sim1` and `complex_sim2` show the simulation of robots going through some complex environments with obstacles.

To run each simulation scenario, you just need to run the corresponding python file. 
# spread_out_sim
![spread_out_sim1-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/c1151031-1a52-4621-9690-326304b47911)


# complex_sim1
 ![narrow_sim1-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/564559c2-b1f4-4322-9ec6-9f74881e25c0)

# complex_sim2
![obstacle_sim1-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/b2b1a954-8686-45b9-9c8b-83128c1fa3a6)


# For more Information...
Please check our Paper: https://arxiv.org/abs/2409.14675








