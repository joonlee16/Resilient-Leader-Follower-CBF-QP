import helper.bootstrap_percolation as dp
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from helper.obstacles import *
from helper.double_integrator import *
from random import randint
import jax
from jax import jit, lax


plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-3,3),ylim=(-3,3)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")


# Sim Parameters    
epsilon = 0.0001              
dt = 0.025
tf = 15
num_steps = int(tf/dt)
leaders = 6
F = 2

#Initialize the robots
robots = []
broadcast_value = randint(0,1000)
y_offset = -0.3 
robots.append( Leaders(broadcast_value, jnp.array([-1.5,y_offset,0,0]),'b',1.0, ax,F))
robots.append( Leaders(broadcast_value, jnp.array([-0.7,y_offset+0.2,0,0]),'b',1.0, ax, F))
robots.append( Leaders(broadcast_value, jnp.array([1,y_offset+0.2,0,0]),'b',1.0, ax, F))
robots.append( Leaders(broadcast_value, jnp.array([0.7,y_offset + 1.5,0,0]),'b',1.0, ax, F))
robots.append( Leaders(broadcast_value, jnp.array([0.2,y_offset-1,0,0]),'b',1.0, ax, F))
robots.append( Malicious([0,1000], jnp.array([-1.1,y_offset - 0.7,0,0]),'r',1.0, ax, F,marker="s"))
robots.append( Malicious([0,1000], jnp.array([1.4,y_offset,0,0]),'r',1.0, ax, F))
robots.append( Agent(jnp.array([-1.2,y_offset - 0.3,0,0]),'g',1.0 , ax, F))
robots.append( Agent(jnp.array([0.7,y_offset + 0.3,0,0]),'g',1.0 , ax, F))
robots.append( Agent(jnp.array([1.2,y_offset - 0.9,0,0]),'g',1.0 , ax, F))
robots.append( Agent(jnp.array([-0.8,y_offset + 1.1,0,0]),'g',1.0 , ax, F))
robots.append( Agent(jnp.array([1.1,y_offset - 0.4,0,0]),'g',1.0 , ax, F))
robots.append( Agent(jnp.array([-0.4,y_offset + 0.2,0,0]),'g',1.0 , ax, F))
robots.append( Agent(jnp.array([-0.1,y_offset + 0.3,0,0]),'g',1.0 , ax, F))

num_robots = n =len(robots)
inter_collision = int(n*(n-1)/2)

############################## CBF Controller ######################################
u1 = cp.Variable((2*n,1))
u1_ref = cp.Parameter((2*n,1),value = np.zeros((2*n,1)) )
num_constraints1  = 1
A1 = cp.Parameter((num_constraints1,2*n),value=np.zeros((num_constraints1,2*n)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 >= b1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )
###################################################################################################
R = 3
r=leaders-1
robustness_history = []
H = [[] for i in range(n-leaders)]

#Setting the goal
goal = []
goal.append(np.array([-100, 0]).reshape(2,-1))
goal.append(np.array([-100, 100]).reshape(2,-1))
goal.append(np.array([100, 100]).reshape(2,-1))
goal.append(np.array([100, 0]).reshape(2,-1))
goal.append(np.array([100, -100]).reshape(2,-1))
goal.append(np.array([-100, -100]).reshape(2,-1))


#Build the parametrized sigmoid functions
q_A = 0.02
q = 0.02
s_A = 2.5
s = 1.3
sigmoid_A = lambda x: (1+q_A)/(1+(1/q_A)*jnp.exp(-s_A*x))-q_A
sigmoid = lambda x: (1+q)/(1+(1/q)*jnp.exp(-s*x))-q


######################Computes the \bar {\pi}_{\mathcal F}######################
@jit 
def barrier_func(x):
    def AA(x):
        A = jnp.zeros((n,n))
        def body_i(i, inputs1):
            def body_j(j, inputs):
                dis = R**2-jnp.sum((x[i]-x[j])**2)
                return jax.lax.cond(dis>=0,lambda x: inputs.at[i,j].set(sigmoid_A(dis**3)), lambda x: inputs.at[i,j].set(0), dis) 
            return lax.fori_loop(0, n, body_j, inputs1)
        A = lax.fori_loop(0, n, body_i, A)

        def bodyD(i, inputs):
            return inputs.at[i,i].set(0.0)
        return lax.fori_loop(0, n, bodyD,A)
    
    def body(i, inputs):
        temp_x = A @ jnp.append( jnp.ones((leaders,1)), inputs, axis=0 )
        state_vector = sigmoid(temp_x[leaders:]-r)
        return state_vector
    
    state_vector = jnp.zeros((n-leaders,1))
    A = AA(x)
    delta = 4
    x = jax.lax.fori_loop(0, delta, body, state_vector) 
    return x[:,0]

barrier_grad = jit(jax.jacrev(barrier_func))
barrier_double_grad = jit(jax.hessian(barrier_func))

def smoothened_strongly_r_robust_simul(robots, R, r):      
    h = barrier_func(robots)
    h_dot = barrier_grad(robots)
    h_ddot =  barrier_double_grad(robots)
    return h, h_dot, h_ddot
###############################################################################

#Set the weight vector \mathbf w
weight = np.array([7]*(num_robots-leaders) + [10]*inter_collision)

#Compiled the construction of robust maintenance HOCBF
compiled = jax.jit(smoothened_strongly_r_robust_simul)

for t in range(num_steps):    
    robots_location = np.array([aa.x.reshape(1,-1)[0] for aa in robots])
    robots_velocity = np.array([aa.v.reshape(1,-1)[0] for aa in robots])

    #Compute the actual robustness
    A = np.zeros((n, n))
    dp.unsmoothened_adjacency(R, A, robots_location)
    f = n-leaders
    robustness_history.append(dp.strongly_r_robust(A,leaders, f))

    #Get the nominal control input \mathbf u_{nom}
    for i in range(leaders):
        vector = goal[i % leaders] - robots[i].location[:2] 
        vector = vector/np.linalg.norm(vector)
        current = robots[i].location[2:4] 
        temp = np.array([[vector[0][0]-current[0]], [vector[1][0]-current[1]]])
        u1_ref.value[2*i] = temp[0][0]
        u1_ref.value[2*i+1] = temp[1][0]

    if t/20 % 1==0:
        #Agents form a network
        for i in range(n):
            for j in range(i+1,n):
                if A[i,j] ==1:
                    robots[i].connect(robots[j])
                    robots[j].connect(robots[i])
        #Agents share their values with neighbors
        for aa in robots:
            aa.propagate()
        # The followers perform W-MSR
        for aa in robots:
            aa.w_msr()
        # All the agents update their LED colors
        for aa in robots:
            aa.set_color()

    # h_{3,c}, gradient, and hessian
    x, der_, double_der_  = compiled(robots_location, R, r)
    x = np.asarray(x);der_ = np.asarray(der_);double_der_ = np.asarray(double_der_)
    print(t, x)

    #Initialize the constraint of QP
    A1.value[0,:] = [0 for i in range(2*num_robots)]
    b1.value[0] = 0

    #Inter-agent collision avoidance
    collision = [];col_alpha = 6
    for i in range(num_robots):
        for j in range(i+1, num_robots):
            h, dh_dxi, dh_dxj, ddh = robots[i].agent_barrier(robots[j], 0.3)
            h_dot = dh_dxi @ robots_velocity[i] + dh_dxj @ robots_velocity[j] + col_alpha*h
            collision.append(h_dot)
            kk = num_robots-leaders+j
            temp = (weight[kk])*np.exp(-weight[kk]*h_dot)
            A1.value[0,2*i:2*i+2]+= temp * dh_dxi[:]
            A1.value[0,2*j:2*j+2]+= temp *dh_dxj[:]
            b1.value[0]-=  temp *(2* robots_velocity[i].reshape(1,-1)[0] @  robots_velocity[i] - 2* robots_velocity[i].reshape(1,-1)[0] @  robots_velocity[j] + col_alpha*dh_dxi @ robots_velocity[i])
            b1.value[0]-=  temp *(2* robots_velocity[j].reshape(1,-1)[0] @  robots_velocity[j] - 2* robots_velocity[i].reshape(1,-1)[0] @  robots_velocity[j] + col_alpha*dh_dxj @ robots_velocity[j])
    
    
    #Calculate the Robustness HOCBF 
    alphas = 2.5
    robustes = []
    for k in range(num_robots-leaders):
        h_dot = der_[k].reshape(1,-1)[0] @ robots_velocity.reshape(-1,1)
        weightee = ((weight[k])*np.exp(-weight[k]*(h_dot+alphas*(x[k]-epsilon))))[0]
        robustes.append((h_dot+alphas*(x[k]-epsilon))[0])
        A1.value[0,:]+= weightee * der_[k].reshape(1,-1)[0]

        temp = []
        for j in range(num_robots):
            temp_x = robots_velocity.reshape(1,-1) @ double_der_[k][j][:][:][0].reshape(-1,1)
            temp_y = robots_velocity.reshape(1,-1) @ double_der_[k][j][:][:][1].reshape(-1,1)
            temp.append(temp_x[0])
            temp.append(temp_y[0]) 
        b1.value[0]-= weightee * ((np.array(temp).reshape(1,-1) + alphas*der_[k].reshape(1,-1)) @ robots_velocity.reshape(-1,1))[0]
    
    for i in range(n-leaders):
            H[i].append(float(x[i]))

    #Composition into \phi(x,w)
    sum_h = 1 - np.sum(np.exp(-weight*np.array(robustes + collision)))
    b1.value[0]-=2*(sum_h)

    #Solve the CBF-QP and get the control input \mathbf u
    cbf_controller.solve(solver=cp.GUROBI)
    if cbf_controller.status!='optimal':
        print("Error: should not have been infeasible here")
        print(A1.value)

    # implement control input \mathbf u and plot the trajectory
    for i in range(num_robots):
        robots[i].step2( u1.value[2*i:2*i+2]) 

        #Colors the trajectories in their current LED colors
        if t>0:
            plt.plot(robots[i].locations[0][t-1:t+1], robots[i].locations[1][t-1:t+1], color = robots[i].LED, zorder=0) 
    
    #Plots the environment and robots
    fig.canvas.draw()
    fig.canvas.flush_events()



plt.ioff()
fig2 = plt.figure()

plt.plot(range(num_steps),robustness_history,label="robustness")
plt.title("Strongly $r$-robustness")
plt.show()

#Plot the evolutions of h_{r,c}'s values
for i in range(n-leaders):
    plt.plot(np.arange(num_steps)*dt, H[i], label="$h_{" + f"{r}," + str(i+1)+ '}$')
plt.plot(np.arange(num_steps)*dt, [0]*num_steps,linestyle='dashed', label="Safety Line", color = 'black')
# plt.legend(loc='upper right')
plt.title("$h_{"+f"{r}"+",c}$ values")
plt.xlabel("$t$")
plt.ylabel("$h_{"+f"{r}"+",c}$")
plt.yticks(np.arange(-0.05, 0.45, 0.1))
plt.show()


#Plot the evolutions of consensus values representing the RGB values
length_of_consensus = len(robots[0].history)*20
for aa in robots:
        temp = np.repeat(np.array(aa.history)/1000,20)
        if issubclass(type(aa), Malicious):
            plt.plot(np.arange(0,length_of_consensus)*dt,temp, "r--")
        elif issubclass(type(aa), Leaders):
            plt.plot(np.arange(0,length_of_consensus)*dt, temp, "b")
        else:
            plt.plot(np.arange(0,length_of_consensus)*dt, temp, "g")
plt.show()