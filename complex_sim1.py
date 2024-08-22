import time
import helper.bootstrap_percolation as dp
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from helper.obstacles import *
from helper.double_integrator import *
from random import randint
from jax import lax, jit, jacrev, hessian


def unsmoothened_adjacency(dif, A, robots):
    n= len(robots)
    for i in range(n):
        for j in range(i+1, n):
            norm = jnp.linalg.norm(robots[i]-robots[j])
            if norm <= dif:
                A[i,j] =1
                A[j,i] =1

plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-5,5),ylim=(-5,7)) 
ax.set_xlabel("X")
ax.set_ylabel("Y")

########################## Make Obatacles ###############################
obstacles = []
index = 0
x1 = -1.2#-1.0
x2 = 1.2 #1.0
radius = 0.6
y_s = 0
y_increment = 0.3
for i in range(int( 5/y_increment )):
    obstacles.append( circle( x1,y_s,radius,ax,0 ) ) # x,y,radius, ax, id
    obstacles.append( circle( x2,y_s,radius,ax,1 ) )
    y_s = y_s + y_increment
obstacles.append(circle(-0.9,0.5,radius,ax,0))
obstacles.append(circle(0.9,1.0,radius,ax,0))
obstacles.append(circle(-0.9,1.9,radius,ax,0))
obstacles.append(circle(0.9,2.5,radius,ax,0))
obstacles.append(circle(0.0,4.0,.3,ax,0))

obstacles.append(circle(-1.4, 0.1,radius,ax,0))
obstacles.append(circle(1.4, 0.1,radius,ax,0))
obstacles.append(circle(-1.6, -0.1,radius,ax,0))
obstacles.append(circle(1.6, -0.1,radius,ax,0))

num_obstacles = len(obstacles)


########################################################################

# Sim Parameters                  
num_steps = 800
robots = []
y_offset = -1.5
leaders = 4
F = 1
broadcast_value = randint(600,766)
robots.append( Leaders(broadcast_value, np.array([-0.7,y_offset,0,0]),'b',1.0, ax,F))
robots.append( Leaders(broadcast_value, np.array([0,y_offset,0,0]),'b',1.0, ax, F))
robots.append( Leaders(broadcast_value, np.array([-0.3,y_offset - 2.1,0,0]),'b',1.0, ax, F))
robots.append( Leaders(broadcast_value, np.array([1.1,y_offset - 2.4,0,0]),'b',1.0, ax, F))
robots.append( Agent(np.array([-1.1,y_offset - 0.5,0,0]),'g',1.0, ax, F))
robots.append( Agent(np.array([-0.7,y_offset - 1.0,0,0]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.8,y_offset - 1.2,0,0]),'g',1.0 , ax, F))
robots.append( Malicious([0,500],np.array([1.1,y_offset - 1.7,0,0]),'r',1.0 , ax, F))
robots.append( Agent(np.array([-1.0,y_offset - 2.1,0,0]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.5,y_offset - 1.6,0,0]),'g',1.0 , ax, F))
robots.append( Agent(np.array([-0.5,y_offset - 1.9,0,0]),'g',1.0 , ax, F))
robots.append( Agent(np.array([-0.8,y_offset - 1.9,0,0]),'g',1.0 , ax, F))
robots.append( Agent(np.array([-0.5,y_offset - 2.4,0,0]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.1,y_offset - 1.3,0,0]),'g',1.0 , ax, F))
robots.append( Agent(np.array([0.5,y_offset - 2.2,0,0]),'g',1.0 , ax, F))


num_robots = n =len(robots)
inter_collision = int(n*(n-1)/2)
############################## Optimization problems ######################################

# ###### 1: CBF Controller
u1 = cp.Variable((2*n,1))
u1_ref = cp.Parameter((2*n,1),value = np.zeros((2*n,1)) )
num_constraints1  = 1
A1 = cp.Parameter((num_constraints1,2*n),value=np.zeros((num_constraints1,2*n)))
b1 = cp.Parameter((num_constraints1,1),value=np.zeros((num_constraints1,1)))
const1 = [A1 @ u1 >= b1]
objective1 = cp.Minimize( cp.sum_squares( u1 - u1_ref  ) )
cbf_controller = cp.Problem( objective1, const1 )
###################################################################################################
dif = 3
# r=1
r= 3


#Setting the goal
goal = []
destinations = [0 for i in range(leaders)]
for k in range(leaders):
    goal.append(np.array([destinations[k % leaders], 10]).reshape(2,-1))
robustness_history = []
H = [[] for i in range(n-leaders)]
u_r =[];u_l=[]
q1 = 0.02
p1 = jnp.log(1/q1)
q2 = 0.02
p2 = jnp.log(1/q2)
s_A = 8
s = 3
relu = lambda x: (1+q1)/(1+jnp.exp(-s_A*x+p1))-q1
relu2 = lambda x: (1+q2)/(1+jnp.exp(-s*x+ p2))-q2

@jit 
def barrier_func(x):
    def AA(x):
        A = jnp.array([[0.0 for i in range(n)] for j in range(n)]) 
        for i in range(n):
            for j in range(i+1, n):
                dis = dif-jnp.linalg.norm(x[i]-x[j])
                A = A.at[j,i].set(relu(dis))
                A = A.at[i,j].set(relu(dis))  
        return A
    def body(i, inputs):
        temp_x = A @ jnp.concatenate([jnp.array([1.0 for p in range(leaders)]),inputs])
        state_vector = relu2(temp_x[leaders:]-r)
        return state_vector
    
    state_vector = jnp.array([0.0 for p in range(n-leaders)])
    A = AA(x)
    delta = 4
    x = lax.fori_loop(0, delta, body, state_vector) 
    return x

barrier_grad = jit(jacrev(barrier_func))
barrier_double_grad = jit(hessian(barrier_func))

def smoothened_strongly_r_robust_simul(robots, dif, r):   
    h = barrier_func(robots)
    h_dot = barrier_grad(robots)
    h_ddot =  barrier_double_grad(robots)
    return h, h_dot, h_ddot


weight = np.array([6]*(num_robots-leaders) + [18]*inter_collision + [18]*(num_obstacles*n))
compiled = jit(smoothened_strongly_r_robust_simul)

counter = 0
while True:   
    robots_location = np.array([aa.x.reshape(1,-1)[0] for aa in robots])
    robots_velocity = np.array([aa.v.reshape(1,-1)[0] for aa in robots])
    A = np.zeros((n, n))
    unsmoothened_adjacency(dif, A, robots_location)
    delta = np.count_nonzero(A)
    robustness_history.append(dp.strongly_r_robust(A,leaders, delta))

    #Get the nominal control input
    for i in range(n):
        vector = goal[i % leaders] - robots[i].location[:2] 
        vector = vector/np.linalg.norm(vector)
        current = robots[i].location[2:4] 
        temp = np.array([[vector[0][0]-current[0]], [vector[1][0]-current[1]]])
        u1_ref.value[2*i] = temp[0][0]
        u1_ref.value[2*i+1] = temp[1][0]
    #Perform W-MSR
    if counter/25 % 1==0:
        for i in range(n):
            for j in range(i+1,n):
                if A[i,j] ==1:
                    robots[i].connect(robots[j])
                    robots[j].connect(robots[i])
        for aa in robots:
            aa.propagate()
        for aa in robots:
            aa.w_msr()
        for aa in robots:
            aa.set_color()

    # h_{3,c}, gradient, and hessian
    
    x, der_, double_der_  = compiled(robots_location, dif, r)
    x=np.asarray(x);der_=np.asarray(der_);double_der_=np.asarray(double_der_)
    print(counter, x)


    A1.value[0,:] = [0 for i in range(2*num_robots)]
    b1.value[0] = 0

    #Obstacle Collision avoidance and Inter-agent collision avoidance
    collision = [];ob=[]
    col_alpha = 1.5; obs_alpha = 4
    for i in range(num_robots):
        for j in range(i+1, num_robots):
            h, dh_dxi, dh_dxj, ddh = robots[i].agent_barrier(robots[j], 0.30)
            h_dot = dh_dxi @ robots_velocity[i] + dh_dxj @ robots_velocity[j] + col_alpha*h
            collision.append(h_dot)
            kk = num_robots-leaders+j
            temp = (weight[kk])*np.exp(-weight[kk]*h_dot)
            A1.value[0,2*i:2*i+2]+= temp * dh_dxi[:]
            A1.value[0,2*j:2*j+2]+= temp *dh_dxj[:]
            b1.value[0]-=  temp *(2* robots_velocity[i].reshape(1,-1)[0] @  robots_velocity[i] - 2* robots_velocity[i].reshape(1,-1)[0] @  robots_velocity[j] + col_alpha*dh_dxi @ robots_velocity[i])
            b1.value[0]-=  temp *(2* robots_velocity[j].reshape(1,-1)[0] @  robots_velocity[j] - 2* robots_velocity[i].reshape(1,-1)[0] @  robots_velocity[j] + col_alpha*dh_dxj @ robots_velocity[j])
        for j in range(num_obstacles):
            h, dh_dxi, dh_dxj, ddh = robots[i].agent_barrier(obstacles[j],obstacles[j].radius+0.1)
            h_dot =  dh_dxi @ robots_velocity[i] + obs_alpha*h
            ob.append(h_dot)
            kk = num_robots-leaders+inter_collision+j
            temp = (weight[kk])*np.exp(-weight[kk]*h_dot)
            A1.value[0,2*i:2*i+2]+= temp * dh_dxi[:]
            b1.value[0]-= temp *(2* robots_velocity[i].reshape(1,-1)[0] @  robots_velocity[i] + obs_alpha*dh_dxi @ robots_velocity[i])
    
    #Robustness HOCBF 
    alphas = 2
    robustes = []
    for k in range(num_robots-leaders):
        h_dot = der_[k].reshape(1,-1)[0] @ robots_velocity.reshape(-1,1)
        weightee = ((weight[k])*np.exp(-weight[k]*(h_dot+alphas*x[k])))[0]
        robustes.append((h_dot+alphas*x[k])[0])
        A1.value[0,:]+= weightee * der_[k].reshape(1,-1)[0]

        temp = []
        for j in range(num_robots):
            temp_x = robots_velocity.reshape(1,-1) @ double_der_[k][j][:][:][0].reshape(-1,1)
            temp_y = robots_velocity.reshape(1,-1) @ double_der_[k][j][:][:][1].reshape(-1,1)
            temp.append(temp_x[0])
            temp.append(temp_y[0]) 
        b1.value[0]-= weightee * ((np.array(temp).reshape(1,-1) + alphas*der_[k].reshape(1,-1)) @ robots_velocity.reshape(-1,1))[0]
        
    for i in range(n-leaders):
        H[i].append(x[i])

    #Composition 
    sum_h = 1 - np.sum(np.exp(-weight*np.array(robustes + collision + ob)))
    b1.value[0]-=2*(sum_h)

    #Solve the CBF-QP and get the control input \mathbf u
    cbf_controller.solve(solver=cp.GUROBI)
    if cbf_controller.status!='optimal':
        print("Error: should not have been infeasible here")
        print(A1.value)
    # implement control input \mathbf u and plot the trajectory
    for i in range(num_robots):
        robots[i].step2( u1.value[2*i:2*i+2]) 
        if counter>0:
            plt.plot(robots[i].locations[0][counter-1:counter+1], robots[i].locations[1][counter-1:counter+1], color = robots[i].LED, zorder=0)            

    fig.canvas.draw()
    fig.canvas.flush_events()  
    for aa in robots_location:
        if aa[1]<=4.0:
            break
    else:
        break
    counter+=1

counter+=1

plt.ioff()
fig2 = plt.figure()

#Plot the robustness history
plt.plot(range(counter),robustness_history,label="robustness")
plt.title("Strongly $r$-robustness")
plt.show()

#Plot the evolutions of h_{r,c}'s values
for i in range(n-leaders):
    plt.plot(range(counter), H[i], label="$h_{" + f"{r}," + str(i+1)+ '}$')
plt.plot(range(counter), [0]*len(range(counter)),linestyle='dashed', label="Safety Line", color = 'black')
# plt.legend(loc='upper right')
plt.title("$h_{"+f"{r}"+",c}$ values")
plt.xlabel("$t$")
plt.ylabel("$h_{"+f"{r}"+",c}$")
plt.yticks(np.arange(-0.1, 1.3, 0.2))
plt.show()