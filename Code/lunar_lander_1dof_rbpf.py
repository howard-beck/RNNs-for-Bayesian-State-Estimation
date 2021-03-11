# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 23:48:46 2021

@author: Howard
"""



import math
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import filterpy as filter
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints
from enum import Enum

#import bpy


import matplotlib.pyplot as plt



from filterpy.monte_carlo import systematic_resample
bpy = {}



def resample_from_index(particles, weights, indexes):
    a = particles.T[indexes]
    particles[:] = a.T
    #weights.resize(len(particles))
    weights.fill (1.0 / N)





class Output(Enum):
    IMAGE = 1
    ALTIMER = 2

OUTPUT_TYPE = Output.ALTIMER

I_sp = 200
g0 = 9.80665
g = -1.622

dt = 1

stdv_Q_T = 20


temp_times = {}
times = {}

def startTimer(name):
    temp_times[name] = time.time_ns()

def endTimer(name):
    dt = (time.time_ns() - temp_times[name]) / 1000000
    
    if hasattr(times, name):
        times[name] += dt
    else:
        times[name] = dt
    
    return dt




if OUTPUT_TYPE == Output.IMAGE:
    # switch on nodes
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
      
    # clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
      
    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')      
    rl.location = 185,285
     
    # create output node
    v = tree.nodes.new('CompositorNodeViewer')   
    v.location = 750,210
    v.use_alpha = False
     
    # Links
    links.new(rl.outputs[0], v.inputs[0])  # link Image output to Viewer input
    
    camera = bpy.data.objects["Camera"]
    camera.location.x = 0
    camera.location.y = 0






P0 = np.array([
    [83.333333333**2, 0,                0],
    [0,               0.06666666666**2, 0],
    [0,               0,                1]
])

PARTICLE_DIM = 1
MARGIN_DIM = 2
STATE_DIM = PARTICLE_DIM + MARGIN_DIM

# covariance of particle state estimate
P0_r = P0[0:PARTICLE_DIM, 0:PARTICLE_DIM]
# covariance of marginalized state estimate
P0_x = P0[PARTICLE_DIM:STATE_DIM, PARTICLE_DIM:STATE_DIM]

# state estimate
zhat0 = np.array([1250, -8, 1179.34])
rhat0 = zhat0[0:PARTICLE_DIM].reshape((PARTICLE_DIM, 1))
xhat0 = zhat0[PARTICLE_DIM:STATE_DIM].reshape((MARGIN_DIM, 1))

r_state = np.random.multivariate_normal(rhat0.reshape(PARTICLE_DIM), P0_r).reshape((PARTICLE_DIM, 1))
x_state = np.random.multivariate_normal(xhat0.reshape(MARGIN_DIM), P0_x).reshape((MARGIN_DIM, 1))

rhat = rhat0
xhat = xhat0



stdv_R_img = 10/255
stdv_R_alt = 1
stdv_R = 0

# 10 / 255 pixels standard deviation
R_img = np.array([[stdv_R_img**2]])
R_alt = np.array([[stdv_R_alt**2]])
R = 0

OUTPUT_DIM = 1
if OUTPUT_TYPE == Output.IMAGE:
    OUTPUT_DIM = 4096
    R = R_img
    stdv_R = stdv_R_img
elif OUTPUT_TYPE == Output.ALTIMER:
    OUTPUT_DIM = 1
    R = R_alt
    stdv_R = stdv_R_alt




# measure image
def y_r_measurement_img(r):
    camera.location.z = r[0] / 1000
    bpy.ops.render.render()
    pixels = bpy.data.images['Viewer Node'].pixels
    arr = np.array(pixels[:])
    return arr[0::4]

# measure altimeter
def y_r_measurement_alt(r):
    return np.array([r[0, 0]])

def y_r_measurement(r):
    if OUTPUT_TYPE == Output.IMAGE:
        return y_r_measurement_img(r)
    elif OUTPUT_TYPE == Output.ALTIMER:
        return y_r_measurement_alt(r)

def noisify_output(y0):
    if OUTPUT_TYPE == Output.IMAGE:
        return y0 + np.random.multivariate_normal([0], R_img, OUTPUT_DIM).reshape(OUTPUT_DIM)
    elif OUTPUT_TYPE == Output.ALTIMER:
        return y0 + np.random.multivariate_normal([0], R_alt)



control = np.zeros((1, 1))
control[0, 0] = 3400

def deriv(r0, x0):
    r_ret = np.zeros(r0.shape)
    x_ret = np.empty(x0.shape)
    
    # y' = v
    r_ret[0] = x0[0]
    # x' = g + T/m
    x_ret[0] = g + control / x0[1]
    # m' = -T / I_sp g0
    x_ret[1] = -control / (I_sp * g0)
    
    return r_ret, x_ret


# assume dW ~ N(mean=0, cov=dt) or similar
def noisy_deriv_term(r0, x0, dW):
    r_ret = np.zeros(r0.shape)
    x_ret = np.empty(x0.shape)
    
    m = x0[1]
    
    # noise applied at the control level
    x_ret[0] = 1/m * dW * stdv_Q_T
    x_ret[1] = -dW * stdv_Q_T / (I_sp * g0)
    
    return r_ret, x_ret

def noisy_update(r0, x0):
    # get number of vectors (useful for noise generation)
    _, n = r0.shape
    
    r = np.copy(r0)
    x = np.copy(x0)
    
    N = 10
    h = dt/N
    sqrth = math.sqrt(h)
    
    S = (np.random.randint(0, 2, (N, n)).dot(2) - 1).dot(sqrth)
    dW = np.random.normal (0, 1, (N, n)).dot(sqrth)
    
    for i in range(N):
        dr1, dx1 = deriv(r, x)
        dr1 *= h
        dx1 *= h
        ddr1, ddx1 = noisy_deriv_term(r, x,  dW[i] - S[i])
        dr1 += ddr1
        dx1 += ddx1
        
        r1 = r + dr1
        x1 = x + dx1 
        
        dr2, dx2 = deriv(r1, x1)
        dr2 *= h
        dx2 *= h
        ddr2, ddx2 = noisy_deriv_term(r1, x1, dW[i] + S[i])
        dr2 += ddr2
        dx2 += ddx2
        
        r += (dr1 + dr2).dot(0.5)
        x += (dx1 + dx2).dot(0.5)
        
    return r, x

# RK4
def deterministic_update(r0, x0):
    k1_r, k1_x = deriv(r0            , x0            )
    k2_r, k2_x = deriv(r0 + k1_r*dt/2, x0 + k1_x*dt/2)
    k3_r, k3_x = deriv(r0 + k2_r*dt/2, x0 + k2_x*dt/2)
    k4_r, k4_x = deriv(r0 + k3_r*dt  , x0 + k3_x*dt  )

    return (r0 + dt/6*(k1_r + 2*k2_r + 2*k3_r + k4_r)), (x0 + dt/6*(k1_x + 2*k2_x + 2*k3_x + k4_x))


N = 1000
Neff = N

particles = np.random.multivariate_normal(rhat0.reshape(PARTICLE_DIM), P0_r, N).T
weights = np.ones(N) / N
kfs = []

particles_full_r = np.random.multivariate_normal(rhat0.reshape(PARTICLE_DIM), P0_r, N).T
particles_full_x = np.random.multivariate_normal(xhat0.reshape(MARGIN_DIM),   P0_x, N).T
weights_full = np.ones(N) / N



# last value of r (condition on this)
r_k0 = np.empty(PARTICLE_DIM)
r_k1_real = np.empty(PARTICLE_DIM)
T_k0 = np.empty((MARGIN_DIM, PARTICLE_DIM))

def x_update(x_k0):
    _, x_k1 = noisy_update(r_k0, x_k0)
    return x_k1

# last value of x
def x_measurement(x_k0):
    r_k1, _ = deterministic_update(r_k0, x_k0)
    return r_k1.reshape(PARTICLE_DIM)

def x_x_noiseless(x_k0, _):
    r_k1, x_k1 = deterministic_update(r_k0, x_k0.reshape((MARGIN_DIM, 1)))
    #print(T_k0.dot(r_k1_real - r_k1))
    #print(x_k1)
    #print(r_k1)
    return (x_k1 + T_k0.dot(r_k1_real - r_k1)).reshape(MARGIN_DIM) # transform to de-correlate noise between x and r



def calc_Qd(rhat, xhat):
    global Ad
    # r',x' = f(r, x) + Gw
    # r,x uses F(r, x) + w_d   w_d ~ Q_d
    Ac = np.array([[0, 1, 0], [0, 0, -control[0,0] / (xhat[1, 0]**2)], [0, 0, 0]])
    G = np.array([[0, stdv_Q_T / kfs[i].x_prior[1], -stdv_Q_T / (I_sp * g0)]])
    Ad, Qd = filter.common.van_loan_discretization(Ac, G, dt)
    
    return Qd

def calc_marginalized_Q_R_T(rhat, xhat):
    Qd = calc_Qd(rhat, xhat)
    
    Q_r = Qd[0:PARTICLE_DIM, 0:PARTICLE_DIM]
    Q_x = Qd[PARTICLE_DIM:STATE_DIM, PARTICLE_DIM:STATE_DIM]
    S = Qd[PARTICLE_DIM:STATE_DIM, 0:PARTICLE_DIM]
    
    T_k0 = S.dot(np.linalg.inv(Q_r))
    
    Q_x -= T_k0.dot(Q_r.dot(T_k0.T)) # noise decorrelation
    
    return Q_x, Q_r, T_k0
    
    
        
x_points = MerweScaledSigmaPoints(MARGIN_DIM, alpha=.001, beta=2., kappa=0)
for i in range(N):
    ukf = UnscentedKalmanFilter(MARGIN_DIM, PARTICLE_DIM, dt, x_measurement, x_x_noiseless, x_points)
    # x estimate
    ukf.x = xhat0.reshape(MARGIN_DIM)
    ukf.x_prior = ukf.x
    # covariance of estimate
    ukf.P_prior = P0_x
    # get around needing to call predict() before update() (not applicable here)
    ukf.sigmas_f = ukf.points_fn.sigma_points(ukf.x, ukf.P_prior)
    
    kfs.append(ukf)
    
    Q_m, R_m, _ = calc_marginalized_Q_R_T(particles[:, i:i+1], xhat0)
    ukf.Q = Q_m
    ukf.R = R_m



def y_z_measurement(z):
    return y_r_measurement(z[0:PARTICLE_DIM].reshape((PARTICLE_DIM, 1)))

def z_z_update(z, _):
    r = z[0:PARTICLE_DIM].reshape((PARTICLE_DIM, 1))
    x = z[PARTICLE_DIM:STATE_DIM].reshape((MARGIN_DIM, 1))
    
    r, x = deterministic_update(r, x)
    
    z_ret = np.empty(z.shape)
    z_ret[0:PARTICLE_DIM] = r.reshape(PARTICLE_DIM)
    z_ret[PARTICLE_DIM:STATE_DIM] = x.reshape(MARGIN_DIM)
    
    return z_ret

#if USE_UKF:
z_points = MerweScaledSigmaPoints(STATE_DIM, alpha=.001, beta=2., kappa=0)
UKF = UnscentedKalmanFilter(STATE_DIM, OUTPUT_DIM, dt, y_z_measurement, z_z_update, z_points)
UKF.P = P0
UKF.x = zhat0.reshape(STATE_DIM)
UKF.Q = calc_Qd(rhat0, xhat0)
UKF.R = R


J = 60
Tmax = J*dt
ts = np.arange(0, Tmax, dt)


startTimer("main")
control[0, 0] = 3400

step_time = 0



Ac = []
Ad = []
G = []
Qd = []
v = []
z_mean = 0
r_mean = 0
x_mean = 0
z_cov = 0
r_cov = 0
x_cov = 0
state = 0

m_dist_rbpf = []
m_dist_ukf = []
m_dist_pf = []

Ts = []

    
dt_est = 0

j = 0

state = np.empty((STATE_DIM, 1))

ys = []
ys_est_rbpf = []
ys_est_ukf = []

for t in ts:
    j += 1
    
    t_go = Tmax - t
    
    yhat = rhat[0,0]
    vhat = xhat[0,0]
    mhat = xhat[1, 0]
    
    ZEM = 50 - (yhat + vhat*t_go + g*t_go**2)
    ZEV = 0 - (vhat + g*t_go)
    
    ac = 6/t_go**2 * ZEM - 2/t_go * ZEV
    T = ac * mhat
    
    if T > 3400:
        T = 3400
    if T < 0:
        T = 0
    
    Ts.append(T)
    control[0, 0] = T
    
    startTimer("state update")
    r_state, x_state = noisy_update(r_state, x_state)
    endTimer("state update")
    
    ys.append(r_state[0, 0])
    
    startTimer("measurement")
    y0 = y_r_measurement(r_state)
    y = noisify_output(y0)
    endTimer("measurement")
    
    # main UKF estimation
    #print(UKF.Q)
    #print(UKF.R)
    UKF.predict()
    #print("UKF P_prior =")
    #print(UKF.P_prior)
    UKF.update(y)
    # calculate process covariance for next iteration
    rhat_ukf = UKF.x_post[0:PARTICLE_DIM].reshape((PARTICLE_DIM, 1))
    xhat_ukf = UKF.x_post[PARTICLE_DIM:STATE_DIM].reshape((MARGIN_DIM, 1))
    UKF.Q = calc_Qd(rhat_ukf, xhat_ukf)
    
    
    startTimer("draw x")
    xs = np.empty((MARGIN_DIM, N))
    # CAN BE PARALLELIZED!!!
    for i in range(N):
        # draw from p(x_t | y_0:t, r_0:t) requires sampling x_t given p(x_t-1 | y_0:t, r_t)
        # last measurement for x_t-1 = posterior
        # thus x_t has the prior covariance
        # in order to draw from p(r_t+1 | r_t)
        xs[:, i] = np.random.multivariate_normal(kfs[i].x, kfs[i].P_prior)
    endTimer("draw x")
    
    particles_full_r, particles_full_x = noisy_update(particles_full_r, particles_full_x)
    weights_full *= scipy.stats.multivariate_normal(y, R).pdf(particles_full_r[0])
    weights_full /= np.sum(weights_full)
    
    startTimer("particle update")
    # propogate particles
    new_particles, _ = noisy_update(particles, xs)
    endTimer("particle update")
    
    startTimer("reweigh")
    ws = np.empty((N, OUTPUT_DIM))
    for i in range(N):
        startTimer("rendering")
        y_part = y_r_measurement(new_particles[:, i:i+1])
        dt_render = endTimer("rendering") / 1000
        
        if dt_est == 0:
            dt_est = dt_render
        else:
            dt_est = 0.7*dt_est + 0.3*dt_render
        
        n_ops = (J - j) * N + (N - i - 1)
        
        # convert estimated time to hours
        t_est = n_ops * dt_est / 3600
        # get hour part
        hours = math.floor(t_est)
        # get remainder
        t_est = (t_est - hours)*60
        # minutes part
        minutes = math.floor(t_est)
        # secodnds
        t_est = (t_est - minutes)*60
        seconds = t_est
        #
        #print(str(i) + " / 1000 | Estimated time left: " + str(hours) + "h" + str(minutes) + "m" + str(seconds) + "s")
        
        e = (y_part - y) / stdv_R
        
        ws[i] = np.e**(-e*e/2)
    scaleW = np.sum(ws) / (N * OUTPUT_DIM)
    #minW = np.min(ws)
    ws /= scaleW
    
    for i in range(N):
        weights[i] *= np.product(ws[i])
    
    weights /= np.sum(weights)
    #print("weights=")
    #print(weights)
    endTimer("reweigh")
    
    
    
    # now, new r_t+1 available --> can find posterior for x_t
    # r_t not a function of x_t, but r_t+1 is a function of x_t
    startTimer("ukf_update")
    for i in range(N):
        # global variable, used to condition state transition from x_t to x_t+1
        r_k0 = particles[:, i:i+1]
        r_k1 = new_particles[:, i]
        # update given new r_k1 to condition on
        kfs[i].update(r_k1)
        
        Q, R_particles, T_k0 = calc_marginalized_Q_R_T(r_k0, kfs[i].x_prior.reshape((MARGIN_DIM, 1)))
        
        kfs[i].Q = Q
        kfs[i].R = R_particles
        
        # real old output for noise decorrelation
        r_k1_real = r_k1.reshape(PARTICLE_DIM, 1)
        
        kfs[i].predict()
    endTimer("ukf_update")
    
    
    
    state[0:PARTICLE_DIM] = r_state
    state[PARTICLE_DIM:STATE_DIM] = x_state
    
    z_mean = np.zeros((STATE_DIM, 1))
    z_cov  = np.zeros((STATE_DIM, STATE_DIM))
    r_mean = np.zeros((PARTICLE_DIM, 1))
    x_mean = np.zeros((MARGIN_DIM, 1))
    r_cov  = np.zeros((PARTICLE_DIM, PARTICLE_DIM))
    x_cov  = np.zeros((MARGIN_DIM, MARGIN_DIM))
    rx_cov = np.zeros((PARTICLE_DIM, MARGIN_DIM))
    
    v = np.empty((STATE_DIM, 1))
    
    particles = new_particles
    
    z_mean_pf = np.zeros((STATE_DIM, 1))
    z_cov_pf = np.zeros((STATE_DIM, STATE_DIM))
    
    startTimer("calc")
    for i in range(N):
        r_est = particles[:, i:i+1]
        x_est = kfs[i].x_prior.reshape((MARGIN_DIM, 1))
        
        
        r_mean += weights[i] * r_est
        x_mean += weights[i] * x_est
        
        r_cov  += weights[i] * r_est.dot(r_est)            
        x_cov  += weights[i] * (kfs[i].P_prior + x_est.dot(x_est.T))
        rx_cov += weights[i] * r_est.dot(x_est.T)
        
        z_pf = np.empty((STATE_DIM, 1))
        z_pf[0:PARTICLE_DIM] = particles_full_r[:, i:i+1]
        z_pf[PARTICLE_DIM:STATE_DIM] = particles_full_x[:, i:i+1]
        
        z_mean_pf += weights_full[i] * z_pf
        z_cov_pf  += weights_full[i] * z_pf.dot(z_pf.T)
        
     
    #r_cov -= r_mean.dot(r_mean.T)
    #x_cov -= x_mean.dot(x_mean.T)
    
    z_mean[0:PARTICLE_DIM,         :] = r_mean
    z_mean[PARTICLE_DIM:STATE_DIM, :] = x_mean
    
    rhat = r_mean
    xhat = x_mean
    
    ys_est_rbpf.append(z_mean[0, 0])
    ys_est_ukf.append(UKF.x[0])
    
    z_cov[0:PARTICLE_DIM, 0:PARTICLE_DIM] = r_cov
    z_cov[PARTICLE_DIM:STATE_DIM, PARTICLE_DIM:STATE_DIM] = x_cov
    z_cov[0:PARTICLE_DIM, PARTICLE_DIM:STATE_DIM] = rx_cov
    z_cov[PARTICLE_DIM:STATE_DIM, 0:PARTICLE_DIM] = rx_cov.T
    
    z_cov -= z_mean.dot(z_mean.T)
    z_cov_pf -= z_mean_pf.dot(z_mean_pf.T)
    
    print("State covariance =")
    print(z_cov)
    print("State mean =")
    print(z_mean)
    print("Ground truth =")
    print(state)
    print("t = " + str(t) + "/" + str(Tmax))
    
    #startTimer("num")
    z_error = z_mean - state
    num = math.sqrt((z_error.T.dot(np.linalg.inv(z_cov)).dot(z_error))[0, 0])
    z_error_ukf = UKF.x.reshape((STATE_DIM, 1)) - state
    num_ukf = math.sqrt((z_error_ukf.T.dot(np.linalg.inv(UKF.P)).dot(z_error_ukf))[0, 0])
    #z_error_pf = z_mean_pf - state
    #oop = z_error_pf.T.dot(np.linalg.inv(z_cov_pf)).dot(z_error_pf)
    print("Mahalanobis distance =")
    print(num)
    #print("Mahalanobis distance UKF =")
    #print(num_ukf)
    #num_pf = math.sqrt(oop[0, 0])
    #print("Mahalanobis distance PF =")
    #print(num_pf)
    #endTimer("num")
    #endTimer("calc")
    print("Thrust = " + str(T))
    
    m_dist_rbpf.append(num)
    m_dist_ukf.append(num_ukf)
    #m_dist_pf.append(num_pf)
    
    Neff = 1/np.sum(weights**2)
    print("Neff =")
    print(Neff)
    
    #if Neff < N/2:
    # always resample
    indexes = systematic_resample(weights)
    resample_from_index(particles, weights, indexes)
    
    indexes = systematic_resample(weights_full)
    resample_from_index(particles_full_r, weights_full, indexes)
    resample_from_index(particles_full_x, weights_full, indexes)
endTimer("main")

print("times =")
print(times)

plt.figure(1)
plt.clf()
fig, ax = plt.subplots(num = 1)
ax.plot(ts, m_dist_rbpf,  'r-', label='RBPF')
ax.plot(ts, m_dist_ukf, 'b-', label='UKF')
#ax.plot(ts, m_dist_pf, 'g-', label='PF')
ax.legend(loc='best')

plt.figure(2)
plt.clf()
fig, ax = plt.subplots(num = 2)
ax.plot(ts, ys,  'r-', label='Real')
ax.plot(ts, ys_est_rbpf, 'b-', label='RBPF')
ax.plot(ts, ys_est_ukf, 'g-', label='UKF')
ax.legend(loc='best')

plt.figure(3)
plt.clf()
fig, ax = plt.subplots(num = 3)
ax.plot(ts, Ts,  'k-', label='Thrust')
ax.legend(loc='best')