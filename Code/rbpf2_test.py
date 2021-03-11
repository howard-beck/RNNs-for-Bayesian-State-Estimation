# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 12:45:03 2021

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


import matplotlib.pyplot as plt



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






def drawEllipse(cov, ax, avg, K=3, color="r-"):
    angles = np.linspace(0, math.pi*2, 100)
    w, v = np.linalg.eig(np.linalg.inv(cov))
    lambda1 = w[0]
    lambda2 = w[1]
    e1 = v[:,0]
    e2 = v[:,1]
    px = avg[0] + K*(1/np.sqrt(lambda1)*np.sin(angles)*e1[0] + 1/np.sqrt(lambda2)*np.cos(angles)*e2[0])
    py = avg[1] + K*(1/np.sqrt(lambda1)*np.sin(angles)*e1[1] + 1/np.sqrt(lambda2)*np.cos(angles)*e2[1])
    ax.plot(px, py, color, label="3-sigma UKF")




X = 0
Y = 1
THETA = 2

STATE_DIM = 3

stdv_Qc = 0.1
Qc = np.zeros((3, 3))
Qc[THETA, THETA] = stdv_Qc**2

stdv_R = 0.1
R = np.eye(2, 2) * stdv_R**2



xhat0 = np.array([2, 0, math.pi/4])
P0 = np.array([
    [0.5**2, 0,      0],
    [0,      0.5**2, 0],
    [0,      0,      (math.pi/36)**2]
])
state = np.random.multivariate_normal(xhat0, P0)

state_r = state[0:2].reshape(2, 1)
state_x = state[2:3].reshape(1, 1)

dt = 0.1



def y_measurement(r):
    return r.reshape(2)

def y_r_pdf(r, y):
    return scipy.stats.multivariate_normal(y, R).pdf(r)



def noiseless_rk4_solver(r0, x0):
    k1_r, k1_x = deriv(r0            , x0            )
    k2_r, k2_x = deriv(r0 + k1_r*dt/2, x0 + k1_x*dt/2)
    k3_r, k3_x = deriv(r0 + k2_r*dt/2, x0 + k2_x*dt/2)
    k4_r, k4_x = deriv(r0 + k3_r*dt  , x0 + k3_x*dt  )

    return (r0 + dt/6*(k1_r + 2*k2_r + 2*k3_r + k4_r)), (x0 + dt/6*(k1_x + 2*k2_x + 2*k3_x + k4_x))

def deriv(r0, x0):
    ret_r = np.empty(r0.shape)
    ret_x = -np.ones(x0.shape)
    
    # r0' = cos x + r0
    # r1' = sin x - r1
    # x' = -1
    ret_r[0] = np.cos(x0) + r0[1]
    ret_r[1] = np.sin(x0) - r0[0]
    
    return ret_r, ret_x

G = np.array([0, 0, stdv_Qc]).reshape(3, 1)

def noisy_rk4_solver(r0, x0):
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
        dx1 += (dW[i] - S[i])*stdv_Qc
        
        r1 = r + dr1
        x1 = x + dx1
        
        dr2, dx2 = deriv(r1, x1)
        dr2 *= h
        dx2 *= h
        dx2 += (dW[i] + S[i])*stdv_Qc
        
        r += (dr1 + dr2).dot(0.5)
        x += (dx1 + dx2).dot(0.5)
        
    return r, x


rhat0 = xhat0[0:2]
P0_r = P0[0:2, 0:2]
P0_x = [[P0[2, 2]]]

N = 100

particles = np.random.multivariate_normal(rhat0, P0_r, N).T
weights   = np.ones(N) / N
kfs = []

# last value of r (condition on this)
r_k0 = np.empty(2)
r_k1_real = np.empty(2)
T_k0 = np.empty((1, 1))

def x_update(x_k0):
    _, x_k1 = noisy_rk4_solver(r_k0, x_k0)
    return x_k1

# last value of x
def x_measurement(x_k0):
    r_k1, _ = noiseless_rk4_solver(r_k0, x_k0)
    return r_k1.reshape(2)

def x_x_noiseless(x_k0, _):
    r_k1, x_k1 = noiseless_rk4_solver(r_k0, x_k0)
    #print(T_k0.dot(r_k1_real - r_k1))
    return (x_k1)# + T_k0.dot(r_k1_real - r_k1)) # transform to de-correlate noise between x and r

points = MerweScaledSigmaPoints(1, alpha=.001, beta=2., kappa=2)
for i in range(N):
    ukf = UnscentedKalmanFilter(1, 1, dt, x_measurement, x_x_noiseless, points)
    # x estimate
    ukf.x = [xhat0[2]]
    ukf.x_prior = ukf.x
    # covariance of estimate
    ukf.P_prior = P0_x
    # get around needing to call predict() before update() (not applicable here)
    ukf.sigmas_f = ukf.points_fn.sigma_points(ukf.x, ukf.P_prior)
    
    kfs.append(ukf)

def calc_Neff():
    return 1/np.sum(weights**2)

ts = np.arange(0, 30*dt, dt)

def z_measurement(z):
    return y_measurement(z[0:2].reshape((2, 1)))

def z_z_noiseless(z, dt):
    r = z[0:2].reshape((2, 1))
    x = z[2].reshape((1, 1))
    
    r, x = noiseless_rk4_solver(r, x)
    
    ret = np.empty(3)
    ret[0:2] = r.reshape(2)
    ret[2] = x.reshape(1)
    
    return ret
    
points = MerweScaledSigmaPoints(3, alpha=.001, beta=2., kappa=0)
UKF = UnscentedKalmanFilter(3, 1, dt, z_measurement, z_z_noiseless, points)
UKF.x = xhat0
UKF.P = P0

z = np.empty((3, 1))
 
# P_{0 | 0} known
# can draw from x_0 to update r_0 into r_1
for t in ts:
    print("Starting x:")
    print(state_x)
    state_r, state_x = noisy_rk4_solver(state_r, state_x)
    print("Ending x:")
    print(state_x)
    
    z[0:2, 0:1] = state_r
    z[2:3, 0:1] = state_x
    
    y = np.random.multivariate_normal(y_measurement(state_r), R)
    
    Ac = np.array([[0, 1, -math.sin(UKF.x[2])], [-1, 0, math.cos(UKF.x[2])], [0, 0, 0]])
    Ad, Qd = filter.common.van_loan_discretization(Ac, G, dt)
    
    UKF.Q = Qd
    
    UKF.predict()
    UKF.update(y)
    
    
    #startTimer("ukf_predict")
    # r_t not a function of x_t, but r_t+1 is a function of x_t
    #for i in range(N):
        # global variable, used to condition state transition from x_t to x_t+1
    #    r_k0 = particles[0:2, i:i+1]
        # predict step
    
    #endTimer("ukf_predict")
    
    #startTimer("discretization")
    #for i in range(N):
        # r',x' = f(r, x) + Gw
        # r,x uses F(r, x) + w_d   w_d ~ Q_d
    #    Ac = np.array([[0, 0, -math.sin(kfs[i].x[0])], [0, 0, -math.cos(kfs[i].x[0])], [0, 0, 0]])
    #    Ad, Qd = filter.common.van_loan_discretization(Ac, G, dt)
        
    #   Q_x = Qd[2:3, 2:3]
    #   Q_r = Qd[0:2, 0:2]
    #   S = Qd[2:3, 0:2]
        
    #   T_k0 = S.dot(np.linalg.inv(kfs[i].R))
        
    #   kfs[i].Q = Q_x - T_k0.dot(R.dot(T_k0.T))
    #   kfs[i].R = Q_r
    #   kfs[i].predict()
    
    #endTimer("discretization")
    
    startTimer("draw_x")
    xs = np.empty(N)
    # CAN BE PARALLELIZED!!!
    for i in range(N):
        # draw from p(x_t | y_0:t, r_0:t) requires sampling x_t given p(x_t-1 | y_0:t, r_t)
        # last measurement for x_t-1 = posterior
        # thus x_t has the prior covariance
        # in order to draw from p(r_t+1 | r_t)
        xs[i] = np.random.multivariate_normal(kfs[i].x, kfs[i].P_prior)
    endTimer("draw_x")
    
    startTimer("update_particles")
    new_particles, _ = noisy_rk4_solver(particles, xs)
    endTimer("update_particles")
    
    startTimer("reweigh")
    weights *= y_r_pdf(new_particles.T, y)
    endTimer("reweigh")
    # normalize weights
    weights /= np.sum(weights)
    
    # now, new r_t+1 available --> can find posterior for x_t
    # r_t not a function of x_t, but r_t+1 is a function of x_t
    startTimer("ukf_update")
    for i in range(N):
        # global variable, used to condition state transition from x_t to x_t+1
        r_k0 = particles[0:2, i:i+1]
        r_k1 = new_particles[0:2, i]
        # updat given new rk1 to condition on
        kfs[i].update(r_k1)
        
        # r',x' = f(r, x) + Gw
        # r,x uses F(r, x) + w_d   w_d ~ Q_d
        Ac = np.array([[0, 1, -math.sin(kfs[i].x_prior[0])], [-1, 0, math.cos(kfs[i].x_prior[0])], [0, 0, 0]])
        Ad, Qd = filter.common.van_loan_discretization(Ac, G, dt)
        
        Q_x = Qd[2:3, 2:3]
        Q_r = Qd[0:2, 0:2]
        S   = Qd[2:3, 0:2]
        
        T_k0 = S.dot(np.linalg.inv(Q_r))
        
        kfs[i].Q = Q_x# - T_k0.dot(Q_r.dot(T_k0.T))
        kfs[i].R = Q_r
        
        # real old output for noise decorrelation
        r_k1_real = r_k1.reshape(2, 1)
        
        kfs[i].predict()
    endTimer("ukf_update")
    
    z_mean = np.zeros((3, 1))
    z_cov  = np.zeros((3, 3))
    
    v = np.empty((3, 1))
    
    particles = new_particles
    
    for i in range(N):
        v[0:2] = particles[0:2, i:i+1]
        v[2]   = kfs[i].x_post
        
        z_mean += weights[i] * v
        z_cov  += weights[i] * v.dot(v.T)
    z_cov -= z_mean.dot(z_mean.T)
    
    z_error = z_mean - z
    num = math.sqrt((z_error.T.dot(np.linalg.inv(z_cov)).dot(z_error))[0, 0])

print(r_k0)

# Ac --> Ad
# x' = Ac x + G w
# x[k + 1] = Ad x + Gd w
# P[k] --> P[k + 1] = Ad P Ad^T + Gd Gd^T
N = 100
P = np.zeros((3, 3))
mean = np.zeros((3, 1))
for i in range(N):
    # dim r = 2
    # dim x = 1
    r_k1, x_k1 = noisy_rk4_solver(r_k0, xhat0[2:3])
    
    v = np.zeros((3, 1))
    v[0:2] = r_k1.reshape((2, 1))
    v[2] = x_k1
    
    mean += v / N
    P += v.dot(v.T) / N

P = P - mean.dot(mean.T)



plt.figure(1)
plt.clf()
fig, ax = plt.subplots(num=1)
drawEllipse(Q_r, ax, [0, 0], color="r-")
drawEllipse(P[0:2, 0:2], ax, [0, 0], color="b-")