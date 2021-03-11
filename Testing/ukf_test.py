import math

import filterpy as filter
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt



THETA = 0
OMEGA = 1

STATE_DIM = 2

stdv_Qc = 100*math.pi/180
Qc = np.zeros((2, 2))
Qc[OMEGA, OMEGA] = stdv_Qc**2

stdv_R = 5*math.pi/180
R = stdv_R**2

xhat0 = np.array([math.pi/4, math.pi/6])

P0 = np.array([(5*math.pi/180)**2, 0, 0, (5*math.pi/180)**2]).reshape((2, 2))
state = xhat0
state[THETA] += np.random.normal(0, 5*math.pi/180);
state[OMEGA] += np.random.normal(0, 5*math.pi/180);

dt = 0.1

def deriv(t, x, noise):
    theta = x[THETA]
    omega = x[OMEGA]
    
    ret = np.empty(STATE_DIM)
    
    ret[0] = omega # theta' = omega
    ret[1] = -math.sin(theta) # omega' = -sin theta
    
    if noise:
        ret[1] += np.random.normal(0, stdv_Qc);
    
    return ret

def derivNoNoise(t, x):
    return deriv(t, x, False)

def derivWithNoise(t, x):
    return deriv(t, x, True)

def noiselessUpdate(x, dt):
    sol = solve_ivp(derivNoNoise, [0, dt], x, method="RK45", t_eval = [dt])
    
    return sol.y.reshape(2)


def measurement(xhat):
    return [xhat[THETA]]

points = MerweScaledSigmaPoints(STATE_DIM, alpha=.1, beta=2., kappa=-1)
ukf = UnscentedKalmanFilter(STATE_DIM, 1, dt, measurement, noiselessUpdate, points)

ukf.x = xhat0
ukf.P = P0
ukf.R = R

ts = np.arange(0, 20, dt)

thetas = []
ys = []
yhats = []

for t in ts:
    sol = solve_ivp(derivWithNoise, [t, t + dt], state, method="RK45", t_eval = [t + dt])
    
    state = sol.y.reshape(2)
    thetas.append(state[THETA])
    
    y = state[THETA] + np.random.normal(0, stdv_R)
    ys.append(y)
    
    # linearize
    Ac = np.array([0, 1, -math.cos(ukf.x[THETA]), 0]).reshape((2, 2))
    Gc = np.array([0, 0, 0, stdv_Qc]).reshape((2, 2))
    
    Ad, Gd = filter.common.van_loan_discretization(Ac, Gc, dt)
    
    ukf.Q = np.multiply(np.transpose(Gd), Gd);
    
    ukf.predict()
    ukf.update(np.array(y))
    
    yhats.append(ukf.x[THETA])

plt.figure(1)
plt.clf()
fig, ax = plt.subplots(num=1)
ax.plot(ts, thetas, 'k-', label='Truth')
ax.plot(ts, ys, 'k--', label='Measurement')
ax.plot(ts, yhats, 'r-', label='Estimate')
ax.legend(loc='best')



# https://pundit.pratt.duke.edu/wiki/Python:Ordinary_Differential_Equations