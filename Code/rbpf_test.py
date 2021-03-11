# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 10:55:20 2021

@author: Howard
"""

import math

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import filterpy as filter
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints


import matplotlib.pyplot as plt

from rao_blackwellized_particle_filter import RaoBlackwellizedParticleFilter



X = 0
Y = 1
THETA = 2

STATE_DIM = 3

stdv_Qc = 1
Qc = np.zeros((3, 3))
Qc[THETA, THETA] = stdv_Qc**2

stdv_R = 0.1
R = np.eye(2, 2) * stdv_R**2



xhat0 = np.array([2, 0, math.pi/2])
P0 = np.array([
    [0.5**2, 0,      0],
    [0,      0.5**2, 0],
    [0,      0,      (math.pi/6)**2]
])
state = np.random.multivariate_normal(xhat0, P0)



dt = 0.1



def deriv(t, x):
    theta = x[THETA]
    
    ret = np.empty(STATE_DIM)
    
    ret[X] = 1*math.cos(THETA)
    ret[Y] = 1*math.sin(THETA)
    ret[THETA] = -1

def deriv_x(t, x):
    return -1

def noiseless_rk4_solver(x0, f, h):
    k1 = f(x0)
    k2 = f(x0 + h/2 * k1)
    k3 = f(x0 + h/2 * k2)
    k4 = f(x0 + h*k3)
    
    return x0 + 1/6*h*(k1 + 2*k2 + 2*k3 + k4)

b = np.array([0, 0, stdv_Qc])

def noisy_rk4_solver(x0, f, h):
    # number of particles to update
    n = len(x0[0])
    
    y = x0
    N = 10
    dt = h/N
    sqrtdt = math.sqrt(dt)
    
    S = (np.random.randint(0, 2, (N, n)).dot(2) - 1).dot(sqrtdt)
    dW = np.random.normal (0, 1, (N, n)).dot(sqrtdt)
    
    for i in range(N):
        K1 = 
        S = 1
        if np.random.random() > 0.5:
            S = -1
        
        dW = np.random.normal(0, sqrtdt)
        
        K1_x = np.cos(y[2])
        K1_y = np.sin(y[2])
        K1_t = 
        
        K1 = dt*f(0, y)        + (dW - S*sqrtdt)*b
        K2 = dt*f(0, y + K1)   + (dW + S*sqrtdt)*b
        y = y + (K1 + K2)/2
        
    return y

def noiselessUpdate(x, dt):
    return noiseless_rk4_solver(x, deriv, dt)
    #return solve_ivp(derivNoNoise, [0, dt], x, method="RK45", t_eval = [dt]).y.reshape(STATE_DIM)

def noisyUpdate(x):
    return noisy_rk4_solver(x, deriv, dt)

ts = np.arange(0, 1, dt)



P0_r = P0[0:2, 0:2]
r0_dist = scipy.stats.multivariate_normal(xhat0[0:2], P0_r)
def pdf_r0(r):
    return r0_dist.pdf(r)

def r_xr(z):
    z1 = noisy_rk4_solver(z, deriv, dt)
    
    return z1[0:2]

def x_x(x):
    return noisy_rk4_solver(x, deriv_x, dt)

def y_r(y, r):
    return scipy.stats.multivariate_normal(y, R)

def q0_sample():
    return np.random.multivariate_normal(xhat0[0:2], P0_r)

def x_x_noiseless(x):
    return noiseless_rk4_solver(x, deriv_x, dt)

N = 10000000

def initialize_kfs(pf):
    for i in range(N):
        ukf = UnscentedKalmanFilter(1, 1, dt, measurement, x_x_noiseless, points)
        pf.kfs.append

rbpf = RaoBlackwellizedParticleFilter(STATE_DIM, 1, N, pdf_r0, r_xr, y_r, q0_sample, initialize_kfs)