# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:08:32 2021

@author: Howard
"""

import math
import numpy as np

import filterpy as filter
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

import matplotlib.pyplot as plt





x = np.array([math.pi/6, -math.pi/12])
Ac = np.array([[0, 1], [-math.cos(x[0]), 0]])
stdv_Qc = 10*math.pi/180
Gc = np.array([[0], [stdv_Qc]])

dt = 1

Ad, Qd = filter.common.van_loan_discretization(Ac, Gc, dt)

stdv_R = 0.5/3
R = np.array([stdv_R**2])



Ad1, Qd1 = filter.common.van_loan_discretization(Ac, Gc, dt/100)

def euler(x0, f, h):
    y = x0
    N = 4
    dt = h/N
    sqrtdt = math.sqrt(dt)
    
    b = np.array([0, stdv_Qc])
    
    for i in range(N):
        y = y + f(y)*dt + b * np.random.normal(0, math.sqrt(dt))
    
    return y

def improved_euler(x0, f, h):
    y = x0
    N = 4
    dt = h/N
    sqrtdt = math.sqrt(dt)
    
    b = np.array([0, stdv_Qc])
    
    for i in range(N):
        S = 1
        if np.random.random() > 0.5:
            S = -1
        
        dW = np.random.normal(0, sqrtdt)
        
        K1 = dt*deriv(y)        + (dW - S*sqrtdt)*b
        K2 = dt*deriv(y + K1)   + (dW + S*sqrtdt)*b
        y = y + (K1 + K2)/2
        
    return y

def rk4(x0, f, h):
    dt = h/4
    dW = np.random.normal(0, math.sqrt(h), 4)
    
    b = np.array([0, stdv_Qc])
    
    k1 = f(x0)*h        + b*dW[0]
    k2 = f(x0 + k1/2)*h + b*dW[1]
    k3 = f(x0 + k2/2)*h + b*dW[2]
    k4 = f(x0 + k3)*h   + b*dW[3]
    
    return x0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    
def rk4_solver(x0, f, h):
    return improved_euler(x0, f, h)
    #k1 = f(x0)
    #k2 = f(x0 + h/2 * k1)
    #k3 = f(x0 + h/2 * k2)
    #k4 = f(x0 + h*k3)
    
    #return x0 + 1/6*h*(k1 + 2*k2 + 2*k3 + k4)
    
    #y = x0
    #N = 4
    #dt = h/N
    #sqrtdt = math.sqrt(dt)
    
    #b = np.array([0, stdv_Qc])
    
    #for i in range(N):
        #y = y + f(y)*dt + b * np.random.normal(0, math.sqrt(dt))
    #    S = 1
    #    if np.random.random() > 0.5:
    #        S = -1
        
    #    dW = np.random.normal(0, sqrtdt)
        
    #    K1 = dt*deriv(y)        + (dW - S*sqrtdt)*b
    #    K2 = dt*deriv((y + K1)) + (dW + S*sqrtdt)*b
    #    y = y + (K1 + K2)/2
    
    #return y
    

def deriv(x):
    ret = np.array([x[1], -math.sin(x[0])])
    #ret = Ac.dot(x)
    
    #ret[1] += np.random.normal(0, stdv_Qc)
    
    return ret

N = 1000
xs = []
ys = []
ps = np.empty((N, 2))

P0 = np.array([
    [0, 0],
    [0, 0]
])

for i in range(N):
    x0 = np.random.multivariate_normal(x, P0)
    x1 = rk4_solver(x0, deriv, dt)
    xs.append(x1[0])
    ys.append(x1[1])
    ps[i] = x1
    
def drawEllipse(cov, ax, avg, K=3):
    angles = np.linspace(0, math.pi*2, 100)
    w, v = np.linalg.eig(np.linalg.inv(cov))
    lambda1 = w[0]
    lambda2 = w[1]
    e1 = v[:,0]
    e2 = v[:,1]
    px = avg[0] + K*(1/np.sqrt(lambda1)*np.sin(angles)*e1[0] + 1/np.sqrt(lambda2)*np.cos(angles)*e2[0])
    py = avg[1] + K*(1/np.sqrt(lambda1)*np.sin(angles)*e1[1] + 1/np.sqrt(lambda2)*np.cos(angles)*e2[1])
    ax.plot(px, py, "r-", label="3-sigma UKF")


xm = np.zeros((1,2))
P =  np.zeros((2, 2))
        
for i in range(N):
    xm += ps[i].dot(1/N)

for i in range(N):
    e = ps[i] - xm
    P += np.dot(e.transpose(), e).dot(1/N)

plt.figure()
plt.clf()
fig, ax = plt.subplots(num = 1)
ax.scatter(xs, ys)

drawEllipse(P, ax, xm.reshape(2), K=3)
drawEllipse(Ad.dot(P0).dot(Ad.T) + Qd, ax, Ad.dot(x), K=3)