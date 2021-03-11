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

import bpy


import matplotlib.pyplot as plt



from filterpy.monte_carlo import systematic_resample



def resample_from_index(particles, weights, indexes):
    a = particles.T[indexes]
    particles[:] = a.T
    #weights.resize(len(particles))
    weights.fill (1.0 / N)





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



class Output(Enum):
    IMAGE = 1
    ALTIMER = 2

STATE_DIM = 3
OUTPUT_DIM = 4096

OUTPUT_TYPE = Output.ALTIMER

I_sp = 200
g0 = 9.80665
g = -1.622

dt = 0.5

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





xhat0 = np.array([1250, -8, 1179.34])

P0 = np.array([
    [83.333333333**2, 0,                0],
    [0,               0.06666666666**2, 0],
    [0,               0,                1]
])

state = np.random.multivariate_normal(xhat0, P0).reshape((STATE_DIM, 1))

# 10 / 255 pixels standard deviation
R_img = np.array([[(10 / 255)**2]])
R_alt = np.array([[1]])




def y_measurement_img(x):
    camera.location.z = x[0] / 1000
    bpy.ops.render.render()
    pixels = bpy.data.images['Viewer Node'].pixels
    arr = np.array(pixels[:])
    return arr[0::4]

def y_measurement_alt(x):
    return x[0]

def y_measurement(x):
    if OUTPUT_TYPE == Output.IMAGE:
        return y_measurement_img(x)
    elif OUTPUT_TYPE == Output.ALTIMER:
        return y_measurement_alt(x)

def noisify_output(y0):
    if OUTPUT_TYPE == Output.IMAGE:
        return y0[0] + np.random.multivariate_normal([0], R_img, OUTPUT_DIM).reshape(OUTPUT_DIM)
    elif OUTPUT_TYPE == Output.ALTIMER:
        return y0[0] + np.random.multivariate_normal([0], R_alt)

#def y_x_pdf(x, y):
#    if OUTPUT_TYPE == Output.IMAGE:
#        return scipy.stats.multivariate_normal(y, R).pdf(x[0].T)
#    elif OUTPUT_TYPE = Output.ALTIMER:
#        return scipy.stats.multivariate_normal(y, R).pdf(x[0].T)

control = np.zeros((1, 1))

def deriv(x0):
    ret = np.empty(x0.shape)
    
    y  = x0[0]
    vy = x0[1]
    m  = x0[2]
    
    ret[0] = vy
    ret[1] = g + control / m
    ret[2] = -control / (I_sp * g0)
    
    return ret


# assume dW ~ N(mean=0, cov=dt) or similar
def noisy_deriv_term(x0, dW):
    ret = np.zeros(x0.shape)
    # noise applied at the control level
    
    m = x0[2]
    
    ret[1] = 1/m * dW * stdv_Q_T
    ret[2] = -dW * stdv_Q_T / (I_sp * g0)
    
    return ret

def noisy_update(x0):
    # get number of vectors (useful for noise generation)
    _, n = x0.shape
    
    x = np.copy(x0)
    
    N = 10
    h = dt/N
    sqrth = math.sqrt(h)
    
    # between - sqrth and + sqrth
    S = (np.random.randint(0, 2, (N, n)).dot(2) - 1).dot(sqrth)
    dW = np.random.normal (0, 1, (N, n)).dot(sqrth)
    
    for i in range(N):
        dx1 = deriv(x)  * h + noisy_deriv_term(x,  dW[i] - S[i])
        x1 = x + dx1
        
        dx2 = deriv(x1) * h + noisy_deriv_term(x1, dW[i] + S[i])
        
        x += (dx1 + dx2).dot(0.5)
        
    return x



N = 1000
Neff = N

particles = np.random.multivariate_normal(xhat0, P0, N).T
weights   = np.ones(N) / N



J = 1
Tmax = J*dt
ts = np.arange(0, Tmax, dt)


startTimer("main")
control[0, 0] = 1

step_time = 0

def start():
    global state
    global particles
    global weights
    global Neff
    global step_time
    
    dt_est = 0
    
    j = 0
    
    for t in ts:
        j += 1
        
        startTimer("state update")
        state = noisy_update(state)
        endTimer("state update")
        
        startTimer("measurement")
        y0 = y_measurement(state[0, 0])
        y = noisify_output(y0)
        endTimer("measurement")
        
        startTimer("particle update")
        new_particles = noisy_update(particles)
        endTimer("particle update")
        
        startTimer("reweigh")
        ws = np.empty((N, OUTPUT_DIM))
        for i in range(N):
            startTimer("rendering")
            y_part = y_measurement(new_particles[0, i])
            dt = endTimer("rendering") / 1000
            #print(str(i) + " / 1000 ")
            
            if dt_est == 0:
                dt_est = dt
            else:
                dt_est = 0.7*dt_est + 0.3*dt
            
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
            
            print(str(i) + " / 1000 | Estimated time left: " + str(hours) + "h" + str(minutes) + "m" + str(seconds) + "s")
            
            e = (y_part - y) / (10/255)
            
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
        
        
        
        z_mean = np.zeros((3, 1))
        z_cov  = np.zeros((3, 3))
        
        v = np.empty((3, 1))
        
        particles = new_particles
        
        startTimer("calc")
        for i in range(N):
            v[0:STATE_DIM] = particles[0:STATE_DIM, i:i+1]
            
            z_mean += weights[i] * v
            z_cov  += weights[i] * v.dot(v.T)
        z_cov -= z_mean.dot(z_mean.T)
        
        print("State covariance =")
        print(z_cov)
        print("State mean =")
        print(z_mean)
        print("Ground truth =")
        print(state)
        
        startTimer("num")
        z_error = z_mean - state
        num = math.sqrt((z_error.T.dot(np.linalg.inv(z_cov)).dot(z_error))[0, 0])
        print("Mahalanobis distance =")
        print(num)
        endTimer("num")
        endTimer("calc")
        
        Neff = 1/np.sum(weights**2)
        print("Neff =")
        print(Neff)
        
        if Neff < N/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
    endTimer("main")
        
    print("times =")
    print(times)

bpy.app.timers.register(start, first_interval=5)