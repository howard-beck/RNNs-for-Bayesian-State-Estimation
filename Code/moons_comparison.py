# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 09:47:39 2021

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

import bpy


import matplotlib.pyplot as plt



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

def y_measurement():
    bpy.ops.render.render()
    pixels = bpy.data.images['Viewer Node'].pixels
    arr = np.array(pixels[:])
    return arr[0::4]



N = 2000

eSum  = 0
e2Sum = 0

dt_est = 0

for i in range(N):
    x = np.random.uniform(-2, 2)
    y = np.random.uniform(-2, 2)
    z = np.random.uniform(0,  1.5)
    
    camera.location.x = x
    camera.location.y = y
    camera.location.z = z
    
    startTimer("render")
    adaptive_y = y_measurement()
    
    camera.location.x += 10 # move to non-adaptive plane
    
    non_adaptive_y = y_measurement()
    dt = endTimer("render") / 1000
    
    # get difference between adaptive and non-adaptive subdivisions
    e = non_adaptive_y - adaptive_y
    # get average of all pixel errors, add it to error sum
    eSum  += np.sum(e) / 4096
    # get average of all square pixel errors, add it to square error sum
    e2Sum += np.sum(e*e) / 4096
    
    
    
    if dt_est == 0:
        dt_est = dt
    else:
        dt_est = 0.7*dt_est + 0.3*dt
    
    n_ops = (N - i - 1)
    
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
    
    print(str(i) + " / " + str(N) + " | Estimated time left: " + str(hours) + "h" + str(minutes) + "m" + str(seconds) + "s")

eAvg  = eSum / N
e2Avg = e2Sum / N

print("average pixel error = ")
print(eAvg)
print("average square pixel error = ")
print(e2Avg)
print("variance = ")
variance = e2Avg - eAvg**2
print(variance)
print("standard deviation = ")
print(math.sqrt(variance))