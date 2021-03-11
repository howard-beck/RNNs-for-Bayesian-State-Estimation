# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 15:12:56 2021

@author: Howard
"""

import time

import numpy as np
import bpy



# switch on nodes
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links
  
# clear default nodes
#for n in tree.nodes:
#    tree.nodes.remove(n)
  
# create input render layer node
#rl = tree.nodes.new('CompositorNodeRLayers')      
#rl.location = 185,285
 
# create output node
#v = tree.nodes.new('CompositorNodeViewer')   
#v.location = 750,210
#v.use_alpha = False
 
# Links
#links.new(rl.outputs[0], v.inputs[0])  # link Image output to Viewer input

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



N = 20

eSum  = 0
e2Sum = 0

dt_est = 0

data = []

startTimer("time")
for i in range(N):
    x = np.random.uniform(-2, 2)
    y = np.random.uniform(-2, 2)
    z = np.random.uniform(0,  1.5)
    
    camera.location.x = x
    camera.location.y = y
    camera.location.z = z
    
    startTimer("render")
    
    bpy.context.scene.render.filepath = "//pics/pic" + str(i)
    bpy.ops.render.render(write_still = True)
    
    #y_m = y_measurement()
    
    #data.append(y_m)
    print("x = " + str(x))
    print("y = " + str(y))
    print("z = " + str(z))
    #print("pixels = " + str(y_m))
    print(i)

dt = endTimer("time") / 1000 / N
print(dt)

np.save("./data.npy", data)