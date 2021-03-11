# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:26:27 2021

@author: Howard
"""

import random
import numpy as np

from filterpy.monte_carlo import systematic_resample



def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))

class Particle:
    def __init__(self, state, weight):
        self.state = state
        self.weight = weight

class ParticleFilter:
    particles = []
    weights = []
    
    def __init__(self, state_dim, N, pdf_x0, transition_sim, measurement_pdf, q0_sampler):
        self.state_dim = state_dim
        self.N = N
        self.xx_sim = transition_sim
        self.yx_pdf = measurement_pdf
        
        # use empty instead of zeros bc its marginally faster
        # https://numpy.org/doc/stable/reference/generated/numpy.empty.html
        self.particles = np.empty((self.N, self.state_dim))
        self.weights = np.empty(self.N)
        
        self.initialize_particles(pdf_x0, q0_sampler)
    
    def initialize_particles(self, pdf_x0, q0_sampler):
        for i in range(self.N):
            state, prob = q0_sampler()
            
            self.particles[i] = state
            
            weight = pdf_x0(state) / prob
            self.weights[i] = weight
            
        self.weights /= np.sum(self.weights)
    
    def calculate_effective_sample_size(self):
        Neff = 0
        
        for w in self.weights:
            #print("weight: " + str(p.weight*self.N))
            Neff += w**2
            
        return 1 / Neff
        
    def update(self, y):
        for i in range(self.N):
            state = self.xx_sim(self.particles[i])
            
            self.particles[i] = state
            # assume 
            self.weights[i] *= self.yx_pdf(y, state)
        
        self.weights /= np.sum(self.weights)
    
    def get_estimate(self):
        state_mean = np.zeros((1, self.state_dim))
        state_cov  = np.zeros((self.state_dim, self.state_dim))
        
        for i in range(self.N):
            state_mean += self.particles[i:i+1] * self.weights[i]
        
        for i in range(self.N):
            e = self.particles[i:i+1] - state_mean
            state_cov += np.dot(e.transpose(), e) * self.weights[i]
            
        return state_mean.reshape(self.state_dim), state_cov
    
    def resample(self):
        indexes = systematic_resample(self.weights)
        resample_from_index(self.particles, self.weights, indexes)