# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:20:26 2021

@author: Howard
"""

import numpy as np
from filterpy.monte_carlo import systematic_resample

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill (1.0 / len(weights))

class RaoBlackwellizedParticleFilter:
    particles = []
    kfs = []
    weights = []
    
    def __init__(self, state_dim, rb_state_dim, N, pdf_r0, r_xr_transition, x_x_transition, y_r_pdf, q0_sampler, initialize_kfs):
        self.state_dim = state_dim
        self.rb_state_dim = rb_state_dim
        self.N = N
        
        self.r_xr_sim = r_xr_transition
        self.x_x_sim = x_x_transition
        self.y_r_pdf = y_r_pdf
        
        self.particles = np.empty((self.N, self.rb_state_dim))
        self.weights = np.empty(self.N)
        
        self.initialize_particles(pdf_r0, q0_sampler)
        initialize_kfs(self)
        
    def initialize_particles(self, pdf_r0, q0_sampler):
        for i in range(self.N):
            state, prob = q0_sampler()
            
            self.particles[i] = state
            
            weight = pdf_r0(state) / prob
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
            full_prior_state = np.concatenate(self.particles[i], self.kfs[i].x)
            r_state = self.r_xr_sim(full_prior_state)
            
            self.particles[i] = r_state
            # assume 
            self.weights[i] *= self.y_r_pdf(y, r_state)
        
        self.weights /= np.sum(self.weights)
    
    def get_estimate(self):
        ma_state_dim = self.state_dim - self.rb_state_dim
        
        rb_state_mean = np.zeros((1, self.rb_state_dim))
        ma_state_mean = np.zeros((1, self.ma_state_dim))
        
        state_cov  = np.zeros((self.state_dim, self.state_dim))
        
        for i in range(self.N):
            rb_state_mean += self.particles[i:i+1] * self.weights[i]
            ma_state_mean += self.kfs[i].x         * self.weights[i]
        
        #for i in range(self.N):
        #    rb_e = self.particles[i:i+1] - rb_state_mean
        #    ma_e = self.kfs[i].x         - ma_state_mean
        #    
        #    state_cov += np.dot(e.transpose(), e) * self.weights[i]
        
        return np.concatenate(rb_state_mean, ma_state_mean).reshape(self.state_dim)#, state_cov
    
    def resample(self):
        indexes = systematic_resample(self.weights)
        resample_from_index(self.particles, self.weights, indexes)