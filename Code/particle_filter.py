import random
import numpy as np

from filterpy.monte_carlo import systematic_resample

class Particle:
    def __init__(self, state, weight):
        self.state = state
        self.weight = weight

class ParticleFilter:
    particles = []
    
    def __init__(self, state_dim, N, pdf_x0, transition_pdf, measurement_pdf, q0_sampler, qn_sampler):
        self.state_dim = state_dim
        self.N = N
        self.xx_pdf = transition_pdf
        self.yx_pdf = measurement_pdf
        
        self.initialize_particles(pdf_x0, q0_sampler)
        self.qn_sampler = qn_sampler
    
    def initialize_particles(self, pdf_x0, q0_sampler):
        states = []
        weights = []
        Z = 0
        
        for i in range(self.N):
            state, prob = q0_sampler()
            
            states.append(state)
            
            weight = pdf_x0(state) / prob
            weights.append(weight)
            Z += weight
        
        for i in range(self.N):
            self.particles.append(Particle(states[i], weights[i] / Z))
    
    def calculate_effective_sample_size(self):
        Neff = 0
        
        for p in self.particles:
            #print("weight: " + str(p.weight*self.N))
            Neff += p.weight**2
            
        return 1 / Neff
        
    def update(self, y):
        new_particle_states = []
        probs = []
        
        for i in range(self.N):
            state, prob = self.qn_sampler(self.particles, y)
            
            new_particle_states.append(state)
            probs.append(prob)
        
        #print("a")
        total_unnormalized_p = 0
        
        unnormalized_ps = []
        
        for i in range(self.N):
            xn = new_particle_states[i]
            
            #print("b")
            unnormalized_p = 0
            
            for particle in self.particles:
                unnormalized_p += particle.weight * self.xx_pdf(xn, particle.state)
                #print("c: " + str(particle.weight))
            
            unnormalized_p *= self.yx_pdf(y, xn) / probs[i]
            unnormalized_ps.append(unnormalized_p)
        
            #print(unnormalized_p)
            total_unnormalized_p += unnormalized_p
        
        #print(total_unnormalized_p)
        for i in range(self.N):
            #print("tup: " + str(total_unnormalized_p))
            self.particles[i] = Particle(new_particle_states[i], (unnormalized_ps[i] / (total_unnormalized_p)))
    
    def get_estimate(self):
        state_mean = np.zeros([self.state_dim, 1])
        state_cov  = np.zeros([self.state_dim, self.state_dim])
        
        for p in self.particles:
            state_mean += p.state * p.weight
        
        for p in self.particles:
            e = p.state - state_mean
            state_cov += np.multiply(e, e.transpose()) * p.weight
            
        return state_mean, state_cov
    
    def resample(self):
        weights = []
        
        for p in self.particles:
            weights.append(p.weight)
        
        #indexes = systematic_resample(weights)
        #resample_from_index(particles, weights, indexes)
            
        p = 0
        ps = []
        
        for i in range(self.N):
            p += self.particles[i].weight
            ps.append(p)
            
        new_particles = []
        
        for i in range(self.N):
            P = random.random()
            
            for j in range(self.N):
                if P < ps[j]:
                    new_particles.append(Particle(self.particles[j].state, 1/self.N))
                    break
            
        self.particles = new_particles