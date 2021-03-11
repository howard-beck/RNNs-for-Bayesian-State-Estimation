import math

import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
import filterpy as filter
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints


import matplotlib.pyplot as plt

from particle_filter2 import ParticleFilter



THETA = 0
OMEGA = 1

STATE_DIM = 2

stdv_Qc = 10*math.pi/180
Qc = np.zeros((2, 2))
Qc[OMEGA, OMEGA] = stdv_Qc**2

stdv_R = 15*math.pi/180
R = np.array(stdv_R**2)



def gaussian_pdf(x, mu, omega2):
    return scipy.stats.norm.pdf(x, mu, math.sqrt(omega2))
    #startTimer("gaussian pdf")
    #ret = 1/(math.sqrt(2*math.pi*omega2)) * math.exp(-0.5*((x - mu)**2 / omega2))
    #endTimer("gaussian pdf")
    #return ret

def mgaussian_pdf(x, mu, Q):
    return scipy.stats.multivariate_normal(mu, Q).pdf(x)
    ##startTimer("mgaussian pdf")
    #ret = 1/(math.sqrt(2*math.pi*np.linalg.det(Q))) * math.exp(-0.5*np.dot(x - mu, np.dot(np.linalg.inv(Q), np.transpose(x - mu)))[0, 0])
    #endTimer("mgaussian pdf")
    #return ret



xhat0 = np.array([math.pi/4, math.pi/6])

P0 = np.array([(5*math.pi/180)**2, 0, 0, (5*math.pi/180)**2]).reshape((2, 2))
state = xhat0
state[THETA] += np.random.normal(0, 5*math.pi/180)
state[OMEGA] += np.random.normal(0, 5*math.pi/180)

dt = 0.1

temp_times = {}
times = {}

def startTimer(name):
    temp_times[name] = time.time_ns()

def endTimer(name):
    dt = time.time_ns() - temp_times[name]
    
    if hasattr(times, name):
        times[name] += dt
    else:
        times[name] = dt



def deriv(t, x, noise):
    theta = x[THETA]
    omega = x[OMEGA]
    
    ret = np.empty(STATE_DIM)
    
    ret[0] = omega # theta' = omega
    ret[1] = -math.sin(theta) # omega' = -sin theta
    
    if noise:
        ret[1] += np.random.normal(0, stdv_Qc)
    
    return ret

def noisy_rk4_solver(x0, f, h):
    #return x0 + h*f(0, x0) + np.array([0, stdv_Qc])*np.random.normal(0, math.sqrt(h))
    y = x0
    N = 10
    dt = h/N
    sqrtdt = math.sqrt(dt)
    
    b = np.array([0, stdv_Qc])
    
    for i in range(N):
        S = 1
        if np.random.random() > 0.5:
            S = -1
        
        dW = np.random.normal(0, sqrtdt)
        
        K1 = dt*f(0, y)        + (dW - S*sqrtdt)*b
        K2 = dt*f(0, y + K1)   + (dW + S*sqrtdt)*b
        y = y + (K1 + K2)/2
        
    return y

def noiseless_rk4_solver(x0, f, h):
    k1 = f(t,       x0)
    k2 = f(t + h/2, x0 + h/2 * k1)
    k3 = f(t + h/2, x0 + h/2 * k2)
    k4 = f(t + h,   x0 + h*k3)
    
    return x0 + 1/6*h*(k1 + 2*k2 + 2*k3 + k4)

def derivNoNoise(t, x):
    return deriv(t, x, False)

def derivWithNoise(t, x):
    return deriv(t, x, True)

def noiselessUpdate(x, dt):
    return noiseless_rk4_solver(x, derivNoNoise, dt)
    #return solve_ivp(derivNoNoise, [0, dt], x, method="RK45", t_eval = [dt]).y.reshape(STATE_DIM)

def noisyUpdate(x):
    return noiseless_rk4_solver(x, derivWithNoise, dt)
    #return x + deriv(0, x, True)*dt
    #return solve_ivp(derivWithNoise, [0, dt], x, method="RK45", t_eval = [dt]).y.reshape(STATE_DIM)



ts = np.arange(0, 1, dt)

thetas = []
ys = []
yhats = []
yhats_ukf = []

omegas = []
omega_hats = []

xhat = xhat0
Pn = P0

def pdf_x0(x):
    return mgaussian_pdf(x, xhat0, P0)

def discretize(x):
    Ac = np.array([0, 1, -math.cos(x[THETA]), 0]).reshape((2, 2))
    Gc = np.array([0, 0, 0, stdv_Qc]).reshape((2, 2))
    
    Ad, Gd = filter.common.van_loan_discretization(Ac, Gc, dt)
    
    return Ad, Gd

def transition_pdf(x1, x0):
    startTimer("transition pdf")
    
    Ad, Gd = discretize(x0)
    
    mu = noiselessUpdate(x0, dt)
    
    Q = np.multiply(np.transpose(Gd), Gd)
    
    prob = mgaussian_pdf(x1, mu, Q)
    
    endTimer("transition pdf")
    return prob

def measurement_pdf(y, x):
    return gaussian_pdf(y, x[THETA], R)

def q0_sampler():
    startTimer("q0 sampler")
    
    state = np.random.multivariate_normal(xhat, P0, 1)
    prob = mgaussian_pdf(state, xhat, P0)
    
    endTimer("q0 sampler")
    return state, prob

def qn_sampler(particles, y):
    startTimer("qn sampler")
    
    Ad, Gd = discretize(xhat)
    
    mu = noiselessUpdate(xhat, dt)
        
    Q = np.multiply(np.transpose(Gd), Gd) + np.dot(Ad, np.dot(Pn, np.transpose(Ad)))
    
    state = np.random.multivariate_normal(mu.reshape(2), Q, 1).reshape(2, 1)
    prob = mgaussian_pdf(state, mu, Q)
    
    endTimer("qn sampler")
    return state, prob
    

N = 1000
pf = ParticleFilter(STATE_DIM, N, pdf_x0, noisyUpdate, measurement_pdf, q0_sampler)

particles_thetas = []
particles_omegas = []
particles_times = []
particles_s = []

points = MerweScaledSigmaPoints(STATE_DIM, alpha=.1, beta=2., kappa=-1)
def measurement(xhat):
    return [xhat[THETA]]
ukf = UnscentedKalmanFilter(STATE_DIM, 1, dt, measurement, noiselessUpdate, points)

ukf.x = xhat0
ukf.P = P0
ukf.R = R

startTimer("main")

NUM = 1

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

for t in ts:
    plt.figure(NUM)
    plt.clf()
    fig, ax = plt.subplots(num = NUM)
    xhat, Pn = pf.get_estimate()
    ax.scatter(pf.particles[0:N, 0], pf.particles[0:N, 1], pf.weights*N)
    drawEllipse(ukf.P, ax, ukf.x)
    drawEllipse(Pn, ax, xhat)
    drawEllipse(Pn, ax, xhat, 1)
    ax.scatter(ukf.x[THETA], ukf.x[OMEGA], c="y", s=5)
    ax.scatter(state[THETA], state[OMEGA], c="r", s=5)
    ax.scatter(xhat[THETA], xhat[OMEGA], c="g", s=5)
    plt.savefig("Particles/Particles-" + str(NUM))
    #plt.xlim([0.6, 1.1])
    #plt.ylim([0, 0.9])
    print(ukf.P);
    
    print(Pn);
    print("--------")
    
    
    NUM += 1
    
    state = noisyUpdate(state)
    
    thetas.append(state[THETA])
    omegas.append(state[OMEGA])
    
    y = state[THETA] + np.random.normal(0, stdv_R)
    ys.append(y)
    
    pf.update(y)
    xhat, Pn = pf.get_estimate()
    
    # UKF stuff
    # linearize
    Ac = np.array([0, 1, -math.cos(ukf.x[THETA]), 0]).reshape((2, 2))
    Gc = np.array([[0], [stdv_Qc]])
    
    Ad, Qd = filter.common.van_loan_discretization(Ac, Gc, dt)
    
    ukf.Q = Qd#np.multiply(np.transpose(Gd), Gd);
    
    ukf.predict()
    ukf.update(np.array(y))
    
    
    
    yhats.append(xhat[THETA])
    omega_hats.append(xhat[OMEGA])
    yhats_ukf.append(ukf.x[THETA])
    
    Neff = pf.calculate_effective_sample_size()
    if Neff < N/2 and True:
        pf.resample()
        print("Resampled " + str(t) + ": " + str(pf.calculate_effective_sample_size()))
    
    print(str(t) + ": " + str(Neff))
    
    for i in range(pf.N):
        p = pf.particles[i]
        w = pf.weights[i]
        
        particles_thetas.append(p[THETA])
        particles_omegas.append(p[OMEGA])
        particles_times.append(t)
        particles_s.append((20*w)**2)
endTimer("main")

plt.figure(NUM)
plt.clf()
fig, ax = plt.subplots(num = NUM)
xhat, Pn = pf.get_estimate()
ax.scatter(pf.particles[0:N, 0], pf.particles[0:N, 1], pf.weights*N)
drawEllipse(ukf.P, ax, ukf.x)
drawEllipse(Pn, ax, xhat)
drawEllipse(Pn, ax, xhat, 1)
ax.scatter(ukf.x[THETA], ukf.x[OMEGA], c="y", s=5)
ax.scatter(state[THETA], state[OMEGA], c="r", s=5)
ax.scatter(xhat[THETA], xhat[OMEGA], c="g", s=5)
plt.savefig("Particles/Particles-" + str(NUM))
NUM += 1
    
plt.figure(N+1)
plt.clf()
fig, ax = plt.subplots(num=N+1)
ax.plot(ts, thetas, 'k-', label='Truth')
ax.plot(ts, ys, 'k--', label='Measurement')
ax.plot(ts, yhats, 'r-', label='PF Estimate')
ax.plot(ts, yhats_ukf, 'g-', label='UKF Estimate')
#ax.scatter(particles_times, particles_thetas, s=particles_s)
ax.legend(loc='best')

plt.savefig("Position_Particles")

plt.figure(N+2)
plt.clf()
fig, ax = plt.subplots(num=N+2)
ax.plot(ts, omegas, 'k-', label='Truth')
ax.plot(ts, omega_hats, 'r-', label='Estimate')
#ax.scatter(particles_times, particles_omegas, s=particles_s)
ax.legend(loc='best')

plt.savefig("Velocity_Particles")

num = 1000
ps = []
Pp = 0
xp = 0
Pvld = 0
xvld = 0

def testing():
    global Pvld
    global Pp
    global ps
    global xp
    
    for i in range(num):
        ps.append(noisyUpdate(state))
    
    xp = np.zeros((1, 2))
    Pp = np.zeros((2, 2))
    
    for i in range(num):
        xp += ps[i].dot(1/num)
    
    for i in range(num):
        e = ps[i] - xp
        Pp += np.dot(e.transpose(), e).dot(1/num)
    
        Ac = np.array([0, 1, -math.cos(state[THETA]), 0]).reshape((2, 2))
        Gc = np.array([[0, 0], [0, stdv_Qc]])
        
        Ad, Pvld = filter.common.van_loan_discretization(Ac, Gc, dt)