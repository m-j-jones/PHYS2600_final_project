#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for PHYS 2600 final project.

"""

# Import libraries
#import numpy as np
#import matplotlib.pyplot as plt
import functions as MC


# Set up parameters
beta_max = 1e-3
beta_min = 1e-10
tau = 5e4
n_cooliter = 1e7
n_stepiter = 5

n_cooling_runs = 5

# Run simulation
MCV = MC.VenusMC(beta_max, beta_min, tau, n_cooliter, n_stepiter)
MCV.MCsimulation()

for i in range(n_cooling_runs):
    MCV2 = MC.VenusMC(beta_max, beta_min, tau, n_cooliter, n_stepiter)
    MCV2.MCsimulation()
    
    if MCV2.resid_array[-1] < MCV.resid_array[-1]:
        print('Residual reduced in simulation ', i+2, ' of ', n_cooling_runs+1)
        print(MCV.resid_array[-1], ' reduced to ', MCV2.resid_array[-1])
        MCV = MCV2
        
print('Final residual: ', MCV.resid_array[-1])

# Analyze results
"""
timestamp = len(MCV.t_array) - round(len(MCV.t_array) * 1)
MCV.state_array = MCV.state_array[:, timestamp:]
MCV.t_array = MCV.t_array[timestamp:]
MCV.beta_array = MCV.beta_array[timestamp:]
MCV.L_array = MCV.L_array[timestamp:]
MCV.resid_array = MCV.resid_array[timestamp:]
"""

# Produce plots
MCV.plot_tresid()
MCV.plot_betaresid()
MCV.plot_tbeta()

MCV.plot_omegaL()
MCV.plot_ML()
MCV.plot_RL()
MCV.plot_vL()
MCV.plot_Mv()


"""
a = 4. # shape
samples = 1000
s = 1 - np.random.power(a, samples)
count, bins, ignored = plt.hist(s, bins=30)
x = np.linspace(0, 1, 100)
y = a*x**(a-1.)
normed_y = samples*np.diff(bins)[0]*y
plt.plot(np.flip(x), normed_y)
plt.show()
"""
