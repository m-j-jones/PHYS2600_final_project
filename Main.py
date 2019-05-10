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
beta_min = 1e-15
tau = 5e4
n_cooliter = 1e7
n_stepiter = 1

n_cooling_runs = 0

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
timestamp = len(MCV.t_array) - round(len(MCV.t_array) * 0.5)
MCV.state_array = MCV.state_array[:, timestamp:]
MCV.t_array = MCV.t_array[timestamp:]
MCV.beta_array = MCV.beta_array[timestamp:]
MCV.L_array = MCV.L_array[timestamp:]
MCV.resid_array = MCV.resid_array[timestamp:]
"""

# Produce plots
MCV.plot_tresid('t_resid')
MCV.plot_betaresid('beta_resid')
MCV.plot_tbeta('t_beta')

MCV.plot_prob(MCV.state_array[0, :], 'Venus initial angular velocity', 'omega_hist')
MCV.plot_prob(MCV.state_array[1, :], 'Impactor mass', 'M_hist')
MCV.plot_prob(MCV.state_array[2, :], 'Impact angle (sort of)', 'R_hist')
MCV.plot_prob(MCV.state_array[3, :], 'Impactor velocity', 'v_hist')

#MCV.plot_scatter(MCV.state_array[0, :], MCV.L_array, 'Venus initial angular velocity', 'Angular momentum')
#MCV.plot_scatter(MCV.state_array[1, :], MCV.L_array, 'Impactor mass (M_V)', 'Angular momentum')
#MCV.plot_scatter(MCV.state_array[2, :], MCV.L_array, 'Impact angle (sort of)', 'Angular momentum')
#MCV.plot_scatter(MCV.state_array[3, :], MCV.L_array, 'Impactor velocity', 'Angular momentum')
#MCV.plot_scatter(MCV.state_array[1, :], MCV.state_array[3, :], 'Impactor mass', 'Impactor velocity')

MCV.plot_hist2d(MCV.state_array[0, :], MCV.L_array, 'Venus initial angular velocity', 'Angular momentum', 'omega_L')
MCV.plot_hist2d(MCV.state_array[1, :], MCV.L_array, 'Impactor mass (M_V)', 'Angular momentum', 'M_L')
MCV.plot_hist2d(MCV.state_array[2, :], MCV.L_array, 'Impact angle (sort of)', 'Angular momentum', 'R_L')
MCV.plot_hist2d(MCV.state_array[3, :], MCV.L_array, 'Impactor velocity', 'Angular momentum', 'v_L')
MCV.plot_hist2d(MCV.state_array[0, :], MCV.state_array[1, :], 'Venus angular velocity', 'Impactor mass', 'omega_M')
MCV.plot_hist2d(MCV.state_array[0, :], MCV.state_array[2, :], 'Venus angular velocity', 'Impact angle proxy', 'omega_R')
MCV.plot_hist2d(MCV.state_array[0, :], MCV.state_array[3, :], 'Venus angular velocity', 'Impactor velocity', 'omega_v')
MCV.plot_hist2d(MCV.state_array[1, :], MCV.state_array[2, :], 'Impactor mass', 'Impact angle proxy', 'M_R')
MCV.plot_hist2d(MCV.state_array[1, :], MCV.state_array[3, :], 'Impactor mass', 'Impactor velocity', 'M_v')
MCV.plot_hist2d(MCV.state_array[2, :], MCV.state_array[3, :], 'Impact angle proxy', 'Impactor velocity', 'R_v')


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
