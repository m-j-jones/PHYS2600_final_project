#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source file for functions used in Main.py

"""

# Import libraries
import numpy as np
from random import random, normal, randint
from numba import jit

# Function definitions

@jit(nopython = True)
def MCstep_jit(state, beta):
    """
    Monte Carlo step
    """
    
    Lfinal = -5.33e31 #kgm^2/s -- retrograde rotation
    omega_max = 2.18166e-4
    I_Earth = 8.04e37
    I_var = 0.25 * I_Earth
    M_mean = 1.791e18
    M_var = M_mean
    R = 6.0518e6 #radius of Venus
    v_max = 20000
    v_mean = 15000
    v_var = 2500
    
    # L = wI + MRv
    Lold = state[0]*state[1] + state[2]*state[3]*state[4]
    resid_old = np.abs(Lfinal - Lold) #TODO figure out if this should be reversed or if it even matters
    
    # pick a random parameter to change
    new_state = state
    choice = randint(0, 4)
    if choice == 0:
        new_state[0] = random() * omega_max
    elif choice == 1:
        new_state[1] = normal(I_Earth, I_var)
    elif choice == 2:
        new_state[2] = normal(M_mean, M_var)
    elif choice == 3:
        new_state[3] = random() * R
    elif choice == 4:
        new_state[4] = normal(v_mean, v_var)
        
    Lnew = new_state[0]*new_state[1] + new_state[2]*new_state[3]*new_state[4]
    resid_new = np.abs(Lfinal - Lold)
    
    dL = resid_new - resid_old
    if (dL <= 0) or (random() > np.exp(-beta*dL)): #check on this -- this is from notes, but Salesman code has -dL/T
        state = new_state
        Lold = Lnew
    
    return state, Lold

class VenusMC (object):
    """
    Class for keeping track of data for Venus impact Monte Carlo simulation
    """
    
    def __init__(self, beta_max, beta_min, n_MCiter=10000, n_stepiter=5, tau=1e4):
        self.beta_max = beta_max #max "temperature" for simulated annealing
        self.beta_min = beta_min #min "temperature" for simulated annealing
        self.n_MCiter = n_MCiter #number of MC cooling steps to do
        self.n_stepiter = n_stepiter #number of MC step iterations to do before cooling
        self.tau = tau #"time" constant for cooling
        self.M = 4.867e24 #mass of Venus in kg
        self.R = 6.0518e6 #radius of Venus in m

    def initialize(self):
        """
        Initialize parameters for use in MC simulation
        """
        
        omega_max = 2.18166e-4
        I_Earth = 8.04e37
        I_var = 0.25 * I_Earth
        M_mean = 1.791e18
        M_var = M_mean
        v_max = 20000
        v_mean = 15000
        v_var = 2500
        
        state0 = np.empty(1, 5)
        state0[0, 0] = random() * omega_max #Venus pre-impact angular velocity -- 0-8hrs/day?
        #state0[0, 1] = random() #Venus moment of inertia -- normal distribution about a reasonable estimated value?
        state0[0, 1] = normal(I_Earth, I_var) #testing with normal distribution about Earth's MOI
        #state0[0, 2] = random() #impactor mass -- based on flux info
        state0[0, 2] = normal(M_mean, M_var) #testing with normal distribution about r=50km impactor with Vesta density
        state0[0, 3] = random() * self.R #sin(theta) of impact -- 0-R_Venus
        #array[0, 4] = random() * v_max #velocity of impactor -- 0-20 km/s? or normal distribution about a chosen value?
        state0[0, 4] = normal(v_mean, v_var)
        
        self.state_array = state0
        
    def MCsimulation(self):
        self.initialize()
        
        
