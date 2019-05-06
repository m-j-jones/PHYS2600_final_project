#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Source file for functions used in Main.py

"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from random import random, randint
from numba import jit

# Function definitions
@jit(nopython = True)
def MCroutine_jit(state0, beta):
    """
    Full MC routine that iteratively runs the MCstep
    """
    
    
    
    return


class VenusMC (object):
    """
    Class for keeping track of data for Venus impact Monte Carlo simulation
    """
    
    def __init__(self, beta_max, beta_min, tau=1e4, n_cooliter=10000, n_stepiter=5):
        self.beta_max = beta_max #max "temperature" for simulated annealing
        self.beta_min = beta_min #min "temperature" for simulated annealing
        self.n_cooliter = n_cooliter #number of MC cooling steps to do
        self.n_stepiter = n_stepiter #number of MC step iterations to do before cooling
        n_tot = int(n_cooliter * n_stepiter)
        self.n_tot = n_tot #total number of computational steps
        self.tau = tau #"time" constant for cooling
        
        self.M = 4.8675e24 #mass of Venus in kg
        self.R = 6.0518e6 #radius of Venus in m
        self.I_Venus = 5.8829e37
        self.I_Vnormal = 0.33
        self.Lfinal = -5.33e31 #kgm^2/s -- retrograde rotation
        self.Lnormal = 0.33 * 2.99e-7
        
        self.state_array = np.zeros((4, 1)) #TODO figure out if there's a better way to initialize number of columns
        self.L_array = np.asarray([])
        self.resid_array = np.asarray([])
        self.beta_array = np.asarray([beta_max])
        self.t_array = np.asarray([0])
        
        self.figsize = [7, 4]
        

    def initialize(self):
        """
        Initialize parameters for use in MC simulation
        """
        
        omega_max = 2.18166e-4 #8 hrs/rotation
        #M_mean = 1.791e18
        #M_var = M_mean
        M_a = 4 #power law constant for mass distribution
        #v_max = 20000
        v_mean = 27000
        v_var = 4000
        
        state0 = np.empty((4, 1))
        state0[0, 0] = random() * omega_max #Venus pre-impact angular velocity -- 0 - +8hrs/rotation
        #state0[1, 0] = np.random.lognormal(M_mean/self.M, M_var/self.M) #testing with normal distribution about r=50km impactor with Vesta density
        #state0[1, 0] = (1 - np.random.power(M_a)) * 0.1
        state0[1, 0] = (random() * 0.4) + 0.1
            #normalized to mass of Venus
        state0[2, 0] = random() #normalized sin(theta) of impact -- 0-R_Venus / R_Venus i.e. 0-1
        state0[3, 0] = np.random.normal(v_mean/self.R, v_var/self.R) #velocity of impactor
        
        # L = wI + MRv -- normalized to mass and radius of Venus
        L0 = state0[0]*self.I_Vnormal + state0[1]*state0[2]*state0[3]
        resid0 = np.abs(self.Lnormal - L0)
        
        self.state_array[:, 0] = state0.ravel()
        self.L_array = np.append(self.L_array, L0)
        self.resid_array = np.append(self.resid_array, resid0)
        
        
    def MCstep(self, state, L, beta, resid):
        """
        Monte Carlo step
        """
        
        omega_max = 2.18166e-4
        #M_mean = 1.791e18
        #M_var = M_mean
        M_a  = 4
        #v_max = 20000
        v_mean = 27000
        v_var = 4000
        
        state = state.reshape(-1, 1)
        
        # pick a random parameter to change
        new_state = state
        choice = randint(0, 3)
        if choice == 0:
            new_state[0, 0] = random() * omega_max
        elif choice == 1:
            #new_state[1, 0] = np.random.normal(M_mean/self.M, M_var/self.M)
            #new_state[1, 0] = (1 - np.random.power(M_a)) * 0.1
            new_state[1, 0] = (random() * 0.4) + 0.1
        elif choice == 2:
            new_state[2, 0] = random()
        elif choice == 3:
            new_state[3, 0] = np.random.normal(v_mean/self.R, v_var/self.R)
        
        # calculate new angular momentum and residual
        Lnew = new_state[0]*self.I_Vnormal + new_state[1]*new_state[2]*new_state[3]
        resid_new = np.abs(self.Lnormal - Lnew)
        
        dL = resid_new - resid
        prob = 1 - np.exp(-0.0025/beta)
        #print("Condition: ", prob, "beta: ", beta)
        if (dL <= 0) or (random() > prob):
            state = new_state
            L = Lnew
            resid = resid_new
        
        return state, L, resid
        
        
    def MCsimulation(self):
        self.initialize()
        t = 0
        beta = self.beta_max
        
        while (beta > self.beta_min) and (t < self.n_tot-1):
            while (t % self.n_stepiter != 0) and (t < self.n_tot-1):
                state = self.state_array[:, t]
                L = self.L_array[t]
                resid = self.resid_array[t]
                state, L, resid = self.MCstep(state, L, beta, resid)
                self.state_array = np.hstack((self.state_array, state))
                self.L_array = np.append(self.L_array, L)
                self.resid_array = np.append(self.resid_array, resid)
                self.beta_array = np.append(self.beta_array, beta)
                self.t_array = np.append(self.t_array, t)
                
                t += 1
            if (t % self.n_stepiter == 0):
                state = self.state_array[:, t]
                L = self.L_array[t]
                resid = self.resid_array[t]
                state, L, resid = self.MCstep(state, L, beta, resid)
                self.state_array = np.hstack((self.state_array, state))
                self.L_array = np.append(self.L_array, L)
                self.resid_array = np.append(self.resid_array, resid)
                self.beta_array = np.append(self.beta_array, beta)
                self.t_array = np.append(self.t_array, t)
                beta = self.beta_max * np.exp(-(t/self.n_stepiter)/self.tau)
                t += 1
                
        print('Final residual: ', self.resid_array[-1])
                
               
    """
    def MCsimulation(self):
        self.initialize()
        t = 0
        beta = self.beta_max
        
        self.MCcooling()
    """
        
            
    def plot_tL(self, fig_name = ''):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor([.97, .97, .9])
        ax1.semilogy(self.t_array, self.L_array)
        #ax1.plot(self.t_array, self.L_array)
        plt.xlabel('Computation Step')
        plt.ylabel('Angular momentum')
        plt.show()
        
    def plot_betaL(self, fig_name = ''):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor([.97, .97, .9])
        ax1.loglog(self.beta_array, self.L_array)
        plt.xlabel('"Temperature" (beta)')
        plt.ylabel('Angular momentum')
        plt.show()
        
    def plot_tresid(self, fig_name = ''):
        fig1 = plt.figure(figsize = self.figsize)
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor([.97, .97, .9])
        #ax1.semilogy(self.t_array, self.resid_array)
        ax1.plot(self.t_array, self.resid_array, color = [.7, .3, .2])
        plt.xlabel('Computation Step')
        plt.ylabel('Angular momentum residual')
        if (fig_name != ''):
            filename = '/Users/mattjones/Dropbox (Brown)/Classes/Computational Physics/Final Project/Images/' + fig_name
            plt.savefig(filename, dpi = 300)
        plt.show()
        
    def plot_betaresid(self, fig_name = ''):
        fig1 = plt.figure(figsize = self.figsize)
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor([.97, .97, .9])
        #ax1.loglog(self.beta_array, self.resid_array)
        ax1.semilogx(self.beta_array, self.resid_array, color = [.7, .3, .2])
        plt.xlabel('"Temperature" (beta)')
        plt.ylabel('Angular momentum residual')
        if (fig_name != ''):
            filename = '/Users/mattjones/Dropbox (Brown)/Classes/Computational Physics/Final Project/Images/' + fig_name
            plt.savefig(filename, dpi = 300)
        plt.show()
        
    def plot_tbeta(self, fig_name = ''):
        fig1 = plt.figure(figsize = self.figsize)
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor([.97, .97, .9])
        ax1.semilogy(self.t_array, self.beta_array, color = [.7, .3, .2])
        plt.xlabel('Computation step')
        plt.ylabel('"Temperature" (beta)')
        plt.show()
        
    
    def plot_prob(self, array, plotttl = '', fig_name = ''):
        fig1 = plt.figure(figsize = self.figsize)
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor([.97, .97, .9])
        ax1.hist(array, bins = 100, density = True, color = [.7, .3, .2])
        plt.title(plotttl)
        
        
        
    def plot_hist2d(self, array1, array2, xlbl = '', ylbl = '', fig_name = ''):
        fig1 = plt.figure(figsize = self.figsize)
        ax1 = fig1.add_subplot(111)
        
        use_logx = False
        use_logy = False
        if (array1.min()/array1.max() <= 0.01):
            use_logx = True
            binsx = np.logspace(np.log10(array1.min()), np.log10(array1.max()), 100)
        else:
            binsx = 100
        if (array2.min()/array2.max() <= 0.01):
            use_logy = True
            binsy = np.logspace(np.log10(array2.min()), np.log10(array2.max()), 100)
        else:
            binsy = 100
        
        counts, xedges, yedges, img = ax1.hist2d(array1, array2, bins = [binsx, binsy], cmap = plt.cm.Spectral)
        plt.colorbar(img, ax = ax1)
        #ax1.set_facecolor([.97, .97, .97])
        #ax1.scatter(array1, array2, alpha = 0.01, s = 20, color = [.7, .3, .2])
        plt.xlim(array1.min(), array1.max())
        plt.ylim(array2.min(), array2.max())
        if use_logx:
            ax1.set_xscale('log')
        if use_logy:
            ax1.set_yscale('log')
        plt.xlabel(xlbl)
        plt.ylabel(ylbl)
        if (fig_name != ''):
            filename = '/Users/mattjones/Dropbox (Brown)/Classes/Computational Physics/Final Project/Images/' + fig_name
            plt.savefig(filename, dpi = 300)
        plt.show()
        
