#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for PHYS 2600 final project.

"""

# Import libraries
#import numpy as np
#import matplotlib.pyplot as plt
#from random import random, randrange
#from copy import deepcopy #is this necessary?
import functions as MC


# Set up parameters
beta_max = 1e30
beta_min = 1e5
tau = 1e5
n_cooliter = 1e10
n_stepiter = 5

# Run simulation
MCV1 = MC.VenusMC(beta_max, beta_min, tau, n_cooliter, n_stepiter)
MCV1.MCsimulation()

# Analyze results


# Produce plots
MCV1.plot_tL()
MCV1.plot_betaL()
