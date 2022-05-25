#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:09:44 2017

@author: meysamhashemi
"""
import matplotlib.pyplot as plt
import simple_model as model
import pymc
mcmc_sampler = pymc.MCMC(model)
mcmc_sampler.sample(1000000, thin=1000, burn=1000)
pymc.plot(mcmc_sampler)
plt.show()

mcmc_sampler = pymc.MCMC(model)
proposal = pymc.Metropolis(model.mixture, proposal_sd=3.)
mcmc_sampler.step_method_dict[model.mixture][0] = proposal
mcmc_sampler.sample(1000000, thin=1000, burn=0,tune_throughout=False)


import simple_model as model
import pymc
import pysmc

# Construct the MCMC sampler
mcmc_sampler = pymc.MCMC(model)
# Construct the SMC sampler
smc_sampler = pysmc.SMC(mcmc_sampler, num_particles=1000,
                        num_mcmc=10, verbose=1)
# Initialize SMC at gamma = 0.01
smc_sampler.initialize(0.01)
# Move the particles to gamma = 1.0
smc_sampler.move_to(1.0)
# Get a particle approximation
p = smc_sampler.get_particle_approximation()
# Plot a histogram
pysmc.hist(p, 'mixture')
plt.show()