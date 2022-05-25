"""
Run SMC on the simple model.

Author:
    Ilias Bilionis

Date:
    9/28/2013

"""

#http://predictivesciencelab.github.io/pysmc/tutorial.html
import simple_model
import pymc
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import pysmc
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np


if __name__ == '__main__':
    # Construct the SMC sampler
    model = simple_model.make_model()
    mcmc = pymc.MCMC(model)
    mcmc.use_step_method(pysmc.RandomWalk, model['mixture'])
    smc_sampler = pysmc.SMC(mcmc, num_particles=1000,
                            num_mcmc=1, verbose=4)
    # Initialize SMC at gamma = 0.01
    smc_sampler.initialize(0.001)
    # Move the particles to gamma = 1.0
    smc_sampler.move_to(1.)
    print smc_sampler.log_Zs
    # Get a particle approximation
    p = smc_sampler.get_particle_approximation()
    print p.mean
    print p.variance
    # Plot a histogram
    data = [p.particles[i]['stochastics']['mixture'] for i in xrange(p.num_particles)]
    data = np.array(data)
    plt.plot(data, np.zeros(data.shape), 'ro', markersize=10)
    pysmc.hist(p, 'mixture')
    plt.show()
