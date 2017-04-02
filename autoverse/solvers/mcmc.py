import emcee
import numpy as np
import json
import logging

from ..logprob import LogProbability
from mpi4py import MPI
from nbodykit.utils import JSONEncoder

class EmceeSolver(object):
    """
    Sample from the posterier distribution using the :mod:`emcee`
    package via the MCMC algorithm
    """
    def __init__(self, params, data, theory, save_bestfit=None):
        """
        Parameters
        ----------
        params : Parameters
            the set of parameters describing the theory model
        data : Measurements
            the data measurements
        theory : 
            the object holding the theory class
        save_bestfit : str, optional
            a string specifying the file name to use to save the best-fitting 
            theory result to disk
        """
        self.logprob = LogProbability(params, data, theory)
        self.save_bestfit = save_bestfit
        
    def run(self, iterations, nwalkers, pool=None):
        """
        Run the mcmc algorithm for the specified number of iterations
        
        Parameters
        ----------
        iterations : int
            the number of iterations to run
        nwalkers : int
            the number of walkers to use to sample
        pool : MPI Pool
            an object with a map function for parallelization
        
        Returns
        -------
        sampler : emcee.EnsembleSampler
            the emcee sampler object
        """
        ndim = len(self.logprob.free)
        objective = lambda theta: self.logprob(theta, return_state=True)
        
        # start parameters in random ball around initial
        p0 = np.array(self.logprob.free)
        p0 = [p0 + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
    
        # initialize the sample
        if MPI.COMM_WORLD.rank == 0: 
            args = (iterations, nwalkers)
            logging.info("running sampler for %d iterations with %d walkers..." %args)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, objective, pool=pool)
        
        # track the maximum lnprob
        max_lnprob = -np.inf
        
        # sample!
        for i, (pos, lp, rstate, blobs) in enumerate(sampler.sample(p0, iterations=iterations)):
            if MPI.COMM_WORLD.rank == 0:
                logging.info("iteration = %d" %(i+1))

            # save the results
            if self.save_bestfit is not None:
                
                if MPI.COMM_WORLD.rank == 0:
                    thismax = lp.max()
                    if thismax > max_lnprob:
                        with open(self.save_bestfit, 'w') as ff:
                            json.dump(blobs[lp.argmax()], ff, cls=JSONEncoder)
                        max_lnprob = thismax

            # clear the results for memory purposes
            sampler.clear_blobs()

        return sampler