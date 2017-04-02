from scipy.optimize import minimize
from ..logprob import LogProbability

import logging
from nbodykit import CurrentMPIComm
import numpy as np
        
class NelderMeadSolver(object):
    """
    Minimize the log probability distribution using the Nelder-Mead
    downhill simplex method
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
        save_bestfit : bool, optional
            whether to save the best-fitting theory result to disk
        """
        self.logprob = LogProbability(params, data, theory)
        self.save_bestfit = save_bestfit
        
    def run(self, **options):
        """
        Run the minimization solver using the Nelder-Mead downhill
        simplex algorithm
        
        Parameters
        ----------
        **options : 
            additional options to pass to the Nelder-Mead algorithm
            via :func:`scipy.optimize.minimize`
        """
        # minimize the negative log-probability
        objective = lambda theta: -1.0*self.logprob(theta)
        
        # track the state of the minimizer
        state = {'iterations':0, 'max_lnprob':-np.inf}
        
        def callback(logprob, theta, val):    
            comm = CurrentMPIComm.get()
            
            # log the parameters and objective value
            if comm.rank == 0:
                theta = "   ".join(["%15.6f" %th for th in theta])
                X = "%.4d   %s   %15.6f" %(state['iterations'], theta, val)
                logging.info(X)
                state['iterations'] += 1
        
            # save the best-fit?
            if val > state['max_lnprob'] and self.save_bestfit is not None:
                state['max_lnprob'] = val
                logprob.theory.save_state(self.save_bestfit)
        
            comm.barrier()
        
        # set the callback
        self.logprob.callback = callback
        
        # call scipy's minimize
        p0 = np.array(self.logprob.free)
        res = minimize(objective, p0, method='Nelder-Mead', options=options)
        return res