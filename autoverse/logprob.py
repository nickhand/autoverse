import numpy as np
from .parameter import Parameters

class LogProbability(object):
    """
    The log probability distribution
    """
    def __init__(self, params, data, theory, callback=None):
        """
        Parameters
        ----------
        params : Parameters
            the set of parameters in the model
        data : Measurements
            the object holding the data measurements
        theory : 
            the object holding the theory class
        callback : callable, optional
            a function to call during each execution of the log probability
        """
        self.callback = callback
        
        # make sure we have all of the needed parameters
        missing = []
        for param in theory.params:
            if param not in params:
                missing.append(params)
        if len(missing):
            raise ValueError("the following theory parameters are missing: %s" %str(missing))
        
        # store data and theory for later
        self.data   = data
        self.theory = theory
        
        # store the parameters and the free parameter names
        self.params = params.copy()
        self.free = Parameters()
        self.free.add_many(*[params[name] for name in params if params[name].vary])
        
        # dict of fixed parameters
        self.fixed = {p:self.params[p].value for p in theory.params if not self.params[p].vary}
        
        # compute the data vector
        self.data_vector = np.concatenate([d.y for d in self.data])
    
    def theory_vector(self):
        """
        Return the concatenated theory array, first updating
        the theory model to the latest free parameters
        """
        # update the theory
        pars = self.free.copy()
        pars.update(self.fixed)
        self.theory.update(**pars)    
        
        # combined theory vector
        theory = []
        for d, th in zip(self.data, self.theory):
            theory.append(th(d.x))
        
        return np.concatenate(theory)
        
    def __call__(self, theta, return_state=False):
        """
        Evaluate the log-probability as the sum of the 
        log of prior distribution values and the
        log likelihood
        
        Parameters
        ----------
        theta : array_like
            the array of free parameter values
        return_state : bool, optional
            if `True`, also return the state of the theory
        """
        # compute the prior
        lp = 0
        for i, par in enumerate(self.free):
            lp += self.free[par].logprior(theta[i])
        
        # return if non-finite prior
        if not np.isfinite(lp):
            if return_state:
                return -np.inf, None
            else:
                return -np.inf
        
        # update the parameters
        for i, name in enumerate(self.free):
            self.free[name].value = theta[i]
            
        # now compute the log probability
        diff = self.theory_vector() - self.data_vector
        lnprob = -0.5 * np.dot(diff, np.dot(self.data.invcov, diff)) + lp
        
        # callback?
        if self.callback is not None:
            self.callback(self, theta, lnprob)
        
        if return_state:
            return lnprob, self.theory.state
        else:
            return lnprob
            
        
