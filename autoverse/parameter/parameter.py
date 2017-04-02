from lmfit import Parameter as BaseParameter
from .priors import PriorBase

class Parameter(BaseParameter):
    """
    Add priors to the lmfit.Parameter class
    """
    def __init__(self, *args, **kwargs):
        self.prior = kwargs.pop('prior', None)
        super(Parameter, self).__init__(*args, **kwargs)
                
    @property
    def prior(self):
        return self._prior
    
    @prior.setter
    def prior(self, val):
        if val is not None and not isinstance(val, PriorBase):
            raise TypeError("prior should be a subclass of PriorBase")
        self._prior = val
    
    def logprior(self, val):
        """
        The log probability of the prior distribution
        """
        lnprior = 0.
        
        # add in the log prior value (can also be -inf)
        if self.prior is not None: 
            lnprior += self.prior.log_pdf(val)
   
        return lnprior