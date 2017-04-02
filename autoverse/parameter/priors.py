import numpy as np
import scipy.stats

# list of valid names and parameters for each distribution
valid_distributions = {'normal' : ['mu', 'sigma'], 'uniform': ['lower', 'upper']}

class BoundDistribution(object):
    """
    Base class to represent a minimum/maximum bound on a distribution
    """
    def __init__(self, sign, value, analytic=False):
        self.sign = sign
        self.value = value
        self.analytic = analytic
        
    def pdf(self, x, k=1000):
        x = self.sign * (x-self.value)
        if not self.analytic:
            return 0. if x < 0 else 1.
        else:
            return 1./(1 + np.exp(-2*k*x))
        
    def log_pdf(self, x, k=1000):
        x = self.sign * (x-self.value)
        if not self.analytic:
            return -np.inf if x < 0 else 0.
        else:
            return -np.logaddexp(0, -2*k*x)
            
    def deriv_log_pdf(self, x, k=1000):
        x = self.sign * (x-self.value)
        ratio = np.exp(-2*k*x - np.logaddexp(0, -2*k*x))
        return 2*k*self.sign * ratio
            
class MinimumBound(BoundDistribution):
    def __init__(self, value, analytic=False):
        if value is None: value = -np.inf
        super(MinimumBound, self).__init__(1., value, analytic=analytic)
        
class MaximumBound(BoundDistribution):
    def __init__(self, value, analytic=False):
        if value is None: value = np.inf
        super(MaximumBound, self).__init__(-1., value, analytic=analytic)
        
class PriorBase(object):
    """
    Base class for representing a `Distribution`
    """
    def __init__(self, name, **params):
        """
        Initialize a distribution.
        
        Parameters
        ----------
        name : str 
            the name of the distribution; must be in `valid_distributions`
        params : dict
            the distribution parameters as a dictionary
        """
        # make the name is valid
        self.name = name.lower()
        if self.name not in valid_distributions.keys():
            args = self.name, valid_distributions.keys()
            raise ValueError("Name '{0}' not valid; must be one of {1}".format(*args))
        
        # make sure we have the right parameters
        if any(k not in valid_distributions[self.name] for k in params):
            raise ValueError("Incorrect parameters for distribution of type {0}".format(self.name))
        
        # add the specific distribution parameters
        self.params = valid_distributions[self.name]
        for k, v in params.items():
            if v is None:
                raise ValueError("`%s` attribute for `%s` prior distribution can not be `None`" %(k, self.name))
            setattr(self, k, v)

    def __str__(self):
        """
        Builtin string method
        """
        name = self.name.title()
        pars = ", ".join(['{}={}'.format(key, getattr(self, key)) for key in sorted(self.params)])
        return "{}({})".format(name, pars)

    def __repr__(self):
        """
        Builtin representation method
        """
        return self.__str__()

class Normal(PriorBase):
    """
    The Normal distribution.
    
    The normal distribution has two parameters in its `params` attribute:
    
        - `mu` (:math:`\mu`): the location parameter (the mean)
        - `sigma` (:math:`\sigma`): the scale parameter (standard deviation)
    """
    def __init__(self, mu, sigma):
        """
        Initiate a normal distribution.
        
        Parameters
        -----------
        mu : float
            location parameter (mean)
        sigma : float
            scale parameter (standard deviation)
        """
        super(Normal, self).__init__('normal', mu=mu, sigma=sigma)
    
    def log_pdf(self, x):
        """
        Return the natural log of the normal PDF at the
        specified domain values
        """        
        x = (x - self.loc)/self.scale
        return -0.5*np.log(2*np.pi*self.scale**2) - 0.5*x**2
        
    def deriv_log_pdf(self, x):
        """
        Return the derivative of the natural log of the 
        normal PDF at the specified domain values
        """
        return (self.loc - x)/self.scale**2
                      
    @property
    def loc(self):
        """
        Return the `loc` parameter used by `numpy`, equal here to `self.mu`
        """
        return self.mu
    
    @property
    def scale(self):
        """
        Return the `scale` parameter used by `numpy`, equal here to `self.sigma`
        """
        return self.sigma
    
class Uniform(PriorBase):
    """
    The Uniform distribution.
    
    If an analytic approximation is requested,
    we return an analytic approximation of the Heaviside step function
    """
    def __init__(self, lower, upper, analytic=False):
        """
        Initiate a normal distribution.
        
        Parameters
        -----------
        lower : float
            the lower limit of the distribution
        upper : float
            the upper limit of the distribution
        analytic : bool, optional
            use an analytic approximation to the uniform distribution; 
            default is False
        """
        super(Uniform, self).__init__('uniform', lower=lower, upper=upper)
        
        self.analytic = analytic
        self._center = 0.5*(self.lower + self.upper)
        self._width = self.upper - self.lower
        
    def _analytic_deriv_pdf(self, x, k=1000):
        """
        Derivative of the analytic uniform pdf
        """
        y = (x-self._center)/self._width
        y_ = 0.25 - y**2
        ratio = np.exp(-2*k*y_ - 2*np.logaddexp(0, -2*k*y_))
        return (1. / self._width**2) * -4*k*y * ratio
        
    def _analytic_log_pdf(self, x, k=1000):
        """
        The log of the analytic approximation of the pdf
        """
        y = (x-self._center)/self._width
        y_ = 0.25 - y**2
        return -np.log(self._width) - np.logaddexp(0, -2*k*y_)
    
    def _analytic_deriv_log_pdf(self, x, k=1000):
        """
        Derivative of the log of the analytic uniform pdf
        """
        y = (x-self._center)/self._width
        y_ = 0.25 - y**2
        
        ratio = np.exp(-2*k*y_ - np.logaddexp(0, -2*k*y_)) # safely compute the ratio for large values
        return (1./self._width) * -4*y*k * ratio
    
    def deriv_pdf(self, x, k=1000):
        """
        Return the derivative of the uniform PDF at the
        specified domain values
        """
        if not self.analytic: 
            raise ValueError("set `analytic = True` for derivative of uniform PDF")
        else:
            return self._analytic_deriv_pdf(x, k=k)
    
    def log_pdf(self, x, k=1000):
        """
        Return the natural log of the uniform PDF at the
        specified domain values, optionally using an analytic
        approximation
        """        
        if not self.analytic: 
            kwargs = {'loc' : self.loc, 'scale' : self.scale}
            return scipy.stats.uniform.logpdf(x, **kwargs)
        else:
            return self._analytic_log_pdf(x, k=k)
        
    def deriv_log_pdf(self, x, k=1000):
        """
        Return the derivative of the natural log of the uniform PDF
        at the specified domain values
        """
        if not self.analytic: 
            raise ValueError("set `analytic = True` for derivative of the log of the uniform PDF")
        else:
            return self._analytic_deriv_log_pdf(x, k=k)
        
    @property
    def loc(self):
        """
        Return the `loc` parameter used by `numpy`, equal here to `self.lower`
        """
        return self.lower
    
    @property
    def scale(self):
        """
        Return the `scale` parameter used by `numpy`, equal here to 
        `self.upper` - `self.lower`
        """
        return self.upper - self.lower