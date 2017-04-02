import numpy as np
from collections import namedtuple

Measurement = namedtuple('Measurement', ['x', 'y', 'yerr'])

class Measurements(object):
    """
    A class to hold a series data measurements, i.e., multipoles
    """
    def __init__(self, x, y, cov):
        """
        Parameters
        ----------
        x : list
            list of independent variables for each measurement
        y : list
            list of dependent variables for each measurement
        cov : array_like or list
            the covariance matrix, either a list of diagonal entries
            for each measurement, or the full covariance for the
            concatenated data vector
        """
        self.x = x
        self.y = y
        assert len(x) == len(y), "data size mismatch"
        
        # store the sizes
        self.sizes = [len(xi) for xi in x]
        self.size = sum(self.sizes)
        
        # check for diagonal errors
        if isinstance(cov, list):
            cov = np.diag(np.concatenate(cov))
        
        # store the covariance and the inverse
        self.cov = cov
        self.invcov = np.linalg.inv(self.cov)
        
        # check covariance shape
        covshape = np.shape(self.cov)
        if covshape != (self.size, self.size):
            raise ValueError("covariance should have shape %s, not %s" %((self.size,self.size),covshape))
            
        # and save yerrors from diagonal of covariance
        self.yerr = np.split(np.diag(self.cov)**0.5, np.cumsum(self.sizes))[:len(self)]
       
    def __len__(self):
        return len(self.x)

    def __iter__(self):
        for i in range(len(self)):
            yield Measurement(x=self.x[i], y=self.y[i], yerr=self.yerr[i])
        
        
            
        
        
        