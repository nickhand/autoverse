import numpy
from scipy.special import legendre
from scipy.interpolate import UnivariateSpline as spline
from nbodykit.algorithms import FFTPower

def compute_gaussian_errors(pkmu, ells):
    """
    Compute the Gaussian errors for the multipoles from the 
    two-dimensional power spectrum ``P(k,mu)``
    """    
    # weight by modes
    modes = numpy.nan_to_num(pkmu['modes'])
    N_1d = modes.sum(axis=-1)
    weights = modes / N_1d[:,None]

    # avg mu
    mu = numpy.nan_to_num(pkmu['mu'])

    # compute the variance
    P = pkmu['power'].real
    var = numpy.asarray([numpy.nansum(weights*((2*ell+1)*P*legendre(ell)(mu))**2, axis=-1) for ell in ells])
    var *= 2/N_1d[None]
    return var**0.5


class PowerMultipoles(object):
    """
    Theoretical modeling of the power spectrum multipoles statistic
    """
    def __init__(self, forward, ells, Nmesh, rsd=[0,0,1], dk=0.005, Nmu=100, kmin=0.):
        """
        Parameters
        ----------
        forward : ForwardModel
            the forward model that gives a galaxy catalog at a specific redshift
        ells : list of int
            the list of multipole numbers to compute
        Nmesh : int
            the mesh size to use to compute the galaxy density field
        rsd : 3-vector, optional
            the unit vector specifying the direction to apply RSD; [0,0,0] is no RSD
        dk : float, optional
            the k-spacing to use when computing the multipoles statistic
        Nmu : int, optional
            number of mu bins to use to compute Gaussian errors
        kmin : float, optional
            the lower edge of the first k bin to use
        """
        self.forward = forward
        self.params = forward.params
        
        self.attrs = {}
        self.attrs['ells']  = ells
        self.attrs['Nmesh'] = Nmesh
        self.attrs['dk']    = dk
        self.attrs['Nmu']   = Nmu
        self.attrs['kmin']  = kmin
        self.attrs['rsd']   = rsd
        
    def update(self, **params):
        """
        Update the theory with new parameters
        
        In this case, these are the "slow" HOD parameters
        """
        self.forward.populate_halos(rsd=self.attrs['rsd'], **params)
        for name in ['result', 'gaussian_error']:
            if hasattr(self, name): delattr(self, name)

    def _compute(self):
        """
        Internal function to compute the multipoles from 
        a galaxy catalog
        """
        kws = {}
        kws['mode'] = '2d'
        kws['BoxSize'] = self.forward.attrs['BoxSize']
        kws['Nmesh'] = self.attrs['Nmesh']
        kws['poles'] = sorted(list(self.attrs['ells']))
        
        for name in ['dk', 'Nmu', 'kmin']:
            kws[name] = self.attrs[name]
            
        # make sure line-of-sight points in the RSD direction
        if numpy.sum(self.attrs['rsd']):
            kws['los'] = self.attrs['rsd']
        
        # compute the multipoles
        self.result = FFTPower(self.forward.hod, **kws)

        # and the Gaussian errors
        self.gaussian_error = compute_gaussian_errors(self.result.power, self.attrs['ells'])
        
    @property
    def state(self):
        """
        The theoretical state, which gives the current measured
        statistic result
        """
        if hasattr(self, 'result'):
            return self.result.__getstate__()
        else:
            return None
    
    def save_state(self, output):
        """
        Save the current state to disk, as a JSON file
        """
        import json
        from nbodykit.utils import JSONEncoder
        from nbodykit import CurrentMPIComm
        
        comm = CurrentMPIComm.get()
        if comm.rank == 0:
            with open(output, 'w') as ff:
                json.dump(self.state, ff, cls=JSONEncoder)
        
    def __call__(self, ell):
        """
        Evaluate the theory for a specified multipole number,
        returning a callable function that returns the :math:`P_\ell(k)`
        """
        iell = self.attrs['ells'].index(ell)        
        if not hasattr(self, 'poles'):
            self._compute()
        
        # the power multipole
        poles = self.result.poles
        Pell = poles['power_%d' %ell].real
        if ell == 0:
            Pell -= self.result.attrs['shotnoise']
        
        valid = ~numpy.isnan(Pell)    
        return spline(poles['k'][valid], Pell[valid], w=1./self.gaussian_error[iell][valid])
            
    def __iter__(self):
        """
        Iterate through each multipole
        """
        for ell in self.attrs['ells']:
            yield self(ell)
