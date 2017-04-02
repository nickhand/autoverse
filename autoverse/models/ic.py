from classylss import ClassParams, Cosmology, transfers
from classylss.power import LinearPS


class InitialConditions(object):
    """
    Specify the initial conditions via the linear power spectrums
    
    Attributes
    ----------
    power : callable
        a callable returning the linear power spectrum in ``(Mpc/h)^3``
        as a function of the wavenumber ``k`` in ``h/Mpc``
    
    TODO: updating ``power`` based on parameter changes
    """
    available_transfers = ['CLASS', 'NoWiggleEH', 'EH']
    
    def __init__(self, cosmo, transfer='CLASS', verbose=False, **kws):
        """
        Parameters
        ----------
        cosmo : nbodykit.cosmology.Cosmology
            a cosmology instance defining the parameters of the initial
            conditions 
        transfer : {'CLASS', 'NoWiggleEH', 'EH'}, optional
            the transfer function to use, either numerical (CLASS) or 
            analytic (Eisenstein & Hu 1998)
        verbose : bool, optional
            the verbosity level to use when running CLASS
        **kws : 
            additional keywords to use when running CLASS
        """
        # check transfer function
        if transfer not in self.available_transfers:
            raise ValueError("available transfer functions: %s" %str(self.available_transfers))
            
        self.cosmo = cosmo
        
        # setup the parameters to give to classylss
        if 'n_s' not in cosmo:
            raise ValueError("please specify 'n_s' in the input cosmology")
        if 'sigma8' not in cosmo:
            raise ValueError("please specify 'sigma8' in the input cosmology")
        kws.setdefault('n_s', cosmo['n_s'])
        kws.setdefault('sigma8', cosmo['sigma8'])
        pars = ClassParams.from_astropy(cosmo.engine, extra=kws)
        
        # determine which transfer
        if transfer == 'NoWiggleEH':
            t = transfers.EH_NoWiggle
        else:
            t = getattr(transfers, transfer)
           
        # initialize the linear power spectrum at z = 0
        self.classcosmo = Cosmology(pars, t, verbose)
        self.power = LinearPS(self.classcosmo, z=0.)
        
        # normalize to the desired sigma8
        self.power.SetSigma8AtZ(cosmo['sigma8'])
        
    def to_mesh(self, BoxSize, Nmesh, seed=None, remove_variance=False):
        """
        Return a Gaussian density field on a mesh
        
        Parameters
        ----------
        BoxSize : scalar, 3-vector
            the size of the box that the mesh covers
        Nmesh : int
            the number of cells per side of the mesh
        seed : int, optional
            fix the random seed to this value
        remove_variance : bool, optional
            remove the variance of the initial Gaussian field
        
        Returns
        -------
        mesh : nbodykit.source.mesh.LinearMesh
            a linear mesh nbodykit Source, with the density field
            set by the linear power spectrum specified by :attr:`power`
        """
        from nbodykit.source.mesh import LinearMesh
        
        kws = {'seed':seed, 'remove_variance':remove_variance}
        mesh = LinearMesh(self.power, BoxSize, Nmesh, **kws)
        mesh.Plin.cosmo = self.cosmo
        mesh.attrs['cosmo'] = self.cosmo
        return mesh
        
        
        
        