from nbodykit.algorithms import FOF
from nbodykit.source.particle import HOD
from fastpm.nbkit import FastPMParticleSource
from .ic import InitialConditions
import numpy as np

class SimulationError(Exception):
    pass

class ForwardModel(object):
    """
    A forward model of the Universe, based on simulations from
    FastPM and halo occupation distribution modeling from
    Zheng et al. 2007 
    """
    params = ['alpha', 'logM0', 'logM1', 'logMmin', 'sigma_logM']
    
    def __init__(self, ic, redshift):
        """
        Parameters
        ----------
        ic : InitialConditions
            an initial conditions class that specifies the linear mesh
            to start the evolution from
        redshift : float
            the desired final redshift to evolve to
        """
        if isinstance(ic, InitialConditions):
            raise TypeError("input ``ic`` object should be a LinearMesh; call to_mesh() first")

        self.ic = ic
        self.cosmo = ic.attrs['cosmo']
        
        # the RSD factor
        H0 = 100.
        self.rsd_factor = (1.+redshift) / (H0 * cosmo.efunc(redshift))
        
        # store the attrs
        self.attrs = {}
        self.attrs['redshift'] = redshift
        self.attrs['rsd_factor'] = self.rsd_factor
        self.attrs['BoxSize'] = self.ic.attrs['BoxSize']
        self.attrs['seed'] = self.ic.attrs['seed']
        
    @classmethod
    def from_halos(cls, halos, seed=None):
        """
        Initialize the forward model from an existing catalog
        of halos
        
        Parameters
        ----------
        halos : HaloCatalog
            an existing halo catalog to initialize the model from
        seed : int, optional
            the random seed to use
        """
        from nbodykit.source.particle import HaloCatalog
        if not isinstance(halos, HaloCatalog):
            raise TypeError("``halos`` should be a Source.HaloCatalog from nbodykit")
        
        # construct empty model
        model = object.__new__(cls)
        
        # add the relevant attributes
        model.halos = halos
        model.cosmo = halos.cosmo # cosmology of the halo catalog
        model.ic    = None
        model.sim   = None
                
        H0 = 100.
        z = halos.attrs['redshift'] # redshift of the halo catalog
        
        # store the attrs
        model.attrs = {}
        model.attrs['redshift'] = z
        model.attrs['rsd_factor'] = (1.+z) / (H0 * model.cosmo.efunc(z))
        model.attrs['BoxSize'] = halos.attrs['BoxSize'] # size of the box the halos are in
        model.attrs['seed'] = seed
        
        return model
    
    def evolve(self, zstart=10., boost=2, Nsteps=5):
        """
        Evolve the initial conditions to the desired redshift
        using the FastPM framework
        
        Attributes
        ----------
        sim : FastPMParticleSource
            the source of dark matter particles at the
            specified redshift, evolved forward from the initial
            conditions
        
        Parameters
        ----------
        zstart : float
            the initial redshift to start the simulation at
        boost : int
            the ratio of the force to particle mesh in FastPM
        Nsteps : int
            the number of time steps to use
        """
        kws = {}
        kw['boost'] = boost
        kws['Nsteps'] = Nsteps
        kws['astart'] = (1+zstart)**(-1)
        kws['aend']  = (1+self.attrs['redshift'])**(-1)
        self.sim = FastPMParticleSource(ic, **kws)

    def find_halos(self, particle_mass, linking_length=0.2, nmin=20):
        """
        Identify halos in the dark matter field using the
        Friends-of-Friends (FoF) algorithm
        
        Attributes
        ----------
        halos : HaloCatalog
            the source of halos, identified from the 
            source of particles in :attr:`sim`
        
        Parameters
        ----------
        particle_mass : float
            the mass of the particles in :attr:`sim` in M_sun/h
        linking_length : float, optional
            the linking length to use when running FoF
        nmin : int, optional
            the minimum number of objects that a halo must contain
        """
        # need to run the FastPM simulation!
        if not hasattr(self, 'sim'):
            raise SimulationError("call evolve() to evolve the dark matter field")
        
        # run FOF on the dark matter field
        fof = FOF(source, linking_length=linking_length, nmin=nmin)
        
        # and convert FOF groups to halos with mass, etc
        self.halos = fof.to_halos(particle_mass, self.cosmo, self.attrs['redshift'])
        
    def populate_halos(self, rsd=[0,0,0], **params):
        """
        Populate the halo catalog with galaxies using the halo occupation
        distribution (HOD) model from Zheng et al. 2007
        
        Repeated calls to this function with different parameters will
        repopulate the existing halo catalog with a new set of galaxies
        
        Parameters
        ----------
        rsd : array_like (3,)
            the vector specifying the line-of-sight to add redshift-space
            distortions (RSD); default of [0,0,0] means no RSD
        **params : 
            the key/value pairs representing the HOD parameters
        """
        # need to find the halos!
        if not hasattr(self, 'halos'):
            raise SimulationError("call find_halos() to find halos before populating galaxies")
        
        # initialize HOD the first time
        if not hasattr(self, 'hod'):
            
            # convert to halotools format
            halocat = self.halos.to_halotools(BoxSize=self.attrs['BoxSize'])
        
            # and now do the HOD population
            self.hod = HOD(halocat, seed=self.attrs['seed'], **params)
            
        # re-populate
        else:
            self.hod.repopulate(seed=self.attrs['seed'], **params)
            
        # add RSD
        if np.sum(rsd):
            self.hod['Position'] += self.attrs['rsd_factor'] * self.hod['Velocity'] * np.array(rsd)    
