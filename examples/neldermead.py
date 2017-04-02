from __future__ import print_function

import argparse
import autoverse
from autoverse import models, priors, Parameter
from nbodykit.lab import *
from nbodykit import setup_logging

setup_logging()

def load_halos(cosmo, z):
    """
    Load an examples Friends-of-Friends halos catalog
    """
    # load the FOF groups
    halos = Source.File(IO.HDFFile, 'data/fof_0.6250.subsample.hdf5')
    BoxSize = halos.attrs['BoxSize']

    # scale the Position/Velocity
    halos['Position'] *= BoxSize
    halos['Velocity'] *= BoxSize

    # velocity in km/s
    H0 = 100.
    rsd_factor = (1.+z) / (H0 * cosmo.efunc(z))
    halos['Velocity'] /= rsd_factor
    
    # scale by particle mass
    m0 = 2.6106e10
    halos['Mass'] = m0 * halos['Length']

    # Halo Catalog 
    halos = Source.HaloCatalog(halos, cosmo=cosmo, redshift=z, mdef='vir')
    halos['Selection'] = halos['Mass'] > 0
    
    return halos
    
def load_data(kmin=0.01, kmax=0.4):
    """
    Load an example clustering measurment of the monopole and 
    quadrupole
    """
    data = numpy.loadtxt('data/ncutsky_poles.dat')
    
    # trim to k-range
    k = data[:,0]
    valid = (k >= kmin)&(k <= kmax)
    data = data[valid]
    
    # monopole/quadrupole
    x = list(data[:,[0,0]].T)
    y = list(data[:,[1,3]].T)
    cov = list((data[:,[2,4]]**2).T)
    
    return autoverse.data.Measurements(x, y, cov)
    

if __name__ == '__main__':
    
    desc = 'use the Nelder-Mead algorithm to model an example monopole/quadrupole'
    parser = argparse.ArgumentParser(description=desc)
    
    h = 'save the best-fitting theory result to this file'
    parser.add_argument('--bestfit', type=str, help=h)
    
    h = 'the kmax of the fit'
    parser.add_argument('--kmax', default=0.4, type=float, help=h)
    
    h = 'the maximum number of iterations to run'
    parser.add_argument('--maxiter', type=int, default=10, help=h)
    
    ns = parser.parse_args()
    
    redshift = 0.613
    cosmo = cosmology.Cosmology(Om0=0.307494, H0=67.74, flat=True)
    
    # initialize the forward model
    forward = models.ForwardModel.from_halos(load_halos(cosmo, redshift))
    
    # and the power spectrum multipoles
    ells = [0, 2]
    multipoles = models.PowerMultipoles(forward, ells, Nmesh=256, rsd=[0,0,1])

    # load the data
    data = load_data(kmax=ns.kmax)
    
    # initialize the parameters
    params = autoverse.Parameters()
    params['alpha'] = Parameter(value=0.76, prior=priors.Uniform(0.5, 1.5))
    params['logM0'] = Parameter(value=13.27, prior=priors.Uniform(12.5, 13.8))
    params['logM1'] = Parameter(value=13.9, prior=priors.Uniform(13.5, 14.5))
    params['logMmin'] = Parameter(value=13.15, prior=priors.Uniform(12.5, 13.5))
    params['sigma_logM'] = Parameter(value=0.38, prior=priors.Uniform(0.2, 0.7))

    # the solver
    solver = autoverse.NelderMeadSolver(params, data=data, theory=multipoles, save_bestfit=ns.bestfit)
    result = solver.run(maxiter=ns.maxiter)
    print("max log probability = ", -1.0 * result.fun)

