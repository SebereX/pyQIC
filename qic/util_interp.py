"""
This module contains the routine to initialize quantities like
curvature and torsion from the magnetix axis shape.
"""

import logging
import numpy as np
from scipy.interpolate import CubicSpline as spline
from scipy.interpolate import BSpline, make_interp_spline, PchipInterpolator

logger = logging.getLogger(__name__)

# Define periodic spline interpolant conversion used in several scripts and plotting
def convert_to_spline(self, array, grid = None, varphi = False, periodic = True, half_period = False):
    """
    General interpolation routine, especially thought for including interpolation of periodic and half-periodic
    functions.
    """
    # Define the input domain : either use the prescribed input domain or the grid provided
    if isinstance(grid, np.ndarray):
        domain = grid
    else:
        domain = self.varphi if varphi else self.phi

    # The domain should be either [0,2pi) or [0,2pi/N) with the endpoints included or not .
    # First detect which domain it corresponds to
    domain_range = [np.abs(domain[-1]-domain[0]-2*np.pi), np.abs(domain[-1]-domain[0]-2*np.pi/self.nfp)]
    domain_extent = np.argmin(domain_range)
    flag_closed = 1 if domain_range[domain_extent] < (domain[-1]-domain[-2])/10 else 0

    # If input data is float, simply copy it down
    if isinstance(array, float):
        sp=spline(np.append(domain,2*np.pi/self.nfp+domain[0]), np.ones(self.nphi + 1)*array, bc_type='periodic')
    # Interpolation taking into account that the function is periodic
    elif periodic:
        # If defined on the field period
        if half_period:
            sgn_half = -1
            if domain_extent == 0:
                if flag_closed:
                    domain_ext = np.concatenate((domain[:-1] - 2*np.pi, domain, domain[1:] + 2*np.pi))
                    array_ext = np.concatenate((array[:-1]*sgn_half**self.nfp, array, array[1:]*sgn_half**self.nfp))
                else:
                    domain_ext = np.concatenate((domain - 2*np.pi, domain, domain + 2*np.pi, [4*np.pi + domain[0]]))
                    array_ext = np.concatenate((array*sgn_half**self.nfp, array, array*sgn_half**self.nfp, [array[0]*sgn_half**self.nfp]))
            else:
                if flag_closed:
                    domain_ext = np.concatenate(tuple(domain[:-1] + 2*np.pi/self.nfp*j for j in range(-1,self.nfp+1,1)))
                    array_ext = np.concatenate(tuple(array[:-1]*sgn_half**j for j in range(-1,self.nfp+1,1)))
                else:
                    domain_ext = np.concatenate(tuple(domain + 2*np.pi/self.nfp*j for j in range(-1,self.nfp+1,1)))
                    array_ext = np.concatenate(tuple(array*sgn_half**j for j in range(-1,self.nfp+1,1)))
            # Spline
            sp_temp = make_interp_spline(domain_ext, array_ext, k=7, axis=0)
            sp = lambda x: sp_temp(x % (2*np.pi))    
        else:
            if flag_closed:
                domain_ext = domain
                array_ext = array
                bc_type = None
            else:
                step = 2*np.pi if domain_extent == 0 else 2*np.pi/self.nfp
                domain_ext = np.append(domain, domain[0] + step)
                array_ext = np.append(array, array[0])
                bc_type = 'periodic'
            sp_temp = make_interp_spline(domain_ext, array_ext, k=7, axis=0, bc_type = bc_type)
            sp = lambda x: sp_temp(x % (2*np.pi/self.nfp))
    else:
        sp = make_interp_spline(domain, array, k = 7)
    return sp

## THINK ALSO OF THE USE IN INPUT INTERPOLATION ##

## COULD PLACE THE SMOOTHING CONSIDERATION HERE ##