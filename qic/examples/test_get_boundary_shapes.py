"""
Test the get_boundary_shapes function, which computes the Cartesian coordinates of the boundary surface for a given near-axis radius.
"""
import time
import numpy as np
from qic import Qic

#############
# LOAD A CONFIGURATION
#############
stel = Qic.from_paper('QI NFP2 Katia smooth')

#############
# GET BOUNDARY SHAPES
#############
ntheta = 100
# Serial, cartesan
timer_start = time.time()
X, Y, Z = stel.get_boundary_shape(r=0.1, ntheta=ntheta, nphi=130, parallel=False, coordinates = 'cartesian')
timer_end = time.time()
print(f"Time taken to compute boundary cartesians: {timer_end - timer_start:.2f} seconds")

# Parallel, cartesian
timer_start = time.time()
X, Y, Z = stel.get_boundary_shape(r=0.1, ntheta=ntheta, nphi=130, parallel=True, coordinates = 'cartesian')
timer_end = time.time()
print(f"Time taken to compute boundary cartesians with parallel=True: {timer_end - timer_start:.2f} seconds")

# Serial, Frenet
timer_start = time.time()
X, Y = stel.get_boundary_shape(r=0.1, ntheta=ntheta, nphi=130, parallel=False, coordinates = 'FS')
timer_end = time.time()
print(f"Time taken to compute boundary shapes in Frenet coordinates: {timer_end - timer_start:.2f} seconds")

# Parallel, Frenet
timer_start = time.time()
X, Y = stel.get_boundary_shape(r=0.1, ntheta=ntheta, nphi=130, parallel=True, coordinates = 'FS')
timer_end = time.time()
print(f"Time taken to compute boundary shapes in Frenet coordinates with parallel=True: {timer_end - timer_start:.2f} seconds")

# Serial, RZ
timer_start = time.time()
R, Z = stel.get_boundary_shape(r=0.1, ntheta=ntheta, nphi=130, parallel=False, coordinates = 'RZ')
timer_end = time.time()
print(f"Time taken to compute boundary shapes in RZ coordinates: {timer_end - timer_start:.2f} seconds")

# Parallel, RZ
timer_start = time.time()
R, Z = stel.get_boundary_shape(r=0.1, ntheta=ntheta, nphi=130, parallel=True, coordinates = 'RZ')
timer_end = time.time()
print(f"Time taken to compute boundary shapes in RZ coordinates with parallel=True: {timer_end - timer_start:.2f} seconds")

