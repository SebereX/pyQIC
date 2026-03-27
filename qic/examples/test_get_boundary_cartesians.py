"""
Test the get_boundary_cartesians function, which computes the Cartesian coordinates of the boundary surface for a given near-axis radius.
"""
import time
import numpy as np
from qic import Qic

#############
# LOAD A CONFIGURATION
#############
stel = Qic.from_paper('QI NFP2 Katia smooth')

#############
# GET BOUNDARY CARTESIANS
#############
ntheta = 100
# Serial, no xsec
timer_start = time.time()
X, Y, Z, _, _ = stel.get_boundary_cartesians(r=0.1, ntheta=ntheta, nphi=130, parallel=False, xsec=False, field_period=False)
timer_end = time.time()
print(f"Time taken to compute boundary cartesians: {timer_end - timer_start:.2f} seconds")

# Serial, with xsec
timer_start = time.time()
X, Y, Z, X_fs, Y_fs = stel.get_boundary_cartesians(r=0.1, ntheta=ntheta, nphi=130, parallel=False, xsec=True, field_period=False)
timer_end = time.time()
print(f"Time taken to compute boundary cartesians with xsec=True: {timer_end - timer_start:.2f} seconds")

# Parallel, no xsec
timer_start = time.time()
X, Y, Z, _, _ = stel.get_boundary_cartesians(r=0.1, ntheta=ntheta, nphi=130, parallel=True, xsec=False, field_period=False)
timer_end = time.time()
print(f"Time taken to compute boundary cartesians with parallel=True: {timer_end - timer_start:.2f} seconds")

# Parallel, with xsec
timer_start = time.time()
X, Y, Z, X_fs, Y_fs = stel.get_boundary_cartesians(r=0.1, ntheta=ntheta, nphi=130, parallel=True, xsec=True, field_period=False)
timer_end = time.time()
print(f"Time taken to compute boundary cartesians with parallel=True and xsec=True: {timer_end - timer_start:.2f} seconds")
