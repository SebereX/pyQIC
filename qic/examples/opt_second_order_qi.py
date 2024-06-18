# Example of the optimisation of d_bar for second order QI

from qic import Qic
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy import optimize, interpolate

# Define a function that in addition to the condition of second order QI, it also requires some elongation constraint
def geo_condition_and_elongation(stel, order):
    res = 0
    if np.max(stel.elongation)>15.0:
        res += np.max(stel.elongation)
    elif np.min(stel.elongation)<1/15.0:
        res += 1/np.min(stel.elongation)
    return stel.min_geo_qi_consistency(order = order) + res

# Some arbitrary initial case with an appropriate axis (with order 1 curvature)
nphi = 301
order = 'r1'
nfp=1
Raxis = {"type": 'fourier',
            "input_value": {"cos": [1.0,0.0,-0.3,0.0,0.01,0.0,0.001],
                            "sin": []}}
Zaxis = {"type": 'fourier',
            "input_value": {"cos": [],
                            "sin":  [0.0,0.0,-0.2,0.0,0.01,0.0,0.001]}}
B0 = {"type": 'fourier',
            "input_value": {"cos": [1.0,0.16], "sin": []}}
# d_over_curvature = {"type": 'fourier', "input_value": {"cos": [0.5,0.01,0.01,0.01,0.01,0.01,0.01],
                                                    #    "sin": []}}
d_over_curvature = {"type": 'spline', "input_value": [0.5,0.5,0.5,0.5,0.5,0.5]}


buffer_opt = {}
buffer_opt["omn_method"] = 'non-zone-fourier'
buffer_opt["delta"] = 0.1
buffer_opt["k_buffer"] = 1
buffer_opt["p_buffer"] = 2

# # Initial point of departure
# stel = Qic(omn_buffer = buffer_opt, Raxis = Raxis, Zaxis = Zaxis, nfp=nfp, B0 = B0, 
#     nphi=nphi, omn=True, order=order, d_over_curvature=d_over_curvature, d = None)

# # Optimise for d_bar without worrying about d_bar crossing the zero and elongation becoming too large
# stel.construct_qi_r2(verbose=1)
# stel.plot()

# Optimise for d_bar, but taking into account elongation
stel = Qic(omn_buffer = buffer_opt, Raxis = Raxis, Zaxis = Zaxis, nfp=nfp, B0 = B0, 
    nphi=nphi, omn=True, order=order, d_over_curvature=d_over_curvature, d = None)
stel.plot()
# params = ['d_over_curvaturec(0)','d_over_curvaturec(1)','d_over_curvaturec(2)','d_over_curvaturec(3)','d_over_curvaturec(4)','d_over_curvaturec(5)','d_over_curvaturec(6)'] #'B2cs(1)','B2sc(0)','B2cs(2)','B2sc(1)','B2cs(3)','B2sc(2)','B2cs(4)','B2sc(3)',
# Just to show that it can be done externally telling what parameters to use
stel.construct_qi_r2(verbose=1, method="Nelder-Mead", params=[],fun_opt = geo_condition_and_elongation)
# stel.construct_qi_r2(verbose=1, method="BFGS", params=params,fun_opt = geo_condition_and_elongation) # Refine the solution
stel.plot()