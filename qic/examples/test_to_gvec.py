"""
Test the to_gvec function.
"""
import numpy as np
from qic import Qic
import matplotlib.pyplot as plt 

###################### 
# LOAD STELLARATOR
#####################
stel = Qic.from_paper('QI NFP2 Katia smooth')
# stel.plot_3d(r=0.1, shaded=True, show=True)

#####################
# CALL THE FUNCTION
#####################
ntheta = 100
nzeta = 130
r = 0.1
parallel = True
outpath = '/home/erodrigu/pyQIC/qic/examples/GVEC_test'
output_prefix = 'test_gvec'

params, dict_gframe = stel.to_gvec(
    ntheta=ntheta,
    nzeta=nzeta,
    r=r,
    outpath=outpath,
    output_prefix=output_prefix,
    verbose = True
    )

print(params)

# ####################
# # PLOT CROSS SECTIONS
# ####################
# Qic.plot_cross_sections_gframe(
#     dict_gframe,
#     n_cross_sections=10,
#     tolerance=1e-5)
# plt.show()

# ################
# # RUN GVEC #
# #############
# import gvec
# # May modify the parameters dictionary to adjust mpol, ntor, or desired quantities
# run = gvec.run(params, runpath="to_gframe_QI_NFP2_Katia")