###################################
# TIMING COMPARISON PARALLEL PLOT #
###################################
from qic import Qic
import time
import matplotlib.pyplot as plt

###################
# Construct field #
###################
stel = Qic.from_paper('QI NFP2 Katia', nphi = 1001)

##################################
# Construct boundary in PARALLEL #
##################################
# Record the start time
start_time = time.time()
# Run
stel.plot_boundary(parallel = True, ntheta_fourier = 30, show = False, plot_3d = False)
# Record end time
end_time = time.time()
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

################################
# Construct boundary in SERIES #
################################
# Record the start time
start_time = time.time() 
# Run
stel.plot_boundary(parallel = False, ntheta_fourier = 30, show = False, plot_3d = False)
# Record the end time
end_time = time.time() 
elapsed_time = end_time - start_time  # Calculate elapsed time
print(f"Elapsed time: {elapsed_time:.4f} seconds")


