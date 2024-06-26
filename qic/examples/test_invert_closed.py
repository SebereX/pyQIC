import numpy as np
from scipy.optimize import minimize, least_squares
from qic.reverse_frenet_serret import invert_frenet_axis

# Get the machine precision for floating-point numbers
eps = np.finfo(float).eps

# Domain size
N = 5001

# nfp 
nfp = 3

# Parameters for kappa and tau
params = [-8.072526644507764, 1.341749529166234, -0.133333320724175]
# [-7.94360471  1.3495362  -0.19056053]

# Length scale
ell = np.linspace(0.0, 1.0, N, endpoint=False)*2*np.pi/nfp  # parameter array

# Curvature and torsion
curvature = 0.5*(1+np.cos(nfp*ell))*np.sin(0.5*nfp*ell)*np.sin(nfp*ell)*params[0]
torsion = params[1] + params[2]*np.cos(3*ell)

class Struct():
    """
    This class is just an empty mutable object to which we can attach
    attributes.
    """
    pass

stel = Struct()
stel.nfp = nfp
stel.flag_half = True
stel.nphi = N
stel.L_in = 2*np.pi/nfp
stel.phi = np.linspace(0, 2*np.pi/stel.nfp, stel.nphi)
stel.varphi = np.linspace(0, 2*np.pi/stel.nfp, stel.nphi)

curvature_func = lambda x: 0.5*(1+np.cos(stel.nfp*x))*np.sin(0.5*stel.nfp*x)*np.sin(stel.nfp*x)*params[0]
torsion_func = lambda x:params[1] + params[2]*np.cos(stel.nfp*x)

curvature_func = lambda x: 0.5*(1+np.cos(x/stel.L_in*2*np.pi))*np.sin(0.5*x/stel.L_in*2*np.pi)*np.sin(x/stel.L_in*2*np.pi)*params[0]
torsion_func = lambda x: params[1] + params[2]*np.cos(x/stel.L_in*2*np.pi)

stel.curvature_in = {"function_ell": curvature_func}
stel.torsion_in = {"function_ell": torsion_func}

mismatch = invert_frenet_axis(stel, curvature, torsion, ell, stel.varphi, plot = False, full_axis = True, func = True) 

def res_closed(params):
    # Curvature and torsion
    curvature = 0.5*(1+np.cos(nfp*ell))*np.sin(0.5*nfp*ell)*np.sin(nfp*ell)*params[0]
    torsion = params[1] + params[2]*np.cos(3*ell)

    curvature_func = lambda x: 0.5*(1+np.cos(x/stel.L_in*2*np.pi))*np.sin(0.5*x/stel.L_in*2*np.pi)*np.sin(x/stel.L_in*2*np.pi)*params[0]
    torsion_func = lambda x: params[1] + params[2]*np.cos(x/stel.L_in*2*np.pi)

    stel.curvature_in = {"function_ell": curvature_func}
    stel.torsion_in = {"function_ell": torsion_func}

    try:
        # Should always call full to correctly identify the axis plane
        mismatch = invert_frenet_axis(stel, curvature, torsion, ell, stel.varphi, plot = False, full_axis = True, minimal = True, func = True) 

        # Tangent difference
        diff_tangent = np.array(mismatch[0])
        res_tangent = np.sum(diff_tangent*diff_tangent)
        # Normal difference
        diff_normal = np.array(mismatch[1])
        res_normal = np.sum(diff_normal*diff_normal)
        # normal difference
        diff_binormal = np.array(mismatch[2])
        res_binormal = np.sum(diff_binormal*diff_binormal)

        diff_pos = np.array(mismatch[3])
        res_pos = np.sum(diff_pos*diff_pos)
        # # R difference
        # diff_R = mismatch[3]
        # res_R = diff_R*diff_R

        # # R difference
        # diff_Z = mismatch[4]
        # res_Z = diff_Z*diff_Z

        # # phi difference
        # diff_phi = mismatch[5]
        # res_phi = diff_phi*diff_phi

        # Total res
        res_tot = res_tangent + res_normal + res_binormal + res_pos

    except:
        diff_tangent = [1.0, 1.0, 1.0]
        diff_normal = [1.0, 1.0, 1.0]
        diff_binormal = [1.0, 1.0, 1.0]
        diff_R = 1.0
        diff_Z = 1.0
        res_tot = 1

    print(res_tot)
    return np.concatenate((diff_tangent, diff_normal, diff_pos))

options = {'maxiter': 1000, 'xtol': 1e-15, 'ftol': 1e-15, 'gtol': 1e-15}
opt = minimize(lambda x: np.linalg.norm(res_closed(x)), params, args = (), method = 'BFGS', options = options)
options = {'method': 'lm', 'max_nfev': 1000, 'xtol': 1e-15, 'ftol': 1e-15, 'gtol': 1e-15}
opt = least_squares(res_closed, params, **options)  

np.set_printoptions(precision=15)

print(params)
print(opt.x)
print(opt.fun)
print(opt.status)



