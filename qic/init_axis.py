"""
This module contains the routine to initialize quantities like
curvature and torsion from the magnetix axis shape.
"""

import logging
import numpy as np
from scipy.interpolate import CubicSpline as spline
from scipy.interpolate import BSpline, make_interp_spline, PchipInterpolator
from .spectral_diff_matrix import spectral_diff_matrix, finite_difference_matrix, construct_periodic_diff_matrix
from .util import fourier_minimum
from .input_structure import evaluate_input_on_grid
from .fourier_interpolation import fourier_interpolation_matrix, make_interp_fourier
from .reverse_frenet_serret import invert_frenet_axis, to_Fourier_axis

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define periodic spline interpolant conversion used in several scripts and plotting
def convert_to_spline(self,array, half_period = False, varphi = False):
    # def sp(x):
    #     fun = make_interp_fourier(array)
    #     return fun(x, self.phi[0], self.nfp)
    domain = self.varphi if varphi else self.phi

    if isinstance(array, float):
        sp=spline(np.append(domain,2*np.pi/self.nfp+domain[0]), np.ones(self.nphi + 1)*array, bc_type='periodic')
    else:
        if half_period:
            phi_ext = np.concatenate(tuple(domain + 2*np.pi/self.nfp*j for j in range(self.nfp)))
            temp = np.concatenate(tuple(array*(-1)**j for j in range(self.nfp)))
            sp = PchipInterpolator(phi_ext, temp, axis=0, extrapolate='periodic')
        else:
            sp=spline(np.append(domain,2*np.pi/self.nfp+domain[0]), np.append(array,array[0]), bc_type='periodic')
    return sp

def self_consistent_ell_from_varphi(self):
    # Picard iteration is used to find varphi and G0
    nphi = self.nphi

    # Evaliuate B0 on grid
    B0 = self.evaluate_input_on_grid(self.B0_in, self.varphi) 

    # Compute |G0|
    abs_G0 =  self.L_in / (2*np.pi/self.nfp) * nphi/np.sum(1/B0)

    # Separate l into l = ltilde + L_in varphi/ (2pi/N), which must be periodic
    rhs = abs_G0/B0 - self.L_in / (2*np.pi/self.nfp)
    ltilde = np.linalg.solve(self.d_d_varphi+self.interpolateTo0, rhs) # Include interpolateTo0 to make matrix invertible, nu = 0 at origin

    # Construct ell
    ell = ltilde + self.varphi * self.L_in / (2*np.pi/self.nfp)

    return B0, ell, abs_G0

def init_axis(self, omn_complete = True):
    """
    Initialize the curvature, torsion, differentiation matrix, etc.
    """
    # First important step is to distinguish between the Frenet and R/Z options
    if self.frenet:
        # Construct directly from the inputs
        varphi = self.varphi
        self.curvature = self.evaluate_input_on_grid(self.curvature_in, varphi, periodic = False)
        self.torsion = self.evaluate_input_on_grid(self.torsion_in, varphi)
        # self.ell = self.evaluate_input_on_grid(self.ell_in, varphi, periodic = False)
        self.axis_length = self.nfp * self.L_in

        # Axis symmetry (not really correct - evaluate_input_on_grid for grid input returns the array w/o change)
        self.lasym_axis = np.max(np.abs(np.abs(self.curvature) - np.abs(self.evaluate_input_on_grid(self.curvature_in, -varphi))))/np.abs(self.curvature).std() or \
                          np.max(np.abs(np.abs(self.torsion) - np.abs(self.evaluate_input_on_grid(self.torsion_in, -varphi))))/np.abs(self.torsion).std()
        
        # Derivative and interpolation
        self.diff_order, self.d_d_varphi = construct_periodic_diff_matrix(self.diff_finite, self.nphi, self.nfp)
        
        # It is also convenient to define interpolation to phi = 0. The grid is not shifted, could simply evaluate at 0
        # self.interpolateTo0 = fourier_interpolation_matrix(self.nphi, 0)
        self.interpolateTo0 = fourier_interpolation_matrix(self.nphi, 0)

        # Construct the self consistent ell, |G0| and B0
        B0, ell, abs_G0 = self_consistent_ell_from_varphi(self)
        self.ell = ell
        
        # Evaluate the axis in cylindrical coordinates (note that it assumes the curve closes; could have
        # some discontinuity, might need an additional optimisation to close the curve)
        # It modifies phi, nu and the Frenet geometry accordingly, using varphi as regular grid
        flag_func = True if ("function_ell" in self.curvature_in) else False
        _ = invert_frenet_axis(self, self.curvature, self.torsion, self.ell, self.varphi, full_axis = True, func = flag_func)
        
        # Obtain axis description as Fourier components : important for output to VMEC (at least approximately)
        ntor = 10
        rc, rs, zc, zs = to_Fourier_axis(self.R0, self.Z0, self.nfp, ntor = ntor, lasym = False, phi_in = self.phi)
        self.Raxis = {"type": "fourier", "input_value": {}}
        self.Zaxis = {"type": "fourier", "input_value": {}}
        self.Raxis["input_value"]["cos"] = rc
        self.Raxis["input_value"]["sin"] = rs
        self.Zaxis["input_value"]["cos"] = zc
        self.Zaxis["input_value"]["sin"] = zs

        # Computing dl/dphi = dl/dvarphi dvarphi/dphi
        # Need to separate secular parts
        d_l_d_varphi = abs_G0 / B0
        d_phi_d_varphi = 1 - np.matmul(self.d_d_varphi, self.nu)
        d_l_d_phi = d_l_d_varphi / d_phi_d_varphi
        
        # Construct the derivative in phi
        self.d_d_phi = np.zeros((self.nphi, self.nphi))
        for j in range(self.nphi):
            self.d_d_phi[j,:] = self.d_d_varphi[j,:] / d_phi_d_varphi[j]
        d2_l_d_phi2 = np.matmul(self.d_d_phi, d_l_d_phi)
        d3_l_d_phi3 = np.matmul(self.d_d_phi, d2_l_d_phi2)

        # Final value for B0
        Bbar = 1 # self.spsi * np.mean(self.B0)
        self.B0_spline = self.convert_to_spline(B0) # splines are in phi

        # Final value for G0
        G0 = self.sG*abs_G0
        abs_G0_over_B0 = np.abs(G0/B0)

        ## Evaluation of d ##
        if not self.omn:
            self.d_bar = self.etabar / curvature
        else:
            d = np.zeros(self.nphi)
            if isinstance(self.d_in, dict):
                d += self.evaluate_input_on_grid(self.d_in, varphi, periodic = False)
            if isinstance(self.d_over_curvature_in, dict):
                dbar = self.evaluate_input_on_grid(self.d_over_curvature_in, varphi) 
                d += dbar * self.curvature
            self.d = d
        
        ## Evaluation of d_bar ##
        if not isinstance(self.d_in, dict):
            # If only d_bar as an input, simply use that
            self.d_bar = dbar
        else:
            self.d_bar = self.d / curvature

        # Define auxiliary quantity
        self.etabar_squared_over_curvature_squared = (B0 / Bbar ) * self.d_bar**2

        ## Given helicity ##
        self.helicity = self.helicity_in
        self.N_helicity = - self.helicity * self.nfp

        # # How to handle stellarator symmetry? For now assume ss
        # self.lasym = False

        self.B0 = B0; self.Bbar = Bbar
        self.G0 = G0; self.abs_G0_over_B0 = abs_G0_over_B0
        self.d_l_d_phi = d_l_d_phi; self.d2_l_d_phi2 = d2_l_d_phi2; self.d3_l_d_phi3 = d3_l_d_phi3; self.d_l_d_varphi = d_l_d_varphi
        self.d_curvature_d_varphi = np.matmul(self.d_d_varphi, self.curvature); self.d_torsion_d_varphi = np.matmul(self.d_d_varphi, self.torsion)
        self.d_curvature_d_varphi_at_0 = self.d_curvature_d_varphi[0]
        self.d_d_d_varphi_at_0 = np.matmul(self.d_d_varphi, self.d)[0]

    else: 
        # When R/Z coordinates are provided, then we need to compute the Frenet frame
        ###############################
        # CONSTRUCT R/Z & DERIVATIVES #
        ###############################
        # This will require taking derivatives. Define differentiation on the regular phi grid
        self.diff_order, self.d_d_phi = construct_periodic_diff_matrix(self.diff_finite, self.nphi, self.nfp)

        # It is also convenient to define interpolation to phi = 0, taking any shift into account
        self.interpolateTo0 = fourier_interpolation_matrix(self.nphi, -self.phi_shift*self.d_phi*self.nfp)

        # Shorthand
        phi = self.phi
        nphi = self.nphi
        d_phi = self.d_phi
        nfp = self.nfp

        # Function to evaluate R/Z and their derivatives
        def compute_R0_Z0_and_derivatives(Raxis, Zaxis, omn_complete):
            # Distinguish the construction depending on how the R/Z functions have been defined
            if Raxis["type"] == 'fourier':
                # Define arrays with axis harmonics making them all of the same length
                nfourier = np.max([len(Raxis["input_value"]["cos"]),
                                   len(Raxis["input_value"]["sin"]), 
                                   len(Zaxis["input_value"]["cos"]), 
                                   len(Zaxis["input_value"]["sin"])])
                rc = np.zeros(nfourier)
                zs = np.zeros(nfourier)
                rs = np.zeros(nfourier)
                zc = np.zeros(nfourier)

                rc[:len(Raxis["input_value"]["cos"])] = Raxis["input_value"]["cos"]
                rs[:len(Raxis["input_value"]["sin"])] = Raxis["input_value"]["sin"]
                zc[:len(Zaxis["input_value"]["cos"])] = Zaxis["input_value"]["cos"]
                zs[:len(Zaxis["input_value"]["sin"])] = Zaxis["input_value"]["sin"]

                # Check whether axis is symmetric
                self.lasym_axis = np.max(np.abs(rs))>0 or np.max(np.abs(zc))>0

                # Complete the rc harmonics if possible to force the curvature to be zero at some points
                if self.omn and omn_complete:
                    def complete_harmonics(rc, rs, zc, zs):
                        """
                        Complete harmonic content to include curvature vanishing points at the centre of the grid.
                        This is what was implemented. Should be possible to quite easily improve.
                        """
                        if not(np.max(np.abs(rs)) == 0.0):
                            raise KeyError("Unable to complete the axis with non-stellarator symmetric R.")
                        if len(rc)>6:
                            rc[6]=-(1 + rc[2] + rc[4] + (rc[2] + 4 * rc[4]) * 4 * nfp * nfp) / (1 + 36 * nfp * nfp)
                            Raxis["input_value"]["cos"][6] = rc[6]
                        elif len(rc)>4:
                            rc[4]=-(1 + rc[2] + 4 * rc[2] * nfp * nfp) / (1 + 16 * nfp * nfp)
                            Raxis["input_value"]["cos"][4] = rc[4]
                        else:
                            rc[2]=-1 / (1 + 4 * nfp * nfp)
                            Raxis["input_value"]["cos"][2] = rc[2]

                    complete_harmonics(rc, rs, zc, zs)

                # Evaluate on the grid (could use evaluate_input_on_grid, but this saves time evaluating cos/sin)      
                R0 = np.zeros(nphi)
                Z0 = np.zeros(nphi)
                R0p = np.zeros(nphi)
                Z0p = np.zeros(nphi)
                R0pp = np.zeros(nphi)
                Z0pp = np.zeros(nphi)
                R0ppp = np.zeros(nphi)
                Z0ppp = np.zeros(nphi)
                for jn in range(0, nfourier):
                    n = jn * nfp
                    sinangle = np.sin(n * phi)
                    cosangle = np.cos(n * phi)
                    R0 += rc[jn] * cosangle + rs[jn] * sinangle
                    Z0 += zc[jn] * cosangle + zs[jn] * sinangle
                    R0p += rc[jn] * (-n * sinangle) + rs[jn] * (n * cosangle)
                    Z0p += zc[jn] * (-n * sinangle) + zs[jn] * (n * cosangle)
                    R0pp += rc[jn] * (-n * n * cosangle) + rs[jn] * (-n * n * sinangle)
                    Z0pp += zc[jn] * (-n * n * cosangle) + zs[jn] * (-n * n * sinangle)
                    R0ppp += rc[jn] * (n * n * n * sinangle) + rs[jn] * (-n * n * n * cosangle)
                    Z0ppp += zc[jn] * (n * n * n * sinangle) + zs[jn] * (-n * n * n * cosangle)

                self.R0_func = self.convert_to_spline(R0)
                self.Z0_func = self.convert_to_spline(Z0)

            elif Raxis["type"] == 'grid':
                # Read inputs as values on phi grid
                R0 = Raxis["input_value"]
                Z0 = Raxis["input_value"]
                
                # Check whether axis is symmetric
                self.R0_func = self.convert_to_spline(R0)
                self.Z0_func = self.convert_to_spline(Z0)
                self.lasym_axis = np.max(np.max(self.R0_func(phi)-self.R0_func(-phi)))/np.std(R0)>1e-3 or \
                                  np.max(np.max(self.Z0_func(phi)+self.Z0_func(-phi)))/np.std(Z0)>1e-3

                # Compute derivatives using d_d_phi 
                R0p = np.matmul(self.d_d_phi, R0)
                R0pp = np.matmul(self.d_d_phi, R0p)
                R0ppp = np.matmul(self.d_d_phi, R0pp)
                Z0p = np.matmul(self.d_d_phi, Z0)
                Z0pp = np.matmul(self.d_d_phi, Z0p)
                Z0ppp = np.matmul(self.d_d_phi, Z0pp)

            elif Raxis["type"] == 'spline':
                # Evaluate the spline on the grid using the evaluate_input_on_grid function
                R0 = self.evaluate_input_on_grid(Raxis, phi, None)
                R0p = self.evaluate_input_on_grid(Raxis, phi, 1)
                R0pp = self.evaluate_input_on_grid(Raxis, phi, 2)
                R0ppp = self.evaluate_input_on_grid(Raxis, phi, 3)
                Z0 = self.evaluate_input_on_grid(Zaxis, phi, None)
                Z0p = self.evaluate_input_on_grid(Zaxis, phi, 1)
                Z0pp = self.evaluate_input_on_grid(Zaxis, phi, 2)
                Z0ppp = self.evaluate_input_on_grid(Zaxis, phi, 3)

                # Check if stellarator symmetric
                self.lasym_axis = np.max(np.max(R0-self.evaluate_input_on_grid(Raxis, -phi)))/np.std(R0)>1e-3 or \
                                  np.max(np.max(Z0-self.evaluate_input_on_grid(Zaxis, -phi)))/np.std(Z0)>1e-3
            else:
                raise ValueError('Please provide the axis as a Fourier array, array or spline dictionary.')
            
            # For now keep this in : the tests have been carried out for stellarator symmetry
            self.lasym = self.lasym_axis
            
            return R0, R0p, R0pp, R0ppp, Z0, Z0p, Z0pp, Z0ppp 
        # Compute R/Z and derivatives
        R0, R0p, R0pp, R0ppp, Z0, Z0p, Z0pp, Z0ppp = compute_R0_Z0_and_derivatives(self.Raxis, self.Zaxis, omn_complete)

        # Compute d_l_d_phi and derivatives
        d_l_d_phi = np.sqrt(R0 * R0 + R0p * R0p + Z0p * Z0p)
        d2_l_d_phi2 = (R0 * R0p + R0p * R0pp + Z0p * Z0pp) / d_l_d_phi
        d3_l_d_phi3 = (R0p * R0p + R0pp * R0pp + Z0pp * Z0pp + R0 * R0pp + R0p * R0ppp + Z0p * Z0ppp - d2_l_d_phi2 * d2_l_d_phi2) / d_l_d_phi

        # Determine G0 and d_l_d_varphi using  
        # B0_over_abs_G0 = nphi / np.sum(d_l_d_phi)
        # abs_G0_over_B0 = 1 / B0_over_abs_G0
        # self.d_l_d_varphi = abs_G0_over_B0
        # G0 = self.sG * abs_G0_over_B0 * self.B0

        ## Find immediate properties of the curve ##
        # Total axis length (taking the nfp into account)
        axis_length = np.trapz(d_l_d_phi, phi) * nfp
        # Mean major radius
        mean_of_R = np.trapz(R0 * d_l_d_phi, phi) * nfp / axis_length
        # Mean vertical displacement
        mean_of_Z = np.trapz(Z0 * d_l_d_phi, phi) * nfp / axis_length

        # standard_deviation_of_R = np.sqrt(np.sum((R0 - mean_of_R) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)
        # standard_deviation_of_Z = np.sqrt(np.sum((Z0 - mean_of_Z) ** 2 * d_l_d_phi) * d_phi * nfp / axis_length)

        ###############################
        # COMPUTE FRENET SERRET FRAME #
        ###############################
        # For these next arrays, the first dimension is the phi grid, and the 2nd dimension represents the components
        # in the cylindrical basis (R, phi, Z).
        d_r_d_phi_cylindrical = np.array([R0p, R0, Z0p]).transpose()
        d2_r_d_phi2_cylindrical = np.array([R0pp - R0, 2 * R0p, Z0pp]).transpose()
        d3_r_d_phi3_cylindrical = np.array([R0ppp - 3 * R0p, 3 * R0pp - R0, Z0ppp]).transpose()

        ## Construct tangent ##
        # Allocate arrays
        tangent_cylindrical = np.zeros((nphi, 3))
        d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            # Employ definition of tangent
            tangent_cylindrical[:,j] = d_r_d_phi_cylindrical[:,j] / d_l_d_phi
            # Construct derivative for later
            d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                            + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)
        
        self.tangent_cylindrical = tangent_cylindrical
            
        ## Construct curvature ##
        # Employ definition of curvature
        curvature = np.sqrt(d_tangent_d_l_cylindrical[:,0] * d_tangent_d_l_cylindrical[:,0] + \
                            d_tangent_d_l_cylindrical[:,1] * d_tangent_d_l_cylindrical[:,1] + \
                            d_tangent_d_l_cylindrical[:,2] * d_tangent_d_l_cylindrical[:,2])

        # rms_curvature = np.sqrt((np.sum(curvature * curvature * d_l_d_phi) * d_phi * nfp) / axis_length)

        ## Construct normal ##
        # Allocate array
        normal_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            # Definition of normal
            normal_cylindrical[:,j] = d_tangent_d_l_cylindrical[:,j] / curvature

        ## Construct binormal as b = t x n ##
        # Allocate array
        binormal_cylindrical = np.zeros((nphi, 3))
        # Compute cross product
        binormal_cylindrical[:,0] = tangent_cylindrical[:,1] * normal_cylindrical[:,2] - tangent_cylindrical[:,2] * normal_cylindrical[:,1]
        binormal_cylindrical[:,1] = tangent_cylindrical[:,2] * normal_cylindrical[:,0] - tangent_cylindrical[:,0] * normal_cylindrical[:,2]
        binormal_cylindrical[:,2] = tangent_cylindrical[:,0] * normal_cylindrical[:,1] - tangent_cylindrical[:,1] * normal_cylindrical[:,0]

        ## Define signed Frenet-Serret frame for QI fields ## 
        def define_sign_curvature():
            """
            Define the signed frame. Here only implemented and working for a single zero curvature point of odd order.
            Would have to implement more general form. Would need to put together with the completion of the axis before
            or some other way.
            """
            sign_curvature_change = np.ones((self.nphi,))
            if self.omn == True:
                nfp_phi_length = int(np.ceil(self.nphi/2))
                sign_curvature_change[nfp_phi_length:2*nfp_phi_length] = (-1)*np.ones((nfp_phi_length-1,))

            return sign_curvature_change
        
        # Sign for defining signed curvature
        sign_curvature_change = define_sign_curvature()

        # Signed curvature
        curvature = curvature * sign_curvature_change

        # Redefined normal and binormal
        for j in range(3):
            normal_cylindrical[:,j]   =   normal_cylindrical[:,j]*sign_curvature_change
            binormal_cylindrical[:,j] = binormal_cylindrical[:,j]*sign_curvature_change

        self.normal_cylindrical = normal_cylindrical 
        self.binormal_cylindrical = binormal_cylindrical

        ## Construct torsion ##
        # We use the same sign convention for torsion as the
        # Landreman-Sengupta-Plunk paper, wikipedia, and
        # mathworld.wolfram.com/Torsion.html.  This sign convention is
        # opposite to Garren & Boozer's sign convention!
        torsion_numerator = (d_r_d_phi_cylindrical[:,0] * (d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,2] - d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,1]) \
                            + d_r_d_phi_cylindrical[:,1] * (d2_r_d_phi2_cylindrical[:,2] * d3_r_d_phi3_cylindrical[:,0] - d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,2]) 
                            + d_r_d_phi_cylindrical[:,2] * (d2_r_d_phi2_cylindrical[:,0] * d3_r_d_phi3_cylindrical[:,1] - d2_r_d_phi2_cylindrical[:,1] * d3_r_d_phi3_cylindrical[:,0]))

        torsion_denominator = (d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,2] - d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,1]) ** 2 \
            + (d_r_d_phi_cylindrical[:,2] * d2_r_d_phi2_cylindrical[:,0] - d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,2]) ** 2 \
            + (d_r_d_phi_cylindrical[:,0] * d2_r_d_phi2_cylindrical[:,1] - d_r_d_phi_cylindrical[:,1] * d2_r_d_phi2_cylindrical[:,0]) ** 2

        torsion = torsion_numerator / torsion_denominator
        
        ###############################################
        # CALCULATE VARPHI and B0 (d for QI as well ) #
        ###############################################
        if self.omn == False:
            # Compute G0 for QS field: in here B0 is a scalar (this was done in __init__)
            G0 = self.sG * np.trapz(self.B0 * d_l_d_phi, self.phi) / (2*np.pi/self.nfp)
            abs_G0_over_B0 = self.sG*G0/self.B0

            # Reference Bbar definition
            self.Bbar = self.spsi * self.B0

            # Initialise varphi
            self.varphi = np.zeros(nphi)

            # Define d_l_d_phi on the unshifted grid  (recall that for QS the grid is not shifted)
            d_l_d_phi_spline = self.convert_to_spline(d_l_d_phi)
            d_l_d_phi_from_zero = d_l_d_phi_spline(phi + d_phi / 4.0)
            for j in range(1, nphi):
                # To get toroidal angle on the full mesh, we need d_l_d_phi on the half mesh.
                self.varphi[j] = self.varphi[j-1] + \
                    (d_l_d_phi_from_zero[j-1] + d_l_d_phi_from_zero[j]) * (0.5 * d_phi * 2 * np.pi / axis_length)
                
            # Length along the axis
            self.d_l_d_varphi = self.sG * G0 / self.B0   

            # Derivative in varphi
            self.d_d_varphi = np.zeros((nphi, nphi))
            for j in range(nphi):
                self.d_d_varphi[j,:] = self.d_d_phi[j,:] * self.sG * G0 / (self.B0 * d_l_d_phi[j])

            # Define an effective dbar
            self.d_bar = self.etabar / curvature

            # Define auxiliary quantity
            self.etabar_squared_over_curvature_squared = (self.B0  / self.Bbar) * self.d_bar**2

        else:
            # Picard iteration is used to find varphi and G0
            # Initialise nu = varphi - phi, which must be periodic
            nu = np.zeros((nphi,))
            num_iter_max = 20   # Max number of iterations
            for j in range(num_iter_max):
                # Nu from previous iteration for reference
                last_nu = nu
                # Update varphi
                varphi = phi + nu
                # In here B0_in is assumed to be provided in varphi
                B0 = self.evaluate_input_on_grid(self.B0_in, varphi) 
                # Construct G0 (everything is in the equally spaced phi grid)
                abs_G0 = np.trapz(B0 * d_l_d_phi, phi) / (2*np.pi/self.nfp)
                # Update nu by inverting d varphi / d phi - 1 = d nu / d phi and 
                # d l/d phi = (abs_G0/B0) d varphi/d phi
                rhs = -1 + d_l_d_phi * B0 / abs_G0
                nu = np.linalg.solve(self.d_d_phi+self.interpolateTo0, rhs) # Include interpolateTo0 to make matrix invertible, nu = 0 at origin
                # Relative error in nu
                norm_change = np.sqrt(sum((nu-last_nu)**2)/nphi)
                logger.debug("  Iteration {}: |change to nu| = {}".format(j, norm_change))
                # Exit iteration if nu converged
                if norm_change < 1e-17:
                    break
            # Final value for varphi
            varphi = phi + nu
            self.varphi = varphi

            # Final value for B0
            B0 = self.evaluate_input_on_grid(self.B0_in, varphi)
            self.B0 = B0
            self.Bbar = self.spsi * np.mean(self.B0)

            # Final value for G0
            G0 = self.sG * np.trapz(self.B0 * d_l_d_phi, phi) / (2*np.pi/self.nfp)
            abs_G0_over_B0 = np.abs(G0/self.Bbar)

            # Length along the axis
            self.d_l_d_varphi = self.sG * G0 / self.B0   

            # Derivative in varphi
            self.d_d_varphi = np.zeros((nphi, nphi))
            for j in range(nphi):
                self.d_d_varphi[j,:] = self.d_d_phi[j,:] * self.sG * G0 / (self.B0[j] * d_l_d_phi[j])

            ## Evaluation of d ##
            d = np.zeros(nphi)
            if isinstance(self.d_in, dict):
                d += self.evaluate_input_on_grid(self.d_in, varphi)
            if isinstance(self.d_over_curvature_in, dict):
                dbar = self.evaluate_input_on_grid(self.d_over_curvature_in, varphi) 
                d += dbar * curvature
            # TEMPORARY : to replicate Rogerio's addition
            d -= self.k_second_order_SS * nfp * self.B0_well_depth * np.sin(nfp * varphi) / B0
            self.d = d

            ## Evaluation of d_bar ##
            if not isinstance(self.d_in, dict):
                # If only d_bar as an input, simply use that
                self.d_bar = dbar
            else:
                self.d_bar = self.d / curvature

            # Define auxiliary quantity
            self.etabar_squared_over_curvature_squared = (self.B0  / self.Bbar) * self.d_bar**2

        ## Compute helicity ##
        self._determine_helicity()
        self.N_helicity = - self.helicity * self.nfp
        self.flag_half = (np.mod(self.helicity, 1) == 0.5) # If half helicity, take into account the flip of sign
    
        # Add all results to self:
        self.G0 = G0; self.abs_G0_over_B0 = abs_G0_over_B0
        self.R0 = R0; self.R0p = R0p; self.R0pp = R0pp; self.R0ppp = R0ppp
        self.Z0 = Z0; self.Z0p = Z0p; self.Z0pp = Z0pp; self.Z0ppp = Z0ppp
        self.d_l_d_phi = d_l_d_phi; self.d2_l_d_phi2 = d2_l_d_phi2; self.d3_l_d_phi3 = d3_l_d_phi3
        self.d_r_d_phi_cylindrical = d_r_d_phi_cylindrical; self.d2_r_d_phi2_cylindrical = d2_r_d_phi2_cylindrical; self.d3_r_d_phi3_cylindrical = d3_r_d_phi3_cylindrical
        self.curvature = curvature; self.torsion = torsion; self.axis_length = axis_length
        self.d_curvature_d_varphi = np.matmul(self.d_d_varphi, curvature); self.d_torsion_d_varphi = np.matmul(self.d_d_varphi, torsion)
        self.d_curvature_d_varphi_at_0 = self.d_curvature_d_varphi[0]
        self.d_d_d_varphi_at_0 = np.matmul(self.d_d_varphi, self.d)[0]
        self.min_R0 = fourier_minimum(self.R0); self.min_Z0 = fourier_minimum(self.Z0)
        self.tangent_cylindrical = tangent_cylindrical; self.normal_cylindrical = normal_cylindrical; self.binormal_cylindrical = binormal_cylindrical

        # For now, I have defined a lasym_axis with axis info
        # # The output is not stellarator-symmetric if (1) R0s is nonzero, (2) Z0c is nonzero, or (3) sigma_initial is nonzero
        # if self.order == 'r1':
            # self.lasym = self.lasym_axis or np.abs(self.sigma0)>0
        # else:
        #     self.lasym = self.lasym_axis or np.abs(self.sigma0)>0 or np.any(np.array(self.B2c_svals) > 0.0) or np.any(np.array(self.B2c_svals) < 0.0)

        # Functions that converts a toroidal angle phi0 on the axis to the axis radial and vertical coordinates
        self.R0_func = self.convert_to_spline(R0)
        self.Z0_func = self.convert_to_spline(Z0)

        # Spline interpolants for the cylindrical components of the Frenet-Serret frame:
        self.normal_R_spline     = self.convert_to_spline(self.normal_cylindrical[:,0])
        self.normal_phi_spline   = self.convert_to_spline(self.normal_cylindrical[:,1])
        self.normal_z_spline     = self.convert_to_spline(self.normal_cylindrical[:,2])
        self.binormal_R_spline   = self.convert_to_spline(self.binormal_cylindrical[:,0])
        self.binormal_phi_spline = self.convert_to_spline(self.binormal_cylindrical[:,1])
        self.binormal_z_spline = self.convert_to_spline(self.binormal_cylindrical[:,2])
        self.tangent_R_spline = self.convert_to_spline(self.tangent_cylindrical[:,0])
        self.tangent_phi_spline = self.convert_to_spline(self.tangent_cylindrical[:,1])
        self.tangent_z_spline = self.convert_to_spline(self.tangent_cylindrical[:,2])

        # Spline interpolant for the magnetic field on-axis as a function of phi (not varphi)
        self.B0_spline = self.convert_to_spline(self.B0)

        # Spline interpolant for nu = varphi-phi
        nu = self.varphi-self.phi
        self.nu_spline = self.convert_to_spline(nu)
        self.nu_spline_of_varphi = spline(np.append(self.varphi,self.varphi[0]+2*np.pi/self.nfp), \
                                            np.append(self.varphi-self.phi,self.varphi[0]-self.phi[0]), bc_type='periodic')
