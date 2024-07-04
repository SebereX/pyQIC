"""
This module contains the calculation for the O(r^2) solution
"""

import logging
import numpy as np
from .util import mu0
from .optimize_nae import min_geo_qi_consistency
from .spectral_diff_matrix import spectral_diff_matrix_extended

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_r2(self):
    """
    Compute the O(r^2) quantities.
    """
    logger.debug('Calculating O(r^2) terms')

    ## Set-up some shorthand ##
    nphi = self.nphi
    Bbar = self.Bbar
    B0_over_abs_G0 = self.B0 / np.abs(self.G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    G0_over_Bbar = self.G0 / self.Bbar
    d_l_d_varphi = abs_G0_over_B0
    X1c = self.X1c
    X1s = self.X1s
    Y1s = self.Y1s
    Y1c = self.Y1c
    iota_N = self.iotaN
    iota = self.iota
    curvature = self.curvature
    torsion = self.torsion
    B0 = self.B0
    G0 = self.G0
    I2 = self.I2
    B1c = self.B1c
    B1s = self.B1s
    p2 = self.p2
    sG = self.sG
    I2_over_Bbar = self.I2 / self.Bbar
    d_d_varphi = self.d_d_varphi.copy()

    # If half helicity axes are considered, we need to introduce a differentiation matrix in the extended domain
    if np.mod(self.helicity, 1) == 0.5:
        if self.diff_finite:
            diff_order = self.diff_order
            d_d_varphi_copy = self.d_d_varphi.copy()
            d_d_varphi_copy[:diff_order,-diff_order:] = -d_d_varphi_copy[:diff_order,-diff_order:]
            d_d_varphi_copy[-diff_order:,:diff_order] = -d_d_varphi_copy[-diff_order:,:diff_order]
        else:
            d_d_varphi_copy = spectral_diff_matrix_extended(self.nphi, xmin = 0, xmax = 2*np.pi/self.nfp)
        d_d_varphi_ext = d_d_varphi_copy
        self.d_d_varphi_ext = d_d_varphi_copy
    else:
        d_d_varphi_ext = self.d_d_varphi.copy()
        self.d_d_varphi_ext = d_d_varphi_ext

    ##############
    # COMPUTE G2 #
    ##############
    # The expression can be found in Eq.(A50) in [Landreman, Sengupta (2019)]
    # Part I: ∫dφ/B0**2/(2π/N) with the integral being over varphi in a period. Could do a sum in the 
    # regular phi grid using dφ = (dφ/dφ_c) dφ_c = dφ_c (dl/dφ_c)/(dl/dφ) 
    average_one_over_B0_squared_over_varphi = np.trapz(np.append(1 / (B0 * B0),1 / (B0[0] * B0[0])), \
                                                       np.append(self.varphi, self.varphi[0]+2*np.pi/self.nfp)) / (2*np.pi/self.nfp)
    # average_one_over_B0_squared_over_varphi = np.sum(1 / (B0 * B0)) / nphi

    # Put all pieces together, Eq.(A50) in [Landreman, Sengupta (2019)] 
    G2 = -mu0 * p2 * G0 * average_one_over_B0_squared_over_varphi - iota * I2

    ################
    # CALCULATE β0 #
    ################
    # We need to compute the form of β0 from equilibrium, as given in Eq.(A51) in [Landreman, Sengupta (2019)]
    # Part I: rhs of the equation, β0' = rhs
    rhs_beta_0_equation = 2 * mu0 * p2 * G0 / Bbar * (1/(B0 * B0) - average_one_over_B0_squared_over_varphi)
    # Integrate in varphi to obtain β0 
    if np.all(np.equal(rhs_beta_0_equation, 0)):
        # If rhs = 0, then set β0 = 0 (choice of integration constant)
        beta_0 = np.zeros(nphi)
    else:
        # If rhs ≠ 0 then compute the integration. Choose β0(φ=0) = 0 as integration condition.
        beta_0 = np.linalg.solve(d_d_varphi + self.interpolateTo0, rhs_beta_0_equation)

    ################
    # CALCULATE β1 #
    ################
    # To calculate β1 we need to solve the equilibrium condition Eq.(A52) from [Landreman, Sengupta (2019)]
    # but separated into the sine/cosine parts in χ: (d/dφ + ι_Ν d/dχ)β_1 = -4 μ0 p2 G0 B1 / Bbar B0**3
    # Want to construct like a linear system and solve for (β1c,β1s): we need
    # (i) the differential operator matrix
    matrix_beta_1 = np.zeros((2 * nphi, 2 * nphi))
    # (ii) the rhs of the equation
    rhs_beta_1 = np.zeros(2 * nphi)

    ## Construct differential matrices, (i) ##
    for j in range(nphi):
        # Differential operation in φ : if the field is half-helicity, then β_1c and β_1s are half-periodic, 
        # so need extended diff matrix
        matrix_beta_1[j, 0:nphi] = d_d_varphi_ext[j, :]  # Terms involving beta_1c
        matrix_beta_1[j+nphi, nphi:(2*nphi)] = d_d_varphi_ext[j, :]  # Terms involving beta_1s

        # Differential operation in χ: did is not differential in φ, but differential in χ crosses sin/cos
        matrix_beta_1[j, j + nphi] = matrix_beta_1[j, j + nphi] + iota_N  # Terms involving beta_1s
        matrix_beta_1[j + nphi, j] = matrix_beta_1[j + nphi, j] - iota_N  # Terms involving beta_1c

    ## Construct rhs of equation, (ii) ##
    temp = -4 * mu0 * p2 * G0 / (Bbar * B0 * B0 * B0)
    rhs_beta_1[0:nphi] = B1c * temp           # cos component
    rhs_beta_1[nphi:2 * nphi] = B1s * temp    # sin component

    ## Solve for β1: invert the linear system ## 
    solution_beta_1 = np.linalg.solve(matrix_beta_1, rhs_beta_1)   
    # For this inversion to be well posed, we need ιN ≠ 0 so that the singularity of inverting d_d_varphi is avoided
    if np.abs(iota_N) < 1e-8:
        print('Warning: |iota_N| is very small so O(r^2) solve will be poorly conditioned. iota_N=', iota_N)
    # Separate the cos/sin components
    beta_1c = solution_beta_1[0:nphi]
    beta_1s = solution_beta_1[nphi:2 * nphi]

    ################
    # CALCULATE Z2 #
    ################
    # We explicitly construct Z2 components following Eqs.(A27-29) in [Landreman, Sengupta (2019)]
    ## Part I: compute auxiliary Vs ##
    # The definition of Vs are found in Eqs.(A24,30-31) in [Landreman, Sengupta (2019)]
    V1 = X1c * X1c + Y1c * Y1c + Y1s * Y1s + X1s * X1s
    V2 = 2 * (Y1s * Y1c + X1s * X1c)
    V3 = X1c * X1c + Y1c * Y1c - Y1s * Y1s - X1s * X1s

    ## Part II: construct Z2 ##
    # Use Eqs.(A27-29) in [Landreman, Sengupta (2019)]
    factor = - 1 / (8 * d_l_d_varphi)
    Z20 = beta_0 * Bbar * d_l_d_varphi / (2 * G0) + factor * np.matmul(d_d_varphi,V1)
    Z2s = factor*(np.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
    Z2c = factor*(np.matmul(d_d_varphi,V3) + 2 * iota_N * V2)

    #################################
    # CALCULATE B2c, B2s, X2c & X2s #
    #################################
    ## Read second order inputs ##
    # Depending on self.omn the input should be X2c/X2s or B2c/B2s
    if not self.omn:
        # If QS, the harmonic components of the magnetic field at second order is provided
        B2c = np.full(nphi, self.B2c_in)
        B2s = np.full(nphi, self.B2s_in)
    else:
        # If QI, the harmonic components of X at second order are provided
        X2c = self.evaluate_input_on_grid(self.X2c_in, self.varphi)
        X2s = self.evaluate_input_on_grid(self.X2s_in, self.varphi)

    ## Construct the other parts (X2c/X2s in QS and B2c/B2s in QI)
    # Auxiliary definitions Eqs.(A37-40) in [Landreman, Sengupta (2019)]
    qs = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * torsion * d_l_d_varphi
    qc = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * torsion * d_l_d_varphi
    rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * torsion * d_l_d_varphi
    rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * torsion * d_l_d_varphi

    if self.omn:
        # If QI, then X2c & X2s are inputs, and need to use Eqs.(A35-36) in [Landreman, Sengupta (2019)] to solve for B2s & B2c
        B2s = 3 * B1c * B1s / (2 * B0) \
                       - B0/(2*G0*G0)*( B0 * B0 * (qc * qs + rc * rs) \
                                      + curvature * G0 * G0 * (curvature * X1c * X1s - 2 * X2s) \
                                      + 2 * B0 * G0 * sG * (np.matmul(d_d_varphi, Z2s) - 2 * Z2c * iota_N) )
 
        B2c  = (1/(4*B0)) * (3 * B1c * B1c - 3 * B1s * B1s + B0_over_abs_G0 * B0_over_abs_G0 \
                                      * ( B0 * B0 * (-qc * qc + qs * qs - rc * rc + rs * rs) \
                                        + G0 * G0 * curvature * (curvature * (-X1c * X1c + X1s * X1s) + 4 * X2c)
                                        - 4 * B0 * G0 * sG * (np.matmul(d_d_varphi, Z2c) + 2 * Z2s * iota_N)))

    else:
        # If QS, then B2c & B2s are inputs, and need to use Eqs.(A35-36) in [Landreman, Sengupta (2019)] to solve for X2s & X2c
        X2s = B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2s) - 2*iota_N*Z2c - B0_over_abs_G0 * ( -abs_G0_over_B0*abs_G0_over_B0*B2s/B0 \
            + 3 * G0 * G0 * B1c * B1s / (2 * B0**4) - X1c * X1s / 2 * (curvature * abs_G0_over_B0)**2 - (qc * qs + rc * rs)/2)) / curvature
        
        X2c = B0_over_abs_G0 * (np.matmul(d_d_varphi,Z2c) + 2*iota_N*Z2s - B0_over_abs_G0 * (-abs_G0_over_B0*abs_G0_over_B0*B2c/B0 \
               + 3 * G0 * G0 * (B1c*B1c-B1s*B1s)/(4*B0**4) - (X1c*X1c - X1s*X1s)/4*(curvature*abs_G0_over_B0)**2 \
               - (qc * qc - qs * qs + rc * rc - rs * rs)/4)) / curvature


    #######################
    # COMPUTE X20 and Y20 #
    #######################
    # From equlibrium, we need to find X20, Y20 and B20 consistently, which involves solving a linear 
    # system of equations; namely, Eqs.(A41-42) in [Landreman, Sengupta (2019)] simultaneously. We use X20 and Y20 as unknowns

    ## Dependence of Y2c and Y2s on X20 and Y20 ##
    # The equations involve Y2c and Y2s, which we separate into their explicit dependence on X20, Y20,
    # and remaining terms from Eqs.(A32-33) in [Landreman, Sengupta (2019)]

    # Y2s:
    Y2s_from_X20 = -(X1s * Y1c + X1c * Y1s) / (X1c * X1c + X1s * X1s + 1e-30) # Included 1e-30 to avoid division by zero (could do differently using d_bar)
    Y2s_inhomogeneous = 1/(2 * B0 * (X1c * X1c + X1s * X1s+ 1e-30))*(Bbar * curvature * sG * (-X1c * X1c + X1s * X1s)\
        + 2 * B0 * (X1s * X2c * Y1c + X1c * X2s * Y1c - X1c * X2c * Y1s + X1s * X2s * Y1s))
    Y2s_from_Y20 = 2 * X1c * X1s / (X1c * X1c + X1s * X1s+ 1e-30)

    # Y2c:
    Y2c_from_X20 = (-X1c * Y1c + X1s * Y1s) / (X1c * X1c + X1s * X1s+ 1e-30) # Included 1e-30 to avoid division by zero
    Y2c_from_Y20 = (X1c * X1c - X1s * X1s) / (X1c * X1c + X1s * X1s+ 1e-30)
    Y2c_inhomogeneous = (Bbar * curvature * sG * X1c * X1s + B0 * (X1c * X2c * Y1c - X1s * X2s * Y1c\
        + X1s * X2c * Y1s + X1c * X2s * Y1s)) / (B0 * (X1c * X1c + X1s * X1s+ 1e-30))

    ## Definitions of auxiliary functions fX and fY ##
    # Expressions for f functions, as in Eqs.(A43-48) in [Landreman, Sengupta (2019)]

    # The forms here omit the contributions from X20 and Y20 to the d/dφ terms for now. 
    # That is, a X2c' as in fXc, Eq.(A45) in [Landreman, Sengupta (2019)], is ignored, and
    # will be handled later. 
    # The derivatives on X2s/X2c, also Y2c/Y2s should be taken in the extended domain to deal 
    # with half helicity cases (only the non Y20 and X20 parts here)

    # fX0
    fX0_from_X20 = -4 * G0_over_Bbar * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
    fX0_from_Y20 = -torsion * abs_G0_over_B0 -4 * G0_over_Bbar * (Y2c_from_Y20 * Z2s - Y2s_from_Y20 * Z2c) \
        - I2_over_Bbar * (-2) * abs_G0_over_B0
    fX0_inhomogeneous = curvature * abs_G0_over_B0 * Z20 - 4 * G0_over_Bbar * (Y2c_inhomogeneous * Z2s - Y2s_inhomogeneous * Z2c) \
        - I2_over_Bbar * 0.5 * curvature * (X1s*Y1s + X1c*Y1c) * abs_G0_over_B0 - 0.5 * beta_0 * curvature * abs_G0_over_B0 * (X1s * Y1c - X1c * Y1s)\
        - 0.5 * abs_G0_over_B0 * (beta_1c * Y1s - beta_1s * Y1c)

    # fXs
    fXs_from_X20 = -torsion * abs_G0_over_B0 * Y2s_from_X20 - 4 * G0_over_Bbar * (Y2c_from_X20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_X20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_X20)
    fXs_from_Y20 = -torsion * abs_G0_over_B0 * Y2s_from_Y20 - 4 * G0_over_Bbar * (-Z2c + Y2c_from_Y20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_Y20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_Y20)
    fXs_inhomogeneous = np.matmul(d_d_varphi_ext, X2s) - 2 * iota_N * X2c - torsion * abs_G0_over_B0 * Y2s_inhomogeneous + curvature * abs_G0_over_B0 * Z2s \
        - 4 * G0_over_Bbar * (Y2c_inhomogeneous * Z20) \
        - I2_over_Bbar * (0.5 * curvature * (X1s * Y1c + X1c * Y1s) - 2 * Y2s_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (-2 * Y2c_inhomogeneous + 0.5 * curvature * (X1c * Y1c - X1s * Y1s)) \
        - 0.5 * abs_G0_over_B0 * (beta_1s * Y1s - beta_1c * Y1c)

    # fXc
    fXc_from_X20 =  -torsion * abs_G0_over_B0 * Y2c_from_X20 - 4 * G0_over_Bbar * (-Y2s_from_X20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_X20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_X20)
    fXc_from_Y20 = -torsion * abs_G0_over_B0 * Y2c_from_Y20 - 4 * G0_over_Bbar * (Z2s - Y2s_from_Y20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_Y20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_Y20)
    fXc_inhomogeneous = np.matmul(d_d_varphi_ext,X2c) + 2 * iota_N * X2s - torsion * abs_G0_over_B0 * Y2c_inhomogeneous + curvature * abs_G0_over_B0 * Z2c \
        - 4 * G0_over_Bbar * (-Y2s_inhomogeneous * Z20) \
        - I2_over_Bbar * (0.5 * curvature * (X1c * Y1c - X1s * Y1s) - 2 * Y2c_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (2 * Y2s_inhomogeneous - 0.5 * curvature * (X1c * Y1s + X1s * Y1c)) \
        - 0.5 * abs_G0_over_B0 * (beta_1c * Y1s + beta_1s * Y1c)

    # fY0
    fY0_from_X20 = torsion * abs_G0_over_B0 - I2_over_Bbar * (2) * abs_G0_over_B0
    fY0_from_Y20 = np.zeros(nphi)
    fY0_inhomogeneous = -4 * G0_over_Bbar * (X2s * Z2c - X2c * Z2s) \
        - I2_over_Bbar * (-0.5 * curvature * (X1s * X1s + X1c * X1c)) * abs_G0_over_B0 \
        - 0.5 * abs_G0_over_B0 * (beta_1s * X1c - beta_1c * X1s)

    # fYs
    fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * G0_over_Bbar * (Z2c)
    fYs_from_Y20 = -2 * iota_N * Y2c_from_Y20
    fYs_inhomogeneous = np.matmul(d_d_varphi_ext,Y2s_inhomogeneous) - 2 * iota_N * Y2c_inhomogeneous + torsion * abs_G0_over_B0 * X2s \
        - 4 * G0_over_Bbar * (-X2c * Z20) - I2_over_Bbar * (-curvature * X1s * X1c + 2 * X2s) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (2 * X2c + 0.5 * curvature*  (X1s * X1s - X1c * X1c)) \
        - 0.5 * abs_G0_over_B0 * (beta_1c * X1c - beta_1s * X1s)

    # fYc:
    fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * G0_over_Bbar * (-Z2s)
    fYc_from_Y20 = 2 * iota_N * Y2s_from_Y20
    fYc_inhomogeneous = np.matmul(d_d_varphi_ext,Y2c_inhomogeneous) + 2 * iota_N * Y2s_inhomogeneous + torsion * abs_G0_over_B0 * X2c \
        - 4 * G0_over_Bbar * (X2s * Z20) - I2_over_Bbar * (0.5 * curvature * (X1s * X1s - X1c * X1c) + 2 * X2c) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (-2 * X2s + curvature * X1s * X1c) \
        + 0.5 * abs_G0_over_B0 * (beta_1c * X1s + beta_1s * X1c)

    ## Construct system of equations ##

    # We shall now place the above in a matrix form and include the derivative terms
    matrix = np.zeros((2 * nphi, 2 * nphi))
    right_hand_side = np.zeros(2 * nphi)
    for j in range(nphi):
        # The system of equations involves two: Eqs.(A41-42) in [Landreman, Sengupta (2019)] which we call I and II respectively
        
        ## Equation I ##
        # NOTE: important to note here that d_d_varphi_ext to the right is actually including derivative of what is to be multiplied
        # by the matrix to the right, but also the pieces sharing the second dimension of d_d_varphi_ext. Need to use extended domain
        # to correctly deal with the half helicity cases

        # Include the dX20/dφ terms: contributions arise from -X1s * fX0 + Y1c * fYs - Y1s * fYc.
        matrix[j, 0:nphi] = (-X1s[j] + Y1c[j] * Y2s_from_X20 - Y1s[j] * Y2c_from_X20) * d_d_varphi_ext[j, :]
        # Include the dY20/dφ terms: contributions arise from  -Y1s * fY0 + Y1c * fYs - Y1s * fYc
        matrix[j, nphi:(2*nphi)] = (-Y1s[j] - Y1s[j] * Y2c_from_Y20 + Y1c[j] * Y2s_from_Y20) * d_d_varphi_ext[j, :]
        # Include the explicit X20 terms
        matrix[j, j] = matrix[j, j] - X1s[j] * fX0_from_X20[j] + X1c[j] * fXs_from_X20[j] - X1s[j] * fXc_from_X20[j] - \
                                      Y1s[j] * fY0_from_X20[j] + Y1c[j] * fYs_from_X20[j] - Y1s[j] * fYc_from_X20[j]
        # Include the explicit X20 terms
        matrix[j, j + nphi] = matrix[j, j + nphi] - X1s[j] * fX0_from_Y20[j] + X1c[j] * fXs_from_Y20[j] - X1s[j] * fXc_from_Y20[j] - \
                                                    Y1s[j] * fY0_from_Y20[j] + Y1c[j] * fYs_from_Y20[j] - Y1s[j] * fYc_from_Y20[j]

        ## Equation II ##
        # Include the explict dX20/dφ terms: contributions arise from -X1c * fX0 + Y1s * fYs + Y1c * fYc
        matrix[j+nphi, 0:nphi] = (-X1c[j] + Y1s[j] * Y2s_from_X20 + Y1c[j] * Y2c_from_X20) * d_d_varphi_ext[j, :]
        # Include the explicit dY20/dφ terms: contributions arise from -Y1c * fY0 + Y1s * fYs + Y1c * fYc
        matrix[j+nphi, nphi:(2*nphi)] = (-Y1c[j] + Y1s[j] * Y2s_from_Y20 + Y1c[j] * Y2c_from_Y20) * d_d_varphi_ext[j, :]
        # Include the explicit X20 terms
        matrix[j + nphi, j] = matrix[j + nphi, j] - X1c[j] * fX0_from_X20[j] + X1s[j] * fXs_from_X20[j] + X1c[j] * fXc_from_X20[j] -\
                                                    Y1c[j] * fY0_from_X20[j] + Y1s[j] * fYs_from_X20[j] + Y1c[j] * fYc_from_X20[j]
        # Include the explicit Y20 terms
        matrix[j + nphi, j + nphi] = matrix[j + nphi, j + nphi] - X1c[j] * fX0_from_Y20[j] + X1s[j] * fXs_from_Y20[j] + \
                                                                  X1c[j] * fXc_from_Y20[j] - Y1c[j] * fY0_from_Y20[j] + \
                                                                  Y1s[j] * fYs_from_Y20[j] + Y1c[j] * fYc_from_Y20[j]

    # Construct the rhs (X20 and Y20 independent terms) with inhomogeneous terms
    # Equation I: from Eq.(A41) in [Landreman, Sengupta (2019)]
    right_hand_side[0:nphi] = -(-X1s * fX0_inhomogeneous + X1c * fXs_inhomogeneous - X1s * fXc_inhomogeneous - \
                                 Y1s * fY0_inhomogeneous + Y1c * fYs_inhomogeneous - Y1s * fYc_inhomogeneous)
    # Equation II: from Eq.(A42) in [Landreman, Sengupta (2019)]
    right_hand_side[nphi:2 * nphi] = -(- X1c * fX0_inhomogeneous + X1s * fXs_inhomogeneous + X1c * fXc_inhomogeneous - \
                                         Y1c * fY0_inhomogeneous + Y1s * fYs_inhomogeneous + Y1c * fYc_inhomogeneous)
    # Solve for unknowns (X20, Y20)
    solution = np.linalg.solve(matrix, right_hand_side)
    X20 = solution[0:nphi]
    Y20 = solution[nphi:2 * nphi]

    ###############
    # COMPUTE B20 #
    ###############
    # Now that we have X20 and Y20 explicitly, we can reconstruct B20 from Eq.(A34) in [Landreman, Sengupta (2019)]
    B20 = B0 * (curvature * X20 - B0_over_abs_G0 * np.matmul(d_d_varphi, Z20)) + 0.75/B0 * (B1c*B1c + B1s*B1s)\
        + (B0/G0)*(G2 + iota * I2) - 0.25 * B0 * curvature * curvature * (X1c*X1c + X1s*X1s)\
        - 0.25 * B0_over_abs_G0 * B0_over_abs_G0 * (qc * qc + qs * qs + rc * rc + rs * rs) * B0
    
    ## Compute associated features of B20 ##
    # Average B20 in varphi
    varphi_ext = np.append(self.varphi, self.varphi[0] + 2*np.pi/self.nfp)
    normalizer = 1 / np.trapz(np.append(self.d_l_d_varphi,self.d_l_d_varphi[0]), varphi_ext)
    self.B20_mean = np.trapz(np.append(B20 * self.d_l_d_varphi, B20[0] * self.d_l_d_varphi[0]), varphi_ext) * normalizer
    # Variation in B20
    self.B20_anomaly = B20 - self.B20_mean
    temp = (B20 - self.B20_mean) * (B20 - self.B20_mean) * self.d_l_d_varphi
    self.B20_residual = np.sqrt(np.trapz(np.append(temp,temp[0]), varphi_ext) * normalizer) / B0
    self.B20_variation = np.max(B20) - np.min(B20)
    
    #####################
    # COMPUTE Y2s & Y2c #
    #####################
    # Now that we have X20 and Y20 explicitly, we can reconstruct Y2s and Y2c
    Y2s = Y2s_inhomogeneous + Y2s_from_X20 * X20 + Y2s_from_Y20 * Y20
    Y2c = Y2c_inhomogeneous + Y2c_from_X20 * X20 + Y2c_from_Y20 * Y20

    ##################
    # SAVE 2nd ORDER #
    ##################
    self.beta_0  = beta_0
    self.beta_1s = beta_1s
    self.beta_1c = beta_1c
    self.G2 = G2

    self.B20 = B20
    self.B2c = B2c
    self.B2s = B2s

    self.X20 = X20
    self.X2s = X2s
    self.X2c = X2c

    self.Y20 = Y20
    self.Y2s = Y2s
    self.Y2c = Y2c

    self.Z20 = Z20
    self.Z2s = Z2s
    self.Z2c = Z2c

    # Some intermediate quantities
    self.V1 = V1
    self.V2 = V2
    self.V3 = V3

    # Additional derived quantities (X2x and Y2x live in the extended domain, in order to handle half helicity)
    self.d_X20_d_varphi = np.matmul(d_d_varphi_ext, X20)
    self.d_X2s_d_varphi = np.matmul(d_d_varphi_ext, X2s)
    self.d_X2c_d_varphi = np.matmul(d_d_varphi_ext, X2c)
    self.d_Y20_d_varphi = np.matmul(d_d_varphi_ext, Y20)
    self.d_Y2s_d_varphi = np.matmul(d_d_varphi_ext, Y2s)
    self.d_Y2c_d_varphi = np.matmul(d_d_varphi_ext, Y2c)
    self.d_Z20_d_varphi = np.matmul(d_d_varphi, Z20)
    self.d_Z2s_d_varphi = np.matmul(d_d_varphi, Z2s)
    self.d_Z2c_d_varphi = np.matmul(d_d_varphi, Z2c)
    self.d2_X1s_d_varphi2 = np.matmul(d_d_varphi, self.d_X1s_d_varphi)
    self.d2_X1c_d_varphi2 = np.matmul(d_d_varphi, self.d_X1c_d_varphi)
    self.d2_Y1c_d_varphi2 = np.matmul(d_d_varphi, self.d_Y1c_d_varphi)
    self.d2_Y1s_d_varphi2 = np.matmul(d_d_varphi, self.d_Y1s_d_varphi)

    # Save untwisted X,Y and Z
    self.X20_untwisted = self.X20
    self.Y20_untwisted = self.Y20
    self.Z20_untwisted = self.Z20

    if self.helicity == 0:
        # If no helicity, χ=θ
        self.X2s_untwisted = self.X2s
        self.X2c_untwisted = self.X2c
        self.Y2s_untwisted = self.Y2s
        self.Y2c_untwisted = self.Y2c
        self.Z2s_untwisted = self.Z2s
        self.Z2c_untwisted = self.Z2c
    else:
        # Need to unravel χ=θ-Νφ
        angle = -self.helicity * self.nfp * self.varphi
        sinangle = np.sin(2*angle)
        cosangle = np.cos(2*angle)
        self.X2s_untwisted = self.X2s *   cosangle  + self.X2c * sinangle
        self.X2c_untwisted = self.X2s * (-sinangle) + self.X2c * cosangle
        self.Y2s_untwisted = self.Y2s *   cosangle  + self.Y2c * sinangle
        self.Y2c_untwisted = self.Y2s * (-sinangle) + self.Y2c * cosangle
        self.Z2s_untwisted = self.Z2s *   cosangle  + self.Z2c * sinangle
        self.Z2c_untwisted = self.Z2s * (-sinangle) + self.Z2c * cosangle

    # Construct some useful splines
    self.B20_spline = self.convert_to_spline(self.B20, varphi = False)
    self.B2c_spline = self.convert_to_spline(self.B2c, varphi = False)
    self.B2s_spline = self.convert_to_spline(self.B2s, varphi = False)

    ######################################
    # REPRESENTATION IN QI ANGULAR BASIS #
    ######################################
    if self.omn:
        # In QI, the 2nd order field may be written as
        # B2 = B20 + B2cQI cos[2*(θ-ιφ+ν)] + B2sQI sin[2*(θ-ιφ+ν)] where ν = ιφ-α and α is self.alpha
        #    = B20 + B2c   cos[2*(θ-Nφ)]   + B2s   sin[2*(θ-Nφ)]

        # Compute helical angle = Nφ-α
        angle = - self.alpha + (-self.helicity * self.nfp * self.varphi) 
        # Apply multiple angle trigonometric formulas
        self.B2cQI = self.B2c * np.cos(2*angle) - self.B2s * np.sin(2*angle)
        self.B2sQI = self.B2s * np.cos(2*angle) + self.B2c * np.sin(2*angle)
        # Construct splines
        self.B2cQI_spline = self.convert_to_spline(self.B2cQI, varphi = False)
        self.B2sQI_spline = self.convert_to_spline(self.B2sQI, varphi = False)

        # Compute some derived quantities
        d_B0_d_varphi = np.matmul(self.d_d_varphi, self.B0)
        d_d_d_varphi = np.matmul(self.d_d_varphi_ext, self.d)
        d_2_B0_d_varphi2 = np.matmul(self.d_d_varphi, d_B0_d_varphi)

        ## Ideal stellarator symmetric QI at 2nd order ##
        # Ideal B2cQI value
        self.B2cQI_ideal = (self.d * self.B0 / d_B0_d_varphi / d_B0_d_varphi /4) * (2*d_B0_d_varphi * \
                            (self.d*d_B0_d_varphi+self.B0*d_d_d_varphi) - self.B0*self.d*d_2_B0_d_varphi2)
        self.B2cQI_ideal_spline = self.convert_to_spline(self.B2cQI_ideal, varphi = False)  # Make spline
        # Compute non-QI parts of B2: B20 should be even
        self.B20QI_deviation = self.B20_spline(self.phi) - self.B20_spline(-self.phi)   # B20(φ) = B20(-φ)
        # B2c should match the ideal value
        self.B2cQI_deviation = self.B2cQI_spline(self.phi) - self.B2cQI_ideal_spline(self.phi) # B2c(φ) = (1/4)*(B0^2 d^2 / B0')'
        # B2s should be odd
        self.B2sQI_deviation = self.B2sQI_spline(self.phi) + self.B2sQI_spline(-self.phi) # B2s(φ) =-B2s(-φ)
        # Single scalar measures of deviation from QI: max value
        self.B20QI_deviation_max = max(abs(self.B20QI_deviation))
        self.B2cQI_deviation_max = max(abs(self.B2cQI_deviation))
        self.B2sQI_deviation_max = max(abs(self.B2sQI_deviation))

    ###########################################
    # COMPUTE ADDITIONAL 2nd ORDER QUANTITIES #
    ###########################################
    # Compute Mercier stability
    self.mercier()

    # Compute ∇∇B tensor
    self.calculate_grad_grad_B_tensor()
    
    # Compute r_c
    self.calculate_r_singularity()


def construct_qi_r2(self, order = 1, verbose = 0, params = [], method = "BFGS", X2s_in = 0, fun_opt = min_geo_qi_consistency):
    """
    Construct a configuration that is QI to second order around the magnetic well minimum in stellarator symmetry, 
    for a fixed axis and B0. This is a bare minimum, and things may be easily changed.
     - order: the order of the zeroes of curvature, important for the QI constraint
     - verbose: non-zero to print the steps of the optimisation
     - params: if decided to input parameters chosen to satisfy the 2nd order qi (only used if they belong
        to the scope of inputs defined). The default is empty, and the function uses only d_bar degrees of freedom (the number
        is prescribed by the size of the input array in self, provided a minimal length)
     - method: method used for the optimisation
     - X2s_in: degree of freedom at second order in a precise QI to second order
     - fun_opt: by default the search minimises min_geo_qi_consistency, which measures how well the constriants are satisfied
        for QI at second order. This allow us to use other functions where other features may also be sought, like reasonable 
        elongation
    """
    logger.debug('Constructing QI to O(r^2)')

    # Change d_bar parameters to find a consistent O(r^2) QI configuration
    if not params:
        params = []
        num_param = 2*order+2

        # Set the parameter lavels to be modified: in this case d_over_curvature input
        if not isinstance(self.d_over_curvature_in, dict):
            raise KeyError('The default d_bar optimisation cannot be carried out, due to d_bar not being an original input')
        else:
            params += self.add_input_string(self.d_over_curvature_in, 'd_over_curvature')
    else:
        new_params = []
        for ind, label in enumerate(params):
            # Check if parameter exists
            if not(label in self.names):
                print('The label ',label, 'does not exist, and will be ignored.')
            else:
                new_params.append(label)
        params = new_params

    # Optimisation is performed
    self.order = "r1" # To leave unnecessary computations out (the way the code is written it does unnecessary things anyway)
    self.optimise_params(params, fun_opt = fun_opt, scale = 0, method = method, verbose = verbose, maxiter = 4000, maxfev = 4000, extras = order) # Order is passed to the function

    # Find the X2s and X2c necessary
    X2c, X2s = evaluate_X2c_X2s_QI(self, X2s_in)
    X2c_in = {"type": 'grid', "input_value": X2c}
    X2s_in = {"type": 'grid', "input_value": X2s}

    # Redefine the configuration (not sure why this is needed; when I try to change X2c and X2s only, and then run stel.calculate() with order 'r2', the solution for alpha is different, any clue?)
    self.__init__(omn = True, order = "r2",
                 nphi = self.nphi, phi_shift = self.phi_shift, nfp=self.nfp, diff_finite = self.diff_finite,
                 frenet = self.frenet, axis_complete = self.axis_complete,
                 Raxis = self.Raxis, Zaxis = self.Zaxis,
                 curvature = self.curvature_in, torsion = self.torsion_in, ell = self.ell_in, L = self.L_in, varphi = self.varphi, helicity = self.helicity_in,
                 B0 = self.B0_in, 
                 sG=self.sG, spsi=self.spsi,
                 d_over_curvature = self.d_over_curvature_in, d = None, alpha_tilde = self.alpha_in, omn_buffer = self.buffer_details,
                 sigma0=self.sigma0, 
                 I2=self.I2, p2=self.p2,
                 B2s = self.B2s_in, B2c = self.B2c_in,
                 X2s = X2s_in, X2c = X2c_in)
        
        # omn_method = self.omn_method, delta=self.delta, p_buffer=self.p_buffer, k_buffer=self.k_buffer, rc=self.rc,zs=self.zs, \
        #           nfp=self.nfp, B0_vals=self.B0_vals, nphi=self.nphi, omn=True, order='r2', d_over_curvature_cvals=self.d_over_curvature_cvals, \
        #           d_over_curvature_spline=self.d_over_curvature_spline, B2c_svals=X2c, B2s_cvals=X2s)

    return self.B2cQI_deviation_max


def evaluate_X2c_X2s_QI(self, X2s_in = 0):
    """
    Construct X2c and X2s (inputs for second order construction) consistent with a configuration that is QI to
    second order. This only works if the residual of min_geo_qi_consistency is 0 (or numerically very small)
    """
    # Check condition
    if min_geo_qi_consistency(self)>0.01:
        print('The QI condition at the turning points does not seem to be satisfied, and thus the second order construction will fail!')
    
    # Define necessary ingredients
    d_d_varphi = self.d_d_varphi
    B0 = self.B0
    dB0 = np.matmul(d_d_varphi, B0)
    B0_over_abs_G0 = self.B0 / np.abs(self.G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    dldphi = abs_G0_over_B0
    X1c = self.X1c
    X1s = self.X1s
    Y1s = self.Y1s
    Y1c = self.Y1c
    iota_N = self.iotaN
    torsion = self.torsion
    curvature = self.curvature

    # Compute some second order quantites
    V2 = 2 * (Y1s * Y1c + X1s * X1c)
    V3 = X1c * X1c + Y1c * Y1c - Y1s * Y1s - X1s * X1s

    factor = - B0_over_abs_G0 / 8
    Z2s = factor*(np.matmul(d_d_varphi,V2) - 2 * iota_N * V3)
    dZ2s = np.matmul(d_d_varphi,Z2s)
    Z2c = factor*(np.matmul(d_d_varphi,V3) + 2 * iota_N * V2)
    dZ2c = np.matmul(d_d_varphi,Z2c)

    qs = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * torsion * abs_G0_over_B0
    qc = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * torsion * abs_G0_over_B0
    rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * torsion * abs_G0_over_B0
    rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * torsion * abs_G0_over_B0

    Tc = B0/dldphi*(dZ2c + 2*iota_N*Z2s + (qc*qc-qs*qs+rc*rc-rs*rs)/4/dldphi)
    Ts = B0/dldphi*(dZ2s - 2*iota_N*Z2c + (qc*qs+rc*rs)/2/dldphi)

    angle = self.alpha - (-self.helicity * self.nfp * self.varphi)
    c2a1 = np.cos(2*angle)
    s2a1 = np.sin(2*angle)

    # Construct the consistent X2c and X2s
    X2c_tilde = (Tc*c2a1 + Ts*s2a1 + B0*B0*np.matmul(d_d_varphi,self.d*self.d/dB0)/4)/B0/curvature
    if X2s_in == 0 or np.size(X2s_in) == self.nphi:
        X2s_tilde = X2s_in  # Needds to be checked for the right even parity
    else:
        X2s_tilde = 0 # For simplicity for the time being

    X2c = X2c_tilde*c2a1 + X2s_tilde*s2a1
    X2s = X2c_tilde*s2a1 - X2s_tilde*c2a1

    return X2c, X2s



