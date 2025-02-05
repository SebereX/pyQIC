"""
This module contains the calculation for the O(r^2) B for r1 construction
"""

import logging
import numpy as np
from .util import mu0
import scipy.integrate as integrate
from .optimize_nae import min_geo_qi_consistency
from .spectral_diff_matrix import spectral_diff_matrix_extended

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def compute_B2_for_r1(self, verbose = False, debug = False):
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

    if not p2 == 0:
        ##############
        # COMPUTE G2 #
        ##############
        # The expression can be found in Eq.(A50) in [Landreman, Sengupta (2019)]
        # Part I: ∫dφ/B0**2/(2π/N) with the integral being over varphi in a period. Could do a sum in the 
        # regular phi grid using dφ = (dφ/dφ_c) dφ_c = dφ_c (dl/dφ_c)/(dl/dφ) 
        average_one_over_B0_squared_over_varphi = np.trapz(np.append(1 / (B0 * B0), 1 / (B0[0] * B0[0])), \
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
    else:
        G2 = - iota * I2
        beta_0 = np.zeros(nphi)
        beta_1c = np.zeros(nphi)
        beta_1s = np.zeros(nphi)

    #######################
    # COMPUTE B_2^0 TILDE #
    #######################

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

    ########################
    # AUXILIARY QUANTITIES #
    ########################
    ## Construct the other parts (X2c/X2s in QS and B2c/B2s in QI)
    # Auxiliary definitions Eqs.(A37-40) in [Landreman, Sengupta (2019)]
    qs = np.matmul(d_d_varphi,X1s) - iota_N * X1c - Y1s * torsion * d_l_d_varphi
    qc = np.matmul(d_d_varphi,X1c) + iota_N * X1s - Y1c * torsion * d_l_d_varphi
    rs = np.matmul(d_d_varphi,Y1s) - iota_N * Y1c + X1s * torsion * d_l_d_varphi
    rc = np.matmul(d_d_varphi,Y1c) + iota_N * Y1s + X1c * torsion * d_l_d_varphi


    #######################
    # COMPUTE X20 and Y20 #
    #######################
    # From equilibrium, we need to find X20 and Y20 consistently, which involves solving a linear 
    # system of equations; namely, Eqs.(A41-42) in [Landreman, Sengupta (2019)] simultaneously. We use X20 and Y20 as unknowns
    # In this case, we need to explicitly write all the dependence of X2c, X2s, Y2c and Y2s on these following Eqs. (C9-C12)
    # in [Landreman (2020)].
    b = X1s*Y1c - X1c*Y1s

    ## Dependence of Y2s on X20 and Y20 ##
    # From Eq. (C11)
    Y2s_from_X20 = (Y1s*Y1s - Y1c*Y1c) / (2 * b)
    Y2s_from_Y20 = (X1c * Y1c - X1s * Y1s) / (2 * b)
    Y2s_inhomogeneous = (Bbar * curvature * sG * (X1s * Y1c + X1c * Y1s)/(2*B0))/ (2 * b)

    ## Dependence of Y2c on X20 and Y20 ##
    # From Eq. (C12)
    Y2c_from_X20 = Y1s * Y1c / b
    Y2c_from_Y20 = -(X1s * Y1c + X1c * Y1s) / (2 * b)
    Y2c_inhomogeneous = (Bbar * curvature * sG * (X1c * Y1c - X1s * Y1s)/(2 * B0)) / (2 * b)

    ## Dependence of X2s on X20 and Y20 ##
    # From Eq. (C9)
    X2s_from_X20 = (X1s * Y1s - X1c * Y1c) / (2 * b)
    X2s_from_Y20 = (X1c*X1c - X1s*X1s) / (2 * b)
    X2s_inhomogeneous = Bbar * sG * curvature * X1s * X1c / B0 / (2 * b)

    ## Dependence of X2c on X20 and Y20 ##
    # From Eq. (C10)
    X2c_from_X20 = (X1s * Y1c + X1c * Y1s) / (2 * b)
    X2c_from_Y20 = - X1c * X1s / b
    X2c_inhomogeneous = Bbar * sG * curvature * (X1c*X1c - X1s*X1s) / (4 * B0 * b)

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
    fXs_from_X20 = - 2 * iota_N * X2c_from_X20 -torsion * abs_G0_over_B0 * Y2s_from_X20 - 4 * G0_over_Bbar * (Y2c_from_X20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_X20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_X20)
    fXs_from_Y20 = - 2 * iota_N * X2c_from_Y20 -torsion * abs_G0_over_B0 * Y2s_from_Y20 - 4 * G0_over_Bbar * (-Z2c + Y2c_from_Y20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_Y20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_Y20)
    fXs_inhomogeneous = np.matmul(d_d_varphi_ext, X2s_inhomogeneous) - 2 * iota_N * X2c_inhomogeneous - torsion * abs_G0_over_B0 * Y2s_inhomogeneous + curvature * abs_G0_over_B0 * Z2s \
        - 4 * G0_over_Bbar * (Y2c_inhomogeneous * Z20) \
        - I2_over_Bbar * (0.5 * curvature * (X1s * Y1c + X1c * Y1s) - 2 * Y2s_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (-2 * Y2c_inhomogeneous + 0.5 * curvature * (X1c * Y1c - X1s * Y1s)) \
        - 0.5 * abs_G0_over_B0 * (beta_1s * Y1s - beta_1c * Y1c)

    # fXc
    fXc_from_X20 = 2 * iota_N * X2s_from_X20 - torsion * abs_G0_over_B0 * Y2c_from_X20 - 4 * G0_over_Bbar * (-Y2s_from_X20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_X20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_X20)
    fXc_from_Y20 = 2 * iota_N * X2s_from_Y20 - torsion * abs_G0_over_B0 * Y2c_from_Y20 - 4 * G0_over_Bbar * (Z2s - Y2s_from_Y20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_Y20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_Y20)
    fXc_inhomogeneous = np.matmul(d_d_varphi_ext, X2c_inhomogeneous) + 2 * iota_N * X2s_inhomogeneous - torsion * abs_G0_over_B0 * Y2c_inhomogeneous + curvature * abs_G0_over_B0 * Z2c \
        - 4 * G0_over_Bbar * (-Y2s_inhomogeneous * Z20) \
        - I2_over_Bbar * (0.5 * curvature * (X1c * Y1c - X1s * Y1s) - 2 * Y2c_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (2 * Y2s_inhomogeneous - 0.5 * curvature * (X1c * Y1s + X1s * Y1c)) \
        - 0.5 * abs_G0_over_B0 * (beta_1c * Y1s + beta_1s * Y1c)

    # fY0
    fY0_from_X20 = torsion * abs_G0_over_B0 - I2_over_Bbar * (2) * abs_G0_over_B0 - \
        4 * G0_over_Bbar * (X2s_from_X20 * Z2c - X2c_from_X20 * Z2s)
    fY0_from_Y20 = -4 * G0_over_Bbar * (X2s_from_Y20 * Z2c - X2c_from_Y20 * Z2s)
    fY0_inhomogeneous = -4 * G0_over_Bbar * (X2s_inhomogeneous * Z2c - X2c_inhomogeneous * Z2s) \
        - I2_over_Bbar * (-0.5 * curvature * (X1s * X1s + X1c * X1c)) * abs_G0_over_B0 \
        - 0.5 * abs_G0_over_B0 * (beta_1s * X1c - beta_1c * X1s)

    # fYs
    fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * G0_over_Bbar * (Z2c) + torsion * abs_G0_over_B0 * X2s_from_X20 \
        - 4 * G0_over_Bbar * (-X2c_from_X20 * Z20) - 2 * I2_over_Bbar * X2s_from_X20 * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * 2 * X2c_from_X20
    fYs_from_Y20 = -2 * iota_N * Y2c_from_Y20 + torsion * abs_G0_over_B0 * X2s_from_Y20 \
        - 4 * G0_over_Bbar * (-X2c_from_Y20 * Z20) - 2 * I2_over_Bbar * X2s_from_Y20 * abs_G0_over_B0  - beta_0 * abs_G0_over_B0 * 2 * X2c_from_Y20
    fYs_inhomogeneous = np.matmul(d_d_varphi_ext,Y2s_inhomogeneous) - 2 * iota_N * Y2c_inhomogeneous + torsion * abs_G0_over_B0 * X2s_inhomogeneous \
        - 4 * G0_over_Bbar * (-X2c_inhomogeneous * Z20) - I2_over_Bbar * (-curvature * X1s * X1c + 2 * X2s_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (2 * X2c_inhomogeneous + 0.5 * curvature*  (X1s * X1s - X1c * X1c)) \
        - 0.5 * abs_G0_over_B0 * (beta_1c * X1c - beta_1s * X1s)

    # fYc:
    fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * G0_over_Bbar * (-Z2s) + torsion * abs_G0_over_B0 * X2c_from_X20 \
        - 4 * G0_over_Bbar * (X2s_from_X20 * Z20) - I2_over_Bbar * 2 * X2c_from_X20 * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (-2 * X2s_from_X20)
    fYc_from_Y20 = 2 * iota_N * Y2s_from_Y20 + torsion * abs_G0_over_B0 * X2c_from_Y20 \
        - 4 * G0_over_Bbar * (X2s_from_Y20 * Z20) - I2_over_Bbar * 2 * X2c_from_Y20 * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (-2 * X2s_from_Y20)
    fYc_inhomogeneous = np.matmul(d_d_varphi_ext, Y2c_inhomogeneous) + 2 * iota_N * Y2s_inhomogeneous + torsion * abs_G0_over_B0 * X2c_inhomogeneous \
        - 4 * G0_over_Bbar * (X2s_inhomogeneous * Z20) - I2_over_Bbar * (0.5 * curvature * (X1s * X1s - X1c * X1c) + 2 * X2c_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (-2 * X2s_inhomogeneous + curvature * X1s * X1c) \
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

        # Include the dX20/dφ terms: contributions arise from -X1s * fX0 + X1c * fXs - X1s * fXc + Y1c * fYs - Y1s * fYc.
        matrix[j, 0:nphi] = (-X1s[j] + X1c[j] * X2s_from_X20 - X1s[j] * X2c_from_X20 + Y1c[j] * Y2s_from_X20 - Y1s[j] * Y2c_from_X20) * d_d_varphi_ext[j, :]
        # Include the dY20/dφ terms: contributions arise from  -Y1s * fY0 + X1c * fXs - X1s * fXc + Y1c * fYs - Y1s * fYc
        matrix[j, nphi:(2*nphi)] = (-Y1s[j] + X1c[j] * X2s_from_Y20 - X1s[j] * X2c_from_Y20 + Y1c[j] * Y2s_from_Y20 - Y1s[j] * Y2c_from_Y20) * d_d_varphi_ext[j, :]
        # Include the explicit X20 terms
        matrix[j, j] = matrix[j, j] - X1s[j] * fX0_from_X20[j] + X1c[j] * fXs_from_X20[j] - X1s[j] * fXc_from_X20[j] - \
                                      Y1s[j] * fY0_from_X20[j] + Y1c[j] * fYs_from_X20[j] - Y1s[j] * fYc_from_X20[j]
        # Include the explicit X20 terms
        matrix[j, j + nphi] = matrix[j, j + nphi] - X1s[j] * fX0_from_Y20[j] + X1c[j] * fXs_from_Y20[j] - X1s[j] * fXc_from_Y20[j] - \
                                                    Y1s[j] * fY0_from_Y20[j] + Y1c[j] * fYs_from_Y20[j] - Y1s[j] * fYc_from_Y20[j]

        ## Equation II ##
        # Include the explict dX20/dφ terms: contributions arise from -X1c * fX0 + X1s * fXs + X1c * fXc + Y1s * fYs + Y1c * fYc
        matrix[j+nphi, 0:nphi] = (-X1c[j] + X1s[j] * X2s_from_X20 + X1c[j] * X2c_from_X20 + \
                                            Y1s[j] * Y2s_from_X20 + Y1c[j] * Y2c_from_X20) * d_d_varphi_ext[j, :]
        # Include the explicit dY20/dφ terms: contributions arise from -Y1c * fY0 + X1s * fXs + X1c * fXc + Y1s * fYs + Y1c * fYc
        matrix[j+nphi, nphi:(2*nphi)] = (-Y1c[j] + X1s[j] * X2s_from_Y20 + X1c[j] * X2c_from_Y20 +\
                                                   Y1s[j] * Y2s_from_Y20 + Y1c[j] * Y2c_from_Y20) * d_d_varphi_ext[j, :]
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

    ###############################
    # COMPUTE X2c, X2s, Y2s & Y2c #
    ###############################
    X2s = X2s_from_X20 * X20 + X2s_from_Y20 * Y20 + X2s_inhomogeneous
    X2c = X2c_from_X20 * X20 + X2c_from_Y20 * Y20 + X2c_inhomogeneous
    Y2s = Y2s_from_X20 * X20 + Y2s_from_Y20 * Y20 + Y2s_inhomogeneous
    Y2c = Y2c_from_X20 * X20 + Y2c_from_Y20 * Y20 + Y2c_inhomogeneous

    # We have at this point completed solving for the tilde versions of
    # X20, X2s, X2c, Y20, Y2s, Y2c.
  
    ####################
    # VERIFY EQUATIONS #
    ####################
    if debug:
        # Check Eqs.(C7)-(C8)
        eqC7 = (X2s * X1s + X2c * X1c) / (X1s * X1s + X1c * X1c) - (Y2s * Y1s + Y2c * Y1c) / (Y1s * Y1s + Y1c * Y1c)
        eqC8 = (-X2c * X1s + X2s * X1c) / (X1s * X1s + X1c * X1c) - (-Y2c * Y1s + Y2s * Y1c) / (Y1s * Y1s + Y1c * Y1c)
        max_eqC7 = np.abs(eqC7).max()
        max_eqC8 = np.abs(eqC8).max()

        if verbose: print("C7 error:", max_eqC7)
        if verbose: print("C8 error:", max_eqC8)
        if (max_eqC7 > 1e-12): raise Warning(f"Large residual in C7 equation {max_eqC7}.")
        if (max_eqC8 > 1e-12): raise Warning("Large residual in C8 equation.")

        # Check Eqs. (A41-42) 
        fX0 = np.matmul(d_d_varphi_ext, X20) - torsion * abs_G0_over_B0 * Y20 + curvature * abs_G0_over_B0 * Z20 \
            - 4*G0/Bbar * (Y2c * Z2s - Y2s * Z2c) \
            - I2 / Bbar * (curvature/2 * (X1s * Y1s + X1c * Y1c) - 2 * Y20) * abs_G0_over_B0 \
            - 0.5 * beta_0 * curvature * abs_G0_over_B0 * (X1s * Y1c - X1c * Y1s) \
            - 0.5 * abs_G0_over_B0 * (beta_1c * Y1s - beta_1s * Y1c)

        fXs = np.matmul(d_d_varphi_ext, X2s) - 2 * iota_N * X2c - torsion * abs_G0_over_B0 * Y2s + curvature * abs_G0_over_B0 * Z2s \
            - 4*G0/Bbar * (-Y20 * Z2c + Y2c * Z20) \
            - I2/Bbar * (curvature/2 * (X1s * Y1c + X1c * Y1s) - 2 * Y2s) * abs_G0_over_B0 \
            - beta_0 * abs_G0_over_B0 * (-2 * Y2c + 0.5 * curvature * (X1c * Y1c - X1s * Y1s)) \
            - 0.5 * abs_G0_over_B0 * (beta_1s * Y1s - beta_1c * Y1c) 

        fXc = np.matmul(d_d_varphi_ext, X2c) + 2 * iota_N * X2s - torsion * abs_G0_over_B0 * Y2c + curvature * abs_G0_over_B0 * Z2c \
            - 4*G0/Bbar * (Y20 * Z2s - Y2s * Z20) \
            - I2/Bbar * (0.5 * curvature * (X1c * Y1c - X1s * Y1s) - 2 * Y2c) * abs_G0_over_B0 \
            - beta_0 * abs_G0_over_B0 * (2 * Y2s - 0.5 * curvature * (X1c * Y1s + X1s * Y1c)) \
            - 0.5 * abs_G0_over_B0 * (beta_1c * Y1s + beta_1s * Y1c)

        fY0 = np.matmul(d_d_varphi_ext, Y20) + torsion * abs_G0_over_B0 * X20 - 4*G0/Bbar * (X2s * Z2c - X2c * Z2s) \
            - I2/Bbar * (-0.5 * curvature * (X1c*X1c + X1s*X1s) + 2*X20) * abs_G0_over_B0 \
            - 0.5 * abs_G0_over_B0 * (beta_1s * X1c - beta_1c * X1s)

        fYs = np.matmul(d_d_varphi_ext, Y2s) - 2 * iota_N * Y2c + torsion * abs_G0_over_B0 * X2s \
            - 4*G0/Bbar * (X20 * Z2c - X2c * Z20) - I2/Bbar * (-curvature * X1s * X1c + 2 * X2s) * abs_G0_over_B0 \
            - beta_0 * abs_G0_over_B0 * (2 * X2c + 0.5 * curvature * (X1s*X1s - X1c*X1c)) \
            - 0.5 * abs_G0_over_B0 * (beta_1c * X1c - beta_1s * X1s)

        fYc = np.matmul(d_d_varphi_ext, Y2c) + 2 * iota_N * Y2s + torsion * abs_G0_over_B0 * X2c \
            - 4*G0/Bbar * (X2s * Z20 - X20 * Z2s) \
            - I2/Bbar * (0.5 * curvature * (X1s*X1s - X1c*X1c) + 2 * X2c) * abs_G0_over_B0 \
            - beta_0 * abs_G0_over_B0 * (-2 * X2s + curvature * X1s * X1c) \
            + 0.5 * abs_G0_over_B0 * (beta_1c * X1s + beta_1s * X1c)


        eqA41 = -X1s * fX0 + X1c * fXs - X1s * fXc - Y1s * fY0 + Y1c * fYs - Y1s * fYc

        eqA42 = -X1c * fX0 + X1s * fXs + X1c * fXc - Y1c * fY0 + Y1s * fYs + Y1c * fYc

        max_eqA41 = np.abs(eqA41).max()
        max_eqA42 = np.abs(eqA42).max()
        if verbose: print("A41 error: ", max_eqA41)
        if verbose: print("A42 error: ", max_eqA42)

        if (max_eqA41 > 1e-6): Warning("Eq (A41) residual is large !!!")
        if (max_eqA42 > 1e-6): Warning("Eq (A42) residual is large !!!")

        # Check Eqs. (A32)-(A33)
        eqA32 = -X1s * Y2s - X1c * Y2c + X1c * Y20 + X2s * Y1s + X2c * Y1c - X20 * Y1c + 0.5 * sG*Bbar/B0 * X1s * curvature

        eqA33 = -X1s * Y2c + X1c * Y2s - X1s * Y20 + X2c * Y1s - X2s * Y1c + X20 * Y1s + 0.5 * sG*Bbar/B0 * X1c * curvature

        max_eqA32 = np.abs(eqA32).max()
        max_eqA33 = np.abs(eqA33).max()
        if verbose: print("A32 error: ", max_eqA32)
        if verbose: print("A33 error: ", max_eqA33)

        if (max_eqA32 > 1e-6): Warning("Eq (A32) residual is large !!!")
        if (max_eqA33 > 1e-6): Warning("Eq (A33) residual is large !!!")

    ############################
    # COMPUTE B2c, B2s and B20 #
    ############################
    # Now that we have X20 and Y20 explicitly, as well as X2c and X2s, we can reconstruct B2 components from Eqs.(A34)-(A36)
    # in [Landreman, Sengupta (2019)]
    d_Z20_d_varphi = np.matmul(d_d_varphi, Z20)
    B20 = B0 * (curvature * X20 - B0_over_abs_G0 * d_Z20_d_varphi) + 0.75/B0 * (B1c*B1c + B1s*B1s)\
        + (B0/G0)*(G2 + iota * I2) - 0.25 * B0 * curvature * curvature * (X1c*X1c + X1s*X1s)\
        - 0.25 * B0_over_abs_G0 * B0_over_abs_G0 * (qc * qc + qs * qs + rc * rc + rs * rs) * B0
    
    B2s = 3 * B1c * B1s / (2 * B0) \
                       - B0/(2*G0*G0)*( B0 * B0 * (qc * qs + rc * rs) \
                                      + curvature * G0 * G0 * (curvature * X1c * X1s - 2 * X2s) \
                                      + 2 * B0 * G0 * sG * (np.matmul(d_d_varphi, Z2s) - 2 * Z2c * iota_N) )
    d_Z2c_d_varphi = np.matmul(d_d_varphi, Z2c)
    B2c  = (1/(4*B0)) * (3 * B1c * B1c - 3 * B1s * B1s + B0_over_abs_G0 * B0_over_abs_G0 \
                                    * ( B0 * B0 * (-qc * qc + qs * qs - rc * rc + rs * rs) \
                                    + G0 * G0 * curvature * (curvature * (-X1c * X1c + X1s * X1s) + 4 * X2c)
                                    - 4 * B0 * G0 * sG * (d_Z2c_d_varphi + 2 * Z2s * iota_N)))
    
    ## This concludes the calculation of B̃_2^(0) ##
    ################################################

    #######################
    # COMPUTE B_0^2 TILDE #
    #######################
    # Q quantity: Eq. (C27) in [Landreman (2020)]
    dX1c = np.matmul(d_d_varphi, X1c)
    dX1s = np.matmul(d_d_varphi, X1s)
    dY1c = np.matmul(d_d_varphi, Y1c)
    dY1s = np.matmul(d_d_varphi, Y1s)
    Q = 0.5 * Bbar * abs_G0_over_B0 / (G0*G0) * (G2 + (iota - iota_N) * I2) + 2 * (X2c * Y2s - X2s * Y2c) \
        + 0.5 * Bbar / G0 * (abs_G0_over_B0 * X20 * curvature - d_Z20_d_varphi) \
        + I2 / (4 * G0) * (-abs_G0_over_B0 * torsion * V1 + Y1c * dX1c - X1c * dY1c + Y1s * dX1s - X1s * dY1s) \
        + beta_0 * Bbar / (4 * G0) * (X1s * dY1c + Y1c * dX1s - X1c * dY1s - Y1s * dX1c)

    # A quantity: Eq. (C28) in [Landreman (2020)]
    A = torsion * (-Z2c * 2 * (X1s *X1c + Y1s * Y1c) - Z2s * (X1s*X1s - X1c*X1c - Y1c*Y1c + Y1s*Y1s)) \
        + Z2s / abs_G0_over_B0 * (X1c * dY1c + Y1s * dX1s - X1s * dY1s - Y1c * dX1c) \
        - Z2c / abs_G0_over_B0 * (X1c * dY1s - Y1s * dX1c + X1s * dY1c - Y1c * dX1s )

    # T quantity: Eq. (C29) in [Landreman (2020)]
    T = 2*(X2s*X2s + X2c*X2c) * (X1s * Y1c - X1c * Y1s) / (X1s*X1s + X1c*X1c)

    # q̃ quantity: Eq. (C26) in [Landreman (2020)]
    q_tilde = Q + 0.5 * A + T

    # B̂ quantity: Eq. (C25) in [Landreman (2020)]
    B_hat = -sG * B0 / Bbar * q_tilde

    # f^(2) bar: the correction to the toroidal angle defined in Eq. (C30) in [Landreman (2020)]
    varphi_temp = np.linspace(0.0, 1.0, self.nphi, endpoint = False) * 2*np.pi/self.nfp
    B_hat_sp = self.convert_to_spline(B_hat, varphi = True)
    B_hat_cent = B_hat_sp(varphi_temp)
    int_varphi_Bhat = integrate.cumtrapz(B_hat_cent, varphi_temp, initial = 0.0)
    int_Bhat = integrate.trapz(np.append(B_hat_cent, B_hat_cent[0]), np.append(varphi_temp, 2*np.pi/self.nfp))
    term_1 = int_varphi_Bhat
    term_2 = 0.5 * (1 - self.nfp*varphi_temp/np.pi) * int_Bhat
    term_3 = - 0.5 * self.nfp / np.pi * integrate.trapz(np.append(int_varphi_Bhat, int_Bhat), np.append(varphi_temp, 2*np.pi/self.nfp))
    f_2_bar = term_1 + term_2 + term_3            
    f_2_bar_sp = self.convert_to_spline(np.append(f_2_bar, int_Bhat - 0.5 * int_Bhat + term_3), \
                                        grid = np.append(varphi_temp, 2*np.pi/self.nfp), periodic = False)
    
    f_2_bar = f_2_bar_sp(self.varphi) 
    # import matplotlib.pyplot as plt
    # varphi_ext = np.linspace(0.0, 1.0, self.nphi, endpoint = True) * 2*np.pi/self.nfp
    # plt.plot(f_2_bar_sp(varphi_ext))
    # plt.plot(-f_2_bar_sp(varphi_ext)[::-1])
    # plt.show()

    # Compute B0^2 tilde
    B02 = B_hat * B0 - f_2_bar * np.matmul(d_d_varphi, B0)

    ############
    # TOTAL B2 #
    ############
    # At a = r we shall put both of these together
    B_tilde_20_tot = B20 + B02
    B_tilde_2c_tot = B2c
    B_tilde_2s_tot = B2s
    
    return B02, B20, B_tilde_2c_tot, B_tilde_2s_tot

