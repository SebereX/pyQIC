"""
This module contains the routines for computing shape gradients of 2nd order features.
"""

import numpy as np
from .util import mu0
from BAD import bounce_int
from matplotlib import rc
import matplotlib.pyplot as plt
from .fourier_interpolation import fourier_interpolation

def compute_L_F_matrices(stel, Y_mat = None, check = False):
    """
    Compute the L and F matrices as well as the RHS defining the 2nd order differential problem. These constitute the main part of the second order construction.
    Args:
        stel: (Qic) containing the magnetic field information
        Y_mat: (dict) to be output containing the Y matrices defining the Y equations. If None, the Y matrices are not computed.
        check: (bool) to check the correct implementation of the L and F matrices
    Returns:
        L_matrix: (array) the L matrix defining the second order differential problem on (X20,Y20)
        F_matrix: (array) the F matrix multiplying (X2c, X2s)
        right_hand_side: the right hand side of the second order differential problem
    """
    ###################
    # INITIALIZE DATA #
    ###################
    # Definitions
    nphi = stel.nphi
    Bbar = stel.Bbar
    B0_over_abs_G0 = stel.B0 / np.abs(stel.G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    G0_over_Bbar = stel.G0 / stel.Bbar
    # 1st order shaping
    X1c = stel.X1c
    X1s = stel.X1s
    Y1s = stel.Y1s
    Y1c = stel.Y1c
    # Derivatives
    d_d_varphi = stel.d_d_varphi
    d_d_varphi_ext = stel.d_d_varphi_ext
    # Iota
    iota_N = stel.iotaN
    # Axis 
    curvature = stel.curvature
    torsion = stel.torsion
    # B0
    B0 = stel.B0
    sG = stel.sG
    I2_over_Bbar = stel.I2 / stel.Bbar
    # Z2
    Z20 = stel.Z20
    Z2s = stel.Z2s
    Z2c = stel.Z2c
    # Pressure functions
    beta_1c = stel.beta_1c
    beta_1s = stel.beta_1s
    beta_0 = stel.beta_0

    ###########
    # COMPUTE #
    ###########
    # Y2s dependence
    Y2s_from_X20 = -(X1s * Y1c + X1c * Y1s) / (X1c * X1c + X1s * X1s + 1e-30)
    Y2s_from_Y20 = 2 * X1c * X1s / (X1c * X1c + X1s * X1s+ 1e-30)
    Y2s_from_X2c = 1/(2 * B0 * (X1c * X1c + X1s * X1s+ 1e-30))*(2 * B0 * (X1s * Y1c - X1c * Y1s))
    Y2s_from_X2s = 1/(2 * B0 * (X1c * X1c + X1s * X1s+ 1e-30))*(2 * B0 * (X1c * Y1c + X1s * Y1s))
    Y2s_inhomogeneous = 1/(2 * B0 * (X1c * X1c + X1s * X1s+ 1e-30))*(Bbar * curvature * sG * (-X1c * X1c + X1s * X1s))


    # Y2c dependence
    Y2c_from_X20 = (-X1c * Y1c + X1s * Y1s) / (X1c * X1c + X1s * X1s+ 1e-30)
    Y2c_from_Y20 = (X1c * X1c - X1s * X1s) / (X1c * X1c + X1s * X1s+ 1e-30)
    Y2c_from_X2c = (B0 * (X1c * Y1c + X1s * Y1s)) / (B0 * (X1c * X1c + X1s * X1s+ 1e-30))
    Y2c_from_X2s = (B0 * (- X1s * Y1c + X1c * Y1s)) / (B0 * (X1c * X1c + X1s * X1s+ 1e-30))
    Y2c_inhomogeneous = (Bbar * curvature * sG * X1c * X1s) / (B0 * (X1c * X1c + X1s * X1s+ 1e-30))

    if isinstance(Y_mat, dict):
        # Save the Y matrices
        Y_mat["Y_0"] = np.concatenate((Y2c_inhomogeneous, Y2s_inhomogeneous))
        Y_mat["Y_bar"] = np.vstack((np.concatenate((np.diag(Y2c_from_X20), np.diag(Y2c_from_Y20)), axis = 1), \
                                   np.concatenate((np.diag(Y2s_from_X20), np.diag(Y2s_from_Y20)), axis = 1)))
        Y_mat["Y_hat"] = np.vstack((np.concatenate((np.diag(Y2c_from_X2c), np.diag(Y2c_from_X2s)), axis = 1), \
                                    np.concatenate((np.diag(Y2s_from_X2c), np.diag(Y2s_from_X2s)), axis = 1)))

    # Note: in the fX* and fY* quantities below, I've omitted the
    # contributions from X20 and Y20 to the d/dzeta terms. These
    # contributions are handled later when we assemble the large
    # matrix.

    # fX0 = fX0* + d_X20_d_varphi
    fX0_from_X20 = -4 * G0_over_Bbar * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
    fX0_from_Y20 = -torsion * abs_G0_over_B0 -4 * G0_over_Bbar * (Y2c_from_Y20 * Z2s - Y2s_from_Y20 * Z2c) \
        - I2_over_Bbar * (-2) * abs_G0_over_B0
    fX0_from_X2c = -4 * G0_over_Bbar * (Y2c_from_X2c * Z2s - Y2s_from_X2c * Z2c)
    fX0_from_X2s = -4 * G0_over_Bbar * (Y2c_from_X2s * Z2s - Y2s_from_X2s * Z2c)
    fX0_inhomogeneous = curvature * abs_G0_over_B0 * Z20 - 4 * G0_over_Bbar * (Y2c_inhomogeneous * Z2s - Y2s_inhomogeneous * Z2c) \
        - I2_over_Bbar * 0.5 * curvature * (X1s*Y1s + X1c*Y1c) * abs_G0_over_B0 - 0.5 * beta_0 * curvature * abs_G0_over_B0 * (X1s * Y1c - X1c * Y1s)\
        - 0.5 * abs_G0_over_B0 * (beta_1c * Y1s - beta_1s * Y1c)

    # fXs = fXs* + d_X2s_d_varphi
    fXs_from_X20 = -torsion * abs_G0_over_B0 * Y2s_from_X20 - 4 * G0_over_Bbar * (Y2c_from_X20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_X20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_X20)
    fXs_from_Y20 = -torsion * abs_G0_over_B0 * Y2s_from_Y20 - 4 * G0_over_Bbar * (-Z2c + Y2c_from_Y20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_Y20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_Y20)
    fXs_from_X2c = -torsion * abs_G0_over_B0 * Y2s_from_X2c - 4 * G0_over_Bbar * (Y2c_from_X2c * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_X2c) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_X2c) \
        - 2 * iota_N
    fXs_from_X2s = -torsion * abs_G0_over_B0 * Y2s_from_X2s - 4 * G0_over_Bbar * (Y2c_from_X2s * Z20) \
        - I2_over_Bbar * (- 2 * Y2s_from_X2s) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (-2 * Y2c_from_X2s)
    fXs_inhomogeneous =  - torsion * abs_G0_over_B0 * Y2s_inhomogeneous + curvature * abs_G0_over_B0 * Z2s \
        - 4 * G0_over_Bbar * (Y2c_inhomogeneous * Z20) \
        - I2_over_Bbar * (0.5 * curvature * (X1s * Y1c + X1c * Y1s) - 2 * Y2s_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (-2 * Y2c_inhomogeneous + 0.5 * curvature * (X1c * Y1c - X1s * Y1s)) \
        - 0.5 * abs_G0_over_B0 * (beta_1s * Y1s - beta_1c * Y1c)

    # fXc = fXc* + d_X2c_d_varphi
    fXc_from_X20 =  -torsion * abs_G0_over_B0 * Y2c_from_X20 - 4 * G0_over_Bbar * (-Y2s_from_X20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_X20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_X20)
    fXc_from_Y20 = -torsion * abs_G0_over_B0 * Y2c_from_Y20 - 4 * G0_over_Bbar * (Z2s - Y2s_from_Y20 * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_Y20) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_Y20)
    fXc_from_X2c =  -torsion * abs_G0_over_B0 * Y2c_from_X2c - 4 * G0_over_Bbar * (-Y2s_from_X2c * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_X2c) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_X2c) 
    fXc_from_X2s =  -torsion * abs_G0_over_B0 * Y2c_from_X2s - 4 * G0_over_Bbar * (-Y2s_from_X2s * Z20) \
        - I2_over_Bbar * (- 2 * Y2c_from_X2s) * abs_G0_over_B0 - beta_0 * abs_G0_over_B0 * (2 * Y2s_from_X2s) \
        + 2 * iota_N 
    fXc_inhomogeneous = - torsion * abs_G0_over_B0 * Y2c_inhomogeneous + curvature * abs_G0_over_B0 * Z2c \
        - 4 * G0_over_Bbar * (-Y2s_inhomogeneous * Z20) \
        - I2_over_Bbar * (0.5 * curvature * (X1c * Y1c - X1s * Y1s) - 2 * Y2c_inhomogeneous) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (2 * Y2s_inhomogeneous - 0.5 * curvature * (X1c * Y1s + X1s * Y1c)) \
        - 0.5 * abs_G0_over_B0 * (beta_1c * Y1s + beta_1s * Y1c)

    # fY0 = fY0* + d_Y20_d_varphi
    fY0_from_X20 = torsion * abs_G0_over_B0 - I2_over_Bbar * (2) * abs_G0_over_B0
    fY0_from_Y20 = np.zeros(nphi)
    fY0_from_X2c = 4 * G0_over_Bbar * Z2s
    fY0_from_X2s = -4 * G0_over_Bbar * Z2c
    fY0_inhomogeneous = - I2_over_Bbar * (-0.5 * curvature * (X1s * X1s + X1c * X1c)) * abs_G0_over_B0 \
        - 0.5 * abs_G0_over_B0 * (beta_1s * X1c - beta_1c * X1s)

    # fYs = fYs* + d_(Y2s-Y2s_inhomogeneous)_d_varphi
    fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * G0_over_Bbar * (Z2c)
    fYs_from_Y20 = -2 * iota_N * Y2c_from_Y20
    fYs_from_X2c = -2 * iota_N * Y2c_from_X2c + 4 * G0_over_Bbar * Z20 - 2 * beta_0 * abs_G0_over_B0 
    fYs_from_X2s = -2 * iota_N * Y2c_from_X2s + torsion * abs_G0_over_B0 - 2 * I2_over_Bbar * abs_G0_over_B0 
    fYs_inhomogeneous = np.matmul(d_d_varphi_ext,Y2s_inhomogeneous) - 2 * iota_N * Y2c_inhomogeneous \
        - I2_over_Bbar * (-curvature * X1s * X1c) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (0.5 * curvature*  (X1s * X1s - X1c * X1c)) \
        - 0.5 * abs_G0_over_B0 * (beta_1c * X1c - beta_1s * X1s)

    # fYc = fYc* + d_(Y2c-Y2c_inhomogeneous)_d_varphi
    fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * G0_over_Bbar * (-Z2s)
    fYc_from_Y20 = 2 * iota_N * Y2s_from_Y20
    fYc_from_X2c = 2 * iota_N * Y2s_from_X2c + torsion * abs_G0_over_B0 - 2 * I2_over_Bbar * abs_G0_over_B0
    fYc_from_X2s = 2 * iota_N * Y2s_from_X2s - 4 * G0_over_Bbar * Z20 + 2 * beta_0 * abs_G0_over_B0
    fYc_inhomogeneous = np.matmul(d_d_varphi_ext,Y2c_inhomogeneous) + 2 * iota_N * Y2s_inhomogeneous \
        - I2_over_Bbar * (0.5 * curvature * (X1s * X1s - X1c * X1c)) * abs_G0_over_B0 \
        - beta_0 * abs_G0_over_B0 * (curvature * X1s * X1c) \
        + 0.5 * abs_G0_over_B0 * (beta_1c * X1s + beta_1s * X1c)

    ############
    # F MATRIX #
    ############
    F_matrix = np.zeros((2 * nphi, 2 * nphi))
    # Compute indices for easier broadcasting
    j_idx = np.arange(nphi)
    j_nphi_idx = j_idx + nphi
    # Handle the terms involving d X_2c / d zeta and d Y_2c / d zeta:
    # ----------------------------------------------------------------
    # Equation 1, terms involving derivatives of X2c:
    # Contributions arise from - X1s * fXc + Y1c * fYs - Y1s * fYc.
    F_matrix[:nphi, :nphi] = (-X1s[:,None] + Y1c[:,None] * Y2s_from_X2c[None,:] - Y1s[:,None] * Y2c_from_X2c[None,:]) * d_d_varphi_ext

    # Equation 1, terms involving derivatives of X2s:
    # Contributions arise from  X1c * fXs + Y1c * fYs - Y1s * fYc.
    F_matrix[:nphi, nphi:] = (X1c[:,None] - Y1s[:,None] * Y2c_from_X2s[None,:] + Y1c[:,None] * Y2s_from_X2s[None,:]) * d_d_varphi_ext

    # Equation 2, terms involving derivatives of X2c:
    # Contributions arise from X1c * fXc + Y1s * fYs + Y1c * fYc
    F_matrix[nphi:, :nphi] = (X1c[:,None] + Y1s[:,None] * Y2s_from_X2c[None,:] + Y1c[:,None] * Y2c_from_X2c[None,:]) * d_d_varphi_ext

    # Equation 2, terms involving derivatives of X2s:
    # Contributions arise from X1s * fXs + Y1s * fYs + Y1c * fYc
    F_matrix[nphi:, nphi:] = (X1s[:,None] + Y1s[:,None] * Y2s_from_X2s[None,:] + Y1c[:,None] * Y2c_from_X2s[None,:]) * d_d_varphi_ext

    # Now handle the terms involving X2c and X2s without d/dzeta derivatives:
    # ----------------------------------------------------------------
    # For X2c
    F_matrix[j_idx,j_idx] += - X1s * fX0_from_X2c + X1c * fXs_from_X2c - X1s * fXc_from_X2c - \
        Y1s * fY0_from_X2c + Y1c * fYs_from_X2c - Y1s * fYc_from_X2c
    F_matrix[j_idx, j_nphi_idx] += - X1s * fX0_from_X2s + X1c * fXs_from_X2s - X1s * fXc_from_X2s - \
        Y1s * fY0_from_X2s + Y1c * fYs_from_X2s - Y1s * fYc_from_X2s
    # For X2s
    F_matrix[j_nphi_idx,j_idx] += - X1c * fX0_from_X2c + X1s * fXs_from_X2c + X1c * fXc_from_X2c \
        - Y1c * fY0_from_X2c + Y1s * fYs_from_X2c + Y1c * fYc_from_X2c
    F_matrix[j_nphi_idx, j_nphi_idx] += - X1c * fX0_from_X2s + X1s * fXs_from_X2s + X1c * fXc_from_X2s \
        - Y1c * fY0_from_X2s + Y1s * fYs_from_X2s + Y1c * fYc_from_X2s
    
    ############
    # L MATRIX #
    ############
    L_matrix = np.zeros((2 * nphi, 2 * nphi))
    j_idx = np.arange(nphi)
    j_nphi_idx = j_idx + nphi

    # Handle the terms involving d X_0 / d zeta and d Y_0 / d zeta:
    # ----------------------------------------------------------------
    # Equation 1, terms involving X0:
    # Contributions arise from -X1s * fX0 + Y1c * fYs - Y1s * fYc.
    L_matrix[:nphi, :nphi] = (-X1s[:,None] + Y1c[:,None] * Y2s_from_X20[None,:] - Y1s[:,None] * Y2c_from_X20[None,:]) * d_d_varphi_ext
    # Equation 1, terms involving Y0:
    # Contributions arise from  -Y1s * fY0 + Y1c * fYs - Y1s * fYc.
    L_matrix[:nphi, nphi:] = (-Y1s[:,None] - Y1s[:,None] * Y2c_from_Y20[None,:] + Y1c[:,None] * Y2s_from_Y20[None,:]) * d_d_varphi_ext
    # Equation 2, terms involving X0:
    # Contributions arise from -X1c * fX0 + Y1s * fYs + Y1c * fYc
    L_matrix[nphi:, :nphi] = (-X1c[:,None] + Y1s[:,None] * Y2s_from_X20[None,:] + Y1c[:,None] * Y2c_from_X20[None,:]) * d_d_varphi_ext
    # Equation 2, terms involving Y0:
    # Contributions arise from -Y1c * fY0 + Y1s * fYs + Y1c * fYc
    L_matrix[nphi:, nphi:] = (-Y1c[:,None] + Y1s[:,None] * Y2s_from_Y20[None,:] + Y1c[:,None] * Y2c_from_Y20[None,:]) * d_d_varphi_ext

    L_matrix[j_idx, j_idx] += - X1s * fX0_from_X20 + X1c * fXs_from_X20 - X1s * fXc_from_X20 - Y1s * fY0_from_X20 + Y1c * fYs_from_X20 - Y1s * fYc_from_X20
    L_matrix[j_idx, j_nphi_idx] += - X1s * fX0_from_Y20 + X1c * fXs_from_Y20 - X1s * fXc_from_Y20 - Y1s * fY0_from_Y20 + Y1c * fYs_from_Y20 - Y1s * fYc_from_Y20
    L_matrix[j_nphi_idx, j_idx] += - X1c * fX0_from_X20 + X1s * fXs_from_X20 + X1c * fXc_from_X20 - Y1c * fY0_from_X20 + Y1s * fYs_from_X20 + Y1c * fYc_from_X20
    L_matrix[j_nphi_idx, j_nphi_idx] += - X1c * fX0_from_Y20 + X1s * fXs_from_Y20 + X1c * fXc_from_Y20 - Y1c * fY0_from_Y20 + Y1s * fYs_from_Y20 + Y1c * fYc_from_Y20
    
    #######
    # RHS #
    #######
    right_hand_side = np.zeros(2 * nphi)
    right_hand_side[0:nphi] = -(-X1s * fX0_inhomogeneous + X1c * fXs_inhomogeneous - X1s * fXc_inhomogeneous - \
                                Y1s * fY0_inhomogeneous + Y1c * fYs_inhomogeneous - Y1s * fYc_inhomogeneous)
    right_hand_side[nphi:2 * nphi] = -(- X1c * fX0_inhomogeneous + X1s * fXs_inhomogeneous + X1c * fXc_inhomogeneous - \
                                    Y1c * fY0_inhomogeneous + Y1s * fYs_inhomogeneous + Y1c * fYc_inhomogeneous)
    
    if check:
        # Check correct implementation
        X2c = stel.X2c
        X2s = stel.X2s
        X2_vec = np.append(X2c, X2s)
        # Solve the 2nd order system using the L and F matrices
        solution = np.linalg.solve(L_matrix, right_hand_side - np.matmul(F_matrix, X2_vec))
        X20 = solution[0:nphi]
        Y20 = solution[nphi:2 * nphi]
        # Check if error is small
        print("Error implementation: ", \
            np.sqrt(np.sum((X20-stel.X20)**2 + (Y20-stel.Y20)**2)/np.sum((stel.X20)**2 + (stel.Y20)**2)))
    
    return L_matrix, F_matrix, right_hand_side

def compute_sensitivity_average_B2(stel, L_matrix, F_matrix, df_dB20 = [], df_dB2c = [], df_dB2s = [], weight = None):
    """
    Compute the sensitivity of ∫ f(B20,B2c,B2s,...) d phi = Σwi to variations in (X2c,X2s). That is the vector G such that 
    δ ∫ w f(B20,B2c,B2s,...) d phi = ∫ (G_X2c,G_X2s) δ(X2c,X2s) d phi.
    Args:
        stel: (Qic) the magnetic field information
        L_matrix: (array) the L matrix defining the second order differential problem on (X20,Y20)
        F_matrix: (array) the F matrix multiplying (X2c, X2s)
        df_dB20: (array) the sensitivity of the function f to variations in B20
        df_dB2c: (array) the sensitivity of the function f to variations in B2c
        df_dB2s: (array) the sensitivity of the function f to variations in B2s
        weight: (array) the weight function w
    Returns:
        G_X2c: (array) the sensitivity of the integral to variations in X2c
        G_X2s: (array) the sensitivity of the integral to variations in X2s
    """
    ################
    # CHECK INPUTS #
    ################
    nphi = stel.nphi
    # Check weight is given, if not 1s
    if (isinstance(weight, np.ndarray) or isinstance(weight, list)) and len(weight) == nphi:
        w = weight
    else:
        w = np.ones(nphi)
    # Check df_dB20, df_dB2c, df_dB2s are given, if not 0s
    if (isinstance(df_dB20, np.ndarray) or isinstance(df_dB20, list)) and len(df_dB20) == nphi:
        df_dB20 = np.array(df_dB20)
    else:
        df_dB20 = np.zeros(nphi)
    if (isinstance(df_dB2c, np.ndarray) or isinstance(df_dB2c, list)) and len(df_dB2c) == nphi:
        df_dB2c = np.array(df_dB2c)
    else:
        df_dB2c = np.zeros(nphi)
    if (isinstance(df_dB2s, np.ndarray) or isinstance(df_dB2s, list)) and len(df_dB2s) == nphi:
        df_dB2s = np.array(df_dB2s)
    else:
        df_dB2s = np.zeros(nphi)
    
    ####################
    # DEFINE VARIABLES #
    ####################
    nphi = stel.nphi
    B0 = stel.B0
    curvature = stel.curvature

    #################### 
    # COMPUTE GRADIENT #
    ####################
    ## G_20 ##
    # Set up the adjoint problem for the sensitivity of the metric \int B20 d phi
    from_X20_to_B20 = B0 * curvature
    rhs_B20_metric = np.append(from_X20_to_B20 * w * df_dB20, np.zeros(nphi))
    # Solve the adjoint problem
    adjoint_solution = np.linalg.solve(L_matrix.transpose(), rhs_B20_metric)
    # Compute the sensitivity vector GB20
    solution = -np.matmul(F_matrix.transpose(), adjoint_solution)
    G_B20_X2c = solution[0:nphi]
    G_B20_X2s = solution[nphi:]

    ## G_2c ##
    G_B2c_X2c = B0*curvature * w * df_dB2c

    ## G_2s ##
    G_B2s_X2s = B0*curvature * w * df_dB2s

    ## Total gradient shape ##
    G_X2c = G_B20_X2c + G_B2c_X2c
    G_X2s = G_B20_X2s + G_B2s_X2s

    return G_X2c, G_X2s

def compute_sensitivity_mag_well(stel, L_matrix = None, F_matrix = None):
    """
    Compute the sensitivity of the magnetic well to variations in (X2c,X2s).
    Args:
        stel: (Qic) the magnetic field information
        L_matrix: (array, optional) the L matrix defining the second order differential problem on (X20,Y20)
        F_matrix: (array, optional) the F matrix multiplying (X2c, X2s)
    Returns:
        G_X2c: (array) the sensitivity of the magnetic well to variations in X2c
        G_X2s: (array) the sensitivity of the magnetic well to variations in X2s
    """
    ###############
    # COMPUTE L,F #
    ###############
    if isinstance(L_matrix, list) or isinstance(L_matrix, np.ndarray):
        pass
    else:
        L_matrix, F_matrix, _ = compute_L_F_matrices(stel)

    ####################
    # COMPUTE GRADIENT #
    ####################
    # To compute the sensitivity and assuming integrals are to be computed using the simplest trapz algorithm, 
    # we need to include the weight factor in the computation of the gradient.
    def trapz_weight(stel):
        """
        Construct weight array for the discretised integral using trapz.
        Args:
            stel: (Qic) the magnetic field information
        Returns:
            weight: (array) the weight function w for quadrature using trapz
        """
        # Input array
        varphi_ext = np.append(stel.varphi, stel.varphi[0] + 2*np.pi/stel.nfp)
        ## Weight function ##
        # Initialise
        weight = np.zeros(stel.nphi)
        # Away from the edges
        weight[1:] = 0.5 * (varphi_ext[2:] - varphi_ext[:-2])
        weight[0] = 0.5 * (2*np.pi/stel.nfp + varphi_ext[1] - varphi_ext[-2])

        return weight


        return weight
    # Compute trapz weights
    weight_integ = trapz_weight(stel)  
    # Compute factors for magnetic well V''
    mhd_factors = -8*np.pi*np.abs(stel.G0/stel.Bbar)/stel.B0**3
    # Compute shape gradient (and normalise accordingly)
    G_X2c, G_X2s = compute_sensitivity_average_B2(stel, L_matrix, F_matrix, weight = weight_integ, df_dB20 = mhd_factors)
    G_X2c *= 1/weight_integ
    G_X2s *= 1/weight_integ

    return G_X2c, G_X2s

def mag_well_reshape(stel, simple = False, check = False, run = True, well = 0.0):
    """
    Reshape the 2nd order near-axis field to achieve a marginal magnetic well.
    Args:
        stel: (Qic) the magnetic field information
        simple: (bool) simple reshaping using the X2c and X2s L2 minimisation; if not simple, use the more sophisticated variational formulation
        check: (bool) check the implementation
        run: (bool) run the self-consistent second order solve using the reshaping.
        well: (float) impose a minimum size of the magnetic well. Default is 0.
    Returns:
        mod_X2c: (array) the modified X2c (if run is False)
        mod_X2s: (array) the modified X2s (if run is False)
        stel: (Qic) the reshaped magnetic field (if run is True)
    """
    ##################
    # SHAPE GRADIENT #
    ##################
    if simple:
        G_X2c, G_X2s = compute_sensitivity_mag_well(stel)
    else:
        # Check that second order object
        assert stel.order == 'r2' or stel.order == 'r3', Warning("No second order object was passed")
        # Ready to obtain Y matrices
        Y_mat_data = {}
        # Compute L and F matrices
        L_matrix, F_matrix, right_hand_side = compute_L_F_matrices(stel, Y_mat = Y_mat_data)
        # Extract Y matrices
        Y_0, Y_bar, Y_hat = Y_mat_data["Y_0"], Y_mat_data["Y_bar"], Y_mat_data["Y_hat"]
        # Compute sensitivity to magnetic well
        G_X2c, G_X2s = compute_sensitivity_mag_well(stel, L_matrix, F_matrix)

    ###################
    # IDEAL RESHAPING #
    ###################
    if simple: 
        ## Simple case where we minimise (X2c)^2 + (X2s)^2 ##
        # Compute ideal scale
        integ = G_X2c**2 + G_X2s**2
        ideal = stel.nfp*np.trapz(np.append(integ, integ[0]), 
                                  np.append(stel.varphi, 2*np.pi/stel.nfp + stel.varphi[0]))
        eps_ideal = stel.d2_volume_d_psi2/ideal if stel.d2_volume_d_psi2 > 0 else 0
        # Construct modified 2nd order shaping
        mod_X2c = stel.X2c - eps_ideal * G_X2c
        mod_X2s = stel.X2s - eps_ideal * G_X2s

    else:
        ## More elaborate case where we minimise the sum of (X20)^2 + (Y20)^2 + (X2c)^2 + (X2s)^2 + (Y2s)^2 + (Y2c)^2 ##
        if check:
            ## Check Y matrices 
            shape_0 = np.concatenate((stel.X20,stel.Y20))
            X2 = np.concatenate((stel.X2c,stel.X2s))
            Y_mat_res = np.matmul(Y_hat, X2) + np.matmul(Y_bar, shape_0) + Y_0
            Y_ideal = np.concatenate((stel.Y2c, stel.Y2s))
            assert np.abs(Y_mat_res - Y_ideal).max() < 1e-10, Warning("Error in Y solve")
        
        nphi = stel.nphi
        ## Construct M matrix ##
        # L⁻¹
        L_inv = np.linalg.inv(L_matrix)
        # L⁻¹F matrix
        L_inv_F = np.matmul(L_inv, F_matrix)
        # Y subpart of M
        Y_sub_mat = Y_hat - np.matmul(Y_bar, L_inv_F) 
        # M matrix
        M_mat = np.eye(2*stel.nphi) + np.matmul(L_inv_F.transpose(), L_inv_F) + np.matmul(Y_sub_mat.transpose(), Y_sub_mat)

        ## Construct vector Λ ##
        # L⁻¹ f
        temp_vec = np.matmul(L_inv, right_hand_side)
        # Λ vector
        temp_mat = L_inv_F - np.matmul(Y_bar.transpose(), Y_sub_mat)
        Lambda = np.matmul(temp_mat.transpose(), temp_vec) - np.matmul(Y_sub_mat.transpose(), Y_0)

        ## (Mᵀ)⁻¹G ##
        G_tot = np.concatenate((G_X2c, G_X2s))

        ## Minimal magnetic well ## (option to impose some size of the well)
        mag_well = stel.d2_volume_d_psi2 + well
        integ = G_X2c * stel.X2c + G_X2s * stel.X2s
        mag_well_min = mag_well - stel.nfp*np.trapz(np.append(integ, integ[0]), np.append(stel.varphi, 2*np.pi/stel.nfp + stel.varphi[0]))

        ## Lagrange multiplier ##
        vec_temp = np.linalg.solve(M_mat, Lambda)
        integ = G_X2c * vec_temp[:nphi] + G_X2s * vec_temp[nphi:]
        num = mag_well_min + stel.nfp*np.trapz(np.append(integ, integ[0]), np.append(stel.varphi, 2*np.pi/stel.nfp + stel.varphi[0]))
        if check:
            M_inv = np.linalg.inv(M_mat)
            integ = np.matmul(M_inv, Lambda)
            integ = G_X2c * integ[:nphi] + G_X2s * integ[nphi:]
            integ_alt = np.matmul(M_inv.transpose(), G_tot)
            integ_alt = Lambda[:nphi] * integ_alt[:nphi] + Lambda[nphi:] * integ_alt[nphi:]
            num_alt = mag_well_min + stel.nfp*np.trapz(np.append(integ, integ[0]), np.append(stel.varphi, 2*np.pi/stel.nfp + stel.varphi[0]))
            assert np.abs(num_alt - num).max() < 1e-10, Warning("numerator lagrange multiuplier error")

        ## Check M inv ##
        vec_temp = np.linalg.solve(M_mat, G_tot)
        integ = G_X2c * vec_temp[:nphi] + G_X2s * vec_temp[nphi:]
        den = stel.nfp*np.trapz(np.append(integ, integ[0]), np.append(stel.varphi, 2*np.pi/stel.nfp + stel.varphi[0]))
        if check:
            assert np.abs(vec_temp - np.matmul(M_inv.transpose(), G_tot)).max() < 1e-10, Warning("Error M inv")
            integ = np.matmul(M_inv, G_tot)
            integ = G_X2c * integ[:nphi] + G_X2s * integ[nphi:]
            den_alt = stel.nfp*np.trapz(np.append(integ, integ[0]), np.append(stel.varphi, 2*np.pi/stel.nfp + stel.varphi[0]))
            assert np.abs(den_alt - den).max() < 1e-10, Warning("denominator lagrange multiuplier error")
        eps_ideal = num/den if mag_well > 0 else 0

        ## Required shaping ##
        shape = np.linalg.solve(M_mat, Lambda - eps_ideal * G_tot)
        mod_X2c = shape[:nphi]
        mod_X2s = shape[nphi:]

        if check:
            # Check integrand
            integ = G_X2c * mod_X2c + G_X2s * mod_X2s
            V_pp_est = stel.nfp*np.trapz(np.append(integ, integ[0]), np.append(stel.varphi, 2*np.pi/stel.nfp + stel.varphi[0]))
            assert np.abs(mag_well_min + V_pp_est).max() < 1e-10, Warning("V'' problems")

    if run:
        # Prepare 2nd order reshaping inputs
        X2c_in = {"type": 'grid', "input_value": mod_X2c}
        X2s_in = {"type": 'grid', "input_value": mod_X2s}
        # Solve re-shaped configuration
        stel.X2c_in = X2c_in
        stel.X2s_in = X2s_in
        stel.calculate_r2()

        return stel
    else:
        return mod_X2c, mod_X2s

def compute_sensitivity_Shafranov_shift(stel, L_matrix = None, check_lin = False):
    """
    Compute the sensitivity of the Shafranov shift to variations in the pressure p2.
    Args:
        stel: (Qic) the magnetic field information
        L_matrix: (array, optional) the L matrix defining the second order differential problem on (X20,Y20)
        check_lin: (bool) check the linearisation on p2
    Returns:
        G_shaf_x_p2: (array) the sensitivity of the `horizontal' Shafranov shift to variations in p2
        G_shaf_y_p2: (array) the sensitivity of the `vertical' Shafranov shift to variations in p2
    """
    #####################
    # INPUT DEFINITIONS #
    #####################
    nphi = stel.nphi
    Bbar = stel.Bbar
    B0_over_abs_G0 = stel.B0 / np.abs(stel.G0)
    abs_G0_over_B0 = 1 / B0_over_abs_G0
    d_l_d_varphi = abs_G0_over_B0
    G0_over_Bbar = stel.G0 / stel.Bbar
    X1c = stel.X1c
    X1s = stel.X1s
    Y1s = stel.Y1s
    Y1c = stel.Y1c
    d_d_varphi = stel.d_d_varphi
    d_d_varphi_ext = stel.d_d_varphi_ext
    iota_N = stel.iotaN
    curvature = stel.curvature
    B0 = stel.B0
    G0 = stel.G0
    B1c = stel.B1c
    B1s = stel.B1s
    sG = stel.sG

    ################
    # CALCULATE β1 #
    ################
    mu0 = 4 * np.pi * 1e-7
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
    temp = -4 * mu0 * G0 / (Bbar * B0 * B0 * B0)
    rhs_beta_1[0:nphi] = B1c * temp           # cos component
    rhs_beta_1[nphi:2 * nphi] = B1s * temp    # sin component

    ## Solve for β1: invert the linear system ## 
    solution_beta_1 = np.linalg.solve(matrix_beta_1, rhs_beta_1)   
    # For this inversion to be well posed, we need ιN ≠ 0 so that the singularity of inverting d_d_varphi is avoided
    if np.abs(iota_N) < 1e-8:
        print('Warning: |iota_N| is very small so O(r^2) solve will be poorly conditioned. iota_N=', iota_N)
    # Separate the cos/sin components
    beta_1c_p2 = solution_beta_1[0:nphi]
    beta_1s_p2 = solution_beta_1[nphi:2 * nphi]

    ###############
    # CALCULATE f # (inhomogeneous term in 2nd order system, proportional to p2)
    ###############
    df_dp2 = np.zeros(2 * nphi)
    df_dp2[0:nphi] = (Bbar*d_l_d_varphi**2)/(2*G0)*beta_1s_p2
    df_dp2[nphi:] = (Bbar*d_l_d_varphi**2)/(2*G0)*beta_1c_p2

    if check_lin:
        # Check some of the terms against the fuller expressions used in the calculate_r2 function
        ################
        # CALCULATE β0 #
        ################
        # Part I: ∫dφ/B0**2/(2π/N) with the integral being over varphi in a period. Could do a sum in the 
        # regular phi grid using dφ = (dφ/dφ_c) dφ_c = dφ_c (dl/dφ_c)/(dl/dφ) 
        average_one_over_B0_squared_over_varphi = np.trapz(np.append(1 / (B0 * B0), 1 / (B0[0] * B0[0])), \
                                                        np.append(stel.varphi, stel.varphi[0]+2*np.pi/stel.nfp)) / (2*np.pi/stel.nfp)
        # We need to compute the form of β0 from equilibrium, as given in Eq.(A51) in [Landreman, Sengupta (2019)] but without p2
        # Part I: rhs of the equation, β0' = rhs
        rhs_beta_0_equation = 2 * mu0 * G0 / Bbar * (1/(B0 * B0) - average_one_over_B0_squared_over_varphi)
        # Integrate in varphi to obtain β0 
        if np.all(np.equal(rhs_beta_0_equation, 0)):
            # If rhs = 0, then set β0 = 0 (choice of integration constant)
            beta_0_p2 = np.zeros(nphi)
        else:
            # If rhs ≠ 0 then compute the integration. Choose β0(φ=0) = 0 as integration condition.
            beta_0_p2 = np.linalg.solve(d_d_varphi + stel.interpolateTo0, rhs_beta_0_equation)

        #####################
        # Z20 p2 dependence #
        #####################
        Z20_p2 = beta_0_p2 * Bbar * d_l_d_varphi / (2 * G0)

        # The cancellation just follows from
        cancel = - 4 * G0_over_Bbar * Z20_p2 + 2 * beta_0_p2 * abs_G0_over_B0
        assert np.abs(cancel).max() < 1e-14, "Issues with the linearisatoin of the equation. Matrices L and F should be independent of second order."

        # Test f
        ###########
        # COMPUTE #
        ###########
        # Y2s dependence
        Y2s_inhomogeneous = 1/(2 * B0 * (X1c * X1c + X1s * X1s+ 1e-30))*(Bbar * curvature * sG * (-X1c * X1c + X1s * X1s))

        # Y2c dependence
        Y2c_inhomogeneous = (Bbar * curvature * sG * X1c * X1s) / (B0 * (X1c * X1c + X1s * X1s+ 1e-30))

        # Note: in the fX* and fY* quantities below, I've omitted the
        # contributions from X20 and Y20 to the d/dzeta terms. These
        # contributions are handled later when we assemble the large
        # matrix.

        # fX0 = fX0* + d_X20_d_varphi
        fX0_inhomogeneous = curvature * abs_G0_over_B0 * Z20_p2 \
            - 0.5 * beta_0_p2 * curvature * abs_G0_over_B0 * (X1s * Y1c - X1c * Y1s)\
            - 0.5 * abs_G0_over_B0 * (beta_1c_p2 * Y1s - beta_1s_p2 * Y1c)

        # fXs = fXs* + d_X2s_d_varphi
        fXs_inhomogeneous = - 4 * G0_over_Bbar * (Y2c_inhomogeneous * Z20_p2) \
            - beta_0_p2 * abs_G0_over_B0 * (-2 * Y2c_inhomogeneous + 0.5 * curvature * (X1c * Y1c - X1s * Y1s)) \
            - 0.5 * abs_G0_over_B0 * (beta_1s_p2 * Y1s - beta_1c_p2 * Y1c)

        # fXc = fXc* + d_X2c_d_varphi
        fXc_inhomogeneous = - 4 * G0_over_Bbar * (-Y2s_inhomogeneous * Z20_p2) \
            - beta_0_p2 * abs_G0_over_B0 * (2 * Y2s_inhomogeneous - 0.5 * curvature * (X1c * Y1s + X1s * Y1c)) \
            - 0.5 * abs_G0_over_B0 * (beta_1c_p2 * Y1s + beta_1s_p2 * Y1c)

        # fY0 = fY0* + d_Y20_d_varphi
        fY0_inhomogeneous = - 0.5 * abs_G0_over_B0 * (beta_1s_p2 * X1c - beta_1c_p2 * X1s)

        # fYs = fYs* + d_(Y2s-Y2s_inhomogeneous)_d_varphi
        fYs_inhomogeneous = - beta_0_p2 * abs_G0_over_B0 * (0.5 * curvature*  (X1s * X1s - X1c * X1c)) \
            - 0.5 * abs_G0_over_B0 * (beta_1c_p2 * X1c - beta_1s_p2 * X1s)

        # fYc = fYc* + d_(Y2c-Y2c_inhomogeneous)_d_varphi
        fYc_inhomogeneous = - beta_0_p2 * abs_G0_over_B0 * (curvature * X1s * X1c) \
            + 0.5 * abs_G0_over_B0 * (beta_1c_p2 * X1s + beta_1s_p2 * X1c)
        
        df_dp2_alt = np.zeros(2 * nphi)
        df_dp2_alt[0:nphi] = -(-X1s * fX0_inhomogeneous + X1c * fXs_inhomogeneous - X1s * fXc_inhomogeneous - \
                                    Y1s * fY0_inhomogeneous + Y1c * fYs_inhomogeneous - Y1s * fYc_inhomogeneous)
        df_dp2_alt[nphi:2 * nphi] = -(- X1c * fX0_inhomogeneous + X1s * fXs_inhomogeneous + X1c * fXc_inhomogeneous - \
                                        Y1c * fY0_inhomogeneous + Y1s * fYs_inhomogeneous + Y1c * fYc_inhomogeneous)
        
        assert np.abs(df_dp2-df_dp2_alt).max() < 1e-14, "Issues with the calculation of f."

    ######################
    # SOLVE FOR GRADIENT #
    ######################
    if L_matrix is None:
        L_matrix,_,_ = compute_L_F_matrices(stel)

    ######################
    # CONSTRUCT GRADIENT #
    ######################
    G_shaf_p2 = np.linalg.solve(L_matrix, df_dp2)
    G_shaf_x_p2 = G_shaf_p2[:nphi]
    G_shaf_y_p2 = G_shaf_p2[nphi:]
    
    return G_shaf_x_p2, G_shaf_y_p2
 





