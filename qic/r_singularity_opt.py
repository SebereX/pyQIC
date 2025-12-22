#!/usr/bin/env python3

"""
Optimized functions for computing the maximum r at which the flux surfaces
become singular.
"""

import logging
import warnings
import numpy as np
from .fqs import quartic_roots

logger = logging.getLogger(__name__)

def calculate_r_singularity_opt(self, high_order=False):
    """
    Optimized calculation of r_singularity.
    """

    s = self
    
    X1c, X1s, Y1s, Y1c = s.X1c, s.X1s, s.Y1s, s.Y1c
    X20, X2s, X2c = s.X20, s.X2s, s.X2c
    Y20, Y2s, Y2c = s.Y20, s.Y2s, s.Y2c
    Z20, Z2s, Z2c = s.Z20, s.Z2s, s.Z2c

    lp = np.abs(s.G0) / s.B0
    curvature, torsion = s.curvature, s.torsion
    nphi = s.nphi

    d_X1c_d_varphi, d_X1s_d_varphi = s.d_X1c_d_varphi, s.d_X1s_d_varphi
    d_Y1s_d_varphi, d_Y1c_d_varphi = s.d_Y1s_d_varphi, s.d_Y1c_d_varphi
    d_X20_d_varphi, d_X2s_d_varphi, d_X2c_d_varphi = s.d_X20_d_varphi, s.d_X2s_d_varphi, s.d_X2c_d_varphi
    d_Y20_d_varphi, d_Y2s_d_varphi, d_Y2c_d_varphi = s.d_Y20_d_varphi, s.d_Y2s_d_varphi, s.d_Y2c_d_varphi
    d_Z20_d_varphi, d_Z2s_d_varphi, d_Z2c_d_varphi = s.d_Z20_d_varphi, s.d_Z2s_d_varphi, s.d_Z2c_d_varphi

    r_singularity_basic_vs_varphi = np.zeros(nphi)
    r_singularity_vs_varphi = np.zeros(nphi)
    r_singularity_residual_sqnorm = np.zeros(nphi)
    r_singularity_theta_vs_varphi = np.zeros(nphi)
    
    g0 = lp * (X1c * Y1s - X1s * Y1c)

    g1c = lp * (-2*X2s*Y1c + 2*X20*Y1s + 2*X2c*Y1s + 2*X1c*Y2s - X1c**2*Y1s*curvature + \
                X1s*(-2*Y20 - 2*Y2c + X1c*Y1c*curvature))
    g1s = lp * (-2*X20*Y1c + 2*X2c*Y1c + 2*X1c*Y20 - 2*X1c*Y2c + 2*X2s*Y1s - 2*X1s*Y2s + \
                X1s**2*Y1c*curvature - X1c*X1s*Y1s*curvature)
    g20 = lp * (2*X20*X1s*Y1c*curvature + 2*X1c*X1s*Y2c*curvature - 2*X1c*X20*Y1s*curvature - \
                X1c**2*Y2s*curvature + X1s**2*Y2s*curvature + X2c*(4*Y2s - (X1s*Y1c + X1c*Y1s)*curvature) + \
                X2s*(-4*Y2c + (X1c*Y1c - X1s*Y1s)*curvature) + 2*X1c*X1s*Z2c*torsion + \
                2*Y1c*Y1s*Z2c*torsion - X1c**2*Z2s*torsion + X1s**2*Z2s*torsion - Y1c**2*Z2s*torsion + \
                Y1s**2*Z2s*torsion) + Y1c*Z2s*d_X1c_d_varphi + Y1c*Z20*d_X1s_d_varphi - Y1c*Z2c*d_X1s_d_varphi +\
                X1s*Z20*d_Y1c_d_varphi + X1s*Z2c*d_Y1c_d_varphi - X1c*Z2s*d_Y1c_d_varphi - X1c*Z20*d_Y1s_d_varphi + \
                X1c*Z2c*d_Y1s_d_varphi + X1s*Z2s*d_Y1s_d_varphi - X1s*Y1c*d_Z20_d_varphi - \
                Y1s*(Z20*d_X1c_d_varphi + Z2c*d_X1c_d_varphi + Z2s*d_X1s_d_varphi - X1c*d_Z20_d_varphi)
    g2c = -lp * (-2*X2c*X1s*Y1c*curvature - 2*X1c*X1s*Y20*curvature + 2*X1c*X2c*Y1s*curvature + X1c**2*Y2s*curvature +\
                X1s**2*Y2s*curvature + X20*(-4*Y2s + X1s*Y1c*curvature + X1c*Y1s*curvature) + X2s*(4*Y20 - \
                (X1c*Y1c + X1s*Y1s)*curvature) - 2*X1c*X1s*Z20*torsion - 2*Y1c*Y1s*Z20*torsion + X1c**2*Z2s*torsion + \
                X1s**2*Z2s*torsion + Y1c**2*Z2s*torsion + Y1s**2*Z2s*torsion) + Y1c*Z2s*d_X1c_d_varphi - \
                Y1c*Z20*d_X1s_d_varphi + Y1c*Z2c*d_X1s_d_varphi + X1s*Z20*d_Y1c_d_varphi + X1s*Z2c*d_Y1c_d_varphi - \
                X1c*Z2s*d_Y1c_d_varphi + X1c*Z20*d_Y1s_d_varphi - X1c*Z2c*d_Y1s_d_varphi - X1s*Z2s*d_Y1s_d_varphi - \
                X1s*Y1c*d_Z2c_d_varphi + Y1s*(-Z20*d_X1c_d_varphi - Z2c*d_X1c_d_varphi + Z2s*d_X1s_d_varphi + \
                X1c*d_Z2c_d_varphi)
    g2s = lp * (2*X1s*X2s*Y1c*curvature - X1c**2*Y20*curvature + X1s**2*Y20*curvature + X1c**2*Y2c*curvature + \
                X1s**2*Y2c*curvature - 2*X1c*X2s*Y1s*curvature + X20*(-4*Y2c + X1c*Y1c*curvature - X1s*Y1s*curvature) + \
                X2c*(4*Y20 - (X1c*Y1c + X1s*Y1s)*curvature) - X1c**2*Z20*torsion + X1s**2*Z20*torsion - \
                Y1c**2*Z20*torsion + Y1s**2*Z20*torsion + X1c**2*Z2c*torsion + X1s**2*Z2c*torsion + \
                Y1c**2*Z2c*torsion + Y1s**2*Z2c*torsion) - Y1s*Z2s*d_X1c_d_varphi - Y1s*Z20*d_X1s_d_varphi - \
                Y1s*Z2c*d_X1s_d_varphi - X1c*Z20*d_Y1c_d_varphi + X1c*Z2c*d_Y1c_d_varphi + X1s*Z2s*d_Y1c_d_varphi + \
                X1s*Z20*d_Y1s_d_varphi + X1s*Z2c*d_Y1s_d_varphi - X1c*Z2s*d_Y1s_d_varphi + X1c*Y1s*d_Z2s_d_varphi + \
                Y1c*(Z20*d_X1c_d_varphi - Z2c*d_X1c_d_varphi + Z2s*d_X1s_d_varphi - X1s*d_Z2s_d_varphi)

    K0 = 2*(g1c*g1c + g1s*g1s)*g20 - 3*(g1c*g1c - g1s*g1s)*g2c + 8*g0*g2c*g2c + 8*g0*g2s*g2s - 6*g1c*g1s*g2s
    K2s = 2*(g1c*g1c + g1s*g1s)*g2s - 4*g1s*g1c*g20
    K2c = -2*(g1c*g1c - g1s*g1s)*g20 + 2*(g1c*g1c + g1s*g1s)*g2c
    K4s = (g1c*g1c - g1s*g1s)*g2s - 16*g0*g2c*g2s + 2*g1c*g1s*g2c
    K4c = (g1c*g1c - g1s*g1s)*g2c - 8*g0*g2c*g2c + 8*g0*g2s*g2s - 2*g1s*g1c*g2s

    coefficients = np.zeros((nphi, 5))
    coefficients[:, 4] = 4*(K4c*K4c + K4s*K4s)
    coefficients[:, 3] = 4*(K4s*K2c - K2s*K4c)
    coefficients[:, 2] = K2s*K2s + K2c*K2c - 4*K0*K4c - 4*K4c*K4c - 4*K4s*K4s
    coefficients[:, 1] = 2*K0*K2s + 2*K4c*K2s - 4*K4s*K2c
    coefficients[:, 0] = (K0 + K4c)*(K0 + K4c) - K2c*K2c

    roots_all = quartic_roots(coefficients[:, ::-1])

    for jphi, roots in enumerate(roots_all):
        real_parts = np.real(roots)
        imag_parts = np.imag(roots)

        rc = 1e+100

        for jr in range(4):
            if np.abs(imag_parts[jr]) > 1e-7:
                continue

            sin2theta = real_parts[jr]

            if np.abs(sin2theta) > 1:
                continue

            abs_cos2theta = np.sqrt(1 - sin2theta * sin2theta)
            residual_if_varpi_plus  = np.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] *   abs_cos2theta + K4s[jphi] * 2 * sin2theta *   abs_cos2theta  + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))
            residual_if_varpi_minus = np.abs(K0[jphi] + K2s[jphi] * sin2theta + K2c[jphi] * (-abs_cos2theta) + K4s[jphi] * 2 * sin2theta * (-abs_cos2theta) + K4c[jphi] * (1 - 2 * sin2theta * sin2theta))

            varpi = -1 if residual_if_varpi_plus > residual_if_varpi_minus else 1
            cos2theta = varpi * abs_cos2theta

            get_cos_from_cos2 = cos2theta > 0
            if get_cos_from_cos2:
                abs_costheta = np.sqrt(0.5*(1 + cos2theta))
            else:
                abs_sintheta = np.sqrt(0.5 * (1 - cos2theta))

            for varsigma in [-1, 1]:
                if get_cos_from_cos2:
                    costheta = varsigma * abs_costheta
                    sintheta = sin2theta / (2 * costheta)
                else:
                    sintheta = varsigma * abs_sintheta
                    costheta = sin2theta / (2 * sintheta)

                if np.abs(costheta*costheta + sintheta*sintheta - 1) > 1e-13:
                    msg = "Error! sintheta={} costheta={} jphi={} jr={} sin2theta={} cos2theta={} abs(costheta*costheta + sintheta*sintheta - 1)={}".format(sintheta, costheta, jphi, jr, sin2theta, cos2theta, np.abs(costheta*costheta + sintheta*sintheta - 1))
                    logger.error(msg)
                    raise RuntimeError(msg)

                linear_solutions = []
                denominator = 2 * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
                if np.abs(denominator) > 1e-8:
                    rr = (g1c[jphi] * sintheta - g1s[jphi] * costheta) / denominator
                    residual = g0[jphi] + rr * (g1c[jphi] * costheta + g1s[jphi] * sintheta) + rr * rr * (g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta)
                    if (rr > 0) and np.abs(residual) < 1e-5:
                        linear_solutions = [rr]

                quadratic_solutions = []
                quadratic_A = g20[jphi] + g2s[jphi] * sin2theta + g2c[jphi] * cos2theta
                quadratic_B = costheta * g1c[jphi] + sintheta * g1s[jphi]
                quadratic_C = g0[jphi]
                radicand = quadratic_B * quadratic_B - 4 * quadratic_A * quadratic_C
                if np.abs(quadratic_A) < 1e-13:
                    rr = -quadratic_C / quadratic_B
                    residual = -g1c[jphi] * sintheta + g1s[jphi] * costheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
                    if rr > 0 and np.abs(residual) < 1e-5:
                        quadratic_solutions.append(rr)
                else:
                    if radicand >= 0:
                        radical = np.sqrt(radicand)
                        for sign_quadratic in [-1, 1]:
                            rr = (-quadratic_B + sign_quadratic * radical) / (2 * quadratic_A)
                            residual = -g1c[jphi] * sintheta + g1s[jphi] * costheta + 2 * rr * (g2s[jphi] * cos2theta - g2c[jphi] * sin2theta)
                            if (rr > 0) and np.abs(residual) < 1e-5:
                                quadratic_solutions.append(rr)

                if len(quadratic_solutions) > 1:
                    quadratic_solutions = [np.min(quadratic_solutions)]

                if len(linear_solutions) > 0 and len(quadratic_solutions) > 0:
                    diff = np.abs(linear_solutions[0] - quadratic_solutions[0])

                rr = -1
                if len(quadratic_solutions) > 0:
                    rr = quadratic_solutions[0]
                elif len(linear_solutions) > 0:
                    rr = linear_solutions[0]

                if rr > 0 and rr < rc:
                    rc = rr
                    sintheta_at_rc = sintheta
                    costheta_at_rc = costheta

        r_singularity_basic_vs_varphi[jphi] = rc
        r_singularity_vs_varphi[jphi] = rc
        r_singularity_residual_sqnorm[jphi] = 0
        r_singularity_theta_vs_varphi[jphi] = 0

    self.r_singularity_vs_varphi = r_singularity_vs_varphi
    self.inv_r_singularity_vs_varphi = 1 / r_singularity_vs_varphi
    self.r_singularity_basic_vs_varphi = r_singularity_basic_vs_varphi
    self.r_singularity = np.min(r_singularity_vs_varphi)    
    self.r_singularity_theta_vs_varphi = r_singularity_theta_vs_varphi
    self.r_singularity_residual_sqnorm = r_singularity_residual_sqnorm
