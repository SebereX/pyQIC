"""
This module contains the routines for estimating measures of omnigeneity breaking such as ε_eff.
"""

from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from BAD import bounce_int

# Plotting details
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 18})
rc('text', usetex=True)


def compute_eps_eff(stel, order = 'r0', N_lam = 100, plot = False, eps_eff_per_lambda = [], ref_R_Bbar = None, check_geo_correction = False, include_Y3 = False):
    if order == 'r0':
        # Compute the leading order contribution to the ripple : constant offset
        ####################
        # FIELD QUANTITIES #
        ####################
        # φ
        varphi = stel.varphi
        varphi_cent = stel.varphi - np.pi/stel.nfp        # α buffer
        # α buffer
        alpha_cent = np.interp(0, varphi_cent, stel.alpha)
        alpha_buf = stel.alpha - stel.iota * varphi_cent - alpha_cent
        print("Central alpha / pi - 0.5: ", alpha_cent/np.pi-0.5)
        sgn_d = np.sign(np.cos(alpha_cent-0.5*np.pi))
        sin_alpha_buff = np.sin(alpha_buf)
        cos_alpha_buff = np.cos(alpha_buf)
        # d
        d = stel.d*sgn_d
        # B0
        B0 = stel.B0
        B_min = B0.min()
        B_max = B0.max()
        # G0
        G0 = stel.G0
        # X1 and Y1
        X1c = stel.X1c
        X1s = stel.X1s
        Y1c = stel.Y1c
        Y1s = stel.Y1s

        ##########
        # λ GRID #
        ##########
        # Make grid in the trapped region
        lam_array = 1/(np.linspace(0.01,1,N_lam,endpoint = False)[::-1] * (B_max-B_min) + B_min)

        #####################
        # H and I INTEGRALS #
        #####################
        # Define the H integral as a function of lambda
        def integral_H(lambda_bounce):
            integral_val = np.zeros(len(lambda_bounce))
            for jlam, lam in enumerate(lambda_bounce):
                # Integrand of H(1)
                f = 1 - lam * B0    # We are going to use the bounce integral of BAD, so put extra factor of (1-lam*B) NOTE: not best becasue normal integration would work
                num_tot = f / B0 * d * sin_alpha_buff * (4/lam/B0 - 1)
                # Compute bounce integral
                temp = bounce_int.bounce_integral_wrapper(f,num_tot,varphi,return_roots=False)
                integral_val[jlam] = temp[0]
            return integral_val

        # Define the I integral as a function of lambda
        def integral_I(lambda_bounce):
            integral_val = np.zeros(len(lambda_bounce))
            for jlam, lam in enumerate(lambda_bounce):
                # Integrand of I(0)
                f = 1 - lam * B0
                num_tot = f / B0 / B0
                temp  = bounce_int.bounce_integral_wrapper(f,num_tot,varphi,return_roots=False)
                integral_val[jlam] = temp[0]
            return integral_val

        # Compute h(1)^2/I(0)
        H_vals = integral_H(lam_array)
        I_vals = integral_I(lam_array)
        integrand_lambda = (H_vals)**2/I_vals

        ###################
        # INTEGRAL OVER λ #
        ###################
        # Compute integral over λ
        integ_lambda = integ.trapz(integrand_lambda * lam_array, lam_array)

        ############################
        # COMPUTE GEOMETRIC FACTOR #
        ############################
        # G_hat needs to be computed
        G_hat_sq = integ.trapz(X1c**2+X1s**2+Y1c**2+Y1s**2, varphi)

        ## More precise argument may be found without approximating the |nabla psi| piece
        # Define the averaging
        def G_hat_sq_int():
            # Make chi grid
            N_chi = 100
            chi_grid = np.linspace(0, 1, N_chi)*2*np.pi

            # Compute integral over phi and alpha (and normalise the alpha part by 2π)
            nablapsi_chi = lambda chi: np.sqrt((X1c*np.sin(chi) - X1s*np.cos(chi))**2 + \
                            (Y1c*np.sin(chi) - Y1s*np.cos(chi))**2)
            G_hat_num = np.sum([integ.trapz(nablapsi_chi(chi)/B0, varphi) for chi in chi_grid])/N_chi
            
            # Normalisation flux average
            norm = np.trapz(1/B0/B0, varphi)

            # Put together
            G_hat_sq_num = G_hat_num**2/norm

            # The factor of two is how we defined G_hat
            return 2*G_hat_sq_num

        # Compute 
        G_hat_sq_num = G_hat_sq_int()

        ###############
        # ε EFFECTIVE #
        ###############
        if not ref_R_Bbar:
            ref_R_Bbar = G0

        fac_eps_eff = np.pi/8/np.sqrt(2)/G_hat_sq_num * ref_R_Bbar**2
        eps_eff_3_2 = fac_eps_eff * integ_lambda
        eps_eff_per_lambda.append([lam_array,fac_eps_eff * integrand_lambda * lam_array])

        if plot:
            print("ε_eff: ", eps_eff_3_2**(2/3))
            plt.plot(1/lam_array, fac_eps_eff * integrand_lambda * lam_array, 'k')
            plt.xlabel(r"$1/\lambda$")
            plt.ylabel(r"$\epsilon_\mathrm{eff}^{3/2}(\lambda)$")
            plt.yscale('log')
            plt.tight_layout()
            plt.show()

        return eps_eff_3_2

    elif order == "r1":
        # See eps_eff notes : we are choosing the normalisation B to be 1 for simplicity
        ####################
        # FIELD QUANTITIES #
        ####################
        varphi = stel.varphi
        varphi_cent = stel.varphi - np.pi/stel.nfp
        d_d_varphi = stel.d_d_varphi
        d_d_varphi_ext = stel.d_d_varphi_ext
        # α buffer
        alpha_cent = np.interp(0, varphi_cent, stel.alpha)
        alpha_buf = stel.alpha - stel.iota * varphi_cent - alpha_cent
        print("Central alpha / pi - 0.5: ", alpha_cent/np.pi-0.5)
        sgn_d = np.sign(np.cos(alpha_cent-0.5*np.pi))
        sin_alpha_buff = np.sin(alpha_buf)
        cos_alpha_buff = np.cos(alpha_buf)
        # d
        d = stel.d*sgn_d
        # B0
        B0 = stel.B0
        B0p = np.matmul(stel.d_d_varphi, B0) + 1e-12
        B_min = B0.min()
        B_max = B0.max()
        # G0
        G0 = stel.G0
        G2 = stel.G2
        # iota
        iota = stel.iota
        iotaN = stel.iotaN
        alpha_buf_per = alpha_buf + iotaN * varphi_cent
        d_alpha_buf_d_varphi = np.matmul(d_d_varphi, alpha_buf_per) - iotaN
        d_sin_alpha_buf_d_varphi = cos_alpha_buff * d_alpha_buf_d_varphi
        d_cos_alpha_buf_d_varphi = -sin_alpha_buff * d_alpha_buf_d_varphi
        # X1 and Y1
        X1c = stel.X1c
        X1s = stel.X1s
        Y1c = stel.Y1c
        Y1s = stel.Y1s
        # B1
        B1c = stel.B1c
        B1s = stel.B1s
        B1_even = B0*d*sin_alpha_buff # cos(α) piece
        # B2 functios
        B20 = stel.B20
        B2c = stel.B2c
        B2s = stel.B2s
        # X2, Y2 and Z2
        X2c = stel.X2c
        X2s = stel.X2s
        Y2c = stel.Y2c
        Y2s = stel.Y2s
        Z2c = stel.Z2c
        Z2s = stel.Z2s


        ##########
        # λ GRID #
        ##########
        # Make grid in the trapped region
        lam_array = 1/(np.linspace(0.01,1,N_lam,endpoint = False)[::-1] * (B_max-B_min) + B_min)

        #####################
        # H and I INTEGRALS #
        #####################
        # Define the H integral as a function of lambda
        def integral_H(lambda_bounce, order = 1):
            if order == 1:
                integral_val = np.zeros(len(lambda_bounce))
                for jlam, lam in enumerate(lambda_bounce):
                    # Integrand of H(1)
                    f = 1 - lam * B0    # We are going to use the bounce integral of BAD, so put extra factor of (1-lam*B) NOTE: not best becasue normal integration would work
                    num_tot = f / B0 * d * sin_alpha_buff * (4/lam/B0 - 1)
                    # Compute bounce integral
                    temp = bounce_int.bounce_integral_wrapper(f,num_tot,varphi,return_roots=False)
                    integral_val[jlam] = temp[0]
                return integral_val
            elif order == 2:
                integral_val = []
                # Construct necessary integrand terms
                ###############
                # INTEGRAND C #
                ###############
                print("Helicity: ", stel.helicity*stel.nfp)
                print("Iota: ", stel.iota)
                print("IotaN: ", stel.iotaN)
                B2cQI = -(B2c*np.cos(2*iotaN*varphi_cent-2*stel.helicity*np.pi) + B2s*np.sin(2*iotaN*varphi_cent-2*stel.helicity*np.pi))
                delta_B2c = 4/B0*(B2cQI -0.25*np.matmul(d_d_varphi, B0*B0*d*d/B0p))
                pref = 1/B0/B0
                C_rest = pref * ((3*B0**2 - B0**3*np.matmul(d_d_varphi,B0p)/B0p**2*d**2) + \
                                 3*B0**3/B0p*d*np.matmul(d_d_varphi_ext,d)) * sin_alpha_buff**2 
                C_rest += pref * B0**3*d**2/B0p * 3*sin_alpha_buff*d_sin_alpha_buf_d_varphi
                C_integ = delta_B2c + C_rest

                for jlam, lam in enumerate(lambda_bounce):
                    # \mathcal{H} without the 1/sqrt(1-lam B)
                    H_it = (1 - lam*B0)/B0/B0 * (4/lam/B0 - 1)
                    F_H = 12/B0/B0/B0*(1-lam*B0/8-1/lam/B0)

                    # For determining bounce
                    f = 1 - lam*B0    # We are going to use the bounce integral of BAD, so put extra factor of (1-lam*B) NOTE: not best becasue normal integration would work

                    ## First integral (1): ʃ(B1_even cos α F_H ΔY1 B0') dφ / G0 = sin 2α ʃ[(B0 d sin αB) F_H (B0 d sin αB)] dφ
                    integrand_1 = 0.5 * F_H * (B0*d*sin_alpha_buff)**2
                    ## Third integral (3): ʃ(B1_even' cos α H_it ΔY1) dφ / G0 = sin 2α ʃ[(B0 d sin αB)' H_it (B0 d sin αB)/B0'] dφ
                    integrand_3 = 0.5 * (np.matmul(d_d_varphi_ext, B0*d)*sin_alpha_buff + B0*d*d_sin_alpha_buf_d_varphi) * H_it * (B0*d*sin_alpha_buff)/B0p
                    ## Second integral (2): split into two parts ʃ(H_it ΔY2 B0') dφ / G0 = -ʃ [H_it B0 (A - A cos 2α + C sin 2α)] dφ
                    integrand_2 = -0.5 * H_it * B0 * C_integ 
                    # Compute bounce integral
                    temp = bounce_int.bounce_integral_wrapper(f,[integrand_1, integrand_2, integrand_3], varphi, multiple=True,return_roots=False)
                    integral_val.append(temp)

                # Integrals : include factor of 1/2 because of going from dB to whole bounce domain
                integral_val = np.squeeze(np.array(integral_val))
                H2_a = integral_val[:,0] + integral_val[:,1] + integral_val[:,2]

                return H2_a
            
            elif order == 3:
                ## Delta Y3, sin α component : this is quite nois and the calaculations clearly struggle from noise and other limitations.
                integral_val = []
                # Definitions (need to filter some of the higher order derivatives due to numerical noise, just checking)
                def filt_data(y, window = 20):
                    filter_kernel = np.ones(window) / window
                    filtered_signal = np.convolve(y, filter_kernel, mode='same')
                    # b, a = butter(order, level)
                    # lfilter(b,a,y)
                    return filtered_signal
                dp = np.matmul(d_d_varphi_ext, d)
                dpp = np.matmul(d_d_varphi_ext, dp)
                B0pp = np.matmul(d_d_varphi, B0p)
                B0ppp = filt_data(np.matmul(d_d_varphi, B0pp))
                s2i = np.sin(2*iotaN*varphi_cent)
                c2i = np.cos(2*iotaN*varphi_cent)
                B2c_qi = -(B2c*c2i + B2s*s2i)
                B2s_qi = B2c*s2i - B2s*c2i
                B2sp = filt_data(np.matmul(d_d_varphi, B2c)*s2i + B2c*2*iotaN*c2i -(np.matmul(d_d_varphi, B2s)*c2i - B2s*2*iotaN*s2i))
                B2cp = filt_data(-np.matmul(d_d_varphi, B2c)*c2i + B2c*2*iotaN*s2i -(np.matmul(d_d_varphi, B2s)*s2i + B2s*2*iotaN*c2i))
                B20p = filt_data(np.matmul(d_d_varphi, B20))
                sin_3alpha_buff = np.sin(3*alpha_buf)
                cos_3alpha_buff = np.cos(3*alpha_buf)
                alBp = d_alpha_buf_d_varphi
                alBpp = np.matmul(d_d_varphi, alBp)
                I2 = stel.I2
                # Different contributions (ignoring the B31s contribution) and multiplying by B0'/G0
                DeltaY3cos = (1/(8*B0p**3))*(-16*B2s*B0p*(B0*B0p*dp + d*(B0p**2 - B0*B0pp)) + B0*d*(B0p**2*(-8*B2sp + 2*(8*B2c_qi + 3*B0*d**2)*alBp) - \
                    18*B0**2*d**2*alBp*B0pp + 3*B0**2*d*B0p*(2*dp*alBp + d*alBpp)))
                DeltaY3sin = (1/(8*B0p**4))*(16*B2c_qi*d*B0p**4 + 8*B0*B0p**3*(d**3*B0p + 2*B2c_qi*dp + d*(2*(G2 + I2*iota)/G0*B0p - \
                    (2*B20p + B2cp - 2*B2s_qi*alBp))) - 13*B0**2*d**2*B0p**2*(-2*B0p*dp + d*B0pp) + B0**3*d*(12*d**2*B0pp**2 + \
                    B0p**2*(8*dp**2 + 3*d**2*alBp**2 + 5*d*dpp) - 3*d*B0p*(6*dp*B0pp + d*B0ppp)))
                DeltaY3cos3 = (B0**2*d**2*(18*alBp*(B0*B0p*dp + d*(B0p**2 - B0*B0pp)) + 5*B0*d*B0p*alBpp))/(8*B0p**3)
                DeltaY3sin3 = (1/(8*B0p**4))*B0*d*(4*B0**2*B0p**2*dp**2 + B0*d*B0p*(18*dp*(B0p**2 - B0*B0pp) + 5*B0*B0p*dpp) + \
                        d**2*(4*B0p**4 + 12*B0**2*B0pp**2 - B0*B0p**2*(9*B0*alBp**2 + 13*B0pp) - 3*B0**2*B0p*B0ppp))
                # Total expression
                DeltaY3 = sin_alpha_buff*DeltaY3sin + cos_alpha_buff*DeltaY3cos + sin_3alpha_buff*DeltaY3sin3 + cos_3alpha_buff*DeltaY3cos3
                def correct_spike(y):
                    # Calculate the z-scores
                    mean = np.mean(y)
                    std_dev = np.std(y)
                    z_scores = (y - mean) / std_dev
                    # Define a threshold for outliers
                    threshold = 10
                    outliers = np.abs(z_scores) > threshold
                    # Replace outliers with the average of their neighbors
                    for i in range(len(y)):  # Avoid the first and last index to prevent out of bounds
                        if outliers[i]:
                            if i == 0:
                                y[i] = y[i+1]
                            elif i == len(y)-1:
                                y[i] = y[i-1]
                            else:
                                y[i] = np.mean([y[i-1], y[i+1]])
                    return y
                DeltaY3 = correct_spike(DeltaY3)

                # Upon performing an integral over phi, note that the expressions are even and thus we simply need to introduce a factor of 0.5later
                for jlam, lam in enumerate(lambda_bounce):
                    # \mathcal{H} without the 1/sqrt(1-lam B)
                    H_it = (1 - lam*B0)/B0 * (4/lam/B0 - 1)
                    # For determining bounce
                    f = 1 - lam*B0    # We are going to use the bounce integral of BAD, so put extra factor of (1-lam*B) NOTE: not best becasue normal integration would work
                    ## Integral: ʃ(H_it ΔY3 B0') dφ / G0
                    integrand = H_it * DeltaY3 
                    # Compute bounce integral
                    temp = bounce_int.bounce_integral_wrapper(f,integrand, varphi, multiple=False,return_roots=False)
                    integral_val.append(temp)

                # Integrals : include factor of 1/2 because of going from dB to whole bounce domain
                integral_val = np.squeeze(np.array(integral_val))
                H3_s = 0.5 * integral_val

                return H3_s

        # Define the I integral as a function of lambda
        def integral_I(lambda_bounce, order = 0):
            if order == 0:
                integral_val = np.zeros(len(lambda_bounce))
                for jlam, lam in enumerate(lambda_bounce):
                    # Integrand of I(0)
                    f = 1 - lam * B0
                    num_tot = f / B0 / B0
                    temp  = bounce_int.bounce_integral_wrapper(f,num_tot,varphi,return_roots=False)
                    integral_val[jlam] = temp[0]
                return integral_val
            elif order == 1:
                integral_val = np.zeros(len(lambda_bounce))
                for jlam, lam in enumerate(lambda_bounce):
                    # Integrand of I(0)
                    f = 1 - lam * B0
                    num_tot = -2*(1-0.75*lam*B0)/B0/B0 * (d*sin_alpha_buff)
                    temp  = bounce_int.bounce_integral_wrapper(f,num_tot,varphi,return_roots=False)
                    integral_val[jlam] = temp[0]
                return integral_val
            elif order == 2:
                integral_val = []
                for jlam, lam in enumerate(lambda_bounce):
                    # F_star integrand
                    f = 1 - lam * B0
                    F_star = (1-0.75*lam*B0)/B0/B0
                    # Integrand of I2_bar
                    I2_bar_integrand = B20 - 0.25*np.matmul(d_d_varphi, B0*B0*d*d/B0p)
                    I2_bar_integrand *= -2*F_star/B0
                    # Integrand of I2_a
                    B2cQI = -(B2c*np.cos(2*iotaN*varphi_cent-2*stel.helicity*np.pi) + B2s*np.sin(2*iotaN*varphi_cent-2*stel.helicity*np.pi))
                    delta_B2c = (B2cQI - \
                                0.25*(np.matmul(d_d_varphi, B0*B0*d*d/B0p)*cos_alpha_buff + B0*B0*d*d/B0p * d_cos_alpha_buf_d_varphi))
                    I2_c_integrand = 2 * delta_B2c * F_star / B0
                    # Compute integrals
                    temp  = bounce_int.bounce_integral_wrapper(f,[I2_bar_integrand, I2_c_integrand],varphi, multiple = True, return_roots=False)
                    integral_val.append(temp)

                # Integrals
                integral_val = np.squeeze(np.array(integral_val))
                I2_bar = integral_val[:,0]
                I2_a = integral_val[:,1]
                return I2_bar, I2_a

        # Compute necessary H pieces
        h_1 = integral_H(lam_array, order = 1)
        H2_a = integral_H(lam_array, order = 2)
        if include_Y3:
            H3_s = integral_H(lam_array, order = 3)

        # Compute necessary I pieces
        I_0 = integral_I(lam_array)
        I_1 = integral_I(lam_array, order = 1)
        I2_bar, I2_a = integral_I(lam_array, order = 2)

        #########################
        # E = H^2/I calculation #
        #########################
        # Leading order : same as when order = 0 is passed
        E_1 = h_1*h_1/I_0
        
        # First order correction : next order corrections to eps_eff^(3/2) neglecting third order effects
        E_2_drift = H2_a*H2_a/I_0
        E_2_resonant = -h_1*H2_a*I_1/I_0/I_0
        E_2_I = h_1*h_1/I_0 * ((I_1/2/I_0)**2 - (I2_bar - 0.5*I2_a)/I_0)

        E_2 = E_2_drift + E_2_resonant + E_2_I
        
        if include_Y3:
            E_3_drift = 2*h_1*H3_s/I_0
            E_2 += E_3_drift


        ###################
        # INTEGRAL OVER λ #
        ###################
        # Compute integral over λ
        integ_lambda_1 = integ.trapz(E_1 * lam_array, lam_array)
        integ_lambda_2 = integ.trapz(E_2 * lam_array, lam_array)

        ############################
        # COMPUTE GEOMETRIC FACTOR # (here we do not consider the O(r2) change of it which we should in principle do)
        ############################
        # Normalisation factors 
        if not ref_R_Bbar:
            ref_R_Bbar = G0

        # G_hat needs to be computed
        G_hat_sq = integ.trapz(X1c**2+X1s**2+Y1c**2+Y1s**2, varphi)

        ## More precise argument may be found without approximating the |nabla psi| piece
        # Define the averaging
        def G_hat_sq_int(check_correction = False):
            # Make chi grid
            N_chi = 100
            chi_grid = np.linspace(0, 1, N_chi)*2*np.pi

            # Compute integral over phi and alpha (and normalise the alpha part by 2π)
            nablapsi_chi = lambda chi: np.sqrt((X1c*np.sin(chi) - X1s*np.cos(chi))**2 + \
                            (Y1c*np.sin(chi) - Y1s*np.cos(chi))**2)
            G_hat_num = np.sum([integ.trapz(nablapsi_chi(chi)/B0, varphi) for chi in chi_grid])/N_chi
            
            # Normalisation flux average
            norm = np.trapz(1/B0/B0, varphi)

            # Put together
            G_hat_sq_num = G_hat_num**2/norm

            if check_correction:
                ######################################
                # HIGHER ORDER PREFACTOR CORRECTIONS #
                ######################################
                # (see eps_eff and geodesic_curvature notes)
                I2 = stel.I2
                lp = G0/B0
                curvature = stel.curvature
                torsion = stel.torsion
                B1n = lambda chi: B1c*np.cos(chi) + B1s*np.sin(chi)
                B2n = lambda chi: B20 + B2c*np.cos(2*chi) + B2s*np.sin(2*chi)
                X1n = lambda chi: X1c*np.cos(chi) + X1s*np.sin(chi)
                dt_Xn1 = lambda chi: -X1c*np.sin(chi) + X1s*np.cos(chi)
                dp_Xn1 = lambda chi: stel.d_X1c_d_varphi*np.cos(chi) + stel.d_X1s_d_varphi*np.sin(chi)
                X2n = lambda chi: stel.X20 + X2c*np.cos(2*chi) + X2s*np.sin(2*chi)
                dt_Xn2 = lambda chi: -2*X2c*np.sin(2*chi) + 2*X2s*np.cos(2*chi)
                Y1n = lambda chi: Y1c*np.cos(chi) + Y1s*np.sin(chi)
                dt_Yn1 = lambda chi: -Y1c*np.sin(chi) + Y1s*np.cos(chi)
                dp_Yn1 = lambda chi: stel.d_Y1c_d_varphi*np.cos(chi) + stel.d_Y1s_d_varphi*np.sin(chi)
                dt_Yn2 = lambda chi: -2*Y2c*np.sin(2*chi) + 2*Y2s*np.cos(2*chi)
                dt_Zn2 = lambda chi: -2*Z2c*np.sin(2*chi) + 2*Z2s*np.cos(2*chi)
                dp_Zn2 = lambda chi: stel.d_Z20_d_varphi + stel.d_Z2c_d_varphi*np.cos(2*chi) + stel.d_Z2s_d_varphi*np.sin(2*chi)
                dt_Xn3 = lambda chi: -stel.X3c1*np.sin(chi) + stel.X3s1*np.cos(chi) -3*stel.X3c3*np.sin(3*chi) + 3*stel.X3s3*np.cos(3*chi)
                dt_Yn3 = lambda chi: -stel.Y3c1*np.sin(chi) + stel.Y3s1*np.cos(chi) -3*stel.Y3c3*np.sin(3*chi) + 3*stel.Y3s3*np.cos(3*chi)

                # |grad psi|^2 = r^2 B0^2 (P1 + r P2 + r^2 P3) : expansion in Mathematica
                P1 = lambda chi: dt_Xn1(chi)**2 + dt_Yn1(chi)**2
                P2 = lambda chi: 2*(P1(chi)*B1n(chi)/B0 + (dt_Xn1(chi)*dt_Xn2(chi) + dt_Yn1(chi)*dt_Yn2(chi)))
                P3 = lambda chi: (1/G0**3)*(2*lp**2*(B0**2*(3*B1n(chi)**2 + 2*B0*B2n(chi))*G0 - B0**4*(G2 + I2*iotaN))*dt_Xn1(chi)**2 - \
                            8*B0**3*B1n(chi)*lp**2*G0*dt_Xn1(chi)*(X1n(chi)*curvature*dt_Xn1(chi) - dt_Xn2(chi)) + \
                                2*lp**2*(B0**2*(3*B1n(chi)**2 + 2*B0*B2n(chi))*G0 - B0**4*(G2 + I2*iotaN))*dt_Yn1(chi)**2 + \
                                B0**4*G0*(dp_Yn1(chi)*dt_Xn1(chi) - dp_Xn1(chi)*dt_Yn1(chi) + lp*torsion*(X1n(chi)*dt_Xn1(chi) + \
                                Y1n(chi)*dt_Yn1(chi)))**2 - 8*B0**3*B1n(chi)*lp**2*G0*dt_Yn1(chi)*(X1n(chi)*curvature*dt_Yn1(chi) - dt_Yn2(chi)) + \
                                B0**4*G0*((lp*X1n(chi)*curvature*dt_Xn1(chi) - lp*dt_Xn2(chi))**2 - 2*lp*dt_Xn1(chi)*(-dp_Zn2(chi)*dt_Xn1(chi) + \
                                dp_Xn1(chi)*dt_Zn2(chi) + lp*(X2n(chi)*curvature*dt_Xn1(chi) + X1n(chi)*curvature*dt_Xn2(chi) - dt_Xn3(chi) - \
                                Y1n(chi)*torsion*dt_Zn2(chi)))) + B0**4*G0*((lp*X1n(chi)*curvature*dt_Yn1(chi) - lp*dt_Yn2(chi))**2 - \
                                2*lp*dt_Yn1(chi)*(-dp_Zn2(chi)*dt_Yn1(chi) + dp_Yn1(chi)*dt_Zn2(chi) + lp*(X2n(chi)*curvature*dt_Yn1(chi) - \
                                dt_Yn3(chi) + X1n(chi)*(curvature*dt_Yn2(chi) + torsion*dt_Zn2(chi))))))/B0**2
                # Construct |grad psi|/B0 = r T1 + r^2 T2 + r^3 T3 
                T1 = lambda chi: np.sqrt(P1(chi))
                T2 = lambda chi: 0.5*P2(chi)/np.sqrt(P1(chi))
                T3 = lambda chi: 0.5*P3(chi)/np.sqrt(P1(chi))-0.125*P2(chi)**2/P1(chi)**1.5

                # D integral : D = ∫ (|nabla psi|/B^2) dφ
                D1 = np.sum([integ.trapz(T1(chi)/B0, varphi) for chi in chi_grid])/N_chi
                # D2 = np.sum([integ.trapz(T2(chi)/B0 - 2*T1(chi)*B1n(chi)/B0**2, varphi) for chi in chi_grid])/N_chi # we expect it to vanish?
                D3 = np.sum([integ.trapz(T3(chi)/B0 - 2*T2(chi)*B1n(chi)/B0**2 + T1(chi)/B0*(3*B1n(chi)**2/B0**2-2*B2n(chi)/B0), varphi)\
                            for chi in chi_grid])/N_chi
                
                # Normalisation : L = ∫ (1/B^2) dφ
                L0 = np.trapz(1/B0/B0, varphi) # Leading order normalisation like before
                # L1 vanishes due to parity upon flux surface integration
                L2 = np.sum([integ.trapz(1/B0**2*(3*B1n(chi)**2/B0**2-2*B2n(chi)/B0), varphi) for chi in chi_grid])/N_chi

                # Function F = L/(2D**2) which is the inverse of G_hat_sq (note we do not have the G0 factor in front by definition)
                F0 = L0/2/D1**2
                F2_over_F0 = L2/L0 - G2/G0 - 2*D3/D1

                print("Relative correction to grad psi: ", F2_over_F0)

            # The factor of two is how we defined G_hat
            return 2*G_hat_sq_num

        # Compute 
        G_hat_sq_num = G_hat_sq_int(check_correction=check_geo_correction)
        print("G_hat_sq: ", G_hat_sq_num)

        ###############
        # ε EFFECTIVE #
        ###############
        fac_eps_eff = np.pi/8/np.sqrt(2)/G_hat_sq_num * ref_R_Bbar**2
        eps_eff_3_2 = fac_eps_eff * integ_lambda_1
        eps_eff_3_2_r2 = fac_eps_eff * integ_lambda_2

        # Make a dictionary with all E components : so can read from outside
        E_all = {}
        E_all["E_1"] = E_1 * fac_eps_eff * lam_array
        E_all["E_2_drift"] = E_2_drift * fac_eps_eff * lam_array
        if include_Y3:
            E_all["E_3_drift"] = E_3_drift * fac_eps_eff * lam_array
        E_all["E_2_resonant"] = E_2_resonant * fac_eps_eff * lam_array
        E_all["E_2_I"] = E_2_I * fac_eps_eff * lam_array
        E_all["E_2"] = E_2 * fac_eps_eff * lam_array
        E_all["lambda"] = lam_array

        eps_eff_per_lambda.append(E_all)

        if plot:
            N_r = 100
            r_array = np.logspace(-2.5,-0.2, N_r)
            plt.plot(r_array, (eps_eff_3_2 + r_array**2 * eps_eff_3_2_r2)**(2/3))
            plt.xlabel(r"$r$")
            plt.ylabel(r"$\epsilon_\mathrm{eff}$")
            plt.yscale('log')
            plt.xscale('log')
            plt.tight_layout()

        return eps_eff_3_2, eps_eff_3_2_r2
    else:
        raise KeyError('eps_eff calculation only available to first order.')


