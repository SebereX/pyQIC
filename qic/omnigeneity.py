"""
This module contains the routines for estimating measures of omnigeneity breaking such as ε_eff.
"""
from tqdm import tqdm
from scipy import interpolate
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
from BAD import bounce_int
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from .optimize_nae import min_geo_qi_consistency

# Plotting details
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 18})
rc('text', usetex=True)


def compute_eps_eff(stel, order = 'r0', N_lam = 100, plot = False, info = None, ref_R_Bbar = None, check_geo_correction = False, include_Y3 = False, mask = None):
    """
    Compute the omnigeneity breaking measure ε_eff for a given stellarator object. The function can compute the leading order (r0) or first order (r1) contribution to the omnigeneity breaking measure ε_eff. The function can also compute the different contributions to the integrand of ε_eff per λ by passing the list partial_info. 
    Args:
        stel (Stellarator): Stellarator object for which to compute ε_eff.
        order (str): Order of the computation. Can be 'r0' or 'r1'.
        N_lam (int): Number of points in the λ grid.
        plot (bool): Whether to plot the integrand of ε_eff per λ. 
        info (list): List to store the integrand of ε_eff per λ and other information. 
        ref_R_Bbar (float): Reference value for the normalisation factor RB̅. If not provided, the normalisation factor is set to the value of G0.
        check_geo_correction (bool): Whether to check the higher order prefactor corrections to the geometric factor G_hat.
        include_Y3 (bool): Whether to include the Y3 contribution in the calculation (barring the third order field, B3).
        mask (array): Mask to apply to the integrand of ε_eff per λ.
    Returns:
        Value of ε_eff^(3/2) (for 0th and 1st order if 'r1') for the given stellarator object.
    """
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
        # # G_hat needs to be computed
        # G_hat_sq = integ.trapz(X1c**2+X1s**2+Y1c**2+Y1s**2, varphi)

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

        if not info is None:
            # Store λ and the integrand of ε_eff per λ 
            info.append([lam_array,fac_eps_eff * integrand_lambda * lam_array])

        if plot:
            # Print ε_eff
            print("ε_eff: ", eps_eff_3_2**(2/3))
            # Plot the integrand of ε_eff per λ
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
            """
            Compute the H_1, H_2 or H_3 (according to order) integral for the omnigeneity breaking measure ε_eff^(3/2).
            Args:
                lambda_bounce (array): Array of λ values.
                order (int): Order of the H integral to compute.
            Returns:
                Array of values with the H integral for the given order.
            """
            if order == 1:
                # Compute the H_1 integral
                integral_val = np.zeros(len(lambda_bounce))
                for jlam, lam in enumerate(lambda_bounce):
                    # Integrand of H_1
                    f = 1 - lam * B0    # We are going to use the bounce integral of BAD, so put extra factor of (1-lam*B) NOTE: not best becasue normal integration would work
                    num_tot = f / B0 * d * sin_alpha_buff * (4/lam/B0 - 1)
                    # Compute bounce integral
                    temp = bounce_int.bounce_integral_wrapper(f,num_tot,varphi,return_roots=False)
                    integral_val[jlam] = temp[0]
                return integral_val
            elif order == 2:
                # Compute the H_2 integral
                integral_val = []
                # Construct necessary integrand terms
                ###############
                # INTEGRAND C #
                ###############
                # Construct the integrand C (directy reated t the 2nd order non-omnigeneous field)
                B2cQI = -(B2c*np.cos(2*iotaN*varphi_cent-2*stel.helicity*np.pi) + B2s*np.sin(2*iotaN*varphi_cent-2*stel.helicity*np.pi))
                delta_B2c = 4/B0*(B2cQI -0.25*np.matmul(d_d_varphi, B0*B0*d*d/B0p*np.cos(2*alpha_buf)))
                # Mask the integrand in λ if necessary
                if mask is None:
                    C_integ = delta_B2c
                else:
                    C_integ = mask * delta_B2c

                # Compute the bounce integrals necessary for H_2 (for each λ)
                for jlam, lam in enumerate(lambda_bounce):
                    # \mathcal{H} without the 1/sqrt(1-lam B)
                    H_it = (1 - lam*B0)/B0/B0 * (4/lam/B0 - 1)
                    # For determining bounce
                    f = 1 - lam*B0    # We are going to use the bounce integral of BAD, so put extra factor of (1-lam*B) NOTE: not best becasue normal integration would work
                    ## Integral
                    integrand_2 = -0.5 * H_it * B0 * C_integ 
                    # Compute bounce integral
                    temp = bounce_int.bounce_integral_wrapper(f, integrand_2, varphi, multiple=False,return_roots=False)
                    integral_val.append(temp)

                # Integrals 
                integral_val = np.squeeze(np.array(integral_val))
                # H2_a = integral_val[:,0] + integral_val[:,1] + integral_val[:,2]
                H2_a = integral_val

                return H2_a
            elif order == 3:
                ## Compute the H_3 integral : ΔY3, sin α component : this is quite noisy and the calaculations clearly struggle from noise and other limitations. NOTE : generally deprecated, but may be used to check certain things
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
            """
            Compute the I_0, I_1 or I_2 (according to order) integral for the omnigeneity breaking measure ε_eff^(3/2).
            Args:
                lambda_bounce (array): Array of λ values.
                order (int): Order of the I integral to compute.
            Returns:
                Array of values with the I integral for the given order. For order 2, returns two arrays with the I2_bar and I2_a integrals.
            """
            if order == 0:
                # Compute the I_0 integral
                integral_val = np.zeros(len(lambda_bounce))
                for jlam, lam in enumerate(lambda_bounce):
                    # Integrand of I(0)
                    f = 1 - lam * B0
                    num_tot = f / B0 / B0
                    # Bounce integral
                    temp  = bounce_int.bounce_integral_wrapper(f,num_tot,varphi,return_roots=False)
                    integral_val[jlam] = temp[0]
                return integral_val
            elif order == 1:
                # Compute the I_1 integral
                integral_val = np.zeros(len(lambda_bounce))
                for jlam, lam in enumerate(lambda_bounce):
                    # Integrand of I(0)
                    f = 1 - lam * B0
                    num_tot = -2*(1-0.75*lam*B0)/B0/B0 * (d*sin_alpha_buff)
                    # Bounce integral
                    temp  = bounce_int.bounce_integral_wrapper(f,num_tot,varphi,return_roots=False)
                    integral_val[jlam] = temp[0]
                return integral_val
            elif order == 2:
                # Compute the I_2 integral
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
                    # Bounce integrals
                    temp  = bounce_int.bounce_integral_wrapper(f,[I2_bar_integrand, I2_c_integrand],varphi, multiple = True, return_roots=False)
                    integral_val.append(temp)

                # Integrals
                integral_val = np.squeeze(np.array(integral_val))
                I2_bar = integral_val[:,0]
                I2_a = integral_val[:,1]
                return I2_bar, I2_a

        # Compute necessary H pieces
        h_1 = integral_H(lam_array, order = 1)      # Leading order contribution to H
        H2_a = integral_H(lam_array, order = 2)     # Second order contribution to H
        if not info is None and include_Y3:
            # Compute H_3 only when info is required and the third order flag is set
            H3_s = integral_H(lam_array, order = 3)

        # Compute necessary I pieces
        I_0 = integral_I(lam_array)
        if not info is None:
            # Compute I_1 and I_2 only when info is required
            I_1 = integral_I(lam_array, order = 1)
            I2_bar, I2_a = integral_I(lam_array, order = 2)

        #########################
        # E = H^2/I calculation #
        #########################
        # Leading order contribution: same as when order = 0 is passed
        E_1 = h_1*h_1/I_0
        
        # Next order corrections to eps_eff^(3/2) neglecting third order effects
        if info is None:
            # Standard dominant second order contribution
            E_2 = H2_a*H2_a/I_0
        else:
            # If info is required, compute the multiple (small) different contributions to the integrand of ε_eff per λ
            E_2_drift = H2_a*H2_a/I_0               # Dominant term    
            E_2_resonant = -h_1*H2_a*I_1/I_0/I_0    # Resonant term
            E_2_I = h_1*h_1/I_0 * ((I_1/2/I_0)**2 - (I2_bar - 0.5*I2_a)/I_0) # I term
            # Total seconf order contribution to E 
            E_2 = E_2_drift + E_2_resonant + E_2_I            
        
        # If required, include the third order contribution to eps_eff^(3/2)
        if include_Y3:
            E_3_drift = 2*h_1*H3_s/I_0
            E_2 += E_3_drift


        ###################
        # INTEGRAL OVER λ #
        ###################
        # Compute integral over λ
        integ_lambda_1 = integ.trapz(E_1 * lam_array, lam_array)
        integ_lambda_2 = integ.trapz(E_2 * lam_array, lam_array)

        # Save information about the integrand of ε_eff per λ. May be used externally for analysis
        if not info is None:
            anal_info = {}
            anal_info["name"] = "Eps_eff integrand anal info"
            anal_info["lambda"] = lam_array
            anal_info["h_1"] = h_1
            anal_info["I_0"] = I_0
            anal_info["E_1"] = h_1*h_1/I_0
            anal_info["int_E"] = integ_lambda_1
            info.append(anal_info)

        ############################
        # COMPUTE GEOMETRIC FACTOR # (here we do not consider the O(r2) change of it which we should in principle do)
        ############################
        # Normalisation factors 
        if not ref_R_Bbar:
            ref_R_Bbar = G0

        # # G_hat needs to be computed
        # G_hat_sq = integ.trapz(X1c**2+X1s**2+Y1c**2+Y1s**2, varphi)

        ## More precise argument may be found without approximating the |nabla psi| piece
        # Define the averaging
        def G_hat_sq_int(check_correction = False):
            """
            Compute the geometric factor G_hat for the omnigeneity breaking measure ε_eff^(3/2), performing the integals over φ and α explicitly. 
            Args:
                check_correction (bool): Whether to check the higher order prefactor corrections to the geometric factor G_hat. If True, the function will save the higher order prefactor corrections to G_hat in info.
            Returns:
                Value of G_hat^(2) for the given stellarator object.
            """
            # Make chi grid
            N_chi = 100
            chi_grid = np.linspace(0, 1, N_chi, endpoint=False)*2*np.pi

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
                D2 = np.sum([integ.trapz(T2(chi)/B0 - 2*T1(chi)*B1n(chi)/B0**2, varphi) for chi in chi_grid])/N_chi # we expect it to vanish?
                D3 = np.sum([integ.trapz(T3(chi)/B0 - 2*T2(chi)*B1n(chi)/B0**2 + T1(chi)/B0*(3*B1n(chi)**2/B0**2-2*B2n(chi)/B0), varphi)\
                            for chi in chi_grid])/N_chi
                
                # Normalisation : L = ∫ (1/B^2) dφ
                L0 = np.trapz(1/B0/B0, varphi) # Leading order normalisation like before
                # L1 vanishes due to parity upon flux surface integration
                L2 = np.sum([integ.trapz(1/B0**2*(3*B1n(chi)**2/B0**2-2*B2n(chi)/B0), varphi) for chi in chi_grid])/N_chi

                # Function F = L/(2D**2) which is the inverse of G_hat_sq (note we do not have the G0 factor in front by definition)
                F0 = L0/2/D1**2
                F1_over_F0 = -2*D2/D1
                F2_over_F0 = L2/L0 - G2/G0 - 2*D3/D1

                if not info is None:
                    G_all = {}
                    G_all["name"] = "G_hat details"     # Label of dictionary
                    G_all["F0"] = F0
                    G_all["F1_over_F0"] = F1_over_F0
                    G_all["F2_over_F0"] = F2_over_F0
                    info.append(G_all)
                else:
                    print("Relative correction to grad psi: ", F2_over_F0)

            # The factor of two is how we defined G_hat
            return 2*G_hat_sq_num

        # Compute G_hat_sq
        G_hat_sq_num = G_hat_sq_int(check_correction=check_geo_correction)

        ###############
        # ε EFFECTIVE #
        ###############
        # Compute prefactor of eps_eff^(3/2)
        fac_eps_eff = np.pi/8/np.sqrt(2)/G_hat_sq_num * ref_R_Bbar**2
        # First and second order contributions to ε_eff^(3/2)
        eps_eff_3_2 = fac_eps_eff * integ_lambda_1
        eps_eff_3_2_r2 = fac_eps_eff * integ_lambda_2

        if not info is None:
            # Make a dictionary with all E components : so can read from outside
            E_all = {}
            E_all["name"] = "Eps_eff details per lambda"
            E_all["fac_eps_eff"] = fac_eps_eff
            E_all["E_1"] = E_1 * fac_eps_eff * lam_array
            E_all["E_2_drift"] = E_2_drift * fac_eps_eff * lam_array
            if include_Y3:
                E_all["E_3_drift"] = E_3_drift * fac_eps_eff * lam_array
            E_all["E_2_resonant"] = E_2_resonant * fac_eps_eff * lam_array
            E_all["E_2_I"] = E_2_I * fac_eps_eff * lam_array
            E_all["E_2"] = E_2 * fac_eps_eff * lam_array
            E_all["lambda"] = lam_array
            E_all["eps_eff_3_2_r2_drift"] = integ.trapz(E_2_drift * lam_array, lam_array) * fac_eps_eff
            E_all["eps_eff_3_2_r2_resonant"] = integ.trapz(E_2_resonant * lam_array, lam_array) * fac_eps_eff
            E_all["eps_eff_3_2_r2_I"] = integ.trapz(E_2_I * lam_array, lam_array) * fac_eps_eff

            info.append(E_all)

        if plot:
            # Plot ε_eff as a function of r
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
        raise KeyError("ε_eff calculation only available up to 'r1'.")

def compute_eps_eff_anal(stel, r = 0.1, alpha = 0.0, N_lam = 100, verbose = False, ref_R_Bbar = None, info = None):
    """
    Compute the omnigeneity breaking measure ε_eff^(3/2) for a given stellarator object at a given radial position r and poloidal angle α using the near-axis field evaluated at r and evaluating all integrals numerically. No asymptotic consideration in the same vain as compute_eps_eff. Its asymptotic form should match that of compute_eps_eff.
    Args:
        stel (stellarator.Stellarator): Stellarator object for which to compute the omnigeneity breaking measure.
        r (float): Radial position at which to evaluate the omnigeneity breaking measure.
        alpha (float): Poloidal angle at which to evaluate the omnigeneity breaking measure.
        N_lam (int): Number of λ values to use in the numerical integration.
        verbose (bool): Whether to print information about the computation.
        ref_R_Bbar (float): Reference value of R Bbar to use in the normalisation of eps_eff. If None, uses G0.
        info (list): If not None, will return additional information about the computation in the list.
    Returns:
        Value of ε_eff^(3/2) at the given radial position and poloidal angle.
    """
    ####################
    # FIELD QUANTITIES #
    ####################
    sgn_half = 1 - 4*np.mod(stel.helicity, 1)
    nfp = stel.nfp
    nphi = stel.nphi
    varphi = stel.varphi
    varphi_ext = np.append(stel.varphi, stel.varphi[0] + 2*np.pi/nfp)
    d_d_varphi = stel.d_d_varphi
    # B0
    B0 = stel.B0
    dB0 = np.matmul(d_d_varphi, B0)
    # G0
    G0 = stel.G0
    I2 = stel.I2
    # iota
    iotaN = stel.iotaN
    # B1
    B1c = stel.B1c
    B1s = stel.B1s
    # X1 and Y1
    X1c = stel.X1c
    X1s = stel.X1s
    Y1c = stel.Y1c
    Y1s = stel.Y1s
    # B2 functios
    B20 = stel.B20
    B2c = stel.B2c
    B2s = stel.B2s

    ###########################
    # FIELDS ALONG FIELD LINE #
    ###########################
    def check_array(fun):
        """
        Check if the input is an array or a single value and return the number of elements and the array.
        Args:
            fun (array or float): Input to check.
        Returns:
            N_f (int): Number of elements in the array.
            fun (array): Array of input values.
        """
        if isinstance(fun, np.ndarray) or isinstance(fun,list):
            N_f = len(fun)
            fun = np.array(fun)
        else:
            N_f = 1
            fun = np.array([fun])
        return N_f, fun

    ## Check inputs ##
    N_r, r = check_array(r)
    N_alpha, alpha = check_array(alpha)  

    def B_fun(r, alpha):
        """
        Construct the magnetic field B along the field line for a given radial position.
        Args:
            r (array): Array of radial positions at which to evaluate the magnetic field.
            alpha (array): Array of field line label alpha at which to evaluate the magnetic field.
        Returns:
            Array of magnetic field values B(r,α) for the given radial positions and poloidal angles. The indices are (r,α,φ).
        """
        ## Create array with B ##
        B_arr = np.zeros((N_alpha,N_r,nphi+1))

        ## χ grid ##
        chi = alpha[:, None] + iotaN * np.append(varphi, varphi[0] + 2*np.pi/nfp)[None, :]

        ## NAE functions  ##
        B0n = np.append(B0, B0[0])
        B1n = np.append(B1c, B1c[0]*sgn_half)[None, :] * np.cos(chi) + np.append(B1s, B1s[0]*sgn_half)[None, :] * np.sin(chi)
        B2n = np.append(B20, B20[0])[None, :] + np.append(B2c, B2c[0])[None, :] * np.cos(2*chi) + np.append(B2s, B2s[0])[None, :] * np.sin(2*chi)

        ## Magnetic field ##
        B_arr = B0n[None, None, :] + r[:, None, None] * B1n[None, :, :] + r[:, None, None] * r[:, None, None] * B2n[None, :, :]

        return B_arr
    
    def K_fun(r, alpha):
        """
        Compute the K = Bx∇B·∇ψ/B² term for the main integral in ε_eff^(3/2) for a given radial position.
        Args:
            r (array): Array of radial positions at which to evaluate the magnetic field.
            alpha (array): Array of field line label alpha at which to evaluate the magnetic field.
        Returns:
            Array of K values K(r,α) for the given radial positions and poloidal angles. The indices are (r,α,φ).
        """
        ## Create array with K = Bx∇B·∇ψ/B² ##
        K_arr = np.zeros((N_alpha,N_r,nphi+1))

        ## χ grid ##
        chi = alpha[:, None] + iotaN * np.append(varphi, varphi[0] + 2*np.pi/nfp)[None, :]

        ## NAE functions  ##
        B0n_dp = np.append(dB0, dB0[0])
        B1n_dt = -np.append(B1c, B1c[0]*sgn_half)[None, :] * np.sin(chi) + np.append(B1s, B1s[0]*sgn_half)[None, :] * np.cos(chi)
        B2n_dt = -2*np.append(B2c, B2c[0])[None, :] * np.sin(2*chi) + 2*np.append(B2s, B2s[0])[None, :] * np.cos(2*chi)

        ## Magnetic field ##
        K_arr = -r[:, None, None] * B1n_dt[None, :, :] + r[:, None, None] * r[:, None, None] * (I2 * B0n_dp[None, None, :] / G0 - B2n_dt[None, :, :])

        return K_arr

    ## Construct arrays for field
    B_arr = B_fun(r, alpha)
    K_arr = K_fun(r, alpha)

    ##################
    # CREATE λ GRIDS #
    ##################
    # For each α it is different: need to compute Bmin and Bmax for each
    Bmax_arr = np.max(B_arr, axis = 2)
    Bmin_arr = np.min(B_arr, axis = 2) 

    ########################
    # COMPUTE ∫ λ ℰ(λ) dλ  #
    ########################
    # (compute for each r,α) #
    int_E_arr = np.zeros((N_r, N_alpha))
    
    with tqdm(desc = "Computing...", total = N_r) as pbar:
        for j_r, r_val in enumerate(r):
            for j_a, alpha_val in enumerate(alpha):
                ##########
                # λ GRID #
                ##########
                B_max = Bmax_arr[j_r, j_a]
                B_min = Bmin_arr[j_r, j_a]
                # Make grid in the trapped region
                lam_array = 1/(np.linspace(0.0,1,N_lam+1,endpoint = False)[1:][::-1] * (B_max-B_min) + B_min)
                assert N_lam == len(lam_array), "Mismatch in length of lambda grid"

                #####################
                # H and I INTEGRALS #
                #####################
                # Define the H integral as a function of lambda
                K_val = K_arr[j_r,j_a,:]
                B_val = B_arr[j_r,j_a,:]

                def H_int(lam_array):
                    """
                    Compute the H integral for the omnigeneity breaking measure ε_eff^(3/2).
                    Args:
                        lam_array (array): Array of λ values.
                    Returns:
                        Array of values with the H integral.
                    """
                    H_int_arr = []
                    for jlam, lam in enumerate(lam_array):
                        # \mathcal{H} without the 1/sqrt(1-lam B)
                        H_it = (1 - lam*B_val)/B_val/B_val * (4/lam/B_val - 1)

                        # For determining bounce
                        f = 1 - lam*B_val    # We are going to use the bounce integral of BAD, so put extra factor of (1-lam*B) NOTE: not best becasue normal integration would work

                        ## Integrand for H (K = Bx∇B·∇ψ/B²)
                        integrand = H_it * K_val

                        ## Compute bounce integral ##
                        temp = bounce_int.bounce_integral_wrapper(f, integrand, varphi_ext, multiple=False, return_roots=False)
                        # Check what is returned
                        assert isinstance(temp, list), "Bounce integral is not returning a list!"
                        if len(temp) > 1:
                            # If multiple wells, say so
                            print(f'Multiple wells! {r_val}: {temp}')
                        if temp == []:
                            # If no bounce integral, say so
                            if verbose: print(f"WARNING! Bounce integral problems for {jlam} out of {N_lam} in λ")
                            temp = [np.nan]
                        H_int_arr.append(temp)

                    return H_int_arr

                # Compute H integral
                H_arr = H_int(lam_array)

                def I_int(lam_array):
                    """
                    Compute the I integral for the omnigeneity breaking measure ε_eff^(3/2).
                    Args:
                        lam_array (array): Array of λ values.
                    Returns:
                        Array of values with the I integral.
                    """
                    I_int_arr = []
                    for jlam, lam in enumerate(lam_array):
                        # integrand without the 1/sqrt(1-lam B)
                        integrand = (1 - lam*B_val)/B_val/B_val

                        # For determining bounce
                        f = 1 - lam*B_val    # We are going to use the bounce integral of BAD, so put extra factor of (1-lam*B) NOTE: not best becasue normal integration would work

                        ## Compute bounce integral ##
                        temp = bounce_int.bounce_integral_wrapper(f, integrand, varphi_ext, multiple=False, return_roots=False)
                        # Check what is returned
                        assert isinstance(temp, list), "Bounce integral is not returning a list!"
                        if temp == []:
                            # If no bounce integral, say so
                            if verbose: print(f"WARNING! Bounce integral problems for {jlam} out of {N_lam} in λ")
                            temp = [np.nan]
                        I_int_arr.append(temp)

                    return I_int_arr
                
                # Compute I integral
                I_arr = I_int(lam_array)

                #############
                # COMPUTE ℰ #
                #############
                def get_rid_of_nans(data):
                    """
                    Function to replace NaN values in an array with interpolated/extrapolated values.
                    Args:
                        data (array): Array with NaN values.
                    Returns:
                        Array with NaN values replaced by interpolated/extrapolated values.
                    """
                    # Identify the indices where NaNs are located
                    nan_indices = np.isnan(data)

                    # Extract non-NaN values and their indices
                    x_non_nan = np.arange(len(data))[~nan_indices]  # Indices of non-NaN values
                    y_non_nan = data[~nan_indices]  # Corresponding non-NaN values

                    # Interpolate or extrapolate the NaN values
                    # Create an interpolating function based on the existing data
                    interp_func = interpolate.interp1d(x_non_nan, y_non_nan, kind='cubic', fill_value="extrapolate")

                    # Replace NaN values with interpolated/extrapolated values
                    data[nan_indices] = interp_func(np.arange(len(data))[nan_indices])

                    return data
                
                # Compute E = H^2/I
                E_arr = np.array([np.sum(np.array(H_arr[i])**2/np.array(I_arr[i])) for i in range(len(H_arr))])
                E_arr = get_rid_of_nans(E_arr)

                ###################
                # INTEGRAL OVER λ #
                ###################
                # Compute integral over λ
                int_E_arr[j_r, j_a] = np.trapz(E_arr * lam_array, x = lam_array)

                if not info is None:
                    # Save information about the integrand of ε_eff per λ. May be used externally for analysis
                    anal_info = {}
                    anal_info["name"] = "Eps_eff integrand anal"
                    anal_info["lambda"] = lam_array
                    anal_info["H"] = get_rid_of_nans(H_arr)
                    anal_info["I"] = get_rid_of_nans(I_arr)
                    anal_info["E"] = E_arr
                    anal_info["int_E"] = int_E_arr[j_r, j_a]
                    info.append(anal_info)
            # Update progress bar
            pbar.update(1)

    #####################
    # COMPUTE α-AVERAGE #
    #####################
    # Normalise to π and not 2π : due to factor of 2 in the definition of G²
    res_int = np.array([np.trapz(np.append(int_E_arr[j_r,:],int_E_arr[j_r,0]), x = np.append(alpha, 2*np.pi))/(np.pi) for j_r in range(N_r)])

    ## More precise argument may be found without approximating the |nabla psi| piece
    # Define the averaging
    def G_hat_sq_int():
        """
        Compute the geometric factor G_hat for ε_eff^(3/2), performing the integals over φ and α explicitly.
        Returns:
            Value of G_hat^2 for the given stellarator object
        """
        # Make chi grid
        N_chi = 100
        chi_grid = np.linspace(0, 1, N_chi, endpoint=False)*2*np.pi

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

    # Normalisation factors
    if not ref_R_Bbar:
        ref_R_Bbar = G0
    fac_eps_eff = np.pi/8/np.sqrt(2)/G_hat_sq_num * ref_R_Bbar**2
    eps_eff_3_2 = fac_eps_eff*res_int/r/r

    return res_int, eps_eff_3_2

def omn_reshape(stel, X2s_in = 0, mask_frac = 0.15, run = True, info = []):
    """
    Construct a reshaped 2nd order configuration to eliminate the non-omnigeneous behaviour where possible.
    Args:
        stel (Qic): Stellarator object for which to construct the reshaped configuration.
        X2s_in (float or array): Input value for the reshaped X2s_tilde. Default is 0.
        mask_frac (float): Fraction of the configuration to mask out at the top and bottom of the well.
        run (bool): Whether to run the 2nd order reshaping of the original Qic configuration. If False, returns reshaped X2c and X2s.
        info (list): If not None, will return additional information about the computation in the list (mismatch at bottom of well).
    Returns:
        Stellarator object with reshaped 2nd order configuration if run is True. Otherwise, returns reshaped X2c and X2s.
    """
    # Shaping and bottom defect
    res, X2c, X2s = min_geo_qi_consistency(stel, X2s_in = X2s_in, order = 1)
    info.append(res)

    ## Shaped configuration
    def create_smoothed_top_hat_mask_from_values(x_grid, rise_values, drop_values, hat_height=1, smooth_width=5):
        """
        Creates a smoothed top-hat mask based on rise and drop positions specified as x values.

        Parameters:
            x_grid (np.ndarray): The x values defining the grid.
            rise_values (list of float): x values where top hats start rising.
            drop_values (list of float): x values where top hats start dropping.
            hat_height (float): Height of the top hat.
            smooth_width (float): Standard deviation for Gaussian smoothing.

        Returns:
            np.ndarray: The mask array with smoothed top hats.
        """
        # Ensure x_grid is sorted
        x_grid = np.asarray(x_grid)
        assert np.all(np.diff(x_grid) > 0), "x_grid must be sorted in ascending order."
        
        # Interpolate rise and drop indices
        rise_indices = np.round(np.interp(rise_values, x_grid, np.arange(len(x_grid)))).astype(int)
        drop_indices = np.round(np.interp(drop_values, x_grid, np.arange(len(x_grid)))).astype(int)
        
        # Initialize the mask array with zeros
        mask = np.zeros_like(x_grid)
        
        # Apply rises
        for idx in rise_indices:
            mask[idx:] += hat_height
        
        # Apply drops
        for idx in drop_indices:
            mask[idx:] -= hat_height
        
        # Smooth the mask
        smoothed_mask = gaussian_filter1d(mask, sigma=smooth_width)
        
        return smoothed_mask

    # Masking of the shaping
    N_mask = 1001
    varphi_mask = np.linspace(0.0, 1.0, N_mask) * 2*np.pi/stel.nfp
    mask = create_smoothed_top_hat_mask_from_values(varphi_mask,
                                                [0,(1 - mask_frac) * np.pi/stel.nfp, (1.9 - mask_frac) * np.pi/stel.nfp], \
                                                [(0.1 + mask_frac) * np.pi/stel.nfp, (1 + mask_frac) * np.pi/stel.nfp])
    mask_per_sp = PchipInterpolator(varphi_mask, mask)
    mask = 0.5*(mask_per_sp(stel.varphi) + mask_per_sp(2*np.pi/stel.nfp-stel.varphi))   # Symmetrise

    # Shaping for the configuration (avoid divergences and nans that could arise from original division by 0)
    X2c[X2c == np.inf] = 0;     X2c[X2c == -np.inf] = 0;     X2c[X2c == np.nan] = 0
    X2s[X2s == np.inf] = 0;     X2s[X2s == -np.inf] = 0;     X2s[X2s == np.nan] = 0

    if run:
        # Create shape
        X2c_in = {"type": 'grid', "input_value": X2c * (1-mask)}
        X2s_in = {"type": 'grid', "input_value": X2s * (1 - mask)}

        # Solve re-shaped configuration
        stel.X2c_in = X2c_in
        stel.X2s_in = X2s_in
        stel.calculate_r2()

        return stel
    
    else:
        return X2c, X2s


