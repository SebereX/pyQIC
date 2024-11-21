"""
This module contains the routine for computing the precession in QI configurations.
"""

import numpy as np
from .util import mu0
from BAD import bounce_int
from matplotlib import rc
import matplotlib.pyplot as plt
from .fourier_interpolation import fourier_interpolation

# Plotting details
rc('font',**{'family':'serif','serif':['Computer Modern Serif'], 'size': 18})
rc('text', usetex=True)

def nae_geo(stel, r, alpha, gridpoints=1001):
    # phi input is in cylindrical coordinates
    # B x grad(B) . grad(psi)
    # alpha = 0
    phi_start   = 0.0
    phi_end     = 2*np.pi/stel.nfp
    phi         = np.linspace(phi_start, phi_end, gridpoints)

    # Extract basic properties from pyQIC
    B0 = stel.B0
    
    vals = {}
    vals["nu"] = stel.varphi - stel.phi

    var_names = ["B0", "B1c", "B1s", "curvature", "torsion","X1c","X1s","Y1c","Y1s","B20","B2c", "B2s","X20","X2c", \
                "X2s", "Y20", "Y2c", "Y2s", "Z20", "Z2c", "Z2s"]
    for name in var_names:
        vals[name] = getattr(stel,name)

    # Compute derivatives
    var_names = ["B0", "B1c", "B1s", "X1c","X1s","Y1c","Y1s","B20","B2c", "B2s", "X20","X2c", \
                "X2s", "Y20", "Y2c", "Y2s", "Z20", "Z2c", "Z2s"]
    dvar_names = ["dB0_dvarphi","dB1c_dvarphi","dB1s_dvarphi","dX1c_dvarphi","dX1s_dvarphi","dY1c_dvarphi","dY1s_dvarphi","dB20_dvarphi","dB2c_dvarphi","dB2s_dvarphi","dX20_dvarphi","dX2c_dvarphi", \
                "dX2s_dvarphi", "dY20_dvarphi", "dY2c_dvarphi", "dY2s_dvarphi", "dZ20_dvarphi", "dZ2c_dvarphi", "dZ2s_dvarphi"]

    for name, dname in zip(var_names,dvar_names):
        vals[dname] = np.matmul(stel.d_d_varphi, vals[name])

    # Evaluate in the input grid specified
    var_splines = {}
    var_names = ["B0", "B1c", "B1s", "dB0_dvarphi","dB1c_dvarphi","dB1s_dvarphi", \
                "nu", "curvature", "torsion", "X1c", "X1s", "Y1c", "Y1s","X20", "X2c","X2s","Y20","Y2c","Y2s","Z20","Z2c","Z2s", \
                "B20","B2c", "B2s","dX1c_dvarphi","dX1s_dvarphi","dY1c_dvarphi","dY1s_dvarphi","dB20_dvarphi",\
                "dB2c_dvarphi","dB2s_dvarphi","dX20_dvarphi","dX2c_dvarphi", "dX2s_dvarphi", "dY20_dvarphi", "dY2c_dvarphi",\
                "dY2s_dvarphi", "dZ20_dvarphi", "dZ2c_dvarphi", "dZ2s_dvarphi"]
    for name in var_names:
        x = vals[name]
        var_splines[name] = stel.convert_to_spline(x)
        vals[name] = var_splines[name](phi)

    varphi = phi + vals["nu"]
    chi = alpha + stel.iotaN * varphi

    B0 = vals["B0"]
    B1 = vals["B1c"] * np.cos(chi) + vals["B1s"] * np.sin(chi)
    B2 = vals["B20"] + vals["B2c"] * np.cos(2*chi) + vals["B2s"] * np.sin(2*chi)
    dB0_dvarphi = vals["dB0_dvarphi"]
    dB1_dvarphi = vals["dB1c_dvarphi"] * np.cos(chi) + vals["dB1s_dvarphi"] * np.sin(chi)
    dB1_dtheta = -vals["B1c"] * np.sin(chi) + vals["B1s"] * np.cos(chi)
    dB2_dvarphi = vals["dB20_dvarphi"] + vals["dB2c_dvarphi"] * np.cos(2*chi) + vals["dB2s_dvarphi"] * np.sin(2*chi)
    dB2_dtheta = -2*vals["B2c"] * np.sin(2*chi) + 2*vals["B2s"] * np.cos(2*chi)

    Y1 = vals["Y1c"] * np.cos(chi) + vals["Y1s"] * np.sin(chi)
    X1 = vals["X1c"] * np.cos(chi) + vals["X1s"] * np.sin(chi)
    Y2 = vals["Y20"] + vals["Y2c"] * np.cos(2*chi) + vals["Y2s"] * np.sin(2*chi)
    X2 = vals["X20"] + vals["X2c"] * np.cos(2*chi) + vals["X2s"] * np.sin(2*chi)
    Z2 = vals["Z20"] + vals["Z2c"] * np.cos(2*chi) + vals["Z2s"] * np.sin(2*chi)
    dX1_dvarphi = vals["dX1c_dvarphi"] * np.cos(chi) + vals["dX1s_dvarphi"] * np.sin(chi)
    dX1_dtheta = -vals["X1c"] * np.sin(chi) + vals["X1s"] * np.cos(chi)
    dY1_dvarphi = vals["dY1c_dvarphi"] * np.cos(chi) + vals["dY1s_dvarphi"] * np.sin(chi)
    dY1_dtheta = -vals["Y1c"] * np.sin(chi) + vals["Y1s"] * np.cos(chi)
    dX2_dtheta = -2*vals["X2c"] * np.sin(2*chi) + 2*vals["X2s"] * np.cos(2*chi)
    dY2_dtheta = -2*vals["Y2c"] * np.sin(2*chi) + 2*vals["Y2s"] * np.cos(2*chi)
    dZ2_dtheta = -2*vals["Z2c"] * np.sin(2*chi) + 2*vals["Z2s"] * np.cos(2*chi)

    # Evaluate the quantities required for AE to the right order
    BxdBdotdpsi_1 = stel.spsi*B0*B0*B0/stel.Bbar*dB1_dtheta*(Y1*dX1_dtheta - X1*dY1_dtheta)
    dldvarphi = stel.G0/B0
    BxdBdotdpsi_2 = stel.spsi*B0*B0/stel.Bbar*(6*B1*dB1_dtheta*(Y1*dX1_dtheta-X1*dY1_dtheta) + B0*(2*Y2*dB1_dtheta*dX1_dtheta + Y1*dB2_dtheta*dX1_dtheta+\
                        Y1*dB1_dtheta*dX2_dtheta-2*X2*dB1_dtheta*dY1_dtheta + 3*X1*X1*vals["curvature"]*dB1_dtheta*dY1_dtheta-\
                        X1*(3*Y1*vals["curvature"]*dB1_dtheta*dX1_dtheta + dB2_dtheta*dY1_dtheta+dB1_dtheta*dY2_dtheta))+\
                        B0/dldvarphi/dldvarphi*dB0_dvarphi*(Y1*dX1_dtheta-X1*dY1_dtheta)*(-dX1_dvarphi*dX1_dtheta-\
                        dY1_dvarphi*dY1_dtheta + dldvarphi*(Y1*vals["torsion"]*dX1_dtheta - X1*vals["torsion"]*dY1_dtheta-\
                        dZ2_dtheta)))
    
    # BxdBdotdpsi_2_alt = (1/(stel.Bbar*stel.G0**3))*stel.spsi*B0**5*dldvarphi*(6*B1*dldvarphi**2*dB1_dtheta*(Y1*dX1_dtheta - X1*dY1_dtheta) + \
    #                     B0*(-dB0_dvarphi*(Y1*dX1_dtheta - X1*dY1_dtheta)*(dX1_dvarphi*dX1_dtheta + dY1_dvarphi*dY1_dtheta) + \
    #                     dldvarphi**2*(2*Y2*dB1_dtheta*dX1_dtheta + Y1*dB2_dtheta*dX1_dtheta + Y1*dB1_dtheta*dX2_dtheta - \
    #                     2*X2*dB1_dtheta*dY1_dtheta + 3*X1**2*vals["curvature"]*dB1_dtheta*dY1_dtheta - X1*(3*Y1*vals["curvature"]*dB1_dtheta*dX1_dtheta +\
    #                     dB2_dtheta*dY1_dtheta + dB1_dtheta*dY2_dtheta)) + dldvarphi*dB0_dvarphi*(Y1*dX1_dtheta - \
    #                     X1*dY1_dtheta)*(Y1*vals["torsion"]*dX1_dtheta - X1*vals["torsion"]*dY1_dtheta - dZ2_dtheta)))
    # print("Res: ", np.sum(np.abs(BxdBdotdpsi_2-BxdBdotdpsi_2_alt)))
    
    bdotdB_0 = B0*dB0_dvarphi/stel.G0
    bdotdB_1 = (dB0_dvarphi*B1 + B0*(dB1_dvarphi + stel.iotaN * dB1_dtheta))/stel.G0
    bdotdB_2 = (B2*dB0_dvarphi + B1*(dB1_dvarphi + stel.iotaN * dB1_dtheta) + B0*(dB2_dvarphi + stel.iotaN * dB2_dtheta -\
                (stel.G2 + stel.iota*stel.I2)/stel.G0*dB0_dvarphi))/stel.G0

    BxdBdotdalpha_m1 = B0*B0*B0/stel.Bbar/stel.Bbar*B1*(X1*dY1_dtheta-Y1*dX1_dtheta)
    BxdBdotdalpha_0 = (1/(stel.Bbar**2*stel.G0**3))*B0**5*dldvarphi*(6*B1**2*dldvarphi**2*(-Y1*dX1_dtheta + X1*dY1_dtheta) + \
                        B0*(2*B2*dldvarphi**2 - dB0_dvarphi*(2*dldvarphi*Z2 + X1*dX1_dvarphi + Y1*dY1_dvarphi))*(-Y1*dX1_dtheta + \
                        X1*dY1_dtheta) + B0*B1*dldvarphi**2*(-2*Y2*dX1_dtheta - Y1*dX2_dtheta + 2*X2*dY1_dtheta - 3*X1**2*vals["curvature"]*dY1_dtheta +\
                        X1*(3*Y1*vals["curvature"]*dX1_dtheta + dY2_dtheta)))

    BxdBdotdpsi = r*BxdBdotdpsi_1 + r*r*BxdBdotdpsi_2
    BxdBdotdalpha = BxdBdotdalpha_m1/r + BxdBdotdalpha_0
    bdotdB = bdotdB_0 + r*bdotdB_1 + r*r*bdotdB_2

    # mod B
    B   = B0 + r * B1 + r*r * B2

    # Choose as Jacobian the Boozer construction: for a more realistic evaluation using the Jacobian resulting from
    # truncating the near-axis model at second order X,Y,Z. We stick to the expressions in the paper for now
    jac_cheeky = (stel.G0+r*r*(stel.G2+stel.iota*stel.I2))/B/B

    # Surface |B| max/min
    Nchi = 100
    Nphi = 100
    chis = np.linspace(0, 2*np.pi, Nchi)
    phis = np.linspace(0, 2*np.pi/stel.nfp, Nphi)
    phis2D, chis2D = np.meshgrid(phis,chis)
    
    # Less memory requirement than computing all cos beforehand (but slower)
    b = np.zeros([Nchi,Nphi])
    b = var_splines["B0"](phis2D) + \
        r*(var_splines["B1c"](phis2D) * np.cos(chis2D) + var_splines["B1s"](phis2D) * np.sin(chis2D)) + \
        r*r*(var_splines["B20"](phis2D) + var_splines["B2c"](phis2D) * np.cos(2*chis2D) + var_splines["B2s"](phis2D) * np.sin(2*chis2D))
    Bmax = np.max(b)
    Bmin = np.min(b)   
    return varphi, jac_cheeky, B, BxdBdotdpsi, BxdBdotdalpha, Bmax, Bmin

def drift_int(stel, varphi, B, BxdBdotdpsi, BxdBdotdalpha, Bmax_in = None, Bmin_in = None, N_k = 1000, verbose=0, alpha_array = [0.0], name = [], k_chib = False, scale_wa = False):
    B_min = B.min()
    B_max = B.max()
    lam_array = 1/(np.linspace(0.001,1,N_k,endpoint = False) * (B_max-B_min) + B_min)

    res = 0 

    wa_array = np.zeros(N_k)
    wpsi_array = np.zeros(N_k)
    p_fac_array = np.zeros(N_k)
    roots_list = np.zeros(N_k)
    norm_array = np.zeros(N_k)
    for jlam, lam in enumerate(lam_array):
        # In a vacuum
        f = 1 - lam * B
        num_rad_drift = 2*BxdBdotdpsi * (1-0.5*lam*B) / B**4 
        rad_drift, roots  = bounce_int.bounce_integral_wrapper(f,num_rad_drift,varphi,return_roots=True)
        num_pol_drift = 2*BxdBdotdalpha * (1-0.5*lam*B)/ B**4
        pol_drift  = bounce_int.bounce_integral_wrapper(f,num_pol_drift,varphi,return_roots=False)
        num_p_fac = 2* (1-lam*B)/ B**3
        p_fac  = bounce_int.bounce_integral_wrapper(f,num_p_fac,varphi,return_roots=False)
        norm_num = 1/B
        norm  = bounce_int.bounce_integral_wrapper(f,norm_num,varphi,return_roots=False)
        # make into list of lists
        if len(roots) < 2:
            roots_list[jlam] = np.nan
            wa_array[jlam] = np.nan
            wpsi_array[jlam] = np.nan
            norm_array[jlam] = np.nan
            continue
        roots_list[jlam] = roots[1]
        wa_array[jlam] = pol_drift[0]
        wpsi_array[jlam] = rad_drift[0]
        p_fac_array[jlam] = p_fac[0]
        norm_array[jlam] = norm[0]
        # print('Root: ', roots, 'Poloidal drift: ', pol_drift, 'Radial drift: ', rad_drift, norm)

    # Decide which k to use: the bounce point definition (as in Rodriguez, Mackenbach (2023)) or usual Roach definition (default)
    if k_chib == True:
        k = np.sin(0.5*stel.nfp*(roots_list-np.pi/stel.nfp))
        k2 = k*k
    else:
        if not Bmax_in == None and  not Bmin_in == None:
            k2 = (1 - lam_array*Bmin_in)*Bmax_in/(Bmax_in-Bmin_in)
        else:
            k2 = (1 - lam_array*B_min)*B_max/(B_max-B_min)
    wa_array = wa_array/norm_array
    wpsi_array = wpsi_array/norm_array

    lB02r = np.trapz(np.ones(stel.nphi), x=stel.varphi)/np.trapz(1/stel.B0/stel.B0, x=stel.varphi)
    mu0 = 4*np.pi*1e-7
    p_fac_array = 2*mu0*stel.p2/lB02r*p_fac_array/norm_array/stel.Bbar # Factor of 2 because of p' -> 2*p2/Bbar
    wa_tot = wa_array + p_fac_array

    if scale_wa == True:
        G0 = stel.G0
        norm = np.trapz(np.ones(stel.nphi), stel.varphi)
        one_over_B0_squared_avrg = np.trapz(1/stel.B0/stel.B0, stel.varphi)/norm
        one_over_B0_avrg = np.trapz(1/stel.B0, stel.varphi)/norm
        A_ratio = 10
        scale = (G0/A_ratio)**2*one_over_B0_avrg**3/one_over_B0_squared_avrg
        wa_tot = scale * wa_tot

    return k2, wa_tot, wpsi_array, wa_array

def plot_curves_k(k2, wa, ax, title = None, right = False):
    ax.plot(k2, wa, color='k', linewidth=1.5, zorder = 10)   
    ax.set_xlabel(r'$k^2$')
    ax.set_xlim([0, 1])
    ax.grid()
    if right:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.set_title(title)

def plot_curves(x, y, ax, title = None, right = False):
    ax.plot(x, y, color='k', linewidth=1.5)   
    if right:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.set_title(title)

def plot_precession_different_alpha(stel, r, num_plots = 15, multiple = True, scale_wa = False, show = True):
    alpha_arr = np.linspace(0.0,np.pi,num_plots)
    wa_array_avrg = np.zeros(1000)
    for alpha in alpha_arr:
        varphi, jac_cheeky, B, BxdBdotdpsi, BxdBdotdalpha, Bmax, Bmin = nae_geo(stel, r, alpha)
        k2, wa_array, wpsi_array, _ = drift_int(stel, varphi, B, BxdBdotdpsi, BxdBdotdalpha, Bmax_in=Bmax, Bmin_in=Bmin, scale_wa=scale_wa)
        wa_array_avrg += wa_array
        if multiple:
            plt.subplot(2,2,1)
            plt.plot(k2, wa_array,color='lightgray')
            plt.subplot(2,2,2)
            plt.plot(k2, wpsi_array,color='lightgray')
            plt.subplot(2,2,3)
            plt.plot(varphi, B,color='lightgray')
        else:
            plt.plot(k2, wa_array,color='lightgray', zorder = 1) # Flip plots
    varphi, jac_cheeky, B, BxdBdotdpsi, BxdBdotdalpha, Bmax, Bmin = nae_geo(stel, r, np.pi/4)
    k2, wa_array, wpsi_array, _ = drift_int(stel, varphi, B, BxdBdotdpsi, BxdBdotdalpha, Bmax_in=Bmax, Bmin_in=Bmin, scale_wa=scale_wa)
    if multiple:
        ax = plt.subplot(2,2,1)
        plot_curves_k(k2, wa_array_avrg/num_plots, plt.gca(), title = r"$\hat{\omega}_\alpha$")
        # plot_curves_k(k, wa_array, ax, title = r"$\omega_\alpha$")
        ax = plt.subplot(2,2,2)
        plot_curves_k(k2, wpsi_array, ax, title = r"$\omega_\psi$", right = True)
        ax = plt.subplot(2,2,3)
        plot_curves(varphi, B, ax)
        plt.xlabel(r"$\varphi$")
        plt.ylabel(r"$|\mathbf{B}|$")
    else:
        wa_av = wa_array_avrg/num_plots
        plot_curves_k(k2, wa_av, plt.gca(), title = r"$\hat{\omega}_\alpha$")
    title = r'$r=' + str(r) + r'$'
    plt.title(title)
    plt.tight_layout()
    if show:
        plt.show()
    return k2, wa_array, wpsi_array

def maxj_at_bottom(self): 

    def eval_at_bottom(y):
        # Interpolate to the bottom of the well (to take into account the shift of the phi grid)
        pos = [np.pi]
        res = fourier_interpolation(y, pos-self.phi_shift*self.d_phi*self.nfp)
        return res
    
    #########################################################
    # Terms that contribute to B20 at the bottom of the well
    ##########################################################
    # Definitions at the bottom of the well (interp needed)
    B0_b = eval_at_bottom(self.B0)
    d_bar_b = eval_at_bottom(self.d_bar)
    a = d_bar_b*d_bar_b*d_bar_b*d_bar_b*B0_b*B0_b/self.Bbar/self.Bbar
    d_dbar = np.matmul(self.d_d_varphi, self.d_bar)
    d2_d_bar = np.matmul(self.d_d_varphi,d_dbar)
    d2_d_bar_b = eval_at_bottom(d2_d_bar)
    d_B0 = np.matmul(self.d_d_varphi, self.B0)
    d2_B0 = np.matmul(self.d_d_varphi,d_B0)
    d2_B0_b = eval_at_bottom(d2_B0)
    dl_dp = eval_at_bottom(self.d_l_d_varphi)

    # Pressure factor
    mu0 = 4*np.pi*1e-7
    f0_p2 = -mu0/B0_b/B0_b

    # dd'' term
    P_dd_over_d2_d_bar = d_bar_b*(1-1/a) #There was a factor of 1/2 here
    P_dd = d2_d_bar_b*P_dd_over_d2_d_bar
    P_dd_norm = P_dd/(4*dl_dp**2)

    # B0'' term
    BodB = self.Bbar/d_bar_b/B0_b
    P_ddB0 = -BodB**2*d2_B0_b/B0_b
    P_ddB0_norm = P_ddB0/(4*dl_dp**2)

    # torsion**2 term
    torsion_b = eval_at_bottom(self.torsion)
    dtl = d_bar_b*torsion_b*dl_dp
    P_t2 = (3-1/a)*dtl*dtl
    P_t2_norm = P_t2/(4*dl_dp**2)

    # I2 term
    P_I2 = -4*dl_dp**2*d_bar_b**2*torsion_b*self.I2/self.Bbar
    P_I2_norm = P_I2/(4*dl_dp**2)

    # QI correction 
    qi_term = np.matmul(self.d_d_varphi, self.B0*self.B0*self.d*self.d/d_B0)
    qi_term = -0.25/B0_b*eval_at_bottom(qi_term)
    P_qi = qi_term * (4*dl_dp**2) # Make dimensionless like other Ps

    # Print information and check B20 construction
    f0_b = f0_p2*self.p2 + (P_dd + P_ddB0 + P_t2 + P_I2)/(4*dl_dp**2)

    # Critical beta
    p2_critical = -(P_qi + P_dd + P_ddB0 + P_t2 + P_I2)/(4*dl_dp**2) / f0_p2

    # Construct near-axis beta0: assume quadratic profile p = p0(1-psi/psi_edge) = p0 + r**2 p2 so 
    # p2 = -Bbar p0 / (2 psi_edge) and for psi_edge we have Eq. (3.4). We may then simply define 
    # beta0 = 2 mu0 p0/ Bbar**2 = - 4 psi_edge mu0 p2/ Bbar**3.
    def define_edge_flux_nae(A_ratio):
        G0 = self.G0
        norm = np.trapz(np.ones(self.nphi), self.varphi)
        one_over_B0_squared_avrg = np.trapz(1/self.B0/self.B0, self.varphi)/norm
        one_over_B0_avrg = np.trapz(1/self.B0, self.varphi)/norm
        psi_edge = 0.5 * (G0/A_ratio)**2 * one_over_B0_avrg**3/one_over_B0_squared_avrg
        return psi_edge
    
    psi_edge = define_edge_flux_nae(A_ratio = 10)
    mu0 = 4*np.pi*1e-7
    beta0_crit = -4 * psi_edge * mu0 * p2_critical / self.Bbar**3

    if self.order == 'r2' or self.order == 'r3':
        f0 = self.B20/self.B0
        f0_b_real = eval_at_bottom(f0)
    else:
        f0_b_real = np.nan

    return [P_dd_norm, P_ddB0_norm, P_t2_norm, P_I2_norm, qi_term, beta0_crit, f0_b, f0_b_real]