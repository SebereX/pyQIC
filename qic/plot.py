"""
This module contains a function to plot a near-axis surface.
"""

import numpy as np
from scipy.interpolate import interp2d, interp1d
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clr
from matplotlib.colors import LightSource
import matplotlib.ticker as tck
from .util import to_Fourier
import mplcursors
from tqdm import tqdm
from multiprocessing import Manager, Pool
from scipy.optimize import fsolve
import plotly.graph_objects as go
import io
from PIL import Image
import os

def plot(self, newfigure=True, show=True, savefig=None, plot_geo = False):
    """
    Generate a matplotlib figure with an array of plots, showing the
    toroidally varying properties of the configuration.

    Args:
        newfigure: Whether to create a new matplotlib figure.
        show: Whether to call matplotlib's ``show()`` function after making the plots.
    """
    # Save current plotting parameters to restore afterwards
    original_rcparams = plt.rcParams.copy()
    if newfigure:
        f = plt.figure(figsize=(14, 7))
        plt.rcParams.update({'font.size': 6})
    else:
        f = plt.gcf()
        

    # Dynamically count the number of subplots needed
    plot_count = 0
    def count_subplot(*args, **kwargs):
        nonlocal plot_count
        plot_count += 1

    # Simulate the plotting logic to count subplots
    if not self.frenet and not self.no_cylindrical:
        for _ in range(8): count_subplot()
    if plot_geo and not self.no_cylindrical:
        for _ in range(2): count_subplot()
    for _ in range(12): count_subplot()  # curvature, torsion, sigma, B0, B1s, B1c, X1s, X1c, Y1c, Y1s, elongation, L_grad_B
    count_subplot()  # 1/L_grad_B
    if self.order != 'r1':
        plot_count -= 2
        for _ in range(18): count_subplot()  # L_grad_grad_B, 1/L_grad_grad_B, beta_1s, beta_1c, V1, V2, V3, X20, X2c, X2s, Y20, Y2c, Y2s, Z20, Z2c, Z2s, r_singularity
    if self.omn:
        for _ in range(2): count_subplot()  # alpha, alpha_no_buffer
        if not self.frenet:
            count_subplot()  # d_alpha_der
            count_subplot()  # d_alpha_der_diff
    if self.order != 'r1':
        for _ in range(8): count_subplot()  # B20, B2cQI, B2sQI, B20QI_deviation, B2cQI_deviation, B2sQI_deviation, B2c, B2s
    if self.omn:
        for _ in range(2): count_subplot()  # d_over_curvature, gamma
    if self.order != 'r1':
        if self.order != 'r2':
            for _ in range(4): count_subplot()  # X3c1, X3s1, Y3c1, Y3s1

    # Compute grid size
    ncols = 8
    nrows = int(np.ceil(plot_count / ncols))
    jplot = 1

    def subplot(title, data=None, y0=False):
        """
        data is assumed to correspond to title, unless specified otherwise.
        Set y0 to True to avoid suppressed 0 on the y axis.
        """
        nonlocal jplot
        if data is None:
            data = eval('self.' + title)
        plt.subplot(nrows, ncols, jplot)
        jplot = jplot + 1
        plt.plot(self.phi, data, label=title)
        plt.xlabel(r'$\phi$')
        plt.title(title)
        if y0:
            plt.ylim(bottom=0)
        plt.xlim((0, self.phi[-1]))

    if not self.frenet and not self.no_cylindrical:
        subplot('R0')
        subplot('Z0')
        subplot('R0p')
        subplot('Z0p')
        subplot('R0pp')
        subplot('Z0pp')
        subplot('R0ppp')
        subplot('Z0ppp')
    if plot_geo and not self.no_cylindrical:
        subplot('R0')
        subplot('Z0')
    subplot('curvature')
    subplot('torsion')
    subplot('sigma')
    subplot('B0')
    subplot('B1s')
    subplot('B1c')
    subplot('X1s')
    subplot('X1c')
    subplot('Y1c')
    subplot('Y1s')
    subplot('elongation', y0=True)
    subplot('L_grad_B', y0=True)
    subplot('1/L_grad_B', data=self.inv_L_grad_B)
    # if self.omn:
    #     subplot('alpha')
    if self.order != 'r1':
        jplot -= 2
        subplot('L_grad_grad_B')
        plt.title('scale lengths')
        plt.legend(loc=0, fontsize=5)
        plt.ylim(0, max(max(self.L_grad_B), max(self.L_grad_grad_B)))
        
        subplot('1/L_grad_grad_B', self.grad_grad_B_inverse_scale_length_vs_varphi)
        plt.title('inv scale lengths')
        plt.legend(loc=0, fontsize=5)
        plt.ylim(0, max(max(self.inv_L_grad_B), max(self.grad_grad_B_inverse_scale_length_vs_varphi)))
        
        subplot('beta_1s')
        subplot('beta_1c')
        subplot('V1')
        subplot('V2')
        subplot('V3')
        subplot('X20')
        subplot('X2c')
        subplot('X2s')
        subplot('Y20')
        subplot('Y2c')
        subplot('Y2s')
        subplot('Z20')
        subplot('Z2c')
        subplot('Z2s')
        data = self.r_singularity_vs_varphi
        data[data > 1e20] = np.nan
        subplot('r_singularity', data=data, y0=True)
    if self.omn:
        subplot('alpha')
        jplot -= 1
        subplot('alpha_no_buffer')
        plt.title('alpha')
        plt.legend(loc=0, fontsize=5)

        if not self.frenet:
            plt.subplot(nrows, ncols, jplot)
            d_alpha_iota_d_varphi_der    = [self.d_alpha_iota_d_varphi]
            d_alpha_notIota_d_varphi_der = [self.d_alpha_notIota_d_varphi]
            d_alpha_der = [d_alpha_iota_d_varphi_der[0] * self.iota + d_alpha_notIota_d_varphi_der[0]]
            ders = [1]
            plt.plot(d_alpha_der[0], label='n=1')
            plt.xlabel(r'$\phi$')
            plt.title(r'$\alpha^{(n)}$')
            self.d_alpha_der_diff = [d_alpha_der[0][-1]-d_alpha_der[0][0]]
            for n in range(2,5):
                d_alpha_iota_d_varphi_der.append(np.matmul(self.d_d_varphi,d_alpha_iota_d_varphi_der[n-2]))
                d_alpha_notIota_d_varphi_der.append(np.matmul(self.d_d_varphi,d_alpha_notIota_d_varphi_der[n-2]))
                d_alpha_der.append(d_alpha_iota_d_varphi_der[n-1] * self.iota + d_alpha_notIota_d_varphi_der[n-1])
                self.d_alpha_der_diff.append(d_alpha_der[n-1][-1]-d_alpha_der[n-1][0])
                ders.append(n)
                plt.plot(d_alpha_der[n-1], label='n='+str(n))
            plt.legend()
            jplot += 1

            plt.subplot(nrows, ncols, jplot)
            plt.plot(ders,self.d_alpha_der_diff)
            plt.xlabel(r'$n=$Order of the derivative')
            plt.title(r'$\alpha^{(n)}(2\pi)-\alpha^{(n)}(0)$')
            jplot += 1
    if self.order != 'r1':
        subplot('B20')
        if self.omn:
            # jplot -= 1
            # subplot('B2QI_exact')
            # plt.legend(loc=0, fontsize=5)
            subplot('B2cQI')
            subplot('B2sQI')
            # jplot -= 1
            # subplot('B2QI_exact')
            # plt.legend(loc=0, fontsize=5)
            subplot('B20QI_deviation')
            subplot('B2cQI_deviation')
            subplot('B2sQI_deviation')
            subplot('B2c')
            subplot('B2s')
    if self.omn:
        subplot('d_over_curvature', data = self.d_bar)
        subplot('gamma')
    if self.order != 'r1':
        if self.order != 'r2':
            subplot('X3c1')
            subplot('X3s1')
            subplot('Y3c1')
            subplot('Y3s1')

    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig+'.pdf')
    if show:
        plt.show()
    # Restore the original rcParams after the plot is displayed
    plt.rcParams.update(original_rcparams)

def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Args:
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, colormap, elev=90, azim=45, dist=7, **kwargs):
    '''
    Construct the surface given a surface in cartesian coordinates
    x_2D_plot, y_2D_plot, z_2D_plot already with phi=[0,2*pi].
    A matplotlib figure with elements fig, ax
    must have been previously created.

    Args:
        ax: matplotlib figure instance
        x_2d_plot: 2D array for the x coordinates of the surface
        y_2d_plot: 2D array for the x coordinates of the surface
        z_2d_plot: 2D array for the x coordinates of the surface
        elev: elevation angle for the camera view
        azim: azim angle for the camera view
        distance: distance parameter for the camera view
    '''
    ax.plot_surface(x_2D_plot, y_2D_plot, z_2D_plot, facecolors=colormap,
                    rstride=1, cstride=1, antialiased=False,
                    linewidth=0, alpha=1., shade=False, **kwargs)
    set_axes_equal(ax)
    ax.set_axis_off()
    ax.dist = dist
    ax.elev = elev
    ax.azim = azim

def create_field_lines(qic, alphas, X_2D, Y_2D, Z_2D, phimax=2*np.pi, nphi=500):
    '''
    Function to compute the (X, Y, Z) coordinates of field lines at
    several alphas, where alpha = theta-iota*varphi with (theta,varphi)
    the Boozer toroidal angles. This function relies on a 2D interpolator
    from the scipy library to smooth out the lines

    Args:
      qic: instance of self
      alphas: array of field line labels alpha
      X_2D: 2D array for the x components of the surface
      Y_2D: 2D array for the y components of the surface
      Z_2D: 2D array for the z components of the surface
      phimax: maximum value for the field line following angle phi
      nphi: grid resolution for the output fieldline
    '''
    phi_array = np.linspace(0,phimax,nphi,endpoint=False)
    fieldline_X = np.zeros((len(alphas),nphi))
    fieldline_Y = np.zeros((len(alphas),nphi))
    fieldline_Z = np.zeros((len(alphas),nphi))
    [ntheta_RZ,nphi_RZ] = X_2D.shape
    phi1D   = np.linspace(0,2*np.pi,nphi_RZ)
    theta1D = np.linspace(0,2*np.pi,ntheta_RZ)
    X_2D_spline = interp2d(phi1D, theta1D, X_2D, kind='cubic')
    Y_2D_spline = interp2d(phi1D, theta1D, Y_2D, kind='cubic')
    Z_2D_spline = interp2d(phi1D, theta1D, Z_2D, kind='cubic')
    for i in range(len(alphas)):
        for j in range(len(phi_array)):
            phi_mod = np.mod(phi_array[j],2*np.pi)
            varphi0=qic.nu_spline(phi_array[j])+2*phi_array[j]-phi_mod
            theta_fieldline=qic.iota*varphi0+alphas[i]
            theta_fieldline_mod=np.mod(theta_fieldline,2*np.pi)
            fieldline_X[i,j] = X_2D_spline(phi_mod,theta_fieldline_mod)[0]
            fieldline_Y[i,j] = Y_2D_spline(phi_mod,theta_fieldline_mod)[0]
            fieldline_Z[i,j] = Z_2D_spline(phi_mod,theta_fieldline_mod)[0]
    return fieldline_X, fieldline_Y, fieldline_Z

def create_subplot_mayavi(mlab, R, alphas, x_2D_plot, y_2D_plot, z_2D_plot,
                          fieldline_X, fieldline_Y, fieldline_Z,
                          Bmag, degrees_array_x, degrees_array_z, shift_array):
    '''
    Plotting routine for a mayavi figure instance that plots
    both the surface and the field lines together. The number
    of surfaces to plot is specified by the length of the
    array degrees_array_x

    Args:
      mlab: mayavi package
      R: scipy rotation vector package
      alphas: array of field line labels alpha
      x_2D_plot: 2D array for the x components of the surface
      y_2D_plot: 2D array for the y components of the surface
      z_2D_plot: 2D array for the z components of the surface
      fieldline_X: 2D array for the x components of the field line
      fieldline_Y: 2D array for the x components of the field line
      fieldline_Z: 2D array for the x components of the field line
      Bmag: 2D array for the magnetic field modulus on the (theta,phi) meshgrid
      degrees_array_x: 1D array with the rotation angles in the x direction for each surface
      degrees_array_z: 1D array with the rotation angles in the z direction for each surface
      shift_array: 1D array with a shift in the y direction for each surface
    '''
    assert len(degrees_array_x) == len(degrees_array_z) == len(shift_array)
    for i in range(len(degrees_array_x)):
        # The surfaces and field lines are rotated first in the
        # z direction and then in the x direction
        rx= R.from_euler('x', degrees_array_x[i], degrees=True)
        rz= R.from_euler('z', degrees_array_z[i], degrees=True)
        # Initialize rotated arrays
        x_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        y_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        z_2D_plot_rotated = np.zeros((x_2D_plot.shape[0],x_2D_plot.shape[1]))
        fieldline_X_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        fieldline_Y_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        fieldline_Z_rotated = np.zeros((fieldline_X.shape[0],fieldline_X.shape[1]))
        # Rotate surfaces
        for th in range(x_2D_plot.shape[0]):
            for ph in range(x_2D_plot.shape[1]):
                [x_2D_plot_rotated[th,ph], y_2D_plot_rotated[th,ph], z_2D_plot_rotated[th,ph]] = rx.apply(rz.apply(np.array([x_2D_plot[th,ph], y_2D_plot[th,ph], z_2D_plot[th,ph]])))
        # Rotate field lines
        for th in range(fieldline_X.shape[0]):
            for ph in range(fieldline_X.shape[1]):
                [fieldline_X_rotated[th,ph], fieldline_Y_rotated[th,ph], fieldline_Z_rotated[th,ph]] = rx.apply(rz.apply(np.array([fieldline_X[th,ph], fieldline_Y[th,ph], fieldline_Z[th,ph]])))
        # Plot surfaces
        mlab.mesh(x_2D_plot_rotated, y_2D_plot_rotated-shift_array[i], z_2D_plot_rotated, scalars=Bmag, colormap='viridis', opacity=1.0)
        # Plot field lines
        for j in range(len(alphas)):
            mlab.plot3d(fieldline_X_rotated[j], fieldline_Y_rotated[j]-shift_array[i], fieldline_Z_rotated[j], color=(0,0,0), line_width=0.002, tube_radius=0.008)

def get_boundary(self, r=0.1, ntheta=40, nphi=130, ntheta_fourier=20, mpol=13, ntor=25, phi1d = None, parallel = True):
    '''
    Function that, for a given near-axis radial coordinate r, outputs
    the [X,Y,Z,R] components of the boundary. The resolution along the toroidal
    angle phi is equal to the resolution nphi for the axis, while ntheta
    is specified by the used.

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the poloidal angle.
      nphi   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      mpol: resolution in poloidal Fourier space
      ntor: resolution in toroidal Fourier space
    '''
    # Get surface shape at fixed off-axis toroidal angle phi
    R_2D, Z_2D, _ = self.Frenet_to_cylindrical(r, ntheta=ntheta_fourier, parallel = parallel)
    # Get Fourier coefficients in order to plot with arbitrary resolution
    RBC, RBS, ZBC, ZBS = to_Fourier(R_2D, Z_2D, self.nfp, mpol=mpol, ntor=ntor, lasym=self.lasym)
    if not self.lasym:
        RBS = np.zeros((int(2*ntor+1),int(mpol+1)))
        ZBC = np.zeros((int(2*ntor+1),int(mpol+1)))

    theta1D = np.linspace(0, 2*np.pi, ntheta)
    if isinstance(phi1d, np.ndarray) or isinstance(phi1d, list):
        phi1D = phi1d
        nphi = len(phi1d)
    else:
        phi1D = np.linspace(0, 2*np.pi, nphi)
    phi2D, theta2D = np.meshgrid(phi1D, theta1D)
    R_2Dnew = np.zeros((ntheta, nphi))
    Z_2Dnew = np.zeros((ntheta, nphi))
    for m in range(mpol + 1):
        for n in range(-ntor, ntor + 1):
            angle = m * theta2D - n * self.nfp * phi2D
            R_2Dnew += RBC[n+ntor,m] * np.cos(angle) + RBS[n+ntor,m] * np.sin(angle)
            Z_2Dnew += ZBC[n+ntor,m] * np.cos(angle) + ZBS[n+ntor,m] * np.sin(angle)

    # X, Y, Z arrays for the whole surface
    x_2D_plot = R_2Dnew * np.cos(phi1D)
    y_2D_plot = R_2Dnew * np.sin(phi1D)
    z_2D_plot = Z_2Dnew

    return x_2D_plot, y_2D_plot, z_2D_plot, R_2Dnew

def plot_boundary(self, r=0.1, ntheta=80, nphi=150, ntheta_fourier=20, nsections=8, mpol=13, ntor=25,
         fieldlines=False, savefig=None, colormap=None, azim_default=None, plot_3d=True, n_field_lines=1,
         show=True, axis = None, legend = True, legend_text = None, parallel = True, **kwargs):
    """
    Plot the boundary of the near-axis configuration. There are two main ways of
    running this function.

    If ``fieldlines=False`` (default), 2 matplotlib figures are generated:

        - A 2D plot with several poloidal planes at the specified radius r with the
          corresponding location of the magnetic axis.

        - A 3D plot with the flux surface and the magnetic field strength
          on the surface.

    If ``fieldlines=True``, both matplotlib and mayavi are required, and
    the following 2 figures are generated:

        - A 2D matplotlib plot with several poloidal planes at the specified radius r with the
          corresponding location of the magnetic axis.

        - A 3D mayavi figure with the flux surface the magnetic field strength
          on the surface and several magnetic field lines.

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the poloidal angle.
      nphi   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      nsections (int): Number of poloidal planes to show.
      fieldlines (bool): Specify if fieldlines are shown. Using mayavi instead of matplotlib due to known bug https://matplotlib.org/2.2.2/mpl_toolkits/mplot3d/faq.html
      savefig (str): Filename prefix for the png files to save.
        Note that a suffix including ``.png`` will be appended.
        If ``None``, no figure files will be saved.
      colormap (cmap): Custom colormap for the 3D plots
      azim_default: Default azimuthal angle for the three subplots in the 3D surface plot
      show: Whether or not to call the matplotlib/mayavi ``show()`` command.
      kwargs: Any additional key-value pairs to pass to matplotlib's plot_surface.

    This function generates plots similar to the ones below:

    .. image:: 3dplot1.png
       :width: 200

    .. image:: 3dplot2.png
       :width: 200

    .. image:: poloidalplot.png
       :width: 200
    """
    if self.no_cylindrical:
        plot_boundary_cartesians(self, r, ntheta, nphi, ntheta_fourier, nsections, mpol, ntor,
         fieldlines, savefig, colormap, azim_default, plot_3d, n_field_lines,
         show, axis, legend, legend_text, parallel, **kwargs)
    else:
        x_2D_plot, y_2D_plot, z_2D_plot, R_2D_plot = self.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier, \
                                                                        mpol = mpol, ntor = ntor, parallel = parallel)
        phi = np.linspace(0, 2 * np.pi, nphi)  # Endpoint = true and no nfp factor, because this is what is used in get_boundary()
        R_2D_spline = interp1d(phi, R_2D_plot, axis=1)
        z_2D_spline = interp1d(phi, z_2D_plot, axis=1)
        ## Poloidal plot
        phi1dplot_RZ = np.linspace(0, 2 * np.pi / self.nfp, nsections, endpoint=False)
        if axis == None:
            fig_poloidal = plt.figure(figsize=(7, 5), dpi=80)
            ax  = plt.gca()
        else:
            ax = axis

        flag_color = False
        if "color" in kwargs:
            color = kwargs["color"]
            kwargs.pop("color")
            flag_color = True

        for i, phi in enumerate(phi1dplot_RZ):
            phinorm = phi * self.nfp / (2 * np.pi)
            if phinorm == 0:
                label = r'$\phi$=0'
            elif phinorm == 0.125:
                label = r'$\phi={\pi}/$' + str(4 * self.nfp)
            elif phinorm == 0.25:
                label = r'$\phi={\pi}/$' + str(2 * self.nfp)
            elif phinorm == 0.375:
                label = r'$\phi={3\pi}/$' + str(4 * self.nfp)
            elif phinorm == 0.5:
                label = r'$\phi=\pi/$' + str(self.nfp)
            elif phinorm == 0.625:
                label = r'$\phi={5\pi}/$' + str(4 * self.nfp)
            elif phinorm == 0.75:
                label = r'$\phi={3\pi}/$' + str(2 * self.nfp)
            elif phinorm == 0.875:
                label = r'$\phi={7\pi}/$' + str(4 * self.nfp)
            else:
                label = '_nolegend_'
            if not flag_color:
                color = next(ax._get_lines.prop_cycler)['color'] if hasattr(ax._get_lines, 'prop_cycler') else 'C0'

            # Plot location of the axis
            if not legend_text is None:
                if i == 0:
                    plt.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=legend_text, color=color)
                else:
                    plt.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, color=color)
            else:
                plt.plot(self.R0_func(phi), self.Z0_func(phi), marker="x", linewidth=2, label=label, color=color)
            if plot_3d == True:
                # Plot poloidal cross-section
                plt.plot(R_2D_spline(phi), z_2D_spline(phi), color=color)
            else:
                plt.plot(R_2D_spline(phi), z_2D_spline(phi), color=color, **kwargs)
        plt.xlabel('R [m]', fontsize=14)
        plt.ylabel('Z [m]', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        if legend:
            plt.legend(loc=2, prop={'size': 8})
        plt.tight_layout()
        ax.set_aspect('equal')
        if savefig != None:
            fig_poloidal.savefig(savefig + '_poloidal.pdf')

        ## 3D plot
        # Set the default azimuthal angle of view in the 3D plot
        # QH stellarators look rotated in the phi direction when
        # azim_default = 0
        if plot_3d==True:
            if azim_default == None:
                if self.omn == True:
                    azim_default = -90
                else:
                    if self.helicity == 0:
                        azim_default = 0
                    else:
                        azim_default = 45
                # Define the magnetic field modulus and create its theta,phi array
                # The norm instance will be used as the colormap for the surface
                theta1D = np.linspace(0, 2 * np.pi, ntheta)
                phi1D = np.linspace(0, 2 * np.pi, nphi)
                phi2D, theta2D = np.meshgrid(phi1D, theta1D)
                # Create a color map similar to viridis 
                Bmag = self.B_mag(r, theta2D, phi2D)
                norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
                if fieldlines==False:
                    if colormap==None:
                        # Cmap similar to quasisymmetry papers
                        # cmap = clr.LinearSegmentedColormap.from_list('qs_papers',['#4423bb','#4940f4','#2e6dff','#0097f2','#00bacc','#00cb93','#00cb93','#7ccd30','#fbdc00','#f9fc00'], N=256)
                        cmap = cm.RdBu_r
                        # Add a light source so the surface looks brighter
                        ls = LightSource(azdeg=0, altdeg=10)
                        cmap_plot = ls.shade(Bmag, cmap, norm=norm)
                    # Create the 3D figure and choose the following parameters:
                    # gsParams: extension in the top, bottom, left right directions for each subplot
                    # elevParams: elevation (distance to the plot) for each subplot
                    fig = plt.figure(constrained_layout=False, figsize=(4.5, 8))
                    gsParams = [[1.02,-0.3,0.,0.85], [1.09,-0.3,0.,0.85], [1.12,-0.15,0.,0.85]]
                    elevParams = [90, 30, 5]
                    for i in range(len(gsParams)):
                        gs = fig.add_gridspec(nrows=3, ncols=1,
                                            top=gsParams[i][0], bottom=gsParams[i][1],
                                            left=gsParams[i][2], right=gsParams[i][3],
                                            hspace=0.0, wspace=0.0)
                        ax = fig.add_subplot(gs[i, 0], projection='3d')
                        create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, cmap_plot, elev=elevParams[i], azim=azim_default, **kwargs)
                    # Create color bar with axis placed on the right
                    cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
                    m = cm.ScalarMappable(cmap=cmap, norm=norm)
                    m.set_array([])
                    cbar = plt.colorbar(m, cax=cbar_ax)
                    cbar.ax.set_title(r'$|B| [T]$')
                    # Save figure
                    if savefig != None:
                        fig.savefig(savefig + '3D.png')
                    if show:
                        # Show figures
                        plt.show()
                else:
                    ## X, Y, Z arrays for the field lines
                    # Plot different field lines corresponding to different alphas
                    # where alpha=theta-iota*varphi with (theta,varphi) the Boozer angles
                    #alphas = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
                    alphas = np.linspace(0.7, 2 * np.pi + 0.7, n_field_lines, endpoint=False)
                    # Create the field line arrays
                    fieldline_X, fieldline_Y, fieldline_Z = create_field_lines(self, alphas, x_2D_plot, y_2D_plot, z_2D_plot)
                    # Define the rotation arrays for the subplots
                    degrees_array_x = [0., 81.]#[0., -66., 81.] # degrees for rotation in x
                    degrees_array_z = [azim_default, azim_default]#[azim_default, azim_default, azim_default] # degrees for rotation in z
                    shift_array   = [-0.8, 1.0]#[-0.9, 0.6, 1.8]
                    # Import mayavi and rotation packages (takes a few seconds)
                    from mayavi import mlab
                    from scipy.spatial.transform import Rotation as R
                    if show:
                        # Show RZ plot
                        plt.show()
                    # Create 3D figure
                    fig_3d = mlab.figure(bgcolor=(1,1,1), size=(580,600))
                    # Create subplots
                    create_subplot_mayavi(mlab, R, alphas, x_2D_plot, y_2D_plot, z_2D_plot,
                                        fieldline_X, fieldline_Y, fieldline_Z,
                                        Bmag, degrees_array_x, degrees_array_z, shift_array)
                    # Create a good camera angle
                    mlab.view(azimuth=0, elevation=0, distance=8.0, focalpoint=(0,0,0), figure=fig_3d)
                    # Create the colorbar and change its properties
                    cb = mlab.colorbar(orientation='vertical', title='|B| [T]', nb_labels=7)
                    cb.scalar_bar.unconstrained_font_size = True
                    cb.label_text_property.font_family = 'times'
                    cb.label_text_property.bold = 0
                    cb.label_text_property.font_size=20
                    cb.label_text_property.color=(0,0,0)
                    cb.title_text_property.font_family = 'times'
                    cb.title_text_property.font_size=20
                    cb.title_text_property.color=(0,0,0)
                    cb.title_text_property.bold = 1
                    # Save figure
                    if savefig != None:
                        mlab.savefig(filename=savefig+'3D_fieldlines.png', figure=fig_3d)
                    if show:
                        # Show mayavi plot
                        mlab.show()
                        # Close mayavi plots
                        mlab.close(all=True)
        else:
            if show:
                # Show figures
                plt.show()
        return ax

def get_boundary_cartesians(self, r=0.1, ntheta=40, nphi=130, parallel=True, xsec = True, field_period = False, theta = None, varphi=None, verbose = False):
    '''
    Function that, for a given near-axis radial coordinate r, outputs
    the [X,Y,Z] components of the boundary using the cartesian coords. The resolution along the toroidal
    angle phi is equal to the resolution nphi for the axis, while ntheta
    is specified by the used.

    Parameters
    ----------
    r : float
        near-axis radius r where to create the surface
    ntheta : int
        Number of grid points to plot in the poloidal angle.
    nphi : int
        Number of grid points to plot in the toroidal angle.
    parallel : bool
        Whether to use parallel computation.
    xsec : bool
        Whether to also compute the cross-section in the Frenet-Serret (X,Y) plane.
    field_period : bool, optional
        Whether to consider one field period or the whole toroidal domain. If True, the toroidal angle will be defined in the range [0, 2*pi/nfp] instead of [0, 2*pi].
    theta : array-like, optional
        Array of poloidal angles to use. If None, a default array will be created in the range [0, 2*pi].
    varphi : array-like, optional
        Array of toroidal angles to use. If None, a default array will be created in the range [0, 2*pi] or [0, 2*pi/nfp] depending on the field_period parameter.
    verbose : bool, optional
        Whether to print verbose output during computation.
    
    Returns
    -------
    X_2D : 2D array
        2D array for the x components of the surface (ntheta, nphi)
    Y_2D : 2D array
        2D array for the y components of the surface (ntheta, nphi)
    Z_2D : 2D array
        2D array for the z components of the surface (ntheta, nphi)
    X_fs_2D : 2D array
        2D array for the x components of the surface in the Frenet-Serret frame (ntheta, nphi)
    Y_fs_2D : 2D array
        2D array for the y components of the surface in the Frenet-Serret frame (ntheta, nphi)
    '''
    if self.nfp > 1 and verbose: print('Should be used only for N=1')

    # Get surface shape parametrised by phi (on-axis) and theta
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=True)
    else:
        ntheta = len(theta)
    if varphi is None:
        varphi = np.linspace(0, 2 * np.pi/self.nfp if field_period else 2 * np.pi, nphi, endpoint=True)
    else:
        nphi = len(varphi)

    X_2D = np.zeros((ntheta, nphi))
    Y_2D = np.zeros((ntheta, nphi))
    X_fs_2D = np.zeros((ntheta, nphi))
    Y_fs_2D = np.zeros((ntheta, nphi))
    Z_2D = np.zeros((ntheta, nphi))

    if not parallel:
        with tqdm(desc='Computing different theta...', total=ntheta, disable=not verbose) as pbar:
            for j_theta in range(ntheta):
                costheta = np.cos(theta[j_theta])
                sintheta = np.sin(theta[j_theta])
                X_at_this_theta = r * (self.X1c_untwisted * costheta + self.X1s_untwisted * sintheta)
                Y_at_this_theta = r * (self.Y1c_untwisted * costheta + self.Y1s_untwisted * sintheta)
                Z_at_this_theta = 0 * X_at_this_theta
                if self.order != 'r1':
                    # We need O(r^2) terms:
                    cos2theta = np.cos(2 * theta[j_theta])
                    sin2theta = np.sin(2 * theta[j_theta])
                    X_at_this_theta += r * r * (self.X20_untwisted + self.X2c_untwisted * cos2theta + self.X2s_untwisted * sin2theta)
                    Y_at_this_theta += r * r * (self.Y20_untwisted + self.Y2c_untwisted * cos2theta + self.Y2s_untwisted * sin2theta)
                    Z_at_this_theta += r * r * (self.Z20_untwisted + self.Z2c_untwisted * cos2theta + self.Z2s_untwisted * sin2theta)
                    if self.order == 'r3':
                        # We need O(r^3) terms:
                        costheta = np.cos(theta[j_theta])
                        sintheta = np.sin(theta[j_theta])
                        cos3theta = np.cos(3 * theta[j_theta])
                        sin3theta = np.sin(3 * theta[j_theta])
                        r3 = r * r * r
                        X_at_this_theta += r3 * (self.X3c1_untwisted * costheta + self.X3s1_untwisted * sintheta
                                                + self.X3c3_untwisted * cos3theta + self.X3s3_untwisted * sin3theta)
                        Y_at_this_theta += r3 * (self.Y3c1_untwisted * costheta + self.Y3s1_untwisted * sintheta
                                                + self.Y3c3_untwisted * cos3theta + self.Y3s3_untwisted * sin3theta)
                        Z_at_this_theta += r3 * (self.Z3c1_untwisted * costheta + self.Z3s1_untwisted * sintheta
                                                + self.Z3c3_untwisted * cos3theta + self.Z3s3_untwisted * sin3theta)
                        
                # If half helicity axes are considered, we need to use the extended domain : 
                # within the 2*pi domain of the axis everything is smooth (as is the signed frame)
                # but not across phi = 0
                X_spline = self.convert_to_spline(X_at_this_theta, half_period=self.flag_half, varphi=True)
                Y_spline = self.convert_to_spline(Y_at_this_theta, half_period=self.flag_half, varphi=True)
                Z_spline = self.convert_to_spline(Z_at_this_theta, varphi=True)
        
                X_cart_sp = lambda p: self.x0_cart_spline(p) + X_spline(p) * self.normal_x_cart_spline(p) + \
                                                                Y_spline(p) * self.binormal_x_cart_spline(p) + \
                                                                Z_spline(p) * self.tangent_x_cart_spline(p)
                X_2D[j_theta, :] = X_cart_sp(varphi)
                
                Y_cart_sp = lambda p: self.y0_cart_spline(p) + X_spline(p) * self.normal_y_cart_spline(p) + \
                                                                Y_spline(p) * self.binormal_y_cart_spline(p) + \
                                                                Z_spline(p) * self.tangent_y_cart_spline(p)
                Y_2D[j_theta, :] = Y_cart_sp(varphi)
                
                Z_cart_sp = lambda p: self.z0_cart_spline(p) + X_spline(p) * self.normal_z_cart_spline(p) + \
                                                                Y_spline(p) * self.binormal_z_cart_spline(p) + \
                                                                Z_spline(p) * self.tangent_z_cart_spline(p)
                Z_2D[j_theta, :] = Z_cart_sp(varphi)

                if xsec:
                    # Find the cross-section normal to the axis
                    axis_point = np.array([self.x0_cart_spline(varphi), self.y0_cart_spline(varphi), self.z0_cart_spline(varphi)])
                    fdot = lambda ph: np.einsum('ij,ij->j', np.array([X_cart_sp(ph), Y_cart_sp(ph), Z_cart_sp(ph)]) - axis_point, \
                        np.array([self.tangent_x_cart_spline(varphi), self.tangent_y_cart_spline(varphi), self.tangent_z_cart_spline(varphi)]))
                    root = fsolve(fdot, varphi)

                    # Project to X,Y locally
                    pos = np.array([X_cart_sp(root), Y_cart_sp(root), Z_cart_sp(root)])
                    normal = np.array([self.normal_x_cart_spline(varphi), self.normal_y_cart_spline(varphi), self.normal_z_cart_spline(varphi)])
                    X_proj = np.einsum('ij,ij->j', pos - axis_point, normal)
                    binormal = np.array([self.binormal_x_cart_spline(varphi), self.binormal_y_cart_spline(varphi), self.binormal_z_cart_spline(varphi)])
                    Y_proj = np.einsum('ij,ij->j', pos - axis_point, binormal)
                                                            
                    # X_fs_2D[j_theta, :] = X_spline(varphi)
                    
                    # Y_fs_2D[j_theta, :] = Y_spline(varphi)

                    X_fs_2D[j_theta, :] = X_proj
                    
                    Y_fs_2D[j_theta, :] = Y_proj

                pbar.update(1)
    else:
        # Defining the attributes
        order = self.order
        flag_half = self.flag_half

        # Real space geometry
        normal_x_cart_spline = self.normal_x_cart_spline
        normal_y_cart_spline = self.normal_y_cart_spline
        normal_z_cart_spline = self.normal_z_cart_spline

        binormal_x_cart_spline = self.binormal_x_cart_spline
        binormal_y_cart_spline = self.binormal_y_cart_spline
        binormal_z_cart_spline = self.binormal_z_cart_spline

        tangent_x_cart_spline = self.tangent_x_cart_spline
        tangent_y_cart_spline = self.tangent_y_cart_spline
        tangent_z_cart_spline = self.tangent_z_cart_spline

        x0_cart_spline = self.x0_cart_spline
        y0_cart_spline = self.y0_cart_spline
        z0_cart_spline = self.z0_cart_spline

        # Frenet-Serret space
        X1c_untwisted = self.X1c_untwisted
        X1s_untwisted = self.X1s_untwisted
        Y1c_untwisted = self.Y1c_untwisted
        Y1s_untwisted = self.Y1s_untwisted
        if order != 'r1':
            # We need O(r^2) terms:
            X20_untwisted = self.X20_untwisted
            X2c_untwisted = self.X2c_untwisted
            X2s_untwisted = self.X2s_untwisted
            Y20_untwisted = self.Y20_untwisted
            Y2c_untwisted = self.Y2c_untwisted
            Y2s_untwisted = self.Y2s_untwisted
            Z20_untwisted = self.Z20_untwisted
            Z2c_untwisted = self.Z2c_untwisted
            Z2s_untwisted = self.Z2s_untwisted
            if self.order == 'r3':
                # We need O(r^3) terms:
                X3c1_untwisted = self.X3c1_untwisted
                X3s1_untwisted = self.X3s1_untwisted
                X3c3_untwisted = self.X3c3_untwisted
                X3s3_untwisted = self.X3s3_untwisted
                Y3c1_untwisted = self.Y3c1_untwisted
                Y3s1_untwisted = self.Y3s1_untwisted
                Y3c3_untwisted = self.Y3c3_untwisted
                Y3s3_untwisted = self.Y3s3_untwisted
                Z3c1_untwisted = self.Z3c1_untwisted
                Z3s1_untwisted = self.Z3s1_untwisted
                Z3c3_untwisted = self.Z3c3_untwisted
                Z3s3_untwisted = self.Z3s3_untwisted


        # Define the splining function
        convert_to_spline = self.convert_to_spline

        def residual_calculation(phi0, varphi, x0_cart_spline, y0_cart_spline, z0_cart_spline,
                                    normal_x_cart_spline, normal_y_cart_spline, normal_z_cart_spline,
                                    binormal_x_cart_spline, binormal_y_cart_spline, binormal_z_cart_spline,
                                    tangent_x_cart_spline, tangent_y_cart_spline, tangent_z_cart_spline,
                                X_spline, Y_spline, Z_spline):
            """
            Residual function with explicit arguments instead of self/qic.
            """
            axis_point = np.array([x0_cart_spline(varphi), y0_cart_spline(varphi), z0_cart_spline(varphi)])

            X_cart = x0_cart_spline(phi0) + X_spline(phi0) * normal_x_cart_spline(phi0) + \
                                                    Y_spline(phi0) * binormal_x_cart_spline(phi0) + \
                                                    Z_spline(phi0) * tangent_x_cart_spline(phi0)
                            
            Y_cart = y0_cart_spline(phi0) + X_spline(phi0) * normal_y_cart_spline(phi0) + \
                                                        Y_spline(phi0) * binormal_y_cart_spline(phi0) + \
                                                        Z_spline(phi0) * tangent_y_cart_spline(phi0)
            
            Z_cart = z0_cart_spline(phi0) + X_spline(phi0) * normal_z_cart_spline(phi0) + \
                                                        Y_spline(phi0) * binormal_z_cart_spline(phi0) + \
                                                        Z_spline(phi0) * tangent_z_cart_spline(phi0)
            
            axis_point = np.array([x0_cart_spline(varphi), y0_cart_spline(varphi), z0_cart_spline(varphi)])

            fdot = np.einsum('ij,ij->j', np.array([X_cart, Y_cart, Z_cart]) - axis_point, \
                np.array([tangent_x_cart_spline(varphi), tangent_y_cart_spline(varphi), tangent_z_cart_spline(varphi)]))

            return fdot
        
        def evaluate_func_par(root, varphi, x0_cart_spline, y0_cart_spline, z0_cart_spline,
                                    normal_x_cart_spline, normal_y_cart_spline, normal_z_cart_spline,
                                    binormal_x_cart_spline, binormal_y_cart_spline, binormal_z_cart_spline,
                                    tangent_x_cart_spline, tangent_y_cart_spline, tangent_z_cart_spline,
                                    X_spline, Y_spline, Z_spline):
            """
            Residual function with explicit arguments instead of self/qic.
            """
            axis_point = np.array([x0_cart_spline(varphi), y0_cart_spline(varphi), z0_cart_spline(varphi)])

            # Project to X,Y locally
            X_pos = x0_cart_spline(root) + X_spline(root) * normal_x_cart_spline(root) + \
                                                        Y_spline(root) * binormal_x_cart_spline(root) + \
                                                        Z_spline(root) * tangent_x_cart_spline(root)            
            Y_pos = y0_cart_spline(root) + X_spline(root) * normal_y_cart_spline(root) + \
                                                        Y_spline(root) * binormal_y_cart_spline(root) + \
                                                        Z_spline(root) * tangent_y_cart_spline(root)            
            Z_pos = z0_cart_spline(root) + X_spline(root) * normal_z_cart_spline(root) + \
                                                        Y_spline(root) * binormal_z_cart_spline(root) + \
                                                        Z_spline(root) * tangent_z_cart_spline(root)

            pos = np.array([X_pos, Y_pos, Z_pos])
            normal = np.array([normal_x_cart_spline(varphi), normal_y_cart_spline(varphi), normal_z_cart_spline(varphi)])
            X_proj = np.einsum('ij,ij->j', pos - axis_point, normal)
            binormal = np.array([binormal_x_cart_spline(varphi), binormal_y_cart_spline(varphi), binormal_z_cart_spline(varphi)])
            Y_proj = np.einsum('ij,ij->j', pos - axis_point, binormal)

            return X_proj, Y_proj

        def worker(j_theta):
            costheta = np.cos(theta[j_theta])
            sintheta = np.sin(theta[j_theta])
            X_at_this_theta = r * (X1c_untwisted * costheta + X1s_untwisted * sintheta)
            Y_at_this_theta = r * (Y1c_untwisted * costheta + Y1s_untwisted * sintheta)
            Z_at_this_theta = 0 * X_at_this_theta
            if order != 'r1':
                # We need O(r^2) terms:
                cos2theta = np.cos(2 * theta[j_theta])
                sin2theta = np.sin(2 * theta[j_theta])
                X_at_this_theta += r * r * (X20_untwisted + X2c_untwisted * cos2theta + X2s_untwisted * sin2theta)
                Y_at_this_theta += r * r * (Y20_untwisted + Y2c_untwisted * cos2theta + Y2s_untwisted * sin2theta)
                Z_at_this_theta += r * r * (Z20_untwisted + Z2c_untwisted * cos2theta + Z2s_untwisted * sin2theta)
                if order == 'r3':
                    # We need O(r^3) terms:
                    costheta  = np.cos(theta[j_theta])
                    sintheta  = np.sin(theta[j_theta])
                    cos3theta = np.cos(3 * theta[j_theta])
                    sin3theta = np.sin(3 * theta[j_theta])
                    r3 = r * r * r
                    X_at_this_theta += r3 * (X3c1_untwisted * costheta + X3s1_untwisted * sintheta
                                            + X3c3_untwisted * cos3theta + X3s3_untwisted * sin3theta)
                    Y_at_this_theta += r3 * (Y3c1_untwisted * costheta + Y3s1_untwisted * sintheta
                                            + Y3c3_untwisted * cos3theta + Y3s3_untwisted * sin3theta)
                    Z_at_this_theta += r3 * (Z3c1_untwisted * costheta + Z3s1_untwisted * sintheta
                                            + Z3c3_untwisted * cos3theta + Z3s3_untwisted * sin3theta)
                        
            # If half helicity axes are considered, we need to use the extended domain : 
            # within the 2*pi domain of the axis everything is smooth (as is the signed frame)
            # but not across phi = 0
            X_spline = convert_to_spline(X_at_this_theta, half_period = flag_half, varphi = True)
            Y_spline = convert_to_spline(Y_at_this_theta, half_period = flag_half, varphi = True)
            Z_spline = convert_to_spline(Z_at_this_theta, varphi = True)

            X_2D_j = x0_cart_spline(varphi) + X_spline(varphi) * normal_x_cart_spline(varphi) + \
                                                        Y_spline(varphi) * binormal_x_cart_spline(varphi) + \
                                                        Z_spline(varphi) * tangent_x_cart_spline(varphi)            
            Y_2D_j = y0_cart_spline(varphi) + X_spline(varphi) * normal_y_cart_spline(varphi) + \
                                                        Y_spline(varphi) * binormal_y_cart_spline(varphi) + \
                                                        Z_spline(varphi) * tangent_y_cart_spline(varphi)            
            Z_2D_j = z0_cart_spline(varphi) + X_spline(varphi) * normal_z_cart_spline(varphi) + \
                                                        Y_spline(varphi) * binormal_z_cart_spline(varphi) + \
                                                        Z_spline(varphi) * tangent_z_cart_spline(varphi)

            # Find the cross-section normal to the axis
            root = fsolve(residual_calculation, varphi, args = (varphi, x0_cart_spline, y0_cart_spline, z0_cart_spline,
                                      normal_x_cart_spline, normal_y_cart_spline, normal_z_cart_spline,
                                      binormal_x_cart_spline, binormal_y_cart_spline, binormal_z_cart_spline,
                                      tangent_x_cart_spline, tangent_y_cart_spline, tangent_z_cart_spline,
                                    X_spline, Y_spline, Z_spline))

            # Project to X,Y locally
            X_proj, Y_proj = evaluate_func_par(root, varphi, x0_cart_spline, y0_cart_spline, z0_cart_spline,
                                      normal_x_cart_spline, normal_y_cart_spline, normal_z_cart_spline,
                                      binormal_x_cart_spline, binormal_y_cart_spline, binormal_z_cart_spline,
                                      tangent_x_cart_spline, tangent_y_cart_spline, tangent_z_cart_spline,
                                    X_spline, Y_spline, Z_spline)
                                                        
            return j_theta, X_2D_j, Y_2D_j, Z_2D_j, X_proj, Y_proj
        
        # Use ThreadPoolExecutor for parallel processing
        with Manager() as manager:
            progress_queue = manager.Queue()

            # Define a processing pool
            n_process = os.cpu_count()
            with Pool(processes = n_process) as pool:
                # Create a tqdm progress bar
                with tqdm(total=ntheta, desc="Processing theta", ncols=100) as pbar:
                    # Start processing the tasks
                    results = []
                    for result in pool.imap(worker, range(ntheta)):
                        # Each time a task completes, update the progress bar
                        results.append(result)  # Collect the result
                        pbar.update(1)

        for j_theta, X_2D_j, Y_2D_j, Z_2D_j, X_proj, Y_proj in results:
            X_2D[j_theta, :], Y_2D[j_theta, :], Z_2D[j_theta, :], X_fs_2D[j_theta, :], Y_fs_2D[j_theta, :] = X_2D_j, Y_2D_j, Z_2D_j, X_proj, Y_proj

    return X_2D, Y_2D, Z_2D, X_fs_2D, Y_fs_2D
    
def plot_boundary_cartesians(self, r=0.1, ntheta=80, nphi=150, ntheta_fourier=20, nsections=8, mpol=13, ntor=25,
         fieldlines=False, savefig=None, colormap=None, azim_default=None, plot_3d=True, n_field_lines=1,
         show=True, axis = None, legend = True, legend_text = None, parallel = True, **kwargs):
    """
    Plot the boundary of the near-axis configuration. There are two main ways of
    running this function.

    If ``fieldlines=False`` (default), 2 matplotlib figures are generated:

        - A 2D plot with several poloidal planes at the specified radius r with the
          corresponding location of the magnetic axis.

        - A 3D plot with the flux surface and the magnetic field strength
          on the surface.

    If ``fieldlines=True``, both matplotlib and mayavi are required, and
    the following 2 figures are generated:

        - A 2D matplotlib plot with several poloidal planes at the specified radius r with the
          corresponding location of the magnetic axis.

        - A 3D mayavi figure with the flux surface the magnetic field strength
          on the surface and several magnetic field lines.

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the poloidal angle.
      nphi   (int): Number of grid points to plot in the toroidal angle.
      ntheta_fourier (int): Resolution in the Fourier transform to cylindrical coordinates
      nsections (int): Number of poloidal planes to show.
      fieldlines (bool): Specify if fieldlines are shown. Using mayavi instead of matplotlib due to known bug https://matplotlib.org/2.2.2/mpl_toolkits/mplot3d/faq.html
      savefig (str): Filename prefix for the png files to save.
        Note that a suffix including ``.png`` will be appended.
        If ``None``, no figure files will be saved.
      colormap (cmap): Custom colormap for the 3D plots
      azim_default: Default azimuthal angle for the three subplots in the 3D surface plot
      show: Whether or not to call the matplotlib/mayavi ``show()`` command.
      kwargs: Any additional key-value pairs to pass to matplotlib's plot_surface.

    This function generates plots similar to the ones below:

    .. image:: 3dplot1.png
       :width: 200

    .. image:: 3dplot2.png
       :width: 200

    .. image:: poloidalplot.png
       :width: 200
    """
    # assert self.nfp ==1,'Should be used only for N=1'

    x_2D_plot, y_2D_plot, z_2D_plot, X_fs_2D, Y_fs_2D = self.get_boundary_cartesians(r=r, ntheta=ntheta, nphi=nphi, \
                                                                    parallel = parallel)
    
    phi = np.linspace(0, 2 * np.pi, nphi)  # Endpoint = true and no nfp factor, because this is what is used in get_boundary()
    X_fs_2D_spline = interp1d(phi, X_fs_2D, axis=1)
    Y_fs_2D_spline = interp1d(phi, Y_fs_2D, axis=1)

    ## Poloidal plot
    phi1dplot_RZ = np.linspace(0, 2 * np.pi / self.nfp, nsections, endpoint=False)
    if axis == None:
        fig_poloidal = plt.figure(figsize=(7, 5), dpi=80)
        ax  = plt.gca()
    else:
        ax = axis

    flag_color = False
    if "color" in kwargs:
        color = kwargs["color"]
        kwargs.pop("color")
        flag_color = True

    for i, phi in enumerate(phi1dplot_RZ):
        phinorm = phi * self.nfp / (2 * np.pi)
        if phinorm == 0:
            label = r'$\phi$=0'
        elif phinorm == 0.125:
            label = r'$\phi={\pi}/$' + str(4 * self.nfp)
        elif phinorm == 0.25:
            label = r'$\phi={\pi}/$' + str(2 * self.nfp)
        elif phinorm == 0.375:
            label = r'$\phi={3\pi}/$' + str(4 * self.nfp)
        elif phinorm == 0.5:
            label = r'$\phi=\pi/$' + str(self.nfp)
        elif phinorm == 0.625:
            label = r'$\phi={5\pi}/$' + str(4 * self.nfp)
        elif phinorm == 0.75:
            label = r'$\phi={3\pi}/$' + str(2 * self.nfp)
        elif phinorm == 0.875:
            label = r'$\phi={7\pi}/$' + str(4 * self.nfp)
        else:
            label = '_nolegend_'
        if not flag_color:
            color = next(ax._get_lines.get_next_color())

        if plot_3d == True:
            # Plot poloidal cross-section
            plt.plot(X_fs_2D_spline(phi), Y_fs_2D_spline(phi), color=color)
        else:
            plt.plot(X_fs_2D_spline(phi), Y_fs_2D_spline(phi), color=color, **kwargs)
    plt.xlabel('X [m]', fontsize=14)
    plt.ylabel('Y [m]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    if legend:
        plt.legend(loc=2, prop={'size': 8})
    plt.tight_layout()
    ax.set_aspect('equal')
    if savefig != None:
        fig_poloidal.savefig(savefig + '_poloidal.pdf')

    ## 3D plot
    # Set the default azimuthal angle of view in the 3D plot
    # QH stellarators look rotated in the phi direction when
    # azim_default = 0
    if plot_3d==True:
        if azim_default == None:
            if self.omn == True:
                azim_default = -90
            else:
                if self.helicity == 0:
                    azim_default = 0
                else:
                    azim_default = 45
            # Define the magnetic field modulus and create its theta,phi array
            # The norm instance will be used as the colormap for the surface
            theta1D = np.linspace(0, 2 * np.pi, ntheta)
            phi1D = np.linspace(0, 2 * np.pi, nphi)
            phi2D, theta2D = np.meshgrid(phi1D, theta1D)
            # Create a color map similar to viridis 
            Bmag = self.B_mag(r, theta2D, phi2D, Boozer_toroidal = True)
            norm = clr.Normalize(vmin=Bmag.min(), vmax=Bmag.max())
            if fieldlines==False:
                if colormap==None:
                    # Cmap similar to quasisymmetry papers
                    # cmap = clr.LinearSegmentedColormap.from_list('qs_papers',['#4423bb','#4940f4','#2e6dff','#0097f2','#00bacc','#00cb93','#00cb93','#7ccd30','#fbdc00','#f9fc00'], N=256)
                    cmap = cm.RdBu_r
                    # Add a light source so the surface looks brighter
                    ls = LightSource(azdeg=0, altdeg=10)
                    cmap_plot = ls.shade(Bmag, cmap, norm=norm)
                # Create the 3D figure and choose the following parameters:
                # gsParams: extension in the top, bottom, left right directions for each subplot
                # elevParams: elevation (distance to the plot) for each subplot
                fig = plt.figure(constrained_layout=False, figsize=(4.5, 8))
                gsParams = [[1.02,-0.3,0.,0.85], [1.09,-0.3,0.,0.85], [1.12,-0.15,0.,0.85]]
                elevParams = [90, 30, 5]
                for i in range(len(gsParams)):
                    gs = fig.add_gridspec(nrows=3, ncols=1,
                                        top=gsParams[i][0], bottom=gsParams[i][1],
                                        left=gsParams[i][2], right=gsParams[i][3],
                                        hspace=0.0, wspace=0.0)
                    ax = fig.add_subplot(gs[i, 0], projection='3d')
                    create_subplot(ax, x_2D_plot, y_2D_plot, z_2D_plot, cmap_plot, elev=elevParams[i], azim=azim_default, **kwargs)
                # Create color bar with axis placed on the right
                cbar_ax = fig.add_axes([0.85, 0.2, 0.03, 0.6])
                m = cm.ScalarMappable(cmap=cmap, norm=norm)
                m.set_array([])
                cbar = plt.colorbar(m, cax=cbar_ax)
                cbar.ax.set_title(r'$|B| [T]$')
                # Save figure
                if savefig != None:
                    fig.savefig(savefig + '3D.png')
                if show:
                    # Show figures
                    plt.show()
            else:
                ## X, Y, Z arrays for the field lines
                # Plot different field lines corresponding to different alphas
                # where alpha=theta-iota*varphi with (theta,varphi) the Boozer angles
                #alphas = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
                alphas = np.linspace(0.7, 2 * np.pi + 0.7, n_field_lines, endpoint=False)
                # Create the field line arrays
                fieldline_X, fieldline_Y, fieldline_Z = create_field_lines(self, alphas, x_2D_plot, y_2D_plot, z_2D_plot)
                # Define the rotation arrays for the subplots
                degrees_array_x = [0., 81.]#[0., -66., 81.] # degrees for rotation in x
                degrees_array_z = [azim_default, azim_default]#[azim_default, azim_default, azim_default] # degrees for rotation in z
                shift_array   = [-0.8, 1.0]#[-0.9, 0.6, 1.8]
                # Import mayavi and rotation packages (takes a few seconds)
                from mayavi import mlab
                from scipy.spatial.transform import Rotation as R
                if show:
                    # Show RZ plot
                    plt.show()
                # Create 3D figure
                fig_3d = mlab.figure(bgcolor=(1,1,1), size=(580,600))
                # Create subplots
                create_subplot_mayavi(mlab, R, alphas, x_2D_plot, y_2D_plot, z_2D_plot,
                                    fieldline_X, fieldline_Y, fieldline_Z,
                                    Bmag, degrees_array_x, degrees_array_z, shift_array)
                # Create a good camera angle
                mlab.view(azimuth=0, elevation=0, distance=8.0, focalpoint=(0,0,0), figure=fig_3d)
                # Create the colorbar and change its properties
                cb = mlab.colorbar(orientation='vertical', title='|B| [T]', nb_labels=7)
                cb.scalar_bar.unconstrained_font_size = True
                cb.label_text_property.font_family = 'times'
                cb.label_text_property.bold = 0
                cb.label_text_property.font_size=20
                cb.label_text_property.color=(0,0,0)
                cb.title_text_property.font_family = 'times'
                cb.title_text_property.font_size=20
                cb.title_text_property.color=(0,0,0)
                cb.title_text_property.bold = 1
                # Save figure
                if savefig != None:
                    mlab.savefig(filename=savefig+'3D_fieldlines.png', figure=fig_3d)
                if show:
                    # Show mayavi plot
                    mlab.show()
                    # Close mayavi plots
                    mlab.close(all=True)
    else:
        if show:
            # Show figures
            plt.show()
    return ax

def plot_and_crop_config_3D_nice(stel, r_ref = 0.1, shaded = True, image_name = "Figures/test.png", r_reg = True, view = 'corner', tag = False):
    """
    Plot a nice 3D figure of the configuration and save it as an image, avoiding clipping.

    Parameters
    ----------
    stel: Qic
        NAE object
    r_ref: float
        Reference minor radius for plotting
    shaded: bool
        Whether to use shaded plotting
    image_name: str
        Name of the output image file
    r_reg: bool
        Whether to use reference r or min(r_ref, 0.5*r_singularity)
    view: str
        View angle for the plot.
    Returns
    -------
    fig: plotly.graph_objects.Figure
        The generated 3D plot figure.
    """
    ####################
    # SET DYNAMIC ZOOM #
    #################### 
    positions = []
    def is_clipped(fig, bg_color=(255,255,255), width=800, height=800):
        """
        Check if any part of the 3D plot is clipped against the borders of the image. It checks if the color of the border pixels
        differs from the background color (indicating clipping).

        Parameters
        ----------
        fig: plotly.graph_objects.Figure
            The 3D plot figure.
        bg_color: tuple
            The RGB background color to consider as non-clipped.
        width: int
            Width of the image in pixels.
        height: int
            Height of the image in pixels.
        Returns
        -------
        bool
            True if any part of the plot is clipped, False otherwise.
        """
        # Convert the figure to an image buffer
        buf = io.BytesIO()
        fig.write_image(buf, format='png', width=width, height=height, scale=1)
        buf.seek(0)

        # Load the image from the buffer and register colors
        img = Image.open(buf).convert('RGBA')
        arr = np.array(img)
        alpha = arr[:,:,3]
        rgb = arr[:,:,:3]

        # Borders
        top = rgb[0,:,:]; top_a = alpha[0,:]
        bottom = rgb[-1,:,:]; bottom_a = alpha[-1,:]
        left = rgb[:,0,:]; left_a = alpha[:,0]
        right = rgb[:,-1,:]; right_a = alpha[:,-1]
        border_rgb = np.concatenate([top, bottom, left, right], axis=0)
        border_a = np.concatenate([top_a, bottom_a, left_a, right_a], axis=0)

        # Check for clipping: non-background color with non-zero alpha  
        mask = (border_a > 0) & (np.any(border_rgb != bg_color, axis=-1))
    
        return np.any(mask)

    # Radius for plotting
    r = np.min([r_ref, 0.5 * stel.r_singularity]) if r_reg else r_ref

    # Iterate plotting with decreasing zoom until no clipping is detected
    max_tries = 10
    tried_zoom = 1.1 # Starting zoom
    for _ in range(max_tries):
        # Plot the configuration
        fig = plot_3d(
            stel, ntheta=200, nphi=300, ntheta_fourier=20, r=r,
            show=False, save_fig=False, zoom=tried_zoom, view=view, positions=positions, shaded=shaded
        )
        fig.update_traces(showscale=False)

        # Remove title & extra padding
        fig.update_layout(
            title=None,  # Remove title
            margin=dict(l=0, r=0, t=0, b=0),  # Remove all extra space
            scene=dict(
                xaxis=dict(visible=False),  # Hide x-axis
                yaxis=dict(visible=False),  # Hide y-axis
                zaxis=dict(visible=False),  # Hide z-axis
            ),
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot area
        )

        # Add text on image corner
        if tag:
            fig.add_annotation(
                text=tag,
                xref="paper", yref="paper",
                x=0.95, y=0.05,
                showarrow=False,
                font=dict(size=16, color="black"),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="black",
                borderwidth=1,
            )

        # Check for clipping
        if not is_clipped(fig):
            break
        tried_zoom *= 1.1  # Zoom out if clipped

    # Save the final image
    fig.write_image(image_name, width=800, height=800, scale=1)

    return fig

def plot_3d(stel, ntheta = 200, nphi = 300, ntheta_fourier = 20, r = 0.1, save_fig = False, fig_folder = "temp/", name = 'temp', zoom = 1.0, show = True, view = 'corner', shaded = False, positions = [], fun_surface = None):
    """
    Plot 3D surface of the NAE boundary using Plotly.

    Parameters
    ----------
        stel: QIC class object
            QIC class object.
        ntheta: int, optional
            Number of theta points for the grid. Default is 200.
        nphi: int, optional
            Number of phi points for the grid. Default is 300.
        ntheta_fourier: int, optional
            Number of Fourier modes for theta, used when cylindrical coordinates are used. Default is 20.
        r: float, optional
            Minor radius for the boundary surface. Default is 0.1.
        save_fig: bool, optional
            Whether to save the figure as a PNG file. Default is False.
        fig_folder: str, optional
            Folder to save the figure if save_fig is True. Default is "temp/".
        name: str, optional
            Name of the figure file if saved. Default is 'temp'.
        zoom: float, optional
            Zoom factor for the camera distance. Default is 1.0.
        show: bool, optional
            Whether to display the figure interactively. Default is True.
        view: str, optional
            View angle for the camera. Options are 'corner', 'top', 'side_x', 'side_y', 'top_side_x', 'top_side_y'. Default is 'corner'.
        shaded: bool, optional
            Use gray shading for the surface if True (else |B|). Default is False.
        positions: list, optional
            List to append the (x,y,z) collocation points defining the surface. Default is empty list.
        fun_surface: function, optional
            If provided, this function will be called to compute the color values for the surface. It should take (r, theta_2D, phi_2D) as input and return a 2D array of the same shape for coloring. If not provided, |B| will be used for coloring when shaded=False.
            
    Returns
    -------
        fig: plotly.graph_objects.Figure
            Plotly figure object representing the 3D surface plot.
    """
    # Choose domain for plotting: whole torus
    phi1D = np.linspace(0,2*np.pi, nphi)

    # Get data for the boundary of NAE
    if True: # Use by default the Cartesian construction (even if cylindrical is available)
        x_2D_plot, y_2D_plot, z_2D_plot, _, _ = stel.get_boundary_cartesians(r=r, ntheta=ntheta, nphi=nphi, parallel = False, xsec = False)
    else:
        x_2D_plot, y_2D_plot, z_2D_plot, _ = stel.get_boundary(r=r, ntheta=ntheta, nphi=nphi, ntheta_fourier=ntheta_fourier, phi1d = phi1D)

    # Create poloidal/toroidal grid
    theta1D = np.linspace(0, 2 * np.pi, ntheta)
    phi2D, theta2D = np.meshgrid(phi1D, theta1D)

    # Compute the surface range
    x_range = np.max(x_2D_plot) - np.min(x_2D_plot)
    y_range = np.max(y_2D_plot) - np.min(y_2D_plot)
    z_range = np.max(z_2D_plot) - np.min(z_2D_plot)
    max_range = max(x_range, y_range, z_range)

    # Set a proportional distance for the camera to ensure the plot fits
    eye_distance = 0.5*max_range * zoom  # Adjust multiplier as needed

    # Choose surface color and shading
    if shaded:
        # Use a constant gray color and enable lighting for shading
        surface_kwargs = dict(
            z=z_2D_plot,
            x=x_2D_plot,
            y=y_2D_plot,
            surfacecolor=np.ones_like(z_2D_plot),  # uniform color
            colorscale=[[0, 'gray'], [1, 'gray']],
            showscale=False,
            lighting=dict(ambient=0.5, diffuse=0.8, fresnel=0.1, specular=0.5, roughness=0.5),
            lightposition=dict(x=100, y=200, z=0)
        )
    else:
        # Use Bmag and RdBu_r colormap
        if fun_surface is not None:
            surfacecolor = fun_surface(r, theta2D, phi2D)
        else:
            surfacecolor = stel.B_mag(r, theta2D, phi2D, Boozer_toroidal = bool(stel.nfp == 1))
        surface_kwargs = dict(
            z=z_2D_plot,
            x=x_2D_plot,
            y=y_2D_plot,
            surfacecolor=surfacecolor,
            colorscale='RdYlBu_r',
            colorbar=dict(tickvals=[surfacecolor.min(), surfacecolor.max()])
        )

    # Create the figure
    fig = go.Figure(data=[go.Surface(**surface_kwargs)])

    # Set the camera view
    if view == 'corner':
        view_props = dict(x=eye_distance, y=eye_distance, z=eye_distance)
    elif view == 'top':
        view_props = dict(x=0, y=0, z=2*eye_distance)
    elif view == 'side_x':
        view_props = dict(x=2*eye_distance, y=0, z=0)
    elif view == 'side_y':
        view_props = dict(x=0, y=2*eye_distance, z=0)
    elif view == 'top_side_x':
        view_props = dict(x=2*eye_distance, y=0, z=2*eye_distance)
    elif view == 'top_side_y':
        view_props = dict(x=0, y=2*eye_distance, z=2*eye_distance)
    elif view == 'custom':
        pass
    else:
        raise ValueError(f"Invalid view '{view}'")
    
    # Update layout to remove axes, set background color to white, and use LaTeX font
    if view == 'custom':
        fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'  # lock aspect ratio to data units
        ),
        autosize=True,
    )
        
    else:
        fig.update_layout(
            title=name,
            scene=dict(
                xaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,  # Hide x-axis labels
                    ticks="",  # Remove ticks
                    visible=False  # Completely hide the x-axis
                ),
                yaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,  # Hide y-axis labels
                    ticks="",  # Remove ticks
                    visible=False  # Completely hide the y-axis
                ),
                zaxis=dict(
                    showgrid=False, 
                    zeroline=False, 
                    showticklabels=False,  # Hide z-axis labels
                    ticks="",  # Remove ticks
                    visible=False  # Completely hide the z-axis
                ),
                camera_eye=view_props,  # Set the camera angle
                aspectmode='data'
            ),
            paper_bgcolor='white',  # Set paper background to white
            plot_bgcolor='white',  # Set plot background to white
            font=dict(family='Computer Modern Serif', size=18),  # Use LaTeX-style font
            width=800,
            height=600
        )

    # Return data
    positions.append((x_2D_plot,y_2D_plot, z_2D_plot))

    # Save the figure if required
    if save_fig:
        fig.write_image(fig_folder + name.replace(' ', '_') + '_3d_plot.png', scale=5)

    # Show figure interactively
    if show:
        fig.show()

    return fig

def B_fieldline(self, r=0.1, alpha=0, phimax=None, nphi=400, show=True, savefig=None):
    '''
    Plot the modulus of the magnetic field B along a field line with
    the Boozer toroidal angle varphi acting as a field-line following
    coordinate

    Args:
      r (float): near-axis radius r where to create the surface
      alpha (float): Field-line label
      phimax (float): Maximum value of the field-line following parameter varphi
      nphi (int): resolution of the phi grid
      show (bool): Whether or not to call the matplotlib ``show()`` command.
    '''
    if phimax is None:
        phimax = 30 * np.pi / abs(self.iota)
    varphi_array = np.linspace(0, phimax, nphi)
    fig_B_fieldline, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.xlabel(r'$\varphi$')
    plt.ylabel(r'$B(\varphi)$')
    plt.title("r = " + str(r) + ", alpha = " + str(alpha))
    theta = alpha + self.iota * varphi_array
    plt.plot(varphi_array/np.pi, self.B_mag(r, theta, varphi_array, Boozer_toroidal=True))
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=2))
    plt.tight_layout()
    if savefig != None:
        fig_B_fieldline.savefig(savefig + '_B_fieldline.pdf')
    if show:
        plt.show()

def B_contour(self, r=0.1, ntheta=100, nphi=120, ncontours=20, B0=1, inter = False, ax = None, show=True, fieldline = False, savefig=None, colorbar = True, fill = False, chi = False, **kwargs):
    '''
    Plot contours of constant B, with B the modulus of the
    magnetic field, as a function of Boozer coordinates theta and varphi

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the Boozer poloidal angle.
      nphi   (int): Number of grid points to plot in the Boozer toroidal angle.
      ncontours (int): number of contours to show in the plot
      show (bool): Whether or not to call the matplotlib ``show()`` command.
    '''
    theta_array=np.linspace(0,2*np.pi,ntheta)
    phi_array=np.linspace(0,2*np.pi/self.nfp,nphi)
    theta_2D, phi_2D = np.meshgrid(theta_array,phi_array)
    # If chi is True, then use theta as chi (careful with the sign)
    if chi:
        magB_2D = self.B_mag(r,theta_2D - self.helicity * self.nfp * phi_2D,phi_2D,Boozer_toroidal=True,B0=B0)
    else:
        magB_2D = self.B_mag(r,theta_2D,phi_2D,Boozer_toroidal=True,B0=B0)
    if ax == None:
        fig_B_contour, ax=plt.subplots(1,1)

    # Put all input parameters together 
    linewidths=2.0
    cmap = 'plasma'
    vars = ['linewidths', 'cmap']
    for var in vars:
        if not var in kwargs:
            kwargs[var] = eval(var)
    if 'colors' in kwargs and 'cmap' in kwargs:
        kwargs.pop('cmap')

    if fill:
        contourplot = ax.contourf(self.nfp*phi_2D/np.pi, theta_2D/np.pi, magB_2D, ncontours, **kwargs)
    else:
        contourplot = ax.contour(self.nfp*phi_2D/np.pi, theta_2D/np.pi, magB_2D, ncontours, **kwargs)
    if colorbar:
        plt.colorbar(contourplot)
    
    if fieldline:
        if chi:
            plt.plot(self.nfp*phi_array/np.pi, 1.0 + self.iotaN*(phi_array-np.pi/self.nfp)/np.pi, 'k')
        else:
            plt.plot(self.nfp*phi_array/np.pi, 1.0 + self.iota*(phi_array-np.pi/self.nfp)/np.pi, 'k')

    ax.set_xlabel(r'$\varphi$')
    if chi:
        ax.set_ylabel(r'$\chi$')
    else:
        ax.set_ylabel(r'$\theta$')    
    def fraction_format(x, pos, nfp):
        if x == 0.0:
            return "0"
        elif np.isclose(int(x)-x,0.):
            if x == 1.0:
                if nfp ==1: return f'$\pi$'
                else: return f'$\pi/{nfp}$'
            else:
                if nfp ==1: return f'{int(x)}$\pi$'
                else: return f'{int(x)}$\pi/{nfp}$'
        else:
            num = int(np.round(2*x))
            if int(num/self.nfp) == num/nfp:
                num = int(np.round(num/nfp))
                if num == 1:
                    return f'$\pi/2$'
                else:
                    return f'{num}$\pi/2$'
            else:
                if num == 1:
                    return f'$\pi/{int(2*nfp)}$'
                else:
                    return f'{num}$\pi/{int(2*nfp)}$'
    ax.xaxis.set_major_formatter(tck.FuncFormatter(lambda x, pos: fraction_format(x, pos, self.nfp)))
    ax.yaxis.set_major_formatter(tck.FuncFormatter(lambda x, pos: fraction_format(x, pos, 1)))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.5))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
    if inter:
        # Add mplcursors cursor with a cu# Add annotation for hover display
        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"), fontsize = 8)
        annot.set_visible(False)

        def update_annot(event):
            # Check if the cursor is in the plot area
            if event.inaxes == ax:
                # Get the cursor's data coordinates
                x, y = event.xdata, event.ydata
                
                # Update the annotation
                annot.xy = (x, y)
                annot.set_text(f"{self.nfp}" + r"$\varphi/\pi =$" + f"{x:.2f}, " + r"$\theta/\pi=$" + f"{y:.2f}")
                annot.set_visible(True)
                plt.gcf().canvas.draw_idle()

        # Connect the motion event to the update function
        plt.gcf().canvas.mpl_connect("motion_notify_event", update_annot)
    plt.tight_layout()
    if savefig != None:
        fig_B_contour.savefig(savefig + '_B_contour.pdf')
    if show:
        plt.show()

    return ax

def B_densityplot(self, r=0.1, ntheta=250, nphi=250, B0=1, show=True, savefig=None):
    '''
    Density plot of B, with B the modulus of the
    magnetic field, as a function of Boozer coordinates theta and varphi

    Args:
      r (float): near-axis radius r where to create the surface
      ntheta (int): Number of grid points to plot in the Boozer poloidal angle.
      nphi   (int): Number of grid points to plot in the Boozer toroidal angle.
      show (bool): Whether or not to call the matplotlib ``show()`` command.
    '''
    theta_array=np.linspace(0,2*np.pi,ntheta)
    phi_array=np.linspace(0,2*np.pi/self.nfp,nphi)
    theta_2D, phi_2D = np.meshgrid(theta_array,phi_array)
    magB_2D = self.B_mag(r,theta_2D,phi_2D,Boozer_toroidal=True,B0=B0)
    # contourplot = ax.contour(phi_2D/np.pi, theta_2D/np.pi, magB_2D, ncontours, cmap=cm.plasma, linewidths=2.0)
    fig_B_densityploy, ax=plt.subplots(1,1)
    contourplot = ax.imshow(magB_2D.transpose(), extent=[0, 2/self.nfp, 0, 2], cmap=cm.plasma, aspect='auto')
    fig_B_densityploy.colorbar(contourplot)
    ax.set_title('|B| for r=' + str(r))
    ax.set_xlabel(r'$\varphi$')
    ax.set_ylabel(r'$\theta$')
    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=0.5/self.nfp))
    ax.yaxis.set_major_locator(tck.MultipleLocator(base=0.5))
    plt.tight_layout()
    if savefig != None:
        fig_B_densityploy.savefig(savefig + '_B_densityploy.pdf')
    if show:
        plt.show()

def plot_axis(self, nphi=100, frenet=True, nphi_frenet=80, frenet_factor=0.12, savefig=None, show=True):
    '''
    Plot axis shape and the Frenet-Serret frame along
    the axis (optional). If frenet is true, creates
    a mayavi instance showing the axis and nphi_frenet
    times 3 vectors, corresponding to the tangent, normal and
    binormal vectors. If frenet is false, creates a
    matplotlib instance with only a single axis shape
    curve shown.

    Args:
      nphi (int): Number of grid points in the axis shape
      frenet (bool): True plots the Frenet-Serret frame, False it doesn't
      nphi_frenet (int): Number of Frenet-Serret vectors to show
      frenet_factor (float): Size of Frenet-Serret vectors
      savefig (string): filename to save resulting figure in png format.
        Note that ``.png`` will be appended.
        If ``None``, no figure file will be saved.
      show (bool): Whether or not to call the matplotlib/mayavi ``show()`` command.
    '''
    # Create array of toroidal angles along the axis
    # where the axis points will be created
    phi_array = np.linspace(0, 2 * np.pi, nphi)
    # Calculate the x, y and z components of the axis
    R0 = self.R0_func(phi_array)
    Z0 = self.Z0_func(phi_array)
    x_plot = R0 * np.cos(phi_array)
    y_plot = R0 * np.sin(phi_array)
    z_plot = Z0
    if frenet:
        # Show Frenet-Serret frame
        # Initiate mayavi instance
        from mayavi import mlab
        fig_axis = mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.), size=(650,500))
        # Plot the magnetic axis
        s = mlab.plot3d(x_plot, y_plot, z_plot, color=(0,0,0), line_width=0.001, tube_radius=0.01)
        # Show the x,y,z axis
        ax = mlab.axes(s,xlabel=r'X [m]',ylabel=r'Y [m]',zlabel=r'Z [m]',line_width=1.0,nb_labels=4)
        ax.axes.font_factor = 1.3
        ax.axes.label_format = '    %4.2f'
        ax.label_text_property.bold = False
        ax.label_text_property.italic = False
        # Create array of toroidal angles where the Frenet-Serret is shown
        phi_array = np.linspace(0, 2 * np.pi, nphi_frenet)
        # Calculate origin and vector arrays for the Frenet-Serret frame
        R0 = self.R0_func(phi_array)
        Z0 = self.Z0_func(phi_array)
        x_plot = R0*np.cos(phi_array)
        y_plot = R0*np.sin(phi_array)
        z_plot = Z0
        # Normal vector (red)
        normal_R   = self.normal_R_spline(phi_array)
        normal_phi = self.normal_phi_spline(phi_array)
        normal_Z   = self.normal_z_spline(phi_array)
        normal_X   = normal_R * np.cos(phi_array) - normal_phi * np.sin(phi_array)
        normal_Y   = normal_R * np.sin(phi_array) + normal_phi * np.cos(phi_array)
        mlab.quiver3d(x_plot, y_plot, z_plot,
                      normal_X, normal_Y, normal_Z,
                      scale_factor=frenet_factor,
                      color=(1, 0, 0), reset_zoom=False, name='Normal')
        # Biormal vector (blue)
        binormal_R   = self.binormal_R_spline(phi_array)
        binormal_phi = self.binormal_phi_spline(phi_array)
        binormal_Z   = self.binormal_z_spline(phi_array)
        binormal_X   = binormal_R * np.cos(phi_array) - binormal_phi * np.sin(phi_array)
        binormal_Y   = binormal_R * np.sin(phi_array) + binormal_phi * np.cos(phi_array)
        mlab.quiver3d(x_plot, y_plot, z_plot,
                      binormal_X, binormal_Y, binormal_Z,
                      scale_factor=frenet_factor,
                      color=(0, 0, 1), reset_zoom=False, name='Binormal')
        # Tangent vector (green)
        tangent_R   = self.tangent_R_spline(phi_array)
        tangent_phi = self.tangent_phi_spline(phi_array)
        tangent_Z   = self.tangent_z_spline(phi_array)
        tangent_X   = tangent_R * np.cos(phi_array) - tangent_phi * np.sin(phi_array)
        tangent_Y   = tangent_R * np.sin(phi_array) + tangent_phi * np.cos(phi_array)
        mlab.quiver3d(x_plot, y_plot, z_plot,
                      tangent_X, tangent_Y, tangent_Z,
                      scale_factor=frenet_factor,
                      color=(0, 1, 0),
                      reset_zoom=False, name='Tangent')
        if self.omn: # Show where curvature is zero if in quasi-isodynamism
            # zero_index = np.where(np.diff(self.sign_curvature_change)==-2)[0][0]
            # phi_at_zero = self.phi[zero_index]
            for i in range(2 * self.nfp):
                phi_at_zero = np.pi * i / self.nfp
                R0_at_zero = self.R0_func(phi_at_zero)
                Z0_at_zero = self.Z0_func(phi_at_zero)
                x_at_zero = R0_at_zero * np.cos(phi_at_zero)
                y_at_zero = R0_at_zero * np.sin(phi_at_zero)
                z_at_zero = Z0_at_zero
                mlab.points3d(x_at_zero, y_at_zero, z_at_zero, color=(0.9,0.9,0.9),scale_factor=0.09,figure=fig_axis)
        # Plot legends
        # Mayavi does not have its own legend function
        # Save figure
        if savefig != None:
            mlab.savefig(savefig + '.png')
        if show:
            # Show figure
            mlab.show()
    else:
        # Do not show Frenet-Serret frame
        # Initiate matplotlib instance
        fig_axis = plt.figure(figsize=(6, 5))
        ax = plt.axes(projection='3d')
        # Plot the magnetic axis
        plt.plot(x_plot, y_plot, z_plot)
        set_axes_equal(ax)
        ax.grid(False)
        ax.set_xlabel('X [m]', fontsize=10)
        ax.set_ylabel('Y [m]', fontsize=10)
        ax.set_zlabel('Z [m]', fontsize=10)
        plt.tight_layout()
        fig_axis.subplots_adjust(left=-0.05, top=1.05)
        # Save figure
        if savefig != None:
            fig_axis.savefig(savefig + '.pdf')
        if show:
            # Show figure
            plt.show()


