#!/usr/bin/env python3

"""
Given the Frenet description of a curve, construct its cylindrical coordinate form.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

def frenet_serret(t, y, kappa_func, tau_func):
    # Frenet-Serret system of equations

    ## Define basis vectors ##
    T = y[:3]
    N = y[3:6]
    B = y[6:9]
    ## Curvature and torsion discrete ##
    kappa = kappa_func(t)
    tau = tau_func(t)
    
    ## Compute derivatives ##
    dT_dt = kappa * N
    dN_dt = -kappa * T + tau * B
    dB_dt = -tau * N
    
    return np.concatenate((dT_dt, dN_dt, dB_dt))

def solve_frenet_serret(kappa, tau, ell):
    # Initial conditions: T0, N0, B0
    T0 = np.array([1, 0, 0])
    N0 = np.array([0, 1, 0])
    B0 = np.array([0, 0, 1])
    
    y0 = np.concatenate((T0, N0, B0))
    
    # Periodic spline interpolation for kappa and tau
    kappa_func = PchipInterpolator(ell, kappa, axis=0, extrapolate='periodic')
    tau_func = PchipInterpolator(ell, tau, axis=0, extrapolate='periodic')

    # Solve ODE
    solution = solve_ivp(frenet_serret, [ell[0], ell[-1]], y0, t_eval=ell, args=(kappa_func, tau_func), method='RK45')
    
    # Separate basis vectors
    T = solution.y[:3].T
    N = solution.y[3:6].T
    B = solution.y[6:9].T
    
    return T, N, B

def integrate_tangent(T, ell):
    # Integrate the tangent vector to get the position
    position = np.zeros((len(ell), 3))
    for i in range(1, len(ell)):
        dL = ell[i] - ell[i-1]
        position[i] = position[i-1] + T[i-1] * dL
    return position

def cartesian_to_cylindrical(position):
    # Construct cylindrical (R, Theta, Z)
    R = np.sqrt(position[:,0]**2 + position[:,1]**2)
    Theta = np.arctan2(position[:,1], position[:,0])
    Z = position[:,2]
    return R, Theta, Z

def align_with_min_z_excursion(position):
    # Center the curve at the origin
    centroid = np.mean(position, axis=0)
    centered_position = position - centroid
    
    # Perform principal component decomposition
    pca = PCA(n_components=3)
    pca.fit(centered_position)
    rotation_matrix = pca.components_ # Orthonormal matrix with the three axes: the latest with lowest variation our Z axis

    # Check that the system is right handed
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[2,:] *= -1

    # Transform curve
    transformed_position = np.einsum('ij,kj->ki', rotation_matrix, centered_position)  
    
    return transformed_position, centroid, rotation_matrix

def invert_frenet_axis(self, curvature, torsion, ell, varphi, plot = False, phi_cyl = None, full_axis = True):
    ## Complete the curve ##
    # The inputs are assumed to be in the [0,2pi/N) grid. We extend them in field period and to include the last point

    if full_axis:
        # Torsion
        tau = torsion.copy()
        if isinstance(tau, list):
            tau += (self.nfp - 1) * torsion
            tau += [tau[0]]   # Enpoint
        elif isinstance(tau, np.ndarray):
            for i in range(self.nfp - 1):
                # Important to add a sign from period to period (because of half helicity)
                tau = np.concatenate((tau, torsion))
            tau = np.append(tau, tau[0])
        else:
            raise TypeError('Invalid type for torsion input.')

        # Curvature: in this case we need to take the negative sign flip between field periods
        kappa = curvature.copy()
        if isinstance(kappa, list):
            kappa = np.array(kappa)
            curvature = np.array(curvature)
        if isinstance(kappa, np.ndarray):
            for j in range(self.nfp - 1):
                # Important to add a sign from period to period (because of half helicity)
                kappa = np.concatenate((kappa, (-1)**(j+1)*curvature))
            kappa = np.append(kappa, kappa[0])
        else:
            raise TypeError('Invalid type for curvature input.')

        # ell
        ell_in = ell.copy()
        ell = ell_in.copy()
        if isinstance(ell, list):
            ell = np.array(ell)
            ell_in = np.array(ell)   # Enpoint
        if isinstance(ell, np.ndarray):
            for i in range(self.nfp - 1):
                ell = np.concatenate((ell, ell_in + (i+1)*self.L_in))
            ell = np.append(ell, self.nfp * self.L_in)
        else:
            raise TypeError('Invalid type for torsion input.')

        # varphi
        varphi_in = varphi.copy()
        varphi = varphi_in.copy()
        if isinstance(varphi, list):
            varphi = np.array(varphi)
            varphi_in = np.array(varphi)   # Enpoint
        if isinstance(ell, np.ndarray):
            for i in range(self.nfp - 1):
                varphi = np.concatenate((varphi, varphi_in + (i+1)*2*np.pi/self.nfp))
            varphi = np.append(varphi, 2*np.pi)
        else:
            raise TypeError('Invalid type for torsion input.')
        
    else:
        tau = torsion.copy()
        kappa = curvature.copy()
        ell = ell.copy()
        varphi = varphi.copy()

    ##############################
    # SOLVE FRENET-SERRET SYSTEM #
    ##############################
    T, N, B = solve_frenet_serret(kappa, tau, ell)
    position = integrate_tangent(T, ell)

    #####################
    # CHOOSE THE Z AXIS #
    #####################
    # Align the curve to minimize Z excursion
    aligned_position, _, rotation_matrix = align_with_min_z_excursion(position)

    ######################################
    # CONVERT TO CYLINDRICAL COORDINATES #
    ######################################
    # Convert curve position to cylindrical coordinates
    R, phi, Z = cartesian_to_cylindrical(aligned_position)

    # Rotate the Frenet basis vectors accordingly
    aligned_T = np.einsum('ji,ki->kj', rotation_matrix, T)
    aligned_N = np.einsum('ji,ki->kj', rotation_matrix, N)
    aligned_B = np.einsum('ji,ki->kj', rotation_matrix, B)

    # Check whether the sense of the axis is in the positive cylindrical angle
    sense_axis = (phi[1] % (2*np.pi)) - (phi[0] % (2*np.pi))
    if sense_axis < 0:
        # Change coordinates accordingly
        Z = -Z
        phi = -phi

        aligned_position[:,1] *= -1
        aligned_T[:,1] *= -1
        aligned_N[:,1] *= -1
        aligned_B[:,1] *= -1

        aligned_position[:,2] *= -1
        aligned_T[:,2] *= -1
        aligned_N[:,2] *= -1
        aligned_B[:,2] *= -1
    
    # Need to translate aligned vectors to cylindrical coordinates
    def change_vector_to_cylindrical(phi, vector):
        new_vector = np.zeros(np.shape(vector))
        new_vector[:,0] = np.cos(phi) * vector[:,0] + np.sin(phi) * vector[:,1] 
        new_vector[:,1] = -np.sin(phi) * vector[:,0] + np.cos(phi) * vector[:,1] 
        new_vector[:,2] = vector[:,2] 

        return new_vector

    aligned_T = change_vector_to_cylindrical(phi, aligned_T)
    aligned_N = change_vector_to_cylindrical(phi, aligned_N)
    aligned_B = change_vector_to_cylindrical(phi, aligned_B)

    # Redefine the cylindrical angle so that the first point is phi = 0 (the cylindrical representation should not change)
    phi = phi - phi[0]

    ################
    # SPLINES OF r #
    ################
    # Function to make splines
    def make_spline(grid, array, periodic = True):
        if periodic:
            wrapped_grid = grid[:-1] % (2*np.pi)
            # ind_wrap = np.argsort(wrapped_grid)
            if len(np.shape(array)) > 1:
                sp = []
                for j in range(3):
                    sp.append([PchipInterpolator(wrapped_grid, array[:-1,j], axis=0, extrapolate='periodic')])
            else:
                sp = PchipInterpolator(wrapped_grid, array[:-1], axis=0, extrapolate='periodic')
        else:
            wrapped_grid = grid[:-1] % (2*np.pi)
            sp = PchipInterpolator(wrapped_grid, array[:-1])
        return sp

    # Periodic spline interpolation for kappa and tau
    self.R0_func = make_spline(phi, R)
    self.Z0_func = make_spline(phi, Z)

    curvature_func = make_spline(phi, kappa)
    torsion_func = make_spline(phi, tau)
    ell_func = make_spline(phi, ell, periodic = False)
    varphi_func = make_spline(phi, varphi, periodic = False)

    # Due to sign, for half helicities, the configurations have sign flips in normal/binormal
    self.normal_R_spline = make_spline(phi, aligned_N[:,0], periodic = False)
    self.normal_phi_spline = make_spline(phi, aligned_N[:,1], periodic = False)
    self.normal_z_spline = make_spline(phi, aligned_N[:,2], periodic = False)
    self.binormal_R_spline = make_spline(phi, aligned_B[:,0], periodic = False)
    self.binormal_phi_spline = make_spline(phi, aligned_B[:,1], periodic = False)
    self.binormal_z_spline = make_spline(phi, aligned_B[:,2], periodic = False)
    self.tangent_R_spline = make_spline(phi, aligned_T[:,0])
    self.tangent_phi_spline = make_spline(phi, aligned_T[:,1])
    self.tangent_z_spline = make_spline(phi, aligned_T[:,2])

    ##############################
    # EVALUATE ON A REGULAR GRID #
    ##############################
    # Create an evenly spaced cylindrical angle in which to sample the geometry of the acis
    if isinstance(phi_cyl, list) or isinstance(phi_cyl, np.ndarray):
        phi_new = np.array(phi_cyl)
        nphi = len(phi_cyl)
    elif hasattr(self, "phi"):
        phi_new = self.phi
        nphi = self.nphi
    else:
        nphi = self.nphi
        phi_new = np.linspace(0, 2*np.pi/self.nfp, nphi) # Note here I choose the period

    # # The alternative would be to use, but leave it for now because not sure what the best final form of approach is
    # phi_new = phi
    # nphi = len(phi)

    # Evaluate geometry
    self.R0 = self.R0_func(phi_new)
    self.Z0 = self.Z0_func(phi_new)

    self.normal_cylindrical = np.zeros((nphi, 3))
    self.normal_cylindrical[:,0] = self.normal_R_spline(phi_new)
    self.normal_cylindrical[:,1] = self.normal_phi_spline(phi_new)
    self.normal_cylindrical[:,2] = self.normal_z_spline(phi_new)

    self.binormal_cylindrical = np.zeros((nphi, 3))
    self.binormal_cylindrical[:,0] = self.binormal_R_spline(phi_new)
    self.binormal_cylindrical[:,1] = self.binormal_phi_spline(phi_new)
    self.binormal_cylindrical[:,2] = self.binormal_z_spline(phi_new)

    self.tangent_cylindrical = np.zeros((nphi, 3))
    self.tangent_cylindrical[:,0] = self.tangent_R_spline(phi_new)
    self.tangent_cylindrical[:,1] = self.tangent_phi_spline(phi_new)
    self.tangent_cylindrical[:,2] = self.tangent_z_spline(phi_new)

    # Resample the curvature and torsion: this gives potential issues with interpolation etc.
    # self.phi = phi # if we insist on a regular varphi grid
    self.curvature = curvature_func(phi_new)
    self.torsion = torsion_func(phi_new)
    self.ell = ell_func(phi_new)
    self.varphi = varphi_func(phi_new)

    #############
    # PLOT AXIS #
    #############
    if plot:
        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plotting normal and binormal vectors as arrows
        origin = [self.R0 * np.cos(phi_new), self.R0 * np.sin(phi_new), self.Z0]

        # print(Theta)
        # Plot curve in cylindrical coordinates
        ax.plot(origin[0],origin[1],origin[2], label='Curve')

        # Plotting normal and binormal vectors as arrows
        stp = 4
        num = int(nphi/stp)

        for i in range(num):
            phi_i = phi_new[i*stp]
            ax.quiver(origin[0][i*stp], origin[1][i*stp], origin[2][i*stp], 
                    self.normal_cylindrical[i*stp,0]*np.cos(phi_i) - self.normal_cylindrical[i*stp,1]*np.sin(phi_i), \
                    self.normal_cylindrical[i*stp,0]*np.sin(phi_i) + self.normal_cylindrical[i*stp,1]*np.cos(phi_i), \
                    self.normal_cylindrical[i*stp,2], 
                    color='r', length=0.05, normalize=True, arrow_length_ratio=0.3)
            ax.quiver(origin[0][i*stp], origin[1][i*stp], origin[2][i*stp], 
                    self.binormal_cylindrical[i*stp,0]*np.cos(phi_i) - self.binormal_cylindrical[i*stp,1]*np.sin(phi_i), \
                    self.binormal_cylindrical[i*stp,0]*np.sin(phi_i) + self.binormal_cylindrical[i*stp,1]*np.cos(phi_i), \
                    self.binormal_cylindrical[i*stp,2], 
                    color='g', length=0.05, normalize=True, arrow_length_ratio=0.3)
            ax.quiver(origin[0][i*stp], origin[1][i*stp], origin[2][i*stp], 
                      self.tangent_cylindrical[i*stp,0]*np.cos(phi_i) - self.tangent_cylindrical[i*stp,1]*np.sin(phi_i), \
                      self.tangent_cylindrical[i*stp,0]*np.sin(phi_i) + self.tangent_cylindrical[i*stp,1]*np.cos(phi_i), \
                      self.tangent_cylindrical[i*stp,2], 
                      color='b', length=0.05, normalize=True, arrow_length_ratio=0.3)
            
        # Set equal scale for all axes
        def _set_axes_radius(ax, origin, radius):
            x, y, z = origin
            ax.set_xlim3d([x - radius, x + radius])
            ax.set_ylim3d([y - radius, y + radius])
            ax.set_zlim3d([z - radius, z + radius])

        def set_axes_equal(ax: plt.Axes):
            """Set 3D plot axes to equal scale.

            Make axes of 3D plot have equal scale so that spheres appear as
            spheres and cubes as cubes.  Required since `ax.axis('equal')`
            and `ax.set_aspect('equal')` don't work on 3D.
            """
            limits = np.array([
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ])
            origin = np.mean(limits, axis=1)
            radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
            _set_axes_radius(ax, origin, radius)

        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Curve with Normal and Binormal Vectors')
        plt.show()

def to_Fourier_axis(R0, Z0, nfp, ntor, lasym, full = True):
    """
    This function takes two 1D arrays (R0 and Z0), which contain
    the values of the radius R and vertical coordinate Z in cylindrical
    coordinates of the magnetic axis and Fourier transform it, outputing
    the resulting cos(theta) and sin(theta) Fourier coefficients

    Args:
        R0: 1D array of the radial coordinate R(phi) of the axis
        Z0: 1D array of the vertical coordinate Z(phi) of the axis
        nfp: number of field periods of the axis
        ntor: resolution in toroidal Fourier space
        lasym: False if stellarator-symmetric, True if not
    """
    nphi = len(R0)
    if full:
        phi_conversion = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    else:
        phi_conversion = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)

    # Harmonic components of axis position
    rc = np.zeros(int(ntor + 1))
    rs = np.zeros(int(ntor + 1))
    zc = np.zeros(int(ntor + 1))
    zs = np.zeros(int(ntor + 1))
    factor = 2 / nphi

    for n in range(1, ntor+1):
        angle = - n * nfp * phi_conversion
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        factor2 = factor
        # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
        # if n == 0: factor2 = factor2 / 2
        rc[n] = np.sum(R0 * cosangle * factor2)
        rs[n] = np.sum(R0 * sinangle * factor2)
        zc[n] = np.sum(Z0 * cosangle * factor2)
        zs[n] = np.sum(Z0 * sinangle * factor2)

    rc[0] = np.sum(R0) / nphi
    zc[0] = np.sum(Z0) / nphi

    if not lasym:
        rs = 0
        zc = 0

    return rc, rs, zc, zs
