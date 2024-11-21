#!/usr/bin/env python3

"""
Given the Frenet description of a curve, construct its cylindrical coordinate form.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from scipy.integrate import solve_ivp, cumulative_trapezoid, quad
from scipy.interpolate import PchipInterpolator, make_interp_spline
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
# from .util_interp import convert_to_spline

logger = logging.getLogger(__name__)

def smooth_fourier(data, nfp, n_harm, phi_out, even = True):
    # Input grid
    nphi = len(data)
    phi_in = np.linspace(0, 2 * np.pi, nphi, endpoint=False)

    # Harmonic components of axis position
    data_harm = np.zeros(int(n_harm + 1))
    data_harm[0] = np.sum(data) / nphi            
    factor = 2 / nphi
    if even:
        smoothed_data = np.full(len(phi_out), fill_value = data_harm[0])
    else:
        smoothed_data = np.zeros(len(phi_out))

    for n in range(1, n_harm+1):
        angle = n * nfp * phi_in
        factor2 = factor
        # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
        # if n == 0: factor2 = factor2 / 2
        if even:
            cosangle = np.cos(angle)
            data_harm[n] = np.sum(data * cosangle * factor2)
            smoothed_data += data_harm[n] * np.cos(n * nfp * phi_out)
        else:
            sinangle = np.sin(angle)
            data_harm[n] = np.sum(data * sinangle * factor2)
            smoothed_data += data_harm[n] * np.sin(n * nfp * phi_out)

    return smoothed_data

def frenet_serret(t, y, kappa_func, tau_func):
    # Frenet-Serret system of equations

    ## Define basis vectors ##
    R = y[:3]
    T = y[3:6]
    N = y[6:9]
    B = y[9:12]
    
    ## Curvature and torsion discrete ##
    kappa = kappa_func(t)
    tau = tau_func(t)
    
    ## Compute derivatives ##
    dR_dt = T
    dT_dt = kappa * N
    dN_dt = -kappa * T + tau * B
    dB_dt = -tau * N
    
    return np.concatenate((dR_dt, dT_dt, dN_dt, dB_dt))

def solve_frenet_serret(kappa, tau, ell):
    # Initial conditions: T0, N0, B0
    R0 = np.array([0, 0, 0])
    T0 = np.array([1, 0, 0])
    N0 = np.array([0, 1, 0])
    B0 = np.array([0, 0, 1])
    
    y0 = np.concatenate((R0, T0, N0, B0))
    
    if callable(kappa) and callable(tau):
        kappa_func = kappa
        tau_func = tau
    else:
        # Periodic spline interpolation for kappa and tau
        kappa_func = PchipInterpolator(ell, kappa, axis=0)
        tau_func = PchipInterpolator(ell, tau, axis=0)

    # Solve ODE
    solution = solve_ivp(frenet_serret, [ell[0], ell[-1]], y0, t_eval=ell, args=(kappa_func, tau_func), \
                         method='DOP853', rtol = 1e-13, atol = 1e-13)
    
    # Separate basis vectors
    R = solution.y[:3].T
    T = solution.y[3:6].T
    N = solution.y[6:9].T
    B = solution.y[9:12].T
    
    return T, N, B, R

def solve_frenet_serret_fun(kappa, tau, ell):
    # Initial conditions: T0, N0, B0
    R0 = np.array([0, 0, 0])
    T0 = np.array([1, 0, 0])
    N0 = np.array([0, 1, 0])
    B0 = np.array([0, 0, 1])
    
    y0 = np.concatenate((R0, T0, N0, B0))
    
    if callable(kappa) and callable(tau):
        kappa_func = kappa
        tau_func = tau
    else:
        # Periodic spline interpolation for kappa and tau
        kappa_func = PchipInterpolator(ell, kappa, axis=0)
        tau_func = PchipInterpolator(ell, tau, axis=0)

    # Solve ODE
    solution = solve_ivp(frenet_serret, [ell[0], ell[-1]], y0, t_eval=ell, args=(kappa_func, tau_func), \
                         method='DOP853', rtol = 1e-13, atol = 1e-13, dense_output=True)
    
    # Separate basis vectors
    R_fun = lambda x: solution.sol(x)[:3].T
    T_fun = lambda x: solution.sol(x)[3:6].T
    N_fun = lambda x: solution.sol(x)[6:9].T
    B_fun = lambda x: solution.sol(x)[9:12].T
    
    return T_fun, N_fun, B_fun, R_fun

def integrate_tangent(T, ell):
    # Integrate the tangent vector to get the position
    position = cumulative_trapezoid(T, ell, initial = 0.0, axis=0)

    return position

def find_position_curve_point(T_fun, ell):
    # Integrate the tangent vector to get the position
    position = np.zeros(3)
    for i in range(3):
        fun = lambda x: T_fun(x)[i]
        position[i] = quad(fun, 0, ell, epsabs = 1e-14, epsrel = 1e-14)[0]
    return position

def find_ss_points(self, T_fun, position_fun = None):
    # Two sets of ss points
    set_1_ss_points = np.arange(self.nfp) * self.L_in

    if callable(position_fun):
        def find_pos_fun(x):
            return position_fun(x)
    else:
        def find_pos_fun(x):
            return find_position_curve_point(T_fun,x)

    # Integrate the tangent vector to get the position
    pos_set_1 = []
    pos_set_2 = []
    for i, L_val in enumerate(set_1_ss_points):
        pos_set_2.append(find_pos_fun(L_val + 0.5*self.L_in))
        if i == 0:
            pos_set_1.append([0.0, 0.0, 0.0])
        else:
            pos_set_1.append(find_pos_fun(L_val))

    return pos_set_1, pos_set_2

def find_centre(points):
    return np.mean(points, axis=0)

def compute_normal_poly(points):
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    normal = np.cross(v1, v2)
    return normal / np.linalg.norm(normal)

def rotation_matrix(rotation_axis, theta):
    # Rotate by angle theta around rotation_axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    r = R.from_rotvec(theta * rotation_axis)
    return r.as_matrix()

def align_to_z_plane(points):
    normal = compute_normal_poly(points)
    z_axis = np.array([0, 0, 1])
    if np.allclose(normal, z_axis):
        return points, np.eye(3)  # Already aligned
    rotation_axis = np.cross(normal, z_axis)
    rotation_angle = np.arccos(np.dot(normal, z_axis))
    R = rotation_matrix(rotation_axis, rotation_angle)
    return np.dot(points, R.T), R

def align_vertex_to_x_axis(points):
    vertex = points[0]
    phi = np.arctan2(vertex[1], vertex[0])
    Rz = np.array([
        [np.cos(-phi), -np.sin(-phi), 0],
        [np.sin(-phi), np.cos(-phi), 0],
        [0, 0, 1]
    ])
    return np.dot(points, Rz.T), Rz

def transform_polygon(points):
    center = find_centre(points)
    points_centred = points - center
    points_aligned_z, R1 = align_to_z_plane(points_centred)
    points_aligned_xy, R2 = align_vertex_to_x_axis(points_aligned_z)
    total_rotation_matrix = np.dot(R2, R1)
    return center, points_aligned_xy, total_rotation_matrix

def transform_vector_field(vectors, rotation_matrix):
    return np.dot(vectors, rotation_matrix.T)

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

def invert_frenet_axis(self, curvature, torsion, ell, varphi, plot = False, full_axis = True, flip = True, minimal = False, func = False):
    ## Complete the curve ##
    # The inputs are assumed to be in the [0,2pi/N) grid. We extend them in field period and to include the last point

    if full_axis:
        # varphi
        varphi_in = varphi.copy()
        varphi = varphi_in.copy()
        if isinstance(varphi, list):
            varphi = np.array(varphi)
            varphi_in = np.array(varphi)   # Enpoint
        if isinstance(varphi, np.ndarray):
            for i in range(self.nfp - 1):
                varphi = np.concatenate((varphi, varphi_in + (i+1)*2*np.pi/self.nfp))
            varphi = np.append(varphi, 2*np.pi)
        else:
            raise TypeError('Invalid type for varphi input.')
        
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

        if func:
            # As a function of ell
            tau = self.torsion_in["function_ell"]
            kappa = self.curvature_in["function_ell"]
        else:
            # Torsion
            tau = torsion.copy()
            if isinstance(tau, list):
                tau += (self.nfp - 1) * torsion
                tau += [tau[0]]   # Enpoint
            elif isinstance(tau, np.ndarray):
                for i in range(self.nfp - 1):
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
                    if flip:
                        # Important to add a sign from period to period (because of half helicity)
                        kappa = np.concatenate((kappa, (-1)**(j+1)*curvature))
                    else:
                        kappa = np.concatenate((kappa, curvature))
                kappa = np.append(kappa, kappa[0])
            else:
                raise TypeError('Invalid type for curvature input.')

    else:
        print("WARNING! Should run with full axis.")
        tau = torsion.copy()
        kappa = curvature.copy()
        ell = ell.copy()
        varphi = varphi.copy()

    ##############################
    # SOLVE FRENET-SERRET SYSTEM #
    ##############################
    T_fun, N_fun, B_fun, position_fun = solve_frenet_serret_fun(kappa, tau, ell)
    T = T_fun(ell)
    N = N_fun(ell)
    B = B_fun(ell)
    position = position_fun(ell)
    
    #####################
    # CHOOSE THE Z AXIS #
    #####################
    # Find set of SS points
    set1, set2 = find_ss_points(self, T_fun, position_fun=position_fun)

    # Find the necessary rotation and shift of the coordinates to frame in cylindrical coordinates
    if self.nfp == 2:
        set1 = np.concatenate(([set1[0]],[set2[0]],[set1[1]],[set2[1]]))
    elif self.nfp == 1:
        print("WARNING! FS alignment and inversion not implemented for nfp = 1.")
    center, ss_points, rotation_matrix = transform_polygon(set1)

    # Functions for transformed positions and vectors
    def aligned_position_fun(ell):
        position = position_fun(ell)
        return np.dot(position - center, rotation_matrix.T)
    
    def aligned_FS_fun(ell):
        if isinstance(ell, float):
            ell = [ell]
        T = T_fun(ell)
        N = N_fun(ell)
        B = B_fun(ell)
        aligned_T = np.einsum('ji,ki->kj', rotation_matrix, T)
        aligned_N = np.einsum('ji,ki->kj', rotation_matrix, N)
        aligned_B = np.einsum('ji,ki->kj', rotation_matrix, B)
        return aligned_T, aligned_N, aligned_B
    
    # Align the curve to minimize Z excursion
    aligned_position = np.dot(position - center, rotation_matrix.T)

    ######################################
    # CONVERT TO CYLINDRICAL COORDINATES #
    ######################################
    # Convert curve position to cylindrical coordinates
    R, phi, Z = cartesian_to_cylindrical(aligned_position)
    R_ss, phi_ss, Z_ss = cartesian_to_cylindrical(ss_points)

    # Output to close curve
    mismatch = [T[-1]-T[0], N[-1]+N[0],[R_ss[1]-R_ss[0], Z_ss[1]-Z_ss[0], phi_ss[1]-phi_ss[0]-2*np.pi/self.nfp], position[-1] - position[0]]

    if minimal:
        return mismatch

    # Rotate the Frenet basis vectors accordingly
    aligned_T = np.einsum('ji,ki->kj', rotation_matrix, T)
    aligned_N = np.einsum('ji,ki->kj', rotation_matrix, N)
    aligned_B = np.einsum('ji,ki->kj', rotation_matrix, B)

    # Check whether the sense of the axis is in the positive cylindrical angle
    phi = np.unwrap(phi)
    phi0 = phi[0]
    sense_axis = phi[1] - phi0

    # Correct sense of the axis
    sgn_change = 1
    if sense_axis < 0:
        # Change coordinates accordingly
        sgn_change = -1
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

    # Redefine the cylindrical angle so that the first point is phi = 0 (the cylindrical representation should not change
    assert np.sign(phi[-1]-phi[0]) > 0
    scl = 1/(phi[-1]-phi[0])*2*np.pi
    phi = (phi-phi[0])*scl

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
    
    ################
    # SPLINES OF r #
    ################
    
    # Periodic spline interpolation for kappa and tau (assume varphi is equally spaced)
    flag_half = self.flag_half

    def save_splines_FS(phi_grid, ell_grid, kappa = kappa, tau = tau):
        # Keep the geometric quantities in cylindrical phi
        self.R0_func = self.convert_to_spline(R, grid = phi_grid)
        self.Z0_func = self.convert_to_spline(Z, grid = phi_grid)

        if func:
            kappa = kappa(ell_grid)
            tau = tau(ell_grid)
            
        # Due to sign, for half helicities, the configurations have sign flips in normal/binormal. We consider a continuous frame within 
        # a whole 2pi turn, and will be discontinuous at phi = 0. Keep it in vylindrical phi.
        self.normal_R_spline = self.convert_to_spline(aligned_N[:,0], grid = phi_grid, half_period = flag_half)
        self.normal_phi_spline = self.convert_to_spline(aligned_N[:,1], grid = phi_grid, half_period = flag_half)
        self.normal_z_spline = self.convert_to_spline(aligned_N[:,2], grid = phi_grid, half_period = flag_half)
        self.binormal_R_spline = self.convert_to_spline(aligned_B[:,0], grid = phi_grid, half_period = flag_half)
        self.binormal_phi_spline = self.convert_to_spline(aligned_B[:,1], grid = phi_grid, half_period = flag_half)
        self.binormal_z_spline = self.convert_to_spline(aligned_B[:,2], grid = phi_grid, half_period = flag_half)
        self.tangent_R_spline = self.convert_to_spline(aligned_T[:,0], grid = phi_grid)
        self.tangent_phi_spline = self.convert_to_spline(aligned_T[:,1], grid = phi_grid)
        self.tangent_z_spline = self.convert_to_spline(aligned_T[:,2], grid = phi_grid)

    save_splines_FS(phi, ell)

    smooth_frame = True
    if smooth_frame:
        smooth_FS_splines(self)
        
    ##############################
    # EVALUATE ON A REGULAR GRID #
    ##############################
    # Do nothing to the curvature and torsion: these are assumed to be given in varphi grid
    # Evaluate phi
    nu_tot = varphi - phi # if smooth_frame else smooth_fourier(varphi - phi, self.nfp, 15, varphi, even = False) # It happens to work better without this smoothing
    self.nu_spline_of_varphi = self.convert_to_spline(nu_tot, grid = varphi, periodic = True)
    self.nu = self.nu_spline_of_varphi(varphi_in)
    phi_out = varphi_in - self.nu
    self.phi = phi_out
    self.nu_spline = self.convert_to_spline(nu_tot, grid = phi, periodic = True)

    # Evaluate geometry
    self.R0 = self.R0_func(phi_out)
    self.Z0 = self.Z0_func(phi_out)

    nphi = self.nphi
    self.normal_cylindrical = np.zeros((nphi, 3))
    self.normal_cylindrical[:,0] = self.normal_R_spline(phi_out)
    self.normal_cylindrical[:,1] = self.normal_phi_spline(phi_out)
    self.normal_cylindrical[:,2] = self.normal_z_spline(phi_out)

    # plt.plot(phi_out, self.normal_z_spline(phi_out))
    # plt.plot(phi_out, aligned_N[:,2])
    # plt.show()

    self.binormal_cylindrical = np.zeros((nphi, 3))
    self.binormal_cylindrical[:,0] = self.binormal_R_spline(phi_out)
    self.binormal_cylindrical[:,1] = self.binormal_phi_spline(phi_out)
    self.binormal_cylindrical[:,2] = self.binormal_z_spline(phi_out)

    self.tangent_cylindrical = np.zeros((nphi, 3))
    self.tangent_cylindrical[:,0] = self.tangent_R_spline(phi_out)
    self.tangent_cylindrical[:,1] = self.tangent_phi_spline(phi_out)
    self.tangent_cylindrical[:,2] = self.tangent_z_spline(phi_out)

    #############
    # PLOT AXIS #
    #############
    if plot:
        # Plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plotting normal and binormal vectors as arrows
        origin = [self.R0 * np.cos(self.phi), self.R0 * np.sin(self.phi), self.Z0]

        # print(Theta)
        # Plot curve in cylindrical coordinates
        ax.plot(origin[0],origin[1],origin[2], label='Curve')

        # Plotting normal and binormal vectors as arrows
        stp = int(nphi/10)
        num = int(nphi/stp)

        for i in range(num):
            phi_i = self.phi[i*stp]
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

    return mismatch

def smooth_FS_splines(stel):
    ##############
    # SMOOTH R/Z #
    ##############
    nfp = stel.nfp
    ntor = 15
    phi_reg = np.linspace(0, 1.0, stel.nphi) * 2*np.pi/stel.nfp
    rc, rs, zc, zs = to_Fourier_axis(stel.R0_func(phi_reg), stel.Z0_func(phi_reg), nfp, ntor = ntor, lasym = True, phi_in = phi_reg)
    R0_temp = np.zeros(stel.nphi)
    Z0_temp = np.zeros(stel.nphi)
    for jn in range(ntor+1):
        n = jn * nfp
        sinangle = np.sin(n * phi_reg)
        cosangle = np.cos(n * phi_reg)
        R0_temp += rc[jn] * cosangle #+ rs[jn] * sinangle
        Z0_temp += zs[jn] * sinangle #+ zc[jn] * cosangle

    stel.R0_func = stel.convert_to_spline(R0_temp, grid = phi_reg)
    stel.Z0_func = stel.convert_to_spline(Z0_temp, grid = phi_reg)

    def smooth_fourier_sp(data, nfp, n_harm, even = True):
        # Input grid
        nphi = len(data)
        # window = np.hamming(len(data))
        # data = window * data

        phi_in = np.linspace(0.0, 1.0, nphi, endpoint=False) * 2*np.pi
        phi_out = phi_in.copy()

        # Harmonic components of axis position
        data_harm = np.zeros(int(n_harm + 1))
        data_harm[0] = np.sum(data) / nphi            
        factor = 2 / nphi

        if even:
            smoothed_data = np.full(len(phi_out), fill_value = data_harm[0])
        else:
            smoothed_data = np.zeros(len(phi_out))

        for n in range(1, n_harm+1):
            angle = n * phi_in
            factor2 = factor
            # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
            # if n == 0: factor2 = factor2 / 2
            if even:
                cosangle = np.cos(angle)
                data_harm[n] = np.sum(data * cosangle * factor2)
                smoothed_data += data_harm[n] * np.cos(n * nfp * (phi_out % (2*np.pi)))
            else:
                sinangle = np.sin(angle)
                data_harm[n] = np.sum(data * sinangle * factor2)
                smoothed_data += data_harm[n] * np.sin(n * nfp * (phi_out % (2*np.pi)))

        def convert_to_spline(phi, array):
            sp_temp = make_interp_spline(phi, array, k=7, axis=0)
            sp = lambda x: sp_temp(x % (2*np.pi))
            return sp

        sp = convert_to_spline(phi_out, smoothed_data)
            
        return sp
    
    # Use the Fourier components of the axis to smooth R0 and Z0 
    N = stel.nphi * 2
    phi_ext = np.linspace(0,2,N, endpoint=False) * 2*np.pi/stel.nfp
    even_parity = [False, True, True, True, False, False, False, True, True]
    half = [True, True, True, True, True, True, False, False, False]
    var_name = ["tangent_R","tangent_phi","tangent_z","normal_R","normal_phi","normal_z","binormal_R","binormal_phi","binormal_z"] # 
    new_sp = {}
    for parity, name in zip(even_parity, var_name):
        array = eval('stel.' + name + '_spline(phi_ext)')
        nfp = stel.nfp/2.0
        sp = smooth_fourier_sp(array, nfp, 20, even = parity)
        new_sp[name] = sp
    
    stel.normal_R_spline_new = new_sp["normal_R"]
    stel.normal_phi_spline = new_sp["normal_phi"]
    stel.normal_z_spline = new_sp["normal_z"]
    stel.binormal_R_spline = new_sp["binormal_R"]
    stel.binormal_phi_spline = new_sp["binormal_phi"]
    stel.binormal_z_spline = new_sp["binormal_z"]
    stel.tangent_R_spline = new_sp["tangent_R"]
    stel.tangent_phi_spline = new_sp["tangent_phi"]
    stel.tangent_z_spline = new_sp["tangent_z"]

    # phi_reg = np.linspace(0.0, 1.0, stel.nphi, endpoint=False) * 2*np.pi
    # plt.plot(phi_reg, stel.R0_func(phi_reg))
    # plt.plot(phi_reg, stel.R0_func_new(phi_reg))
    # plt.show()


def to_Fourier_axis(R0, Z0, nfp, ntor, lasym, phi_in = None):
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
        phi_in: phi grid in [0, 2pi/N) onto which to evaluate
    """
    # Create an evenly spaced cylindrical angle if no phi provided:
    # the sampling is otherwise in whatever is provided, and use trapz to integrate over
    # a potentially not uniform grid
    if isinstance(phi_in, list) or isinstance(phi_in, np.ndarray):
        phi_conversion = np.array(phi_in)
        nphi = len(phi_conversion)
        assert len(R0) == nphi
        assert len(Z0) == nphi
    else:
        nphi = len(R0)
        phi_conversion = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)

    # Harmonic components of axis position
    rc = np.zeros(int(ntor + 1))
    rs = np.zeros(int(ntor + 1))
    zc = np.zeros(int(ntor + 1))
    zs = np.zeros(int(ntor + 1))
    factor = 2 / (2*np.pi/nfp)
    phi_ext = np.append(phi_conversion, phi_conversion[0] + 2*np.pi/nfp)
    R0_ext = np.append(R0, R0[0])
    Z0_ext = np.append(Z0, -Z0[0])

    for n in range(1, ntor+1):
        # The angle with positive sign (when using it for VMEC will need sign)
        angle = n * nfp * phi_ext
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        factor2 = factor
        # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
        # if n == 0: factor2 = factor2 / 2
        rc[n] = np.trapz(R0_ext * cosangle * factor2, phi_ext)
        rs[n] = np.trapz(R0_ext * sinangle * factor2, phi_ext)
        zc[n] = np.trapz(Z0_ext * cosangle * factor2, phi_ext)
        zs[n] = np.trapz(Z0_ext * sinangle * factor2, phi_ext)

    rc[0] = np.trapz(R0_ext, phi_ext) / (2 * np.pi / nfp)
    zc[0] = np.trapz(Z0_ext, phi_ext) / (2 * np.pi / nfp)

    if not lasym:
        rs = rs * 0.0
        zc = zc * 0.0

    return rc, rs, zc, zs
