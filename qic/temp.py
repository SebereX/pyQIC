#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

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

def align_with_min_z_excursion(position, T):
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

def invert_frenet_axis(self, curvature, torsion, ell, varphi, plot = False, nphi = None, full_axis = True):
    ## Complete the curve ##
    # The inputs are assumed to be in the [0,2pi/N) grid. We extend them in field period and to include the last point

    if full_axis:
        # Torsion
        tau = torsion.copy()
        if isinstance(tau,list):
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
    aligned_position, _, rotation_matrix = align_with_min_z_excursion(position, T)

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
        
        aligned_position[:,2] *= -1
        aligned_T[:,2] *= -1
        aligned_N[:,2] *= -1
        aligned_B[:,2] *= -1

        aligned_position[:,1] *= -1
        aligned_T[:,1] *= -1
        aligned_N[:,1] *= -1
        aligned_B[:,1] *= -1

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
    # Function to make splines
    def make_spline(grid,array, periodic = True):
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
    if not isinstance(nphi, int):
        nphi = self.nphi
    phi_new = np.linspace(0, 2*np.pi, nphi) # Note here I choose the whole configuration instead of 1/nfp

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
        # ax.plot(origin[0],origin[1],origin[2], label='Curve')

        # Plotting normal and binormal vectors as arrows
        stp = 8
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


# Example usage
kappa = [0.00000000e+00,1.87563887e-01,3.74215561e-01,5.59043387e-01
,7.41134068e-01,9.19573450e-01,1.09344554e+00,1.26183358e+00
,1.42382004e+00,1.57848779e+00,1.72492139e+00,1.86221035e+00
,1.98945140e+00,2.10575278e+00,2.21023976e+00,2.30206239e+00
,2.38040334e+00,2.44448886e+00,2.49360217e+00,2.52709861e+00
,2.54442425e+00,2.54513719e+00,2.52893168e+00,2.49566515e+00
,2.44538749e+00,2.37837157e+00,2.29514380e+00,2.19651244e+00
,2.08359268e+00,1.95782178e+00,1.82096374e+00,1.67509670e+00
,1.52259140e+00,1.36605534e+00,1.20826387e+00,1.05207054e+00
,9.00302222e-01,7.55647753e-01,6.20542253e-01,4.97062537e-01
,3.86839994e-01,2.90995910e-01,2.10103202e-01,1.44173051e-01
,9.26716047e-02,5.45547816e-02,2.83209553e-02,1.20758091e-02
,3.60507801e-03,4.52665667e-04,-1.53524722e-21,-4.52670715e-04
,-3.60510003e-03,-1.20757783e-02,-2.83209548e-02,-5.45546917e-02
,-9.26714529e-02,-1.44173992e-01,-2.10104211e-01,-2.90997427e-01
,-3.86842020e-01,-4.97062955e-01,-6.20543504e-01,-7.55648199e-01
,-9.00302926e-01,-1.05206900e+00,-1.20826269e+00,-1.36605505e+00
,-1.52258786e+00,-1.67509594e+00,-1.82095741e+00,-1.95782092e+00
,-2.08359231e+00,-2.19651128e+00,-2.29514353e+00,-2.37837136e+00
,-2.44538707e+00,-2.49566517e+00,-2.52893171e+00,-2.54513719e+00
,-2.54442421e+00,-2.52709845e+00,-2.49360191e+00,-2.44448840e+00
,-2.38040259e+00,-2.30206171e+00,-2.21023832e+00,-2.10575198e+00
,-1.98944954e+00,-1.86220931e+00,-1.72491938e+00,-1.57848609e+00
,-1.42381824e+00,-1.26183197e+00,-1.09344446e+00,-9.19572192e-01
,-7.41134524e-01,-5.59044027e-01,-3.74215542e-01,-1.87563887e-01] # example curvature values

tau = [2.29004055,2.29016788,2.29055003,2.29118744,2.29208088,2.29323139
,2.29464031,2.29630921,2.29823994,2.30043454,2.30289526,2.30562446
,2.30862461,2.3118982,2.31544763,2.31927517,2.32338279,2.32777203
,2.33244384,2.33739832,2.34263452,2.34815012,2.35394114,2.36000149
,2.36632263,2.37289306,2.3796979,2.38671839,2.39393139,2.40130906
,2.40881863,2.4164223,2.42407703,2.43173514,2.43934486,2.44685109
,2.45419652,2.46132272,2.46817162,2.47468673,2.48081438,2.48650466
,2.4917122,2.49639678,2.50052346,2.50406267,2.50699004,2.50928619
,2.51093651,2.51193071,2.51226277,2.51193071,2.5109365,2.5092862
,2.50699004,2.50406268,2.50052347,2.49639671,2.49171213,2.48650457
,2.48081426,2.47468671,2.46817155,2.4613227,2.45419649,2.44685117
,2.43934492,2.43173516,2.4240772,2.41642234,2.40881896,2.40130911
,2.39393141,2.38671847,2.37969792,2.37289308,2.36632267,2.36000149
,2.35394113,2.34815012,2.34263449,2.33739829,2.33244381,2.327772
,2.32338275,2.31927514,2.31544757,2.31189817,2.30862457,2.30562444
,2.30289522,2.30043452,2.29823991,2.29630919,2.2946403,2.29323138
,2.29208088,2.29118744,2.29055003,2.29016788]   # example torsion values

ell = [0.,0.00957647,0.0191613,0.0287629,0.03838969,0.04805021
,0.05775305,0.06750696,0.07732082,0.08720365,0.09716469,0.10721335
,0.11735923,0.12761216,0.13798218,0.14847949,0.15911451,0.16989776
,0.18083986,0.19195145,0.2032431,0.21472515,0.22640762,0.23829998
,0.25041095,0.26274823,0.27531823,0.28812574,0.30117362,0.31446248
,0.32799036,0.34175254,0.35574138,0.36994629,0.38435386,0.39894816
,0.41371119,0.42862345,0.44366461,0.45881429,0.47405273,0.48936148
,0.50472388,0.52012551,0.53555438,0.55100095,0.5664581,0.58192087
,0.59738615,0.61285227,0.62831856,0.64378485,0.65925097,0.67471625
,0.69017902,0.70563617,0.72108274,0.7365116,0.75191324,0.76727564
,0.78258438,0.79782282,0.8129725,0.82801367,0.84292592,0.85768895
,0.87228326,0.88669083,0.90089574,0.91488458,0.92864677,0.94217465
,0.95546351,0.9685114,0.98131891,0.99388891,1.00622619,1.01833716
,1.03022952,1.04191199,1.05339404,1.06468569,1.07579728,1.08673938
,1.09752263,1.10815764,1.11865496,1.12902497,1.1392779,1.14942378
,1.15947243,1.16943347,1.17931631,1.18913016,1.19888407,1.20858692
,1.21824743,1.22787423,1.23747582,1.24706066]  # parameter array

class Struct():
    """
    This class is just an empty mutable object to which we can attach
    attributes.
    """
    pass

stel = Struct()
stel.nfp = 5
stel.nphi = 501
stel.L_in = 1.25663712

invert_frenet_axis(stel, kappa, tau, ell, ell, plot = True, nphi = None, full_axis = True)
print(np.einsum('ij,ij->i', stel.binormal_cylindrical, stel.tangent_cylindrical))
