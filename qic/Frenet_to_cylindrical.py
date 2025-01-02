"""
This module contains the routines to compute
a given flux surface shape at a fixed
off-axis cylindrical toroidal angle
"""
import os
import numpy as np
from scipy.optimize import root_scalar
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm

  
def Frenet_to_cylindrical(self, r, ntheta=20, parallel = True):
    """
    Function to convert the near-axis coordinate system to
    a cylindrical one for a surface at a particular radius,
    outputing the following arrays: R(theta,phi),
    phi0(theta,phi) and Z(theta,phi) with R,phi, Z cylindrical
    coordinates and theta Boozer coordinate and phi0 the 
    cylindrical coordinate on axis.

    Args:
        r:  near-axis radius r of the desired boundary surface
        ntheta: resolution in the poloidal angle theta
    """
    if parallel:
        R_2D, Z_2D, phi0_2D = Frenet_to_cylindrical_parallel(self, r, ntheta)
    else:
        R_2D, Z_2D, phi0_2D = Frenet_to_cylindrical_no_parallel(self, r, ntheta)
            
    return R_2D, Z_2D, phi0_2D

def Frenet_to_cylindrical_parallel(self, r, ntheta=20):
    """
    Function to convert the near-axis coordinate system to
    a cylindrical one for a surface at a particular radius,
    outputing the following arrays: R(theta,phi),
    phi0(theta,phi) and Z(theta,phi) with R,phi, Z cylindrical
    coordinates and theta Boozer coordinate and phi0 the 
    cylindrical coordinate on axis.

    Args:
        r:  near-axis radius r of the desired boundary surface
        ntheta: resolution in the poloidal angle theta
    """
    nphi_conversion = self.nphi
    nfp = self.nfp

    theta = np.linspace(0,2*np.pi,ntheta,endpoint=False)
    phi_conversion = np.linspace(0,2*np.pi/self.nfp,nphi_conversion,endpoint=False)
    R_2D = np.zeros((ntheta,nphi_conversion))
    Z_2D = np.zeros((ntheta,nphi_conversion))
    phi0_2D = np.zeros((ntheta,nphi_conversion))

    # Defining the attributes
    order = self.order
    flag_half = self.flag_half

    # Real space geometry
    normal_R_spline = self.normal_R_spline
    normal_phi_spline = self.normal_phi_spline
    binormal_R_spline = self.binormal_R_spline
    binormal_phi_spline = self.binormal_phi_spline
    tangent_R_spline = self.tangent_R_spline
    tangent_phi_spline = self.tangent_phi_spline
    R0_func = self.R0_func
    Z0_func = self.Z0_func
    normal_z_spline = self.normal_z_spline
    binormal_z_spline = self.binormal_z_spline
    tangent_z_spline = self.tangent_z_spline
    
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

    def Frenet_to_cylindrical_residual_func_par(phi0, phi_target, R0_func, normal_R_spline, normal_phi_spline,
                                            binormal_R_spline, binormal_phi_spline, tangent_R_spline, tangent_phi_spline,
                                            X_spline, Y_spline, Z_spline, order):
        """
        Residual function with explicit arguments instead of self/qic.
        """
        sinphi0 = np.sin(phi0)
        cosphi0 = np.cos(phi0)
        R0_at_phi0 = R0_func(phi0)
        X_at_phi0 = X_spline(phi0)
        Y_at_phi0 = Y_spline(phi0)
        normal_R = normal_R_spline(phi0)
        normal_phi = normal_phi_spline(phi0)
        binormal_R = binormal_R_spline(phi0)
        binormal_phi = binormal_phi_spline(phi0)

        normal_x = normal_R * cosphi0 - normal_phi * sinphi0
        normal_y = normal_R * sinphi0 + normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y

        if order != 'r1':
            Z_at_phi0 = Z_spline(phi0)
            tangent_x = tangent_R_spline(phi0) * cosphi0 - tangent_phi_spline(phi0) * sinphi0
            tangent_y = tangent_R_spline(phi0) * sinphi0 + tangent_phi_spline(phi0) * cosphi0
            total_x += Z_at_phi0 * tangent_x
            total_y += Z_at_phi0 * tangent_y

        residual = np.arctan2(total_y, total_x) - phi_target
        if residual > np.pi:
            residual -= 2 * np.pi
        if residual < -np.pi:
            residual += 2 * np.pi
        return residual

    def Frenet_to_cylindrical_1_point_par(
        phi0, R0_func, Z0_func, normal_R_spline, normal_phi_spline, normal_z_spline,
        binormal_R_spline, binormal_phi_spline, binormal_z_spline,
        tangent_R_spline, tangent_phi_spline, tangent_z_spline,
        X_spline, Y_spline, Z_spline, order
    ):
        """
        Computes the cylindrical coordinate components R and Z for an associated point at r > 0.
        
        Args:
            phi0 (float): Toroidal angle on the axis.
            R0_func (callable): Function returning the radial coordinate at phi0.
            Z0_func (callable): Function returning the Z-coordinate at phi0.
            normal_R_spline, normal_phi_spline, normal_z_spline (callables): Splines for the normal components.
            binormal_R_spline, binormal_phi_spline, binormal_z_spline (callables): Splines for the binormal components.
            tangent_R_spline, tangent_phi_spline, tangent_z_spline (callables): Splines for the tangent components.
            X_spline, Y_spline, Z_spline (callables): Splines for the near-axis coordinates.
            order (str): The computation order ('r1', 'r2', etc.).

        Returns:
            tuple: (R, Z, phi) - Cylindrical coordinates.
        """
        sinphi0 = np.sin(phi0)
        cosphi0 = np.cos(phi0)
        R0_at_phi0 = R0_func(phi0)
        z0_at_phi0 = Z0_func(phi0)
        X_at_phi0 = X_spline(phi0)
        Y_at_phi0 = Y_spline(phi0)
        Z_at_phi0 = Z_spline(phi0)
        normal_R = normal_R_spline(phi0)
        normal_phi = normal_phi_spline(phi0)
        normal_z = normal_z_spline(phi0)
        binormal_R = binormal_R_spline(phi0)
        binormal_phi = binormal_phi_spline(phi0)
        binormal_z = binormal_z_spline(phi0)

        normal_x = normal_R * cosphi0 - normal_phi * sinphi0
        normal_y = normal_R * sinphi0 + normal_phi * cosphi0
        binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
        binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

        total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
        total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y
        total_z = z0_at_phi0 + X_at_phi0 * normal_z + Y_at_phi0 * binormal_z

        if order != 'r1':
            tangent_R = tangent_R_spline(phi0)
            tangent_phi = tangent_phi_spline(phi0)
            tangent_z = tangent_z_spline(phi0)

            tangent_x = tangent_R * cosphi0 - tangent_phi * sinphi0
            tangent_y = tangent_R * sinphi0 + tangent_phi * cosphi0

            total_x += Z_at_phi0 * tangent_x
            total_y += Z_at_phi0 * tangent_y
            total_z += Z_at_phi0 * tangent_z

        total_R = np.sqrt(total_x ** 2 + total_y ** 2)
        total_phi = np.arctan2(total_y, total_x)

        return total_R, total_z, total_phi

    def process_theta(j_theta):
        """
        Worker function to process a single theta value.
        """
        costheta = np.cos(theta[j_theta])
        sintheta = np.sin(theta[j_theta])
        X_at_this_theta = r * (X1c_untwisted * costheta + X1s_untwisted * sintheta)
        Y_at_this_theta = r * (Y1c_untwisted * costheta + Y1s_untwisted * sintheta)
        Z_at_this_theta = 0 * X_at_this_theta
        if order != 'r1':
            cos2theta = np.cos(2 * theta[j_theta])
            sin2theta = np.sin(2 * theta[j_theta])
            X_at_this_theta += r * r * (X20_untwisted + X2c_untwisted * cos2theta + X2s_untwisted * sin2theta)
            Y_at_this_theta += r * r * (Y20_untwisted + Y2c_untwisted * cos2theta + Y2s_untwisted * sin2theta)
            Z_at_this_theta += r * r * (Z20_untwisted + Z2c_untwisted * cos2theta + Z2s_untwisted * sin2theta)
            if order == 'r3':
                cos3theta = np.cos(3 * theta[j_theta])
                sin3theta = np.sin(3 * theta[j_theta])
                r3 = r * r * r
                X_at_this_theta += r3 * (X3c1_untwisted * costheta + X3s1_untwisted * sintheta +
                                         X3c3_untwisted * cos3theta + X3s3_untwisted * sin3theta)
                Y_at_this_theta += r3 * (Y3c1_untwisted * costheta + Y3s1_untwisted * sintheta +
                                         Y3c3_untwisted * cos3theta + Y3s3_untwisted * sin3theta)
                Z_at_this_theta += r3 * (Z3c1_untwisted * costheta + Z3s1_untwisted * sintheta +
                                         Z3c3_untwisted * cos3theta + Z3s3_untwisted * sin3theta)

        X_spline = convert_to_spline(X_at_this_theta, half_period=flag_half, varphi=False)
        Y_spline = convert_to_spline(Y_at_this_theta, half_period=flag_half, varphi=False)
        Z_spline = convert_to_spline(Z_at_this_theta, varphi=False)

        R_row = np.zeros(nphi_conversion)
        Z_row = np.zeros(nphi_conversion)
        phi0_row = np.zeros(nphi_conversion)

        for j_phi in range(nphi_conversion):
            phi_target = phi_conversion[j_phi]
            phi0_rootSolve_min = phi_target - 1.0 / nfp
            phi0_rootSolve_max = phi_target + 1.0 / nfp
            res = root_scalar(
                Frenet_to_cylindrical_residual_func_par,
                xtol=1e-17,
                rtol=1e-15,
                maxiter=2000,
                args=(
                    phi_target, R0_func, normal_R_spline, normal_phi_spline, binormal_R_spline, binormal_phi_spline,
                    tangent_R_spline, tangent_phi_spline, X_spline, Y_spline, Z_spline, order),
                bracket=[phi0_rootSolve_min, phi0_rootSolve_max],
                x0=phi_target
            )
            phi0_solution = res.root
            final_R, final_z, _ = final_R, final_z, final_phi = Frenet_to_cylindrical_1_point_par(
                    phi0_solution, R0_func, Z0_func,
                    normal_R_spline, normal_phi_spline, normal_z_spline,
                    binormal_R_spline, binormal_phi_spline, binormal_z_spline,
                    tangent_R_spline, tangent_phi_spline, tangent_z_spline,
                    X_spline, Y_spline, Z_spline, order
                )
            R_row[j_phi] = final_R
            Z_row[j_phi] = final_z
            phi0_row[j_phi] = phi0_solution


        return j_theta, R_row, Z_row, phi0_row

    # Use ThreadPoolExecutor for parallel processing\
    from multiprocessing import Manager
    with Manager() as manager:
        progress_queue = manager.Queue()

        # Define a processing pool
        n_process = os.cpu_count()
        with Pool(processes = n_process) as pool:
            # Create a tqdm progress bar
            with tqdm(total=ntheta, desc="Processing theta", ncols=100) as pbar:
                # Start processing the tasks
                results = []
                for result in pool.imap(process_theta, range(ntheta)):
                    # Each time a task completes, update the progress bar
                    results.append(result)  # Collect the result
                    pbar.update(1)

    # with ThreadPoolExecutor() as executor:
    #     results = list(executor.map(process_theta, range(ntheta)))

    # Combine results into the final arrays
    for j_theta, R_row, Z_row, phi0_row in results:
        R_2D[j_theta, :] = R_row
        Z_2D[j_theta, :] = Z_row
        phi0_2D[j_theta, :] = phi0_row
            
    return R_2D, Z_2D, phi0_2D

def Frenet_to_cylindrical_residual_func(phi0, phi_target, qic, X_spline, Y_spline, Z_spline):
    """
    This function takes a point on the magnetic axis with a given
    toroidal angle phi0, computes the actual toroidal angle phi
    for an associated point at r>0 and finds the difference between
    this phi and the target value of phi

    Args:
        phi0 (float): toroidal angle on the axis
        phi_target (float): standard cylindrical toroidal angle
    """
    sinphi0 = np.sin(phi0)
    cosphi0 = np.cos(phi0)
    R0_at_phi0   = qic.R0_func(phi0)
    X_at_phi0    = X_spline(phi0)
    Y_at_phi0    = Y_spline(phi0)
    normal_R     = qic.normal_R_spline(phi0)
    normal_phi   = qic.normal_phi_spline(phi0)
    binormal_R   = qic.binormal_R_spline(phi0)
    binormal_phi = qic.binormal_phi_spline(phi0)

    normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
    normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
    binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
    binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

    total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
    total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y

    if qic.order != 'r1':
        Z_at_phi0    = Z_spline(phi0)
        tangent_R    = qic.tangent_R_spline(phi0)
        tangent_phi  = qic.tangent_phi_spline(phi0)

        tangent_x = tangent_R * cosphi0 - tangent_phi * sinphi0
        tangent_y = tangent_R * sinphi0 + tangent_phi * cosphi0

        total_x = total_x + Z_at_phi0 * tangent_x
        total_y = total_y + Z_at_phi0 * tangent_y

    Frenet_to_cylindrical_residual = np.arctan2(total_y, total_x) - phi_target
    # We expect the residual to be less than pi in absolute value, so if it is not, the reason must be the branch cut:
    if (Frenet_to_cylindrical_residual >  np.pi): Frenet_to_cylindrical_residual = Frenet_to_cylindrical_residual - 2*np.pi
    if (Frenet_to_cylindrical_residual < -np.pi): Frenet_to_cylindrical_residual = Frenet_to_cylindrical_residual + 2*np.pi
    return Frenet_to_cylindrical_residual

def Frenet_to_cylindrical_1_point(phi0, qic, X_spline, Y_spline, Z_spline):
    """
    This function takes a point on the magnetic axis with a given
    toroidal angle phi0 and computes the cylindrical coordinate
    components R and Z for an associated point at r>0

    Args:
        phi0: toroidal angle on the axis
    """
    sinphi0 = np.sin(phi0)
    cosphi0 = np.cos(phi0)
    R0_at_phi0   = qic.R0_func(phi0)
    z0_at_phi0   = qic.Z0_func(phi0)
    X_at_phi0    = X_spline(phi0)
    Y_at_phi0    = Y_spline(phi0)
    Z_at_phi0    = Z_spline(phi0)
    normal_R     = qic.normal_R_spline(phi0)
    normal_phi   = qic.normal_phi_spline(phi0)
    normal_z     = qic.normal_z_spline(phi0)
    binormal_R   = qic.binormal_R_spline(phi0)
    binormal_phi = qic.binormal_phi_spline(phi0)
    binormal_z   = qic.binormal_z_spline(phi0)

    normal_x   =   normal_R * cosphi0 -   normal_phi * sinphi0
    normal_y   =   normal_R * sinphi0 +   normal_phi * cosphi0
    binormal_x = binormal_R * cosphi0 - binormal_phi * sinphi0
    binormal_y = binormal_R * sinphi0 + binormal_phi * cosphi0

    total_x = R0_at_phi0 * cosphi0 + X_at_phi0 * normal_x + Y_at_phi0 * binormal_x
    total_y = R0_at_phi0 * sinphi0 + X_at_phi0 * normal_y + Y_at_phi0 * binormal_y

    total_z = z0_at_phi0           + X_at_phi0 * normal_z + Y_at_phi0 * binormal_z

    if qic.order != 'r1':
        tangent_R   = qic.tangent_R_spline(phi0)
        tangent_phi = qic.tangent_phi_spline(phi0)
        tangent_z   = qic.tangent_z_spline(phi0)

        tangent_x = tangent_R * cosphi0 - tangent_phi * sinphi0
        tangent_y = tangent_R * sinphi0 + tangent_phi * cosphi0

        total_x = total_x + Z_at_phi0 * tangent_x
        total_y = total_y + Z_at_phi0 * tangent_y
        total_z = total_z + Z_at_phi0 * tangent_z

    total_R = np.sqrt(total_x * total_x + total_y * total_y)
    total_phi=np.arctan2(total_y, total_x)

    return total_R, total_z, total_phi
    
def Frenet_to_cylindrical_no_parallel(self, r, ntheta=20):
    """
    Function to convert the near-axis coordinate system to
    a cylindrical one for a surface at a particular radius,
    outputing the following arrays: R(theta,phi),
    phi0(theta,phi) and Z(theta,phi) with R,phi, Z cylindrical
    coordinates and theta Boozer coordinate and phi0 the 
    cylindrical coordinate on axis.

    Args:
        r:  near-axis radius r of the desired boundary surface
        ntheta: resolution in the poloidal angle theta
    """
    nphi_conversion = self.nphi
    theta = np.linspace(0,2*np.pi,ntheta,endpoint=False)
    phi_conversion = np.linspace(0,2*np.pi/self.nfp,nphi_conversion,endpoint=False)
    R_2D = np.zeros((ntheta,nphi_conversion))
    Z_2D = np.zeros((ntheta,nphi_conversion))
    phi0_2D = np.zeros((ntheta,nphi_conversion))
    with tqdm(desc = 'Computing different theta...', total = ntheta) as pbar:
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
                    costheta  = np.cos(theta[j_theta])
                    sintheta  = np.sin(theta[j_theta])
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
            X_spline = self.convert_to_spline(X_at_this_theta, half_period = self.flag_half, varphi = False)
            Y_spline = self.convert_to_spline(Y_at_this_theta, half_period = self.flag_half, varphi = False)
            Z_spline = self.convert_to_spline(Z_at_this_theta, varphi = False)
            # print('*')
            # import matplotlib.pyplot as plt
            # phi_ext = np.linspace(-1,1,10000)*np.pi
            # plt.plot(phi_ext, Z_spline(phi_ext)*self.tangent_R_spline(phi_ext))
            # plt.plot(phi_ext, Z_spline(phi_ext)*self.tangent_phi_spline(phi_ext))
            # plt.plot(phi_ext, Z_spline(phi_ext)*self.tangent_z_spline(phi_ext))
            # plt.show()
            for j_phi in range(nphi_conversion):
                # Solve for the phi0 such that r0 + X n + Y b has the desired phi
                phi_target = phi_conversion[j_phi]
                phi0_rootSolve_min = phi_target - 1.0 / self.nfp
                phi0_rootSolve_max = phi_target + 1.0 / self.nfp
                res = root_scalar(Frenet_to_cylindrical_residual_func, xtol=1e-17, rtol=1e-15, maxiter=2000,\
                                args=(phi_target, self, X_spline, Y_spline, Z_spline), bracket=[phi0_rootSolve_min, phi0_rootSolve_max], \
                                x0=phi_target)
                phi0_solution = res.root
                final_R, final_z, _ = Frenet_to_cylindrical_1_point(phi0_solution, self, X_spline, Y_spline, Z_spline)
                R_2D[j_theta,j_phi] = final_R
                Z_2D[j_theta,j_phi] = final_z
                phi0_2D[j_theta,j_phi] = phi0_solution
            pbar.update(1)
            
    return R_2D, Z_2D, phi0_2D
   
def to_RZ(self,points):
    """
    Function to convert a set of points in (r,theta,phi0) coordinates
    where r=sqrt(2*psi/B0) is the near-axis radius, theta is the
    Boozer poloidal angle and phi0 is the cylindrical angle phi
    on the axis to cylindrical coordinates (R,Z)

    Args:
        points: an array of floats with dimension Nx3 with N the
        number of points to evaluate with each points having
        the (r,theta,phi0) values to evaluate
    """
    R_final = []
    Z_final = []
    Phi_final = []
    for point in points:
        r      = point[0]
        theta  = point[1]
        phi0   = point[2]
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        X_at_this_theta = r * (self.X1c_untwisted * costheta + self.X1s_untwisted * sintheta)
        Y_at_this_theta = r * (self.Y1c_untwisted * costheta + self.Y1s_untwisted * sintheta)
        Z_at_this_theta = 0 * X_at_this_theta
        if self.order != 'r1':
            # We need O(r^2) terms:
            cos2theta = np.cos(2 * theta)
            sin2theta = np.sin(2 * theta)
            X_at_this_theta += r * r * (self.X20_untwisted + self.X2c_untwisted * cos2theta + self.X2s_untwisted * sin2theta)
            Y_at_this_theta += r * r * (self.Y20_untwisted + self.Y2c_untwisted * cos2theta + self.Y2s_untwisted * sin2theta)
            Z_at_this_theta += r * r * (self.Z20_untwisted + self.Z2c_untwisted * cos2theta + self.Z2s_untwisted * sin2theta)
            if self.order == 'r3':
                # We need O(r^3) terms:
                cos3theta = np.cos(3 * theta)
                sin3theta = np.sin(3 * theta)
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
        # NOTE : it has a weird blip on the leftmost point
        X_spline = self.convert_to_spline(X_at_this_theta, half_period = self.flag_half, varphi = False)
        Y_spline = self.convert_to_spline(Y_at_this_theta, half_period = self.flag_half, varphi = False)
        Z_spline = self.convert_to_spline(Z_at_this_theta, varphi = False)
        R, Z, Phi = Frenet_to_cylindrical_1_point(phi0, self, X_spline, Y_spline, Z_spline)
        R_final.append(R)
        Z_final.append(Z)
        Phi_final.append(Phi)

    return R_final, Z_final, Phi_final
