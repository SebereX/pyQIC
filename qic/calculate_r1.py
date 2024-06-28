"""
This module contains the functions for solving the sigma equation
and computing diagnostics of the O(r^1) solution.
"""

import logging
import numpy as np
from scipy.linalg import solve
from .util import fourier_minimum
from .newton import newton
from scipy.interpolate import CubicSpline as spline

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def _residual(self, x):
    """
    Residual in the sigma equation, used for Newton's method.  x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    # Solving for x = [iota, sigma(1), sigma(2), ...] 
    sigma = np.copy(x[1::])
    iota = x[0]

    if self.omn == True:
        # Right sign of helicity 
        helicity = - self.helicity

        # Distinguish between the case when alpha is an input and not
        if isinstance(self.alpha_in, dict):
            #########################
            # REFERENCE IDEAL ALPHA #
            #########################
            # Initialise alpha to ideal value (had to change +1/2 to -1/2 for agreement)
            self.alpha_no_buffer = np.pi*(2*helicity-1/2) + iota * (self.varphi - np.pi/self.nfp)

            ###############
            # INPUT ALPHA #
            ###############
            # No change in alpha, which is provided as input to __init__. Note the separation in secular and non-secular parts
            non_secular_part = self.evaluate_input_on_grid(self.alpha_in, self.varphi)
            self.alpha = non_secular_part + self.varphi * helicity * self.nfp

            # We will need to compute gamma = iota - d_alpha_d_varphi
            # To compute d_alpha_d_varphi, we need to separate the non-symmetric piece: we assume this is the helicity part
            self.gamma = iota - helicity * self.nfp - np.matmul(self.d_d_varphi, non_secular_part)
        else:
            # Need to find alpha to satisfy the QI condition in stellarator symmetry, but with the addition of a buffer
            # region to immpose periodicity of the solution
            #########################
            # REFERENCE IDEAL ALPHA #
            #########################
            # Initialise alpha to ideal value 
            self.alpha_no_buffer = np.pi*(2*helicity+1/2) + iota * (self.varphi - np.pi/self.nfp)

            #############################
            # CONSTRUCT ALPHA W/ BUFFER #
            #############################
            # Construct alpha following different prescriptions
            buffer_method = self.buffer_details["omn_method"]
            # Construct alpha: this includes the part proportional to iota and iota independent as well as their derivatives
            # which are stored in self
            _make_buffer(self, buffer_method, iota)
            
            #############################
            # CONSTRUCT GAMMA W/ BUFFER #
            #############################
            # Calculate gamma = iota - alpha' (using the derivatives properly taken)
            self.gamma_iota    = 1 - self.d_alpha_iota_d_varphi
            self.gamma_notIota = - self.d_alpha_notIota_d_varphi
            self.gamma         = self.gamma_iota * iota + self.gamma_notIota
    else:
        # If QS, then gamma = iota_N = iota + stel.helicity * stel.nfp
        self.gamma = iota + self.helicity * self.nfp + np.zeros(self.nphi) # - np.matmul(self.d_d_varphi, self.alpha)
    
    ############
    # RESIDUAL #
    ############
    # Part I : value of the sigma equation
    r = np.matmul(self.d_d_varphi, sigma) + self.gamma * \
        (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + 1 + sigma * sigma) \
        - 2 * self.etabar_squared_over_curvature_squared * (- self.torsion + self.I2 / self.Bbar) * self.G0 / self.B0
    # Part II : value of sigma at phi = 0 (to make it match with the initial sigma condition)
    sigma_at_0 = np.matmul(self.interpolateTo0, sigma)

    return np.append(r,sigma_at_0-self.sigma0)

def _make_buffer(self, buffer_method, iota):
    """
    Construct alpha including the necessary deviation from exact omnigeneity to be periodic.
    """
    # Define right sign for helicity
    helicity = - self.helicity

    # Only run the first time in the sigma solve iteration, then use the form of alpha
    if not hasattr(self, "alpha_iota"):
        ## Standard piecewise buffer region ## [Plunk et al., 2019]
        if buffer_method == 'buffer':
            # Reference ideal alpha, split into a piece proportional to iota and one independent
            self.alpha_iota = self.varphi - np.pi/self.nfp
            self.alpha_notIota = 0 * self.varphi # added varphi term to get proper sized array (missing a shift for QI)

            # Define locations of buffer given delta
            location_section_I_II   = np.argmin(np.abs(self.varphi-self.delta))
            location_section_II_III = np.argmin(np.abs(self.varphi-(2*np.pi/self.nfp-self.delta)))+1
            varphiI   = self.varphi[0:location_section_I_II]
            varphiIII = self.varphi[location_section_II_III::]

            # Calculate alpha_iota (the part proportional to iota) on buffer regions
            alpha_I_II = self.delta - np.pi/self.nfp
            alpha_0 = 0
            alpha_1 = -(2*alpha_0 - 2*alpha_I_II + 1*self.delta) / self.delta
            alpha_3 = 2*(alpha_0 - alpha_I_II + 1*self.delta) / (self.delta * self.delta * self.delta)
            alpha_4 = -(alpha_0 - alpha_I_II + 1*self.delta) / (self.delta * self.delta * self.delta * self.delta)

            # Modify the ideal alpha_iota defined outside
            self.alpha_iota[0:location_section_I_II]   =   alpha_0 + alpha_1 * varphiI + alpha_3 * (varphiI **3) + \
                                                        alpha_4 * (varphiI **4)
            self.alpha_iota[location_section_II_III::] = -(alpha_0 + alpha_1 * (2*np.pi/self.nfp-varphiIII) + \
                                                        alpha_3 * ((2*np.pi/self.nfp-varphiIII) **3) + alpha_4 * ((2*np.pi/self.nfp-varphiIII) **4))

            # Overall shift of alpha (necessary for QI condition)
            alpha_shift = np.pi*(2*helicity+1/2)
            # Slope of secular term for difference between the two ends of alpha
            n_for_alpha = helicity

            # Calculate alpha_notIota on buffer regions
            alpha_I_II = 0
            alpha_0 =  - np.pi * n_for_alpha
            alpha_1 = -(2*alpha_0 - 2*alpha_I_II + 0*self.delta) / self.delta
            alpha_3 = 2*(alpha_0 - alpha_I_II + 0*self.delta) / (self.delta * self.delta * self.delta) 
            alpha_4 = -(alpha_0 - alpha_I_II + 0*self.delta) / (self.delta * self.delta * self.delta * self.delta)
            # Modify the ideal alpha_notIota defined outside
            self.alpha_notIota[0:location_section_I_II] = alpha_0 + alpha_1 * varphiI + alpha_3 * (varphiI **3) + alpha_4 * (varphiI **4)
            self.alpha_notIota[location_section_II_III::] = -(alpha_0 + alpha_1 * (2*np.pi/self.nfp-varphiIII) + \
                                            alpha_3 * ((2*np.pi/self.nfp-varphiIII) **3) + alpha_4 * ((2*np.pi/self.nfp-varphiIII) **4))
            # Add shift
            self.alpha_notIota = self.alpha_notIota + alpha_shift
            # self.alpha_at_zero = alpha_shift + alpha_0

            # Compute derivatives for contructing gamma (separating secular parts)
            self.d_alpha_iota_d_varphi = np.matmul(self.d_d_varphi, self.alpha_iota)
            self.d_alpha_notIota_d_varphi = n_for_alpha * self.nfp + \
                np.matmul(self.d_d_varphi, self.alpha_notIota - self.varphi * n_for_alpha * self.nfp) # We have to treat the secular part separately here since d_d_varphi assumes periodicity
        
        ## Alternative buffer region attempting a more smooth alpha ## [Camacho et al., 2022]
        elif buffer_method == 'non-zone':
            # k parameter for degree of smoothness (note that the resulting function is not C^∞)
            self.k_buffer = self.buffer_details["k_buffer"]
            k = self.k_buffer
            # Reference ideal alpha, split into a piece proportional to iota and one independent
            self.alpha_iota = self.varphi - np.pi/self.nfp
            self.alpha_notIota = 0 * self.varphi # added varphi term to get proper sized array (missing a shift for QI)
            # Modify the ideal alpha_iota and alpha_notIota (including the QI shift)
            self.alpha_iota += -(np.pi / self.nfp) * (self.varphi*self.nfp/np.pi - 1)**(2*k+1)
            self.alpha_notIota += np.pi*(2*helicity + 1/2 + helicity * (self.varphi*self.nfp/np.pi - 1)**(2*k+1))
            # Compute derivatives (in this case the powers may be differentiated)
            self.d_alpha_iota_d_varphi = 1 - (np.pi / self.nfp) * (2*k+1) * (self.nfp/np.pi) * (self.varphi*self.nfp/np.pi - 1)**(2*k)
            self.d_alpha_notIota_d_varphi = np.pi * helicity * (2*k+1) * (self.nfp/np.pi) * (self.varphi*self.nfp/np.pi - 1)**(2*k)
        
        ## Alternative buffer region attempting an even more smooth alpha ## [Camacho et al., 2022]
        elif buffer_method == 'non-zone-smoother':
            # k and p parameters for degree of smoothness (note that the resulting function is not C^∞)
            self.k_buffer = self.buffer_details["k_buffer"]
            k = self.k_buffer
            self.p_buffer = self.buffer_details["p_buffer"]
            p = self.p_buffer
            # Reference ideal alpha, split into a piece proportional to iota and one independent
            self.alpha_iota = self.varphi - np.pi/self.nfp
            self.alpha_notIota = 0 * self.varphi # added varphi term to get proper sized array (missing a shift for QI)
            # Define shorthand parameters
            nu = (2*k+1)*(2*k)/((2*p+1)*(2*p))
            m = helicity
            n = self.nfp
            a_not_iota =  ((np.pi * m) * (np.pi/n)**(-2*k-1)) * (1/(1-nu))
            a_iota     = -((np.pi / n) * (np.pi/n)**(-2*k-1)) * (1/(1-nu))
            b_not_iota = -((np.pi * m) * (np.pi/n)**(-2*p-1)) * (nu/(1-nu))
            b_iota     =  ((np.pi / n) * (np.pi/n)**(-2*p-1)) * (nu/(1-nu))
            # Construct alpha by modifying the ideal alphas
            self.alpha_iota    += a_iota     * (self.varphi-np.pi/self.nfp)**(2*k+1) + b_iota     * (self.varphi-np.pi/self.nfp)**(2*p+1)
            self.alpha_notIota += a_not_iota * (self.varphi-np.pi/self.nfp)**(2*k+1) + b_not_iota * (self.varphi-np.pi/self.nfp)**(2*p+1)
            self.alpha_notIota += np.pi*(2*helicity + 1/2)  # Add QI shift
            # Take the derivatives (in this case the powers may be differentiated)
            self.d_alpha_iota_d_varphi    = 1 + a_iota     * (2*k+1) * (self.varphi-np.pi/self.nfp)**(2*k) + b_iota     * (2*p+1) * (self.varphi-np.pi/self.nfp)**(2*p)
            self.d_alpha_notIota_d_varphi =     a_not_iota * (2*k+1) * (self.varphi-np.pi/self.nfp)**(2*k) + b_not_iota * (2*p+1) * (self.varphi-np.pi/self.nfp)**(2*p)

        ## Alternative buffer region using a smoother alpha using a Fourier representation ## (Rogerio method)
        elif buffer_method == 'non-zone-fourier':
            # Buffer properties 
            self.k_buffer = self.buffer_details["k_buffer"]
            # Shorthand definition
            x = self.varphi
            Pi = np.pi
            # Run through different cases 
            if self.nfp==1:
                if self.k_buffer==1:
                    self.alpha_iota = -(125*(1728*np.sin(x) + 216*np.sin(2*x) + 64*np.sin(3*x) + 27*np.sin(4*x)) + 1728*np.sin(5*x))/(18000*np.pi*np.pi)
                    self.alpha_notIota =  -(125*(72*np.pi*np.pi*(np.pi + 2*x) + 1728*np.sin(x) + 216*np.sin(2*x) + 64*np.sin(3*x) + 27*np.sin(4*x)) + 1728*np.sin(5*x))/(18000*np.pi*np.pi)
                elif self.k_buffer==3:
                    self.alpha_iota = (-21*(8*(120 - 20*Pi**2 + Pi**4) + (15 + 2*Pi**2*(-5 + Pi**2))*np.cos(x))*np.sin(x))/(2.*Pi**6) - \
                        (28*(40 - 60*Pi**2 + 27*Pi**4)*np.sin(3*x))/(243.*Pi**6) - \
                        (21*(15 - 40*Pi**2 + 32*Pi**4)*np.sin(4*x))/(512.*Pi**6) - \
                        (84*(24 + 25*Pi**2*(-4 + 5*Pi**2))*np.sin(5*x))/(15625.*Pi**6)
                    self.alpha_notIota =          -Pi/2. - x - (84*(120 - 20*Pi**2 + Pi**4)*np.sin(x))/Pi**6 - \
                        (21*(15 + 2*Pi**2*(-5 + Pi**2))*np.sin(2*x))/(4.*Pi**6) - \
                        (28*(40 - 60*Pi**2 + 27*Pi**4)*np.sin(3*x))/(243.*Pi**6) - \
                        (21*(15 - 40*Pi**2 + 32*Pi**4)*np.sin(4*x))/(512.*Pi**6) - \
                        (84*(24 + 25*Pi**2*(-4 + 5*Pi**2))*np.sin(5*x))/(15625.*Pi**6)
                else: 
                    logging.raiseExceptions("Not implemented yet")
            elif self.nfp==2:
                if self.k_buffer==1:
                    self.alpha_iota = -(6*np.sin(self.nfp*x) + 3*np.sin(2*self.nfp*x)/4 + 2*np.sin(3*self.nfp*x)/9 + 3*np.sin(4*self.nfp*x)/32 + 6*np.sin(5*self.nfp*x)/125) / (np.pi * np.pi)
                    self.alpha_notIota = helicity * self.nfp * x + np.pi/2 * (1 + 2*helicity) - helicity * 2 * self.alpha_iota
                else: 
                    logging.raiseExceptions("Not implemented yet")
            elif self.nfp==3:
                if self.k_buffer==1:
                    self.alpha_iota = -(4*np.sin(self.nfp*x) + np.sin(2*self.nfp*x)/2 + 4*np.sin(3*self.nfp*x)/27 + np.sin(4*self.nfp*x)/16 + 4*np.sin(5*self.nfp*x)/125) / (np.pi * np.pi)
                    self.alpha_notIota = helicity * self.nfp * x + np.pi/2 * (1 + 2*helicity) - helicity * 2 * self.alpha_iota
                else: 
                    logging.raiseExceptions("Not implemented yet")
            else: 
                logging.raiseExceptions("Not implemented yet")
            # Compute derivatives separating the secular part of the expressions
            self.d_alpha_iota_d_varphi = np.matmul(self.d_d_varphi, self.alpha_iota)
            self.d_alpha_notIota_d_varphi = helicity * self.nfp + np.matmul(self.d_d_varphi, self.alpha_notIota - self.varphi * helicity * self.nfp)

        ## Alternative buffer region using the simplest Fourier form ##
        elif buffer_method == 'simple-fourier':
            # Buffer properties 
            self.k_buffer = self.buffer_details["k_buffer"]
            # Prepare functions to construct alpha
            def construct_alpha_iota(k_order, nfp, phi):
                # Build the shape of the region once (it is independent of the iota)
                def build_constraint_matrix(k_order):
                    # Number of modes
                    N_modes = k_order

                    # Mode array
                    mode_array = np.arange(1, N_modes+1)

                    # Mode grid (including the nfp for N)
                    N, J = np.meshgrid(mode_array, mode_array)

                    # Matrix S
                    S_mat = N**(2*J-1)*(-1)**(J-1)

                    return S_mat

                # Find Fourier coefficients alpha
                def construct_fourier_alpha(k_order, nfp):
                    # Number of modes
                    N_modes = k_order

                    if not isinstance(k_order, int) or k_order <= 0:
                        raise TypeError('k_order input is not an integer or <= 0!')
                    if N_modes == 1:
                        coeffs = np.array([1])
                    else:
                        # RHS of linear system
                        rhs = np.zeros(N_modes)
                        rhs[0] = 1
                        
                        # Find constraint matrix
                        S_mat = build_constraint_matrix(k_order)

                        # Solve the system of equations
                        coeffs = np.array(solve(S_mat, rhs))

                    return coeffs/nfp

                # Evaluate Fourier coefficients in a grid
                def evaluate_fourier_grid(coeffs, nfp, nphi = 100, phi_in = None, derivative = 0):
                    ## Define grid for evaluation ##
                    if isinstance(phi_in, list) or isinstance(phi_in, np.ndarray):
                        # If a grid is provided, use it
                        phi = phi_in
                    else:
                        # If no grid is provided, construct grid
                        phi = np.linspace(-1, 1, nphi, endpoint = False) * np.pi/nfp

                    ## Evaluate on grid ##
                    # Number of coefficients
                    N_coeff = len(coeffs)

                    if np.mod(derivative, 2) == 0:
                        # Sum over Fourier components
                        eval_coeffs = np.sum([coeffs[j]*((j+1)*nfp)**derivative*(-1)**(derivative/2)*np.sin((j+1)*nfp*phi) for j in range(N_coeff)], axis = 0)
                    else:
                        # Sum over Fourier components
                        eval_coeffs = np.sum([coeffs[j]*((j+1)*nfp)**derivative*(-1)**((derivative-1)/2)*np.cos((j+1)*nfp*phi) for j in range(N_coeff)], axis = 0)

                    return phi, eval_coeffs
                
                # Construct Fourier components of alpha_iota
                coeffs = construct_fourier_alpha(k_order, nfp)
                # Evaluate alpha_iota
                _, alpha_iota = evaluate_fourier_grid(coeffs, self.nfp, phi_in = phi, derivative = 0)
                # Evaluate d_alpha_iota_d_varphi
                _, d_alpha_iota_d_varphi = evaluate_fourier_grid(coeffs, self.nfp, phi_in = phi, derivative = 1)

                return alpha_iota, d_alpha_iota_d_varphi
            
            # Construct alpha_iota
            alpha_iota, d_alpha_iota_d_varphi = construct_alpha_iota(self.k_buffer, self.nfp, self.varphi-np.pi/self.nfp)
            # Construct alpha_notIota
            alpha_notIota = np.pi * (2*helicity + 1/2) + helicity * self.nfp * ((self.varphi-np.pi/self.nfp) - alpha_iota)
            d_alpha_notIota_d_varphi = helicity * self.nfp * (1 - d_alpha_iota_d_varphi)

            # Save to self
            self.alpha_iota = alpha_iota
            self.d_alpha_iota_d_varphi = d_alpha_iota_d_varphi
            self.alpha_notIota = alpha_notIota
            self.d_alpha_notIota_d_varphi = d_alpha_notIota_d_varphi
        
        else:
            raise KeyError('Unrecognised buffer region completion! Must be one of buffer, non-zone, non-zone-smoother, non-zone-fourier or simple-fourier.')
    
    # Calculate alpha putting the iota and not_iota pieces together
    self.alpha = self.alpha_iota * iota + self.alpha_notIota

    return

def _jacobian(self, x):
    """
    Compute the Jacobian matrix for solving the sigma equation. x is
    the state vector, corresponding to sigma on the phi grid,
    except that the first element of x is actually iota.
    """
    sigma = np.copy(x[1::])

    # d (Riccati equation) / d sigma:
    jac = np.copy(self.d_d_varphi)
    for j in range(self.nphi):
        jac[j, j] += self.gamma [j] * 2 * sigma[j]

    # d (Riccati equation) / d iota:
    if self.omn == True:
        if isinstance(self.alpha_in, dict):
            # If alpha is fixed, then the variation of the Riccati equation only explicit iota in gamma = iota - alpha'
            gamma_iota = 1
        else:
            gamma_iota = self.gamma_iota
    else:
        gamma_iota = 1
    jac = np.append(np.transpose([gamma_iota * (self.etabar_squared_over_curvature_squared * self.etabar_squared_over_curvature_squared + \
                                                1 + sigma * sigma)]),jac,axis=1)

    # d (sigma[0]-sigma0) / dsigma:
    jac = np.append(jac,[np.append(0,self.interpolateTo0)],axis=0)

    #logger.debug("_jacobian called with x={}, jac={}".format(x, jac))
    return jac

def solve_sigma_equation(self):
    """
    Solve the sigma equation.
    """
    x0 = np.full(self.nphi+1, self.sigma0)
    x0[0] = 0 # Initial guess for iota

    """
    soln = scipy.optimize.root(self._residual, x0, jac=self._jacobian, method='lm')
    self.iota = soln.x[0]
    self.sigma = np.copy(soln.x)
    self.sigma[0] = self.sigma0
    """
    self.sigma = newton(self._residual, x0, jac=self._jacobian)
    self.iota = self.sigma[0]
    self.iotaN = self.iota + self.helicity * self.nfp
    self.sigma = self.sigma[1::]

def _determine_helicity(self):
    """
    Determine the integer N associated with the type of quasisymmetry
    by counting the number of times the normal vector rotates
    poloidally as you follow the axis around toroidally. If non-integer
    elicity (because of sign flip in QI), this does not work
    """
    quadrant = np.zeros(self.nphi + 1)
    for j in range(self.nphi):
        if self.normal_cylindrical[j,0] >= 0:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 1
            else:
                quadrant[j] = 4
        else:
            if self.normal_cylindrical[j,2] >= 0:
                quadrant[j] = 2
            else:
                quadrant[j] = 3
    quadrant[self.nphi] = quadrant[0]

    counter = 0
    for j in range(self.nphi):
        if quadrant[j] == 4 and quadrant[j+1] == 1:
            counter += 1
        elif quadrant[j] == 1 and quadrant[j+1] == 4:
            counter -= 1
        else:
            counter += quadrant[j+1] - quadrant[j]

    # It is necessary to flip the sign of axis_helicity in order
    # to maintain "iota_N = iota + axis_helicity" under the parity
    # transformations.
    counter *= self.spsi * self.sG
    self.helicity = counter / 4

def r1_diagnostics(self):
    """
    Compute various properties of the O(r^1) solution, once sigma and
    iota are solved for.
    """
    ################
    # CONSTRUCT B1 #
    ################
    if self.omn:
        # Make spline for d (in phi)
        self.d_spline = self.convert_to_spline(self.d)
        # self.alpha_tilde = self.alpha # -self.N_helicity*self.varphi

        # Cos/sin of alpha (defined respect to θ)
        self.cos_alpha_spline = self.convert_to_spline(np.cos(self.alpha))
        self.sin_alpha_spline = self.convert_to_spline(np.sin(self.alpha))

        # Define angle as: θ - α = χ - (α - Νφ) = χ - angle <- angle is actually periodic
        angle = self.alpha - (-self.helicity * self.nfp * self.varphi)
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)

        # First order form of B: B=B0*[1+d cos(θ-α)]. Define QI labels as harmonics in θ-α
        B1sQI    = 0  
        B1cQI    = self.B0 * self.d

        # Construct B1 components in the helical basis
        self.B1c = (B1cQI * cosangle - B1sQI * sinangle)
        self.B1s = (B1sQI * cosangle + B1cQI * sinangle)
        
    else:
        # Cos/sin of alpha (defined respect to θ) - for ideal QS it is Nφ
        self.alpha = (-self.helicity * self.nfp * self.varphi)
        self.cos_alpha_tilde_spline = self.convert_to_spline(np.cos(self.alpha))
        self.sin_alpha_tilde_spline = self.convert_to_spline(np.sin(self.alpha))

        # Define angle as: θ = χ - (α - Νφ) = χ - angle <- angle is actually periodic
        angle = 0
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)

        # Construct B1 components in the helical basis
        self.B1c = self.etabar * self.B0
        self.B1s = 0.0

    # Make splines for B1 : if float, it also works 
    self.B1c_spline = self.convert_to_spline(self.B1c)
    self.B1s_spline = self.convert_to_spline(self.B1s)

    ################
    # CONSTRUCT X1 #
    ################
    # We consider the definition of X1 in the helical angle χ = θ - Νφ
    # X1 = X1c*cos(χ) + X1s*sin(χ)
    # Instead of using the expressions in Eq.(A22) from [Landreman, Sengupta (2019)] explicitly
    # self.X1c = self.B1c / (self.curvature * self.B0)
    # self.X1s = self.B1s / (self.curvature * self.B0)
    # we use d_bar = d/κ to try to avoid by zero
    self.X1c = self.d_bar * cosangle
    self.X1s = self.d_bar * sinangle

    ################
    # CONSTRUCT Y1 #
    ################
    # Defining everything in χ
    # Y1 = Y1c*cos(χ) + Y1s*sin(χ)
    # Instead of using Eq.(A25) from [Landreman, Sengupta (2019)]
    # self.Y1s = self.sG * self.Bbar * self.curvature * ( self.B1c + self.B1s * self.sigma) / \
    #         (self.B1c * self.B1c + self.B1s * self.B1s)# + 1e-30)
    # self.Y1c = self.sG * self.Bbar * self.curvature * (-self.B1s + self.B1c * self.sigma) / \
    #         (self.B1c * self.B1c + self.B1s * self.B1s)# + 1e-30)
    # we instead express things in terms of d_bar = d/κ to avoid division by 0
    self.Y1s = self.sG * self.Bbar * ( cosangle + sinangle * self.sigma) / (self.B0 * self.d_bar)
    self.Y1c = self.sG * self.Bbar * (-sinangle + cosangle * self.sigma) / (self.B0 * self.d_bar)

    ###############
    # UNTWISTED θ #
    ###############
    # If helicity is nonzero, then the original X1s/X1c/Y1s/Y1c variables are defined with respect to a "poloidal" angle χ that
    # is actually helical. Here we convert to an untwisted poloidal angle.
    if self.helicity == 0:
        # If not helicity χ = θ
        self.X1s_untwisted = self.X1s
        self.X1c_untwisted = self.X1c
        self.Y1s_untwisted = self.Y1s
        self.Y1c_untwisted = self.Y1c
    else:
        # Define angle = Nφ 
        angle = -self.helicity * self.nfp * self.varphi
        # Evaluate cos/sin
        sinangle = np.sin(angle)
        cosangle = np.cos(angle)
        # Use multiple angle formulas to decompose sin(θ-Νφ) and cos(θ-Νφ)
        self.X1s_untwisted = (self.X1s *   cosangle  + self.X1c * sinangle)
        self.X1c_untwisted = (self.X1s * (-sinangle) + self.X1c * cosangle)
        self.Y1s_untwisted = (self.Y1s *   cosangle  + self.Y1c * sinangle)
        self.Y1c_untwisted = (self.Y1s * (-sinangle) + self.Y1c * cosangle)

    # Compute some derivatives for later
    self.d_X1c_d_varphi = np.matmul(self.d_d_varphi, self.X1c)
    self.d_X1s_d_varphi = np.matmul(self.d_d_varphi, self.X1s)
    self.d_Y1s_d_varphi = np.matmul(self.d_d_varphi, self.Y1s)
    self.d_Y1c_d_varphi = np.matmul(self.d_d_varphi, self.Y1c)

    ############################
    # COMPUTE ELLIPSE FEATURES #
    ############################
    ## Commpute elongation ##
    # Use (X,Y) for elongation in the plane perpendicular to the magnetic axis.
    p = self.X1s * self.X1s + self.X1c * self.X1c + self.Y1s * self.Y1s + self.Y1c * self.Y1c
    q = self.X1s * self.Y1c - self.X1c * self.Y1s
    self.elongation = (p + np.sqrt(p * p - 4 * q * q)) / (2 * np.abs(q))
    varphi_ext = np.append(self.varphi, self.varphi[0] + 2*np.pi/self.nfp)
    self.mean_elongation = np.trapz(np.append(self.elongation * self.d_l_d_varphi,self.elongation[0] * self.d_l_d_varphi[0]), varphi_ext) /\
          np.trapz(np.append(self.d_l_d_varphi,self.d_l_d_varphi[0]), varphi_ext)
    # index = np.argmax(self.elongation)
    self.max_elongation = -fourier_minimum(-self.elongation)

    ## Other ellipse features ##
    # Area of the ellipse in the plane perpendicular to the magnetic axis
    self.ellipse_area = np.pi * self.sG * self.Bbar / self.B0 + 2 * (self.X1c * self.Y1c - self.X1s * self.Y1s)

    ## Compute the grad B tensor ##
    self.calculate_grad_B_tensor()


