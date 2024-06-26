"""
This module contains the top-level routines for the quasisymmetric
stellarator construction.
"""

import logging
import numpy as np
from scipy.io import netcdf
import matplotlib.pyplot as plt

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Qic():
    """
    This is the main class for representing the quasisymmetric
    stellarator construction.
    """
    
    # Import methods that are defined in separate files:
    from .init_axis import init_axis, convert_to_spline
    from .calculate_r1 import _residual, _jacobian, solve_sigma_equation, \
        _determine_helicity, r1_diagnostics
    from .grad_B_tensor import calculate_grad_B_tensor, calculate_grad_grad_B_tensor, \
        Bfield_cylindrical, Bfield_cartesian, grad_B_tensor_cartesian, \
        grad_grad_B_tensor_cylindrical, grad_grad_B_tensor_cartesian
    from .calculate_r2 import calculate_r2, construct_qi_r2, evaluate_X2c_X2s_QI
    from .calculate_r3 import calculate_r3
    from .mercier import mercier
    from .plot import plot, get_boundary, B_fieldline, B_contour, plot_axis
    from .r_singularity import calculate_r_singularity
    from .plot import plot, plot_boundary, get_boundary, B_fieldline, B_contour, plot_axis, B_densityplot
    from .fourier_interpolation import fourier_interpolation
    from .Frenet_to_cylindrical import Frenet_to_cylindrical, to_RZ
    from .optimize_nae import optimise_params, min_geo_qi_consistency
    from .to_vmec import to_vmec
    from .util import B_mag
    from .input_structure import evaluate_input_on_grid
    
    def __init__(self, 
                 omn = True, order = "r1",
                 nphi = 31, phi_shift = 1/3.0, nfp=1, diff_finite = False,
                 frenet = False, axis_complete = True,
                 Raxis = {"type": 'fourier', "input_value": {"cos": [1.0, 0.1], "sin": []}},
                 Zaxis = {"type": 'fourier', "input_value": {"cos": [], "sin": [0.0, 0.05]}},
                 curvature = None, torsion = None, ell = None, L = None, varphi = None, helicity = None,
                 B0 = {"type": 'scalar', "input_value": 1.0},
                 sG=1, spsi=1,
                 d_over_curvature = {"type": 'scalar', "input_value": 0.1}, d =  {"type": 'scalar', "input_value": 1.0}, k_second_order_SS = 0.0,
                 alpha_tilde = None, omn_buffer = {"omn_method": 'buffer', "k_buffer": 1, "p_buffer": 2, "delta": np.pi/5},
                 sigma0=0., 
                 I2=0., p2=0.,
                 B2s = 0.0, B2c = 0.0,
                 X2s = {"type": 'scalar', "input_value": 0.0}, X2c = {"type": 'scalar', "input_value": 0.0}):
        """
        Create a near-axis stellarator.
        """
        #####################
        # TYPE OF NEAR-AXIS #
        #####################
        # Check whether the near-axis construction is for a QI field (omn = True) or QS one (omn = False)
        self.omn = omn
        # Order to which near-axis is to be computed
        self.order = order

        #####################
        # GRID CONSTRUCTION #
        #####################
        # Number of field periods
        self.nfp = nfp

        # Force nphi to be odd even in the case of Frenet; this is important for the appropriate behaviour of the derivatives in the 
        # Fourier basis later in the sigma solve. 
        nphi_in = nphi
        if np.mod(nphi, 2) == 0:
            nphi -= 1
        self.nphi = nphi

        # Make phi grid
        phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        # Define grid step
        self.d_phi = phi[1] - phi[0]

        # Consider shift of the grid to potentially avoid dividing by zero at points of vanishing curvature
        if omn==True:
            self.phi_shift = phi_shift
            if phi_shift==0:
                print('WARNING! phi_shift = 0 may lead to problems wherever the curvature vanishes. It is recommended to use 1/3. ')
            # Define shifted array
            self.phi = phi + self.phi_shift*self.d_phi
            if frenet:
                # The input is considered to be in the unshifted grid always, but we need to change varphi so that we evaluate
                # on the appropriate shifted grid later.
                varphi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        else:
            # If QS set shift to 0
            self.phi_shift = 0
            self.phi = phi

        # Input that determines whether to use the spectral differentiation matrix or finite differences:
        # False - spectral, 2 - 2nd order centred differences, 4 - 4nd order centred differences,
        # 6 - 6nd order centred differences, else 4th order. It is important for higher order and the non-smoothness
        # of the construction
        self.diff_finite = diff_finite

        ###################
        # AXIS PARAMETERS #
        ###################
        # Flag for Frenet or cylindrical description of axis
        self.frenet = frenet

        # If frenet = False, then usual cylindrical description of the axis
        if not frenet:
            # Complete axis to enforce vanishing curvature
            if axis_complete == True and Raxis["type"] == 'fourier':
                self.axis_complete = True
            else:
                self.axis_complete = False

            # Read the input R and Z axis parameters 
            self.Raxis = Raxis
            self.Zaxis = Zaxis

            self.curvature_in = None
            self.torsion_in = None
            self.ell_in = None
            self.L_in = None # Length of curve in period
            self.helicity_in = None
            
        else:
            # In Frenet mode, the inputs are curvature, torsion and ell, all in the varphi (Boozer phi) coordinate
            self.curvature_in = curvature
            self.torsion_in = torsion
            self.ell_in = ell
            self.L_in = L # Length of curve in period
            self.varphi = varphi
            self.helicity_in = helicity
            self.flag_half = (np.mod(helicity, 1) == 0.5) # If half helicity, to take into account the flip of sign
            self.axis_complete = False

            self.Raxis = None
            self.Zaxis = None

        self.lasym = True # Temporary fix
        ##########################
        # MAGNETIC FIELD ON AXIS #
        ##########################
        # Read B0 as an input
        if isinstance(B0, dict):
            self.B0_in = B0
        else:
            # TO DO: I could use a translation in case an array or scalar is provide. 
            raise KeyError('Please provide B0 as a dictionary.')

        # If QS, make sure it is a scalar
        if omn == False:
            if not (B0["type"] == 'scalar'):
                raise ValueError('Input B0 is not a scalar as it should for a QS field.')
            self.B0 = B0["input_value"]
        else:
            # If QI, evaluate B0 for now on the phi grid and store in B0
            self.B0 = self.evaluate_input_on_grid(B0, self.phi)

            # Define the well depth
            self.B0_well_depth = 0.5 * (self.B0.max() - self.B0.min())

        ####################
        # SIGN DEFINITIONS #
        ####################
        # Check they are only signs
        if sG != 1 and sG != -1:
            raise ValueError('sG must be +1 or -1')
        
        if spsi != 1 and spsi != -1:
            raise ValueError('spsi must be +1 or -1')
        
        # Store signs
        self.sG = sG
        self.spsi = spsi

        #################
        # FIRST ORDER B #
        #################
        # Distinguish between QS and QI
        if not omn:
            # If QS, define etabar from d
            if d["type"] == 'scalar':
                self.etabar = d["input_value"]
                self.d_in = d
                self.d = self.evaluate_input_on_grid(d, self.phi)
                self.d_over_curvature_in = None
            else:
                raise ValueError('Please provide a value for etabar in d as a scalar.')
        # If it is QI
        else:
            self.etabar = 0.0
            self.k_second_order_SS = k_second_order_SS
            # If d_over_curvature and d are both provided then use both (the total d will be the sum of both) 
            if isinstance(d_over_curvature, dict):
                self.d_over_curvature_in = d_over_curvature
            else:
                self.d_over_curvature_in = None

            if isinstance(d, dict):
                self.d_in = d
                self.d = self.evaluate_input_on_grid(d, self.phi) # Temporary value
                print('WARNING! Both d and d_bar have been provided, which could give unwanted problems.')
            else:
                self.d_in = None
            if self.d_over_curvature_in == None:
                if self.d_in == None:
                    raise ValueError('The first order B1 field is not specified; please provide d or d_over_curvature as dictionaries.')
                else:
                    raise Warning("Careful with providing d as an input for it must match curvature.")
                    
        # Define the function alpha (or buffer regions if needed) for QI
        if self.omn:
            # If alpha is provided as an input
            if isinstance(alpha_tilde, dict):
                if helicity == None:
                    raise ValueError('Missing helicity input if alpha is given (note that alpha must be provided as a periodic function and the code adds the Nφ iece).')
                self.alpha_in = alpha_tilde
                self.alpha = self.evaluate_input_on_grid(alpha_tilde, self.phi) - self.varphi * self.helicity_in * self.nfp

            # If alpha is not provided, then we need details of the buffer construction
            else:
                self.alpha_in = None
                self.buffer_details = omn_buffer
                if "delta" in omn_buffer:
                    delta = omn_buffer["delta"]
                else:
                    delta = 0
                self.delta = delta # min(abs(delta),0.95*np.pi/nfp)
        else:
            self.alpha_in = None

        #######################
        # FIRST ORDER SHAPING #
        #######################
        # Value for σ(0), whose non-zero value braks stellarator symmetry
        self.sigma0 = sigma0

        #######################
        # SECOND ORDER INPUTS #
        #######################
        # Equilibrium scalars
        self.I2 = I2
        self.p2 = p2

        # Inputs of B2 or X2 depending on whether QI or QS is considered
        if not omn:
            # If QS, we consider the inputs to be B2 (which are scalars and not dict)
            # For now define these as _in, to see if I can use B2s and B2c for both cases as outputs
            self.B2c_in = B2c
            self.B2s_in = B2s
            self.X2s_in = None
            self.X2c_in = None
        else:
            # If QI, the inputs should be X2s and X2c, and we shall ignore the input values for B2
            self.X2s_in = X2s
            self.X2c_in = X2c
            self.B2c_in = None
            self.B2s_in = None

        ##########
        # OTHERS #
        ##########
        self.min_R0_threshold = 0.3
        self.min_Z0_threshold = 0.3

        ###########################
        # COMPLETE INITIALISATION #
        ###########################
        # Create names for optimisation
        self._set_names()
        # Run calculations
        self.calculate()

    def change_nfourier(self, nfourier_new):
        """
        Resize the arrays of Fourier amplitudes. You can either increase
        or decrease nfourier.
        """
        rc_old = self.rc
        rs_old = self.rs
        zc_old = self.zc
        zs_old = self.zs
        index = np.min((self.nfourier, nfourier_new))
        self.rc = np.zeros(nfourier_new)
        self.rs = np.zeros(nfourier_new)
        self.zc = np.zeros(nfourier_new)
        self.zs = np.zeros(nfourier_new)
        self.rc[:index] = rc_old[:index]
        self.rs[:index] = rs_old[:index]
        self.zc[:index] = zc_old[:index]
        self.zs[:index] = zs_old[:index]
        nfourier_old = self.nfourier
        self.nfourier = nfourier_new
        self._set_names()
        # No need to recalculate if we increased the Fourier
        # resolution, only if we decreased it.
        self.calculate()

    def calculate(self):
        """
        Driver for the main calculations.
        """
        self.init_axis(omn_complete = self.axis_complete)
        self.solve_sigma_equation()
        self.r1_diagnostics()
        if self.order != 'r1':
            self.calculate_r2()
            if self.order == 'r3':
                self.calculate_r3()
    
    def get_dofs(self):
        """
        Return a 1D numpy vector of all possible optimizable
        degrees-of-freedom, for simsopt.
        """
        def select_contents_inputs(input_dict):
            if isinstance(input_dict, dict):
                if input_dict["type"] == 'scalar':
                    val = [input_dict["input_value"]]
                elif input_dict["type"] == 'fourier':
                    val = input_dict["input_value"]["cos"].copy()
                    val = np.concatenate((val, input_dict["input_value"]["sin"]))
                else:
                    val = input_dict["input_value"].copy()
            else:
                # In case None or otherwise, do not add anything
                val = []

            return np.array(val)
        
        dofs = np.array([])
        if self.frenet:
            # varphi is left fixed, but curvature, torsion and ell may change. Note that the curve may not close!
            dofs = np.concatenate((dofs, select_contents_inputs(self.curvature_in)))
            dofs = np.concatenate((dofs, select_contents_inputs(self.torsion_in)))
            dofs = np.concatenate((dofs, select_contents_inputs(self.ell_in)))
        else:
            dofs = np.concatenate((dofs, select_contents_inputs(self.Raxis)))
            dofs = np.concatenate((dofs, select_contents_inputs(self.Zaxis)))

        dofs = np.concatenate((dofs, select_contents_inputs(self.B0_in)))
        dofs = np.concatenate((dofs, select_contents_inputs(self.d_in)))
        dofs = np.concatenate((dofs, select_contents_inputs(self.d_over_curvature_in)))
        dofs = np.concatenate((dofs, select_contents_inputs(self.alpha_in)))
        if not self.omn:
            dofs = np.concatenate((dofs, np.array([self.B2c_in, self.B2s_in])))
        else:
            dofs = np.concatenate((dofs, select_contents_inputs(self.X2c_in)))
            dofs = np.concatenate((dofs, select_contents_inputs(self.X2s_in)))
        dofs = np.concatenate((dofs, np.array([self.sigma0, self.p2, self.I2])))
        if not isinstance(self.alpha_in, dict):
            dofs = np.concatenate((dofs, np.array([self.delta])))

        assert dofs.ndim == 1 and dofs.size != 1
        assert dofs.size == len(self.names)
        return dofs

    def set_dofs(self, x, re_evaluate = 'all'):
        """
        For interaction with simsopt, set the optimizable degrees of
        freedom from a 1D numpy vector.
        """
        assert len(x) == len(self.names)

        def set_input(input_dict, x, init):
            if isinstance(input_dict, dict):
                if input_dict["type"] == 'scalar':
                    input_dict["input_value"] = x[init]
                    len_input = 1
                elif input_dict["type"] == 'fourier':
                    length_cos = len(input_dict["input_value"]["cos"])
                    input_dict["input_value"]["cos"] = x[init:init+length_cos]
                    length_sin = len(input_dict["input_value"]["sin"])
                    input_dict["input_value"]["sin"] = x[init+length_cos:init+length_cos+length_sin]
                    len_input = length_cos + length_sin
                else:
                    len_input = len(input_dict["input_value"])
                    input_dict["input_value"] = x[init:init+len_input]
            else:
                # In case None or otherwise, do not add anything
                len_input = 0
            return init + len_input
        
        ind = 0
        if self.frenet:
            # varphi is left fixed, but curvature, torsion and ell may change. Note that the curve may not close!
            ind = set_input(self.curvature_in, x, ind)
            ind = set_input(self.torsion_in, x, ind)
            ind = set_input(self.ell_in, x, ind)
        else:
            ind = set_input(self.Raxis, x, ind)
            ind = set_input(self.Zaxis, x, ind)
        
        ind = set_input(self.B0_in, x, ind)
        ind = set_input(self.d_in, x, ind)
        ind = set_input(self.d_over_curvature_in, x, ind)
        ind = set_input(self.alpha_in, x, ind)
        if not self.omn:
            self.B2c_in = x[ind]
            self.B2s_in = x[ind+1]
            ind = ind + 2
        else:
            ind = set_input(self.X2c_in, x, ind)
            ind = set_input(self.X2s_in, x, ind)

        self.sigma0, self.p2, self.I2 = x[ind:ind+3]
        ind = ind+3

        if not isinstance(self.alpha_in, dict):
            self.delta = x[ind]
            self.buffer_details["delta"] = x[ind]
            ind = ind+1

        assert ind == len(self.names)
        
        ##########################
        # SET NEW DERIVED VALUES #
        ##########################
        # Not sure if this is needed or legacy
        # Set B0
        if self.omn == False:
            self.B0 = self.B0_in["input_value"]
        else:
            self.B0 = self.evaluate_input_on_grid(self.B0_in, self.phi)
            # Define the well depth
            self.B0_well_depth = 0.5 * (self.B0.max() - self.B0.min())

        # Set d
        if not self.omn:
            # If QS, define etabar from d
            self.etabar = self.d_in["input_value"]
            self.d = self.evaluate_input_on_grid(self.d_in, self.phi)
        else:
            # If it is QI
            if isinstance(self.d_in, dict):
                self.d = self.evaluate_input_on_grid(self.d_in, self.phi)
        
        if self.omn:
            # If alpha is provided as an input
            if isinstance(self.alpha_in, dict):
                if self.helicity_in == None:
                    raise ValueError('Missing helicity input if alpha is given (note that alpha must be provided as a periodic function and the code adds the Nφ iece).')
                self.alpha = self.evaluate_input_on_grid(self.alpha_in, self.phi) - self.varphi * self.helicity_in * self.nfp

        # Unsure if needed
        # self.B1s = self.B0 * self.d * np.sin(self.alpha)
        # self.B1c = self.B0 * self.d * np.cos(self.alpha)

        # Compute the quantities thereof, regardless of the properties changed (this could be unnecesaily cumbersome)
        if re_evaluate == 'all' or re_evaluate == 'r1':
            self.calculate()
        elif re_evaluate == 'r2':
            self.calculate_r2()
            if self.order == 'r3':
                self.calculate_r3()
        elif re_evaluate == 'r3':
            self.calculate_r3()
        else:
            raise NameError("The order specified for re_evaluate in set_dofs is not recognised: it must be one of all, r1, r2, or r3")

        logger.info('set_dofs called with x={}. Now iota={}, elongation={}'.format(x, self.iota, self.max_elongation))

    def add_input_string(self, input_dict, name_str):
        """
        Different addition of parameters depending on the type of input
        """
        if isinstance(input_dict, dict):
            if input_dict["type"] == 'scalar':
                new_str = [name_str]
            elif input_dict["type"] == 'fourier':
                length_cos = len(input_dict["input_value"]["cos"])
                cos_str = [name_str + 'c({})'.format(j) for j in range(length_cos)]
                length_sin = len(input_dict["input_value"]["sin"])
                sin_str = [name_str + 's({})'.format(j) for j in range(length_sin)]
                new_str = cos_str + sin_str
            else:
                length_array = len(input_dict["input_value"])
                new_str = [name_str + '({})'.format(j) for j in range(length_array)]
        else:
            # In case None or otherwise, do not add anything
            new_str = []

        return new_str
     
    def _set_names(self):
        """
        For simsopt, sets the list of names for each degree of freedom.
        """        
        names = []
        if self.frenet:
            # varphi is left fixed, but curvature, torsion and ell may change. Note that the curve may not close!
            names += self.add_input_string(self.curvature_in, 'curv')
            names += self.add_input_string(self.torsion_in, 'tors')
            names += self.add_input_string(self.ell_in, 'ell')
        else:
            names += self.add_input_string(self.Raxis, 'r')
            names += self.add_input_string(self.Zaxis, 'z')
            # names += ['rc({})'.format(j) for j in range(self.nfourier)]
            # names += ['zs({})'.format(j) for j in range(self.nfourier)]
            # names += ['rs({})'.format(j) for j in range(self.nfourier)]
            # names += ['zc({})'.format(j) for j in range(self.nfourier)]
        
        names += self.add_input_string(self.B0_in, 'B0')
        names += self.add_input_string(self.d_in, 'd')
        names += self.add_input_string(self.d_over_curvature_in, 'd_over_curvature')
        names += self.add_input_string(self.alpha_in, 'alpha')
        if not self.omn:
            names += ['B2c','B2s']
        else:
            names += self.add_input_string(self.X2c_in, 'X2c')
            names += self.add_input_string(self.X2s_in, 'X2s')
        names += ['sigma0', 'p2', 'I2']
        if not isinstance(self.alpha_in, dict):
            names += ['delta']

        self.names = names

    @classmethod
    def from_paper(cls, name, **kwargs):
        """
        Get one of the configurations that has been used in our papers.
        Available values for ``name`` are
        ``"QI"``,
        ``"r1 section 5.1"``,
        ``"r1 section 5.2"``,
        ``"r1 section 5.3"``,
        ``"r2 section 5.1"``,
        ``"r2 section 5.2"``,
        ``"r2 section 5.3"``,
        ``"r2 section 5.4"``,
        ``"r2 section 5.5"``,
        These last 5 configurations can also be obtained by specifying an integer 1-5 for ``name``.
        The configurations that begin with ``"r1"`` refer to sections in 
        Landreman, Sengupta, and Plunk, Journal of Plasma Physics 85, 905850103 (2019).
        The configurations that begin with ``"r2"`` refer to sections in 
        Landreman and Sengupta, Journal of Plasma Physics 85, 815850601 (2019).
        The QI configuration refers to section 8.2 of
        Plunk, Landreman, and Helander, Journal of Plasma Physics 85, 905850602 (2019)

        You can specify any other arguments of the ``qic`` constructor
        in ``kwargs``. You can also use ``kwargs`` to override any of
        the properties of the configurations from the papers. For
        instance, you can modify the value of ``etabar`` in the first
        example using

        .. code-block::

          q = qic.Qic.from_paper('r1 section 5.1', etabar=1.1)
        """
        default_buffer_dict = {"omn_method": 'buffer', "k_buffer": 1, "p_buffer": 2, "delta": np.pi/5}

        def add_default_args(kwargs_old, **kwargs_new):
            """
            Take any key-value arguments in ``kwargs_new`` and treat them as
            defaults, adding them to the dict ``kwargs_old`` only if
            they are not specified there.
            """
            for key in kwargs_new:
                if key not in kwargs_old:
                    kwargs_old[key] = kwargs_new[key]

                    
        if name == "r1 section 5.1":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.1 """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1, 0.045], "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [], "sin": [0, -0.045]}}
            d = {"type": 'scalar',
                     "input_value": -0.9}
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, omn = False, nfp=3, d = d)
                
        elif name == "r1 section 5.2":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.2 """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1, 0.265], "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [], "sin": [0, -0.21]}}
            d = {"type": 'scalar',
                     "input_value": -2.25}
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, omn = False, nfp=4, d = d)
                
        elif name == "r1 section 5.3":
            """ The configuration from Landreman, Sengupta, Plunk (2019), section 5.3 """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1, 0.042], "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [0, -0.025], "sin": [0, -0.042]}}
            d = {"type": 'scalar',
                     "input_value": -1.1}
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, omn = False, nfp=3, d = d, sigma0=-0.6)
                
        elif name == "r2 section 5.1" or name == '5.1' or name == 1:
            """ The configuration from Landreman & Sengupta (2019), section 5.1 """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1, 0.155, 0.0102], "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [], "sin": [0, 0.154, 0.0111]}}
            d = {"type": 'scalar',
                     "input_value": 0.64}
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, omn = False, nfp=2, d = d, B2c=-0.00322, order='r3')
            
        elif name == "r2 section 5.2" or name == '5.2' or name == 2:
            """ The configuration from Landreman & Sengupta (2019), section 5.2 """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1, 0.173, 0.0168, 0.00101], "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [], "sin": [0, 0.159, 0.0165, 0.000985]}}
            d = {"type": 'scalar',
                     "input_value": 0.632}
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, omn = False, nfp=2, d = d, B2c = -0.158, order='r3')
                             
        elif name == "r2 section 5.3" or name == '5.3' or name == 3:
            """ The configuration from Landreman & Sengupta (2019), section 5.3 """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1, 0.09], "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [], "sin": [0, -0.09]}}
            d = {"type": 'scalar',
                     "input_value": 0.95}
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, omn = False, nfp=2, d = d, B2c = -0.7, I2=0.9, p2=-600000., order='r3')
                             
        elif name == "r2 section 5.4" or name == '5.4' or name == 4:
            """ The configuration from Landreman & Sengupta (2019), section 5.4 """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1, 0.17, 0.01804, 0.001409, 5.877e-05], "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [], "sin": [0, 0.1581, 0.01820, 0.001548, 7.772e-05]}}
            d = {"type": 'scalar',
                     "input_value": 1.569}
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, omn = False, nfp=4, d = d, B2c = 0.1348, order='r3')
                             
        elif name == "r2 section 5.5" or name == '5.5' or name == 5:
            """ The configuration from Landreman & Sengupta (2019), section 5.5 """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1, 0.3], "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [], "sin": [0, 0.3]}}
            d = {"type": 'scalar',
                     "input_value": 2.5}
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, omn = False, nfp=5, d = d, sigma0=0.3, B2c = 1., I2=1.6, \
                             B2s=3., p2=-0.5e7, order='r3')        
        
        elif name == "QI" or name == "QI r1 Plunk" or name == "QI Plunk":
            """ The configuration from Plunk, Landreman & Helander (2019), section 8.2 """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0, 0.0,-0.2 ], "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [], "sin": [ 0.0, 0.0, 0.35]}}
            B0 = {"type": 'fourier',
                     "input_value": {"cos": [1.0, 0.1], "sin": []}}
            d = {"type": 'fourier',
                     "input_value": {"cos": [], "sin": [0.0, 1.08, 0.26, 0.46]}}
            buffer_opt = default_buffer_dict.copy()
            buffer_opt["delta"] = 0.1 * 2*np.pi
            nphi    = 151
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, nfp=1, B0 = B0, d = d, d_over_curvature = None, nphi = nphi, \
                             omn_buffer = buffer_opt, omn = True)

        elif name == "QI r1 Jorge" or name == "QI NFP1 r1 Jorge" or name == "QI Jorge" or name == 6:
            """ The configuration from Jorge et al (2022) """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,0.0,-0.4056622889934463,0.0,0.07747378220100756,0.0,-0.007803860877024245,0.0,0.0,0.0,0.0,0.0,0.0 ],
                                     "sin": []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [],
                                     "sin": [ 0.0,0.0,-0.24769666390049602,0.0,0.06767352436978152,0.0,-0.006980621303449165,0.0,-0.0006816270917189934,0.0,-1.4512784317099981e-05,0.0,-2.839050532138523e-06 ]}}
            B0 = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,0.16915531046156507 ], "sin": []}}
            d = {"type": 'fourier',
                     "input_value": {"cos": [], 
                                     "sin": [ 0.0,0.003563114185517955,0.0002015921485566435,-0.0012178616509882368,-0.00011629450296628697,-8.255825435616736e-07,3.2011540526397e-06 ]}}
            d_over_curvature = {"type": 'scalar', "input_value": 0.5183783762725197}
        
            buffer_opt = default_buffer_dict.copy()
            buffer_opt["omn_method"] = 'non-zone'
            buffer_opt["delta"] = 0.1
            buffer_opt["k_buffer"] = 3
            nfp     = 1
            nphi    = 201

            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, nfp=nfp, B0 = B0, d = d, d_over_curvature = d_over_curvature, nphi = nphi, \
                             omn_buffer = buffer_opt, omn = True)

        elif name == "QI NFP1 r2":
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,0.0,-0.41599809655680886,0.0,0.08291443961920232,0.0,-0.008906891641686355,0.0,0.0,0.0,0.0,0.0,0.0 ],
                                     "sin": [ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ],
                                     "sin": [ 0.0,0.0,-0.28721210154364263,0.0,0.08425262593215394,0.0,-0.010427621520053335,0.0,-0.0008921610906627226,0.0,-6.357200965811029e-07,0.0,2.7316247301500753e-07 ]}}
            B0 = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,0.15824229612567256 ], "sin": []}}
            d = {"type": 'fourier',
                     "input_value": {"cos": [], 
                                     "sin": [ 0.0,-0.00023993050759319644,1.6644294162908823e-05,0.00012071143120099562,-1.1664837950174757e-05,-2.443821681789672e-05,2.0922298879435957e-06 ]}}
            d_over_curvature = {"type": 'scalar', "input_value": 0.48654821249917474}
        
            X2c = {"type": 'fourier',
                     "input_value": {"cos": [ -0.0007280714400220894,0.20739775852289746,0.05816363701644946,0.06465766308954603,0.006987357785313118,1.2229700694973357e-07,-3.057497440766065e-09,0.0 ], 
                                     "sin": [ 0.0,0.0,0.0,0.0 ]}}
            X2s = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,0.27368018673599265,-0.20986698715787325,0.048031641735420336,0.07269565329289157,1.3981498114634812e-07,-9.952017662433159e-10 ]}}
            buffer_opt = default_buffer_dict.copy()
            buffer_opt["omn_method"] = 'non-zone'
            buffer_opt["delta"] = 0.1
            buffer_opt["k_buffer"] = 3
            buffer_opt["p_buffer"] = 2
            sigma0 = 0.0
            nfp     = 1
            p2      =  0.0
            nphi    = 201
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, nfp=nfp, B0 = B0, d = d, d_over_curvature = d_over_curvature, nphi = nphi, \
                             omn_buffer = buffer_opt, sigma0 = sigma0, p2 = p2, X2c = X2c, X2s = X2s, omn = True, order = 'r3')
            
        elif name == "QI NFP2 r2":
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,0.0,-0.07764451554933544,0.0,0.005284971468552636,0.0,-0.00016252676632564814,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ],
                                     "sin":  [ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ]}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ],
                                     "sin": [ 0.0,0.0,-0.06525233925323416,0.0,0.005858113288916291,0.0,-0.0001930489465183875,0.0,-1.21045713465733e-06,0.0,-6.6162738585035e-08,0.0,-1.8633251242689778e-07,0.0,1.4688345268925702e-07,0.0,-8.600467886165271e-08,0.0,4.172537468496238e-08,0.0,-1.099753830863863e-08 ]}}
            B0 = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,0.12735237900304514 ], "sin": []}}
            
            d = {"type": 'fourier',
                     "input_value": {"cos": [], 
                                     "sin": [ 0.0,-5.067489975338647,0.2759212337742016,-0.1407115065170644,0.00180521570352059,-0.03135134464554904,0.009582569807320895,-0.004780243312143034,0.002241790407060276,-0.0006738437017134619,0.00031559081192998053 ]}}
            d_over_curvature = {"type": 'scalar', "input_value": -0.14601620836497467}
            k_second_order_SS = -25.137439389881692

            X2c = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0018400322140812674,-0.0013637739279265815,-0.0017961063281748597,-0.000855123667865997,-0.001412983361026517,-0.0010676686588779228,-0.0008117922713651492,-0.0002878689335032291,-0.0002515272886665927,-7.924709175875918e-05,-4.919421452969814e-05,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,2.7062914673236698,-0.9151373916194634,0.021394010521077745,-0.017469913902854437,0.03186670312840335,0.021102584055813403,0.0024194864183551515,-0.0059152315287890125,0.003709416127750524,0.010027743000785166,0.0,0.0,0.0,0.0 ]}}
            X2s = {"type": 'fourier',
                     "input_value": {"cos": [ 0.4445604502180231,0.13822067284200223,-0.561756934579829,0.2488873179399463,-0.14559282723014635,0.020548052084815048,-0.011070304464557718,0.004342889373034949,-0.0015730819049237866,0.0035406584522436986,0.002831887060104115,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,0.0012174780422017702,0.00026317725313621535,0.0002235661375254599,0.0006235230087895861,0.00021429298911807877,8.428032911991958e-05,-0.000142566391046771,-3.194627950185967e-05,-0.0001119389848119665,-6.226472957451552e-05 ]}}
            
            buffer_opt = default_buffer_dict.copy()
            buffer_opt["omn_method"] = 'non-zone-fourier'
            buffer_opt["delta"] = 0.8
            buffer_opt["k_buffer"] = 1
            buffer_opt["p_buffer"] = 2
            sigma0 = 0.0
            nfp     = 2
            p2      = 0.0
            nphi    = 201
            
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, nfp=nfp, B0 = B0, d = d, d_over_curvature = d_over_curvature, k_second_order_SS = k_second_order_SS, nphi = nphi, \
                             omn_buffer = buffer_opt, sigma0 = sigma0, p2 = p2, X2c = X2c, X2s = X2s, omn = True, order = 'r3')
            
        elif name == "LandremanPaul2021QA" or name == "precise QA":
            """
            A fit of the near-axis model to the quasi-axisymmetric
            configuration in Landreman & Paul, arXiv:2108.03711 (2021).

            The fit was performed to the boozmn data using the script
            20200621-01-Extract_B0_B1_B2_from_boozxform
            """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1.0038581971135636, 0.18400998741139907, 0.021723381370503204, 0.0025968236014410812, 0.00030601568477064874, 3.5540509760304384e-05, 4.102693907398271e-06, 5.154300428457222e-07, 4.8802742243232844e-08, 7.3011320375259876e-09],
                                     "sin":  []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [],
                                     "sin": [0.0, -0.1581148860568176, -0.02060702320552523, -0.002558840496952667, -0.0003061368667524159, -3.600111450532304e-05, -4.174376962124085e-06, -4.557462755956434e-07, -8.173481495049928e-08, -3.732477282851326e-09]}}
            B0 = {"type": 'scalar', "input_value": 1.006541121335688}
            d = {"type": 'scalar', "input_value": -0.6783912804454629}
            d_over_curvature = None
            B2c = 0.26859318908803137

            add_default_args(kwargs,
                             omn = False, nfp=2, nphi=99, order='r3',
                             Raxis = Raxis, Zaxis = Zaxis, B0 = B0, d=d, d_over_curvature = d_over_curvature,
                             B2c = B2c)

        elif name == "precise QA+well":
            """
            A fit of the near-axis model to the precise quasi-axisymmetric
            configuration from SIMSOPT with magnetic well.

            The fit was performed to the boozmn data using the script
            20200621-01-Extract_B0_B1_B2_from_boozxform
            """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1.0145598919163676, 0.2106377247598754, 0.025469267136340394, 0.0026773601516136727, 0.00021104172568911153, 7.891887175655046e-06, -8.216044358250985e-07, -2.379942694112007e-07, -2.5495108673798585e-08, 1.1679227114962395e-08, 8.961288962248274e-09],
                                     "sin":  []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [],
                                     "sin": [0.0, -0.14607192982551795, -0.021340448470388084, -0.002558983303282255, -0.0002355043952788449, -1.2752278964149462e-05, 3.673356209179739e-07, 9.261098628194352e-08, -7.976283362938471e-09, -4.4204430633540756e-08, -1.6019372369445714e-08]}}
            B0 = {"type": 'scalar', "input_value": 1.0117071561808106}
            d = {"type": 'scalar', "input_value": -0.5064143402495729}
            d_over_curvature = None
            B2c = -0.2749140163639202

            add_default_args(kwargs,
                             omn = False, nfp=2, nphi=99, order='r3',
                             Raxis = Raxis, Zaxis = Zaxis, B0 = B0, d=d, d_over_curvature = d_over_curvature,
                             B2c = B2c)
            
        elif name == "LandremanPaul2021QH" or name == "precise QH":
            """
            A fit of the near-axis model to the quasi-helically symmetric
            configuration in Landreman & Paul, arXiv:2108.03711 (2021).

            The fit was performed to the boozmn data using the script
            20211001-02-Extract_B0_B1_B2_from_boozxform
            """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1.0033608429348413, 0.19993025252481125, 0.03142704185268144, 0.004672593645851904, 0.0005589954792333977, 3.298415996551805e-05, -7.337736061708705e-06, -2.8829857667619663e-06, -4.51059545517434e-07],
                                     "sin":  []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [],
                                     "sin": [0.0, 0.1788824025525348, 0.028597666614604524, 0.004302393796260442, 0.0005283708386982674, 3.5146899855826326e-05, -5.907671188908183e-06, -2.3945326611145963e-06, -6.87509350019021e-07]}}
            B0 = {"type": 'scalar', "input_value": 1.003244143729638}
            d = {"type": 'scalar', "input_value": -1.5002839921360023}
            d_over_curvature = None
            B2c = 0.37896407142157423

            add_default_args(kwargs,
                             omn = False, nfp=4, nphi=99, order='r3',
                             Raxis = Raxis, Zaxis = Zaxis, B0 = B0, d=d, d_over_curvature = d_over_curvature,
                             B2c = B2c)

        elif name == "precise QH+well":
            """
            A fit of the near-axis model to the precise quasi-helically symmetric
            configuration from SIMSOPT with magnetic well.

            The fit was performed to the boozmn data using the script
            20211001-02-Extract_B0_B1_B2_from_boozxform
            """
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [1.000474932581454, 0.16345392520298313, 0.02176330066615466, 0.0023779201451133163, 0.00014141976024376502, -1.0595894482659743e-05, -2.9989267970578764e-06, 3.464574408947338e-08],
                                     "sin":  []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [],
                                     "sin": [0.0, 0.12501739099323073, 0.019051257169780858, 0.0023674771227236587, 0.0001865909743321566, -2.2659053455802824e-06, -2.368335337174369e-06, -1.8521248561490157e-08]}}
            B0 = {"type": 'scalar', "input_value": 0.999440074325872}
            d = {"type": 'scalar', "input_value": -1.2115187546668142}
            d_over_curvature = None
            B2c = 0.6916862277166693

            add_default_args(kwargs,
                             omn = False, nfp=4, nphi=99, order='r3',
                             Raxis = Raxis, Zaxis = Zaxis, B0 = B0, d=d, d_over_curvature = d_over_curvature,
                             B2c = B2c)
            

        elif name == "QI NFP2 Katia" or name == "QI NFP2 DirectConstruction":
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,0.0,-1/17 ],
                                     "sin":  [ 0.0,0.0,0.0 ]}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0,0.0,0.0 ],
                                     "sin": [ 0.0,0.8/2.04,0.01/2.04 ]}}
            B0 = {"type": 'fourier',
                     "input_value": {"cos": [1.0, 0.15], "sin": []}}
            
            d = None
            d_over_curvature = {"type": 'scalar', "input_value": 0.73}

            X2c = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0018400322140812674,-0.0013637739279265815,-0.0017961063281748597,-0.000855123667865997,-0.001412983361026517,-0.0010676686588779228,-0.0008117922713651492,-0.0002878689335032291,-0.0002515272886665927,-7.924709175875918e-05,-4.919421452969814e-05,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,2.7062914673236698,-0.9151373916194634,0.021394010521077745,-0.017469913902854437,0.03186670312840335,0.021102584055813403,0.0024194864183551515,-0.0059152315287890125,0.003709416127750524,0.010027743000785166,0.0,0.0,0.0,0.0 ]}}
            X2s = {"type": 'fourier',
                     "input_value": {"cos": [ 0.4445604502180231,0.13822067284200223,-0.561756934579829,0.2488873179399463,-0.14559282723014635,0.020548052084815048,-0.011070304464557718,0.004342889373034949,-0.0015730819049237866,0.0035406584522436986,0.002831887060104115,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,0.0012174780422017702,0.00026317725313621535,0.0002235661375254599,0.0006235230087895861,0.00021429298911807877,8.428032911991958e-05,-0.000142566391046771,-3.194627950185967e-05,-0.0001119389848119665,-6.226472957451552e-05 ]}}
            
            buffer_opt = default_buffer_dict.copy()
            buffer_opt["omn_method"] = 'non-zone'
            buffer_opt["delta"] = 0.0
            buffer_opt["k_buffer"] = 2

            sigma0 = 0.0
            nfp     = 2
            p2      = 0.0
            nphi    = 201
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, nfp=nfp, B0 = B0, d = d, d_over_curvature = d_over_curvature, nphi = nphi, \
                             omn_buffer = buffer_opt, sigma0 = sigma0, p2 = p2, X2c = X2c, X2s = X2s, omn = True, order = 'r3')
               
        elif name == "QI NFP2 Katia smooth":
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,0.0,-1/17 ],
                                     "sin":  []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [],
                                     "sin": [ 0.0,0.8/2.04,0.01/2.04 ]}}
            B0 = {"type": 'fourier',
                     "input_value": {"cos": [1.0, 0.15], "sin": []}}
            
            d = None
            d_over_curvature = {"type": 'scalar', "input_value": 0.73}

            X2c = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0018400322140812674,-0.0013637739279265815,-0.0017961063281748597,-0.000855123667865997,-0.001412983361026517,-0.0010676686588779228,-0.0008117922713651492,-0.0002878689335032291,-0.0002515272886665927,-7.924709175875918e-05,-4.919421452969814e-05,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,2.7062914673236698,-0.9151373916194634,0.021394010521077745,-0.017469913902854437,0.03186670312840335,0.021102584055813403,0.0024194864183551515,-0.0059152315287890125,0.003709416127750524,0.010027743000785166,0.0,0.0,0.0,0.0 ]}}
            X2s = {"type": 'fourier',
                     "input_value": {"cos": [ 0.4445604502180231,0.13822067284200223,-0.561756934579829,0.2488873179399463,-0.14559282723014635,0.020548052084815048,-0.011070304464557718,0.004342889373034949,-0.0015730819049237866,0.0035406584522436986,0.002831887060104115,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,0.0012174780422017702,0.00026317725313621535,0.0002235661375254599,0.0006235230087895861,0.00021429298911807877,8.428032911991958e-05,-0.000142566391046771,-3.194627950185967e-05,-0.0001119389848119665,-6.226472957451552e-05 ]}}
            
            buffer_opt = default_buffer_dict.copy()
            buffer_opt["omn_method"] = 'non-zone-smoother'
            buffer_opt["delta"] = 0.0
            buffer_opt["k_buffer"] = 2
            buffer_opt["p_buffer"] = 1

            sigma0 = 0.0
            nfp     = 2
            p2      = 0.0
            nphi    = 201
            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, nfp=nfp, B0 = B0, d = d, d_over_curvature = d_over_curvature, nphi = nphi, \
                             omn_buffer = buffer_opt, sigma0 = sigma0, p2 = p2, X2c = X2c, X2s = X2s, omn = True, order = 'r3')
                
        elif name == "QI NFP3 Katia" or name == "QI NFP3 DirectConstruction":
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,  9.075485257221899e-02, -2.058279495912439e-02, -1.106766494783158e-02, -1.644390251809640e-03 ],
                                     "sin":  [ 0.0,0.0,0.0,0.0,0.0 ]}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0,0.0,0.0,0.0 ],
                                     "sin": [ 0.0,0.36,0.02,0.01 ]}}
            B0 = {"type": 'fourier',
                     "input_value": {"cos": [1.0, 0.25], "sin": []}}
            
            d = None
            d_over_curvature = {"type": 'scalar', "input_value": 0.73}

            X2c = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0018400322140812674,-0.0013637739279265815,-0.0017961063281748597,-0.000855123667865997,-0.001412983361026517,-0.0010676686588779228,-0.0008117922713651492,-0.0002878689335032291,-0.0002515272886665927,-7.924709175875918e-05,-4.919421452969814e-05,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,2.7062914673236698,-0.9151373916194634,0.021394010521077745,-0.017469913902854437,0.03186670312840335,0.021102584055813403,0.0024194864183551515,-0.0059152315287890125,0.003709416127750524,0.010027743000785166,0.0,0.0,0.0,0.0 ]}}
            X2s = {"type": 'fourier',
                     "input_value": {"cos": [ 0.4445604502180231,0.13822067284200223,-0.561756934579829,0.2488873179399463,-0.14559282723014635,0.020548052084815048,-0.011070304464557718,0.004342889373034949,-0.0015730819049237866,0.0035406584522436986,0.002831887060104115,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,0.0012174780422017702,0.00026317725313621535,0.0002235661375254599,0.0006235230087895861,0.00021429298911807877,8.428032911991958e-05,-0.000142566391046771,-3.194627950185967e-05,-0.0001119389848119665,-6.226472957451552e-05 ]}}
            
            buffer_opt = default_buffer_dict.copy()
            buffer_opt["omn_method"] = 'non-zone'
            buffer_opt["delta"] = 0.0
            buffer_opt["k_buffer"] = 2

            sigma0 = 0.0
            nfp     = 3
            p2      = 0.0
            nphi    = 201

            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, nfp=nfp, B0 = B0, d = d, d_over_curvature = d_over_curvature, nphi = nphi, \
                             omn_buffer = buffer_opt, sigma0 = sigma0, p2 = p2, X2c = X2c, X2s = X2s, omn = True, order = 'r3')
            
        elif name == "QI NFP3 Katia smooth":
            Raxis = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,  9.075485257221899e-02, -2.058279495912439e-02, -1.106766494783158e-02, -1.644390251809640e-03 ],
                                     "sin":  []}}
            Zaxis = {"type": 'fourier',
                     "input_value": {"cos": [],
                                     "sin": [ 0.0,0.36,0.02,0.01 ]}}
            B0 = {"type": 'fourier',
                     "input_value": {"cos": [1.0, 0.25], "sin": []}}
            
            d = None
            d_over_curvature = {"type": 'scalar', "input_value": 0.73}

            X2c = {"type": 'fourier',
                     "input_value": {"cos": [ 0.0018400322140812674,-0.0013637739279265815,-0.0017961063281748597,-0.000855123667865997,-0.001412983361026517,-0.0010676686588779228,-0.0008117922713651492,-0.0002878689335032291,-0.0002515272886665927,-7.924709175875918e-05,-4.919421452969814e-05,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,2.7062914673236698,-0.9151373916194634,0.021394010521077745,-0.017469913902854437,0.03186670312840335,0.021102584055813403,0.0024194864183551515,-0.0059152315287890125,0.003709416127750524,0.010027743000785166,0.0,0.0,0.0,0.0 ]}}
            X2s = {"type": 'fourier',
                     "input_value": {"cos": [ 0.4445604502180231,0.13822067284200223,-0.561756934579829,0.2488873179399463,-0.14559282723014635,0.020548052084815048,-0.011070304464557718,0.004342889373034949,-0.0015730819049237866,0.0035406584522436986,0.002831887060104115,0.0,0.0,0.0,0.0 ], 
                                     "sin": [ 0.0,0.0012174780422017702,0.00026317725313621535,0.0002235661375254599,0.0006235230087895861,0.00021429298911807877,8.428032911991958e-05,-0.000142566391046771,-3.194627950185967e-05,-0.0001119389848119665,-6.226472957451552e-05 ]}}
            
            buffer_opt = default_buffer_dict.copy()
            buffer_opt["omn_method"] = 'non-zone-smoother'
            buffer_opt["delta"] = 0.0
            buffer_opt["k_buffer"] = 2
            buffer_opt["p_buffer"] = 1

            sigma0 = 0.0
            nfp     = 3
            p2      = 0.0
            nphi    = 201

            add_default_args(kwargs, Raxis = Raxis, Zaxis = Zaxis, nfp=nfp, B0 = B0, d = d, d_over_curvature = d_over_curvature, nphi = nphi, \
                             omn_buffer = buffer_opt, sigma0 = sigma0, p2 = p2, X2c = X2c, X2s = X2s, omn = True, order = 'r3')
            
        else:
            raise ValueError('Unrecognized configuration name')

        return cls(**kwargs)

    # @classmethod
    # def from_cxx(cls, filename):
    #     """
    #     Load a configuration from a ``qsc_out.<extension>.nc`` output file
    #     that was generated by the C++ version of QSC. Almost all the
    #     data will be taken from the output file, over-writing any
    #     calculations done in python when the new Qic object is
    #     created.
    #     """
    #     def to_string(nc_str):
    #         """ Convert a string from the netcdf binary format to a python string. """
    #         temp = [c.decode('UTF-8') for c in nc_str]
    #         return (''.join(temp)).strip()
        
    #     f = netcdf.netcdf_file(filename, mmap=False)
    #     nfp = f.variables['nfp'][()]
    #     nphi = f.variables['nphi'][()]
    #     rc = f.variables['R0c'][()]
    #     rs = f.variables['R0s'][()]
    #     zc = f.variables['Z0c'][()]
    #     zs = f.variables['Z0s'][()]
    #     I2 = f.variables['I2'][()]
    #     B0 = f.variables['B0'][()]
    #     spsi = f.variables['spsi'][()]
    #     sG = f.variables['sG'][()]
    #     etabar = f.variables['eta_bar'][()]
    #     sigma0 = f.variables['sigma0'][()]
    #     order_r_option = to_string(f.variables['order_r_option'][()])
    #     if order_r_option == 'r2.1':
    #         order_r_option = 'r3'
    #     if order_r_option == 'r1':
    #         p2 = 0.0
    #         B2c = 0.0
    #         B2s = 0.0
    #     else:
    #         p2 = f.variables['p2'][()]
    #         B2c = f.variables['B2c'][()]
    #         B2s = f.variables['B2s'][()]

    #     q = cls(nfp=nfp, nphi=nphi, rc=rc, rs=rs, zc=zc, zs=zs,
    #             B0=B0, sG=sG, spsi=spsi,
    #             etabar=etabar, sigma0=sigma0, I2=I2, p2=p2, B2c=B2c, B2s=B2s, order=order_r_option)
        
    #     def read(name, cxx_name=None):
    #         if cxx_name is None: cxx_name = name
    #         setattr(q, name, f.variables[cxx_name][()])

    #     [read(v) for v in ['R0', 'Z0', 'R0p', 'Z0p', 'R0pp', 'Z0pp', 'R0ppp', 'Z0ppp',
    #                        'sigma', 'curvature', 'torsion', 'X1c', 'Y1c', 'Y1s', 'elongation']]
    #     if order_r_option != 'r1':
    #         [read(v) for v in ['X20', 'X2c', 'X2s', 'Y20', 'Y2c', 'Y2s', 'Z20', 'Z2c', 'Z2s', 'B20']]
    #         if order_r_option != 'r2':
    #             [read(v) for v in ['X3c1', 'Y3c1', 'Y3s1']]
                    
    #     f.close()
    #     return q
        
    def min_R0_penalty(self):
        """
        This function can be used in optimization to penalize situations
        in which min(R0) < min_R0_constraint.
        """
        return np.max((0, self.min_R0_threshold - self.min_R0)) ** 2

    def min_Z0_penalty(self):
        """
        This function can be used in optimization to penalize situations
        in which min(Z0) < min_Z0_constraint.
        """
        return np.max((0, self.min_Z0_threshold - self.min_Z0)) ** 2
        
    @classmethod
    def from_boozxform(cls, booz_xform_file, max_s_for_fit = 0.4, N_phi = 200, max_n_to_plot = 2, show=False,
                         vmec_file=None, rc=[], rs=[], zc=[], zs=[], nNormal=None, input_stel=None, savefig=False, phi_a = []):#, order='r2', sigma0=0, I2=0, p2=0, omn=False):
        """
        NEED TO MODIFY Load a configuration from a VMEC and a BOOZ_XFORM output files
        """
        # Read properties of BOOZ_XFORM output file
        f = netcdf.netcdf_file(booz_xform_file,'r',mmap=False)
        bmnc = f.variables['bmnc_b'][()]
        ixm = f.variables['ixm_b'][()]
        ixn = f.variables['ixn_b'][()]
        jlist = f.variables['jlist'][()]
        ns = f.variables['ns_b'][()]
        nfp = f.variables['nfp_b'][()]
        Psi = f.variables['phi_b'][()]
        if phi_a:
            Psi_a = phi_a
        else:
            Psi_a = np.abs(Psi[-1])
        iotaVMECt=f.variables['iota_b'][()][1]
        f.close()

        if vmec_file!=None:
            # Read axis-shape from VMEC output file
            f = netcdf.netcdf_file(vmec_file,'r',mmap=False)
            am = f.variables['am'][()]
            rc = f.variables['raxis_cc'][()]
            zs = -f.variables['zaxis_cs'][()]
            Phi_v = f.variables['phi'][()]
            Psi_a = np.abs(Phi_v[-1])
            try:
                rs = -f.variables['raxis_cs'][()]
                zc = f.variables['zaxis_cc'][()]
                logger.info('Non stellarator symmetric configuration')
            except:
                rs=[]
                zc=[]
                logger.info('Stellarator symmetric configuration')
            f.close()
        elif rc!=[]:
            # Read axis-shape from input parameters
            rc=rc
            rs=rs
            zc=zc
            zs=zs
        else:
            print("Axis shape not specified")
            # Calculate nNormal
        if nNormal==None:
            Raxis = {"type": 'fourier', "input_value": {"cos": rc, "sin": rs}}
            Zaxis = {"type": 'fourier', "input_value": {"cos": zc, "sin": zs}}
            stel = Qic(Raxis = Raxis, Zaxis = Zaxis, nfp=nfp)
            nNormal = stel.iotaN - stel.iota
        else:
            nNormal = nNormal

        # Prepare coordinates for fit
        s_full = np.linspace(0,1,ns)
        ds = s_full[1] - s_full[0]
        #s_half = s_full[1:] - 0.5*ds
        s_half = s_full[jlist-1] - 0.5*ds
        mask = s_half < max_s_for_fit
        s_fine = np.linspace(0,1,400)
        sqrts_fine = s_fine
        phi = np.linspace(0,2*np.pi / nfp, N_phi)
        B0  = np.zeros(N_phi)
        B1s = np.zeros(N_phi)
        B1c = np.zeros(N_phi)
        B20 = np.zeros(N_phi)
        B2s = np.zeros(N_phi)
        B2c = np.zeros(N_phi)

        # Perform fit
        numRows=3
        numCols=max_n_to_plot*2+1
        if show: fig=plt.figure(num=None, figsize=(16, 9), dpi=80, facecolor='w', edgecolor='k')
        for jmn in range(len(ixm)):
            m = ixm[jmn]
            n = ixn[jmn] / nfp
            if m>2:
                continue
            doplot = (np.abs(n) <= max_n_to_plot) & show
            row = m
            col = n+max_n_to_plot
            if doplot:
                plt.subplot(int(numRows),int(numCols),int(row*numCols + col + 1))
                plt.plot(np.sqrt(s_half), bmnc[:,jmn],'.-')
                # plt.xlabel(r'$\sqrt{s}$')
                plt.title('bmnc(m='+str(m)+' n='+str(n)+')')
            if m==0:
                # For m=0, fit a polynomial in s (not sqrt(s)) that does not need to go through the origin.
                degree = 1
                p = np.polyfit(s_half[mask], bmnc[mask,jmn], degree)
                B0 += p[-1] * np.cos(n*nfp*phi)
                B20 += p[-2] * np.cos(n*nfp*phi)
                if doplot:
                    plt.plot(np.sqrt(s_fine), np.polyval(p, s_fine),'r')
            if m==1:
                # For m=1, fit a polynomial in sqrt(s) to an odd function
                x1 = np.sqrt(s_half[mask])
                y1 = bmnc[mask,jmn]
                x2 = np.concatenate((-x1,x1))
                y2 = np.concatenate((-y1,y1))
                degree = 1
                p = np.polyfit(x2,y2, degree)
                B1c += p[-2] * (np.sin(n*nfp*phi) * np.sin(nNormal*phi) + np.cos(n*nfp*phi) * np.cos(nNormal*phi))
                B1s += p[-2] * (np.sin(n*nfp*phi) * np.cos(nNormal*phi) - np.cos(n*nfp*phi) * np.sin(nNormal*phi))
                if doplot:
                    plt.plot(sqrts_fine, np.polyval(p, sqrts_fine),'r')
            if m==2:
                # For m=2, fit a polynomial in s (not sqrt(s)) that does need to go through the origin.
                x1 = s_half[mask]
                y1 = bmnc[mask,jmn]
                degree = 1
                p = np.polyfit(x1,y1, degree)
                B2c += p[-2] * (np.sin(n*nfp*phi) * np.sin(2*nNormal*phi) + np.cos(n*nfp*phi) * np.cos(2*nNormal*phi))
                B2s += p[-2] * (np.sin(n*nfp*phi) * np.cos(2*nNormal*phi) - np.cos(n*nfp*phi) * np.sin(2*nNormal*phi))
                if doplot:
                    plt.plot(np.sqrt(s_fine), np.polyval(p, s_fine),'r')
        if show:
            plt.show()
        # Convert expansion in sqrt(s) to an expansion in r
        BBar = np.mean(B0)
        sqrt_s_over_r = np.sqrt(np.pi * BBar / np.abs(Psi_a))
        B1s *= -sqrt_s_over_r
        B1c *= sqrt_s_over_r
        B20 *= sqrt_s_over_r*sqrt_s_over_r
        B2c *= sqrt_s_over_r*sqrt_s_over_r
        B2s *= -sqrt_s_over_r*sqrt_s_over_r
        eta_bar = np.mean(B1c) / BBar

        # NEEDS A WAY TO READ I2 FROM VMEC OR BOOZ_XFORM

        # if p2==0 and vmec_file!=None:
        #     r  = np.sqrt(Psi_a/(np.pi*BBar))
        #     p2 = - am[0]/r/r

        # if omn:
        #     if order=='r1':
        #         q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar,sigma0=sigma0,I2=I2)
        #     else:
        #         q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar,sigma0=sigma0,I2=I2, B2c=np.mean(B2c), B2s=np.mean(B2s), order=order, p2=p2)
        # else:
        #     if order=='r1':
        #         q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar,sigma0=sigma0, I2=I2)
        #     else:
        #         q = cls(rc=rc,rs=rs,zc=zc,zs=zs,etabar=eta_bar,nphi=N_phi,nfp=nfp,B0=BBar,sigma0=sigma0, I2=I2, B2c=np.mean(B2c), B2s=np.mean(B2s), order=order, p2=p2)

        # q.B0_boozxform_array=B0
        # q.B1c_boozxform_array=B1c
        # q.B1s_boozxform_array=B1s
        # q.B20_boozxform_array=B20
        # q.B2c_boozxform_array=B2c
        # q.B2s_boozxform_array=B2s
        # q.iotaVMEC = iotaVMECt
        if show:
            try:
                # name = vmec_file[5:-3]
                name = 'temp'

                figB0=plt.figure(figsize=(5, 5), dpi=80)
                plt.plot(input_stel.varphi, input_stel.B0, 'r--', label=r'$B_0$ Near-axis')
                plt.plot(phi, B0,            'b-' , label=r'$B_0$ VMEC')
                plt.xlabel(r'$\phi$', fontsize=18)
                plt.legend(fontsize=14)
                if savefig: figB0.savefig('B0_VMEC'+name+'.pdf')

                figB1=plt.figure(figsize=(5, 5), dpi=80)
                plt.plot(input_stel.varphi, input_stel.B1c, 'r--', label=r'$B_{1c}$ Near-axis')
                plt.plot(phi, B1c,            'r-' , label=r'$B_{1c}$ VMEC')
                plt.plot(input_stel.varphi, input_stel.B1s, 'b--', label=r'$B_{1s}$ Near-axis')
                plt.plot(phi, B1s,            'b-' , label=r'$B_{1s}$ VMEC')
                plt.xlabel(r'$\phi$', fontsize=18)
                plt.legend(fontsize=14)
                if savefig: figB1.savefig('B1_VMEC'+name+'.pdf')

                figB2=plt.figure(figsize=(5, 5), dpi=80)
                if input_stel.order != 'r1':
                    plt.plot(input_stel.varphi, input_stel.B20, 'r--', label=r'$B_{20}$ Near-axis')
                    plt.plot(input_stel.varphi, input_stel.B2c, 'b--', label=r'$B_{2c}$ Near-axis')
                    plt.plot(input_stel.varphi, input_stel.B2s, 'g--', label=r'$B_{2s}$ Near-axis')
                plt.plot(phi, B20,            'r-' , label=r'$B_{20}$ VMEC')
                plt.plot(phi, B2c,            'b-' , label=r'$B_{2c}$ VMEC')
                plt.plot(phi, B2s,            'g-' , label=r'$B_{2s}$ VMEC')
                plt.xlabel(r'$\phi$', fontsize=18)
                plt.legend(fontsize=14)
                if savefig: figB2.savefig('B2_VMEC'+name+'.pdf')

                if show: plt.show()

                plt.close(figB0)
                plt.close()
            except Exception as e:
                print(e)


        return [B0,B1c,B1s,B20,B2c,B2s,iotaVMECt]