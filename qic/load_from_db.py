"""
Scripts to load QI configs from the database.
"""

import ijson
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import make_interp_spline

class Struct():
    pass

# Location of the database files
_DATA_FOLDER_PATH = "/home/../mnt/d/Research/Stellerator/Datsshare_sync/DB/Paper_scripts/"

@classmethod
def from_db(cls, name, **kwargs):
    """
    Load a configuration from the built-in database of configurations.
    """
    kwargs = load_config_from_db(name, **kwargs)
    return cls(**kwargs)

##################
# DATABASE FILES #
##################
def load_config_from_db(config_name, no_alpha = False, solve_geo = True, **kwargs):
    """
    Load a specific configuration from database providing the config name and construct the corresponding QIC object. The config name should be in the format "Nx_xxx_xxx", where N is the number of field periods, and xxx are the identifiers for the database file and configuration ID.

    Parameters
    ----------
        config_name: str
            The name of the configuration, in the format "Nx_xxx_xxx".
        no_alpha: bool, optional
            Whether to use the alpha_tilde input or resolve for it. Default is False.
        solve_geo: bool, optional
            Whether to construct the axis in R3. Default is True.
        **kwargs: 
            additional arguments to pass to the QIC class.

    Returns
    -------
        stel_qic: QIC class object.
            QIC class object corresponding to the loaded configuration.
    """
    # Undo the config name to obtain the nfp, db_file and config_id
    _, db_file, config_id = undo_config_name(config_name)

    # Load the configuration from the database file and construct the QIC object
    props_stel = load_config_id_file_from_db(db_file, config_id, no_alpha = no_alpha, solve_geo = solve_geo, **kwargs)

    return props_stel

def undo_config_name(config_name, path_header = "/home/IPP-HGW/rodre/Documents/qi_database/"):
    """
    Undo a config name to obtain the nfp, db_file and config_id.
    Parameters
    ----------
    config_name : str
        The name of the configuration, in the format "Nx_xxx_xxx".

    Returns
    -------
    nfp : int
        The number of field periods of the configuration.
    db_file : str
        The name of the database file, which contains the configuration ID.
    config_id : int
        The ID of the configuration in the database.
    """
    parts = config_name.split("_")
    nfp = int(parts[0][1:])
    db_file = f"{path_header}data/N{nfp}/QI-Database-N{nfp}-m-0.5-kappa23-Broad-Sweep-{parts[1]}.json"
    config_id = int(parts[2])

    return nfp, db_file, config_id

def load_config_id_file_from_db(db_file, config_id, verbose = False, no_alpha = False, solve_geo = True, **kwargs):
    """
    Load a specific configuration from database file and construct the corresponding QIC object.

    Parameters
    ----------
        db_file: str
            Path to the database file in JSON format.
        config_id: int
            Configuration index to load from the database.
        verbose: bool, optional
            Whether to print verbose output. Default is False.
        no_alpha: bool, optional
            Whether to use the alpha_tilde input or resolve for it. Default is False.
        solve_geo: bool, optional
            Whether to construct the axis in R3. Default is True.
        **kwargs: 
            additional arguments to pass to the QIC class.

    Returns
    -------
        stel_qic: QIC class object.
            QIC class object corresponding to the loaded configuration.
    """
    # Repath file if necessary
    if "IPP-HGW" in db_file:
        db_file = repath_file(db_file)

    # Open the file and parse it incrementally
    with open(db_file, 'rb') as file:
        # Create a generator for the objects in the JSON file
        objects = ijson.items(file, 'item', use_float=True)
        
        # Loop over the items and check for the target configurationIndex
        for current_index, obj in enumerate(objects):
            if current_index == config_id:
                config = {nam: np.array(val) for nam, val in obj}
                break

    # Make stel object
    stel, _ = make_stel_object(config)

    # Run pyQIC
    if verbose:
        print('Running pyQIC...')
    props_stel = construct_qic(stel, model = False, smooth = False, axis_complete = False, # Do not close the axis further\ 
                             solve_geo = solve_geo,     # Construct the axis in R3
                             Bbar = 1, verbose = verbose, no_alpha = no_alpha, # Whether to skip alpha_tilde or recalculate it
                             **kwargs)
    return props_stel

def repath_file(file_path):
    """
    Repath file from processed database to original database, new address. It will change the path 
    /home/IPP-HGW/rodre/Documents/qi_database/data/...
    to the desired one.
    Parameters
    ----------
    file_path : str
        The original file path to be repathed.

    Returns
    -------
    new_file_path : str
        The repathed file path.
    """
    # Define the old and new base paths
    old_base_path = "/home/IPP-HGW/rodre/Documents/qi_database/"
    new_base_path = _DATA_FOLDER_PATH

    # Replace the old base path with the new base path
    if file_path.startswith(old_base_path):
        new_file_path = file_path.replace(old_base_path, new_base_path, 1)
        return new_file_path
    else:
        raise ValueError("The provided file path does not start with the expected base path.")

def make_stel_object(config):
    """
    Create a class object with the digested configuration parameters using config info obtained from .json.
    Parameters:
    ----------
    config : dict
        Configuration dictionary containing the necessary parameters.
    Returns:
    -------
    stel : Struct
        Class object with the configuration parameters as attributes.
    _: int
        Placeholder return value (0).
    """
    # Read the inputs from the configuration dictionary    
    nphi = config["nphi"]
    nfp = config["nfp"]
    sigma = config["sigma"]
    varphi = config["varphi"]
    d_l_d_varphi = config["d_l_d_varphi"]
    d = config["d"]/np.sqrt(2)  # Normalise by sqrt(2) because of the difference in Gabe's and Matt's notation
    d_over_curvature = config["dbar"]/np.sqrt(2)    # Normalise by sqrt(2) because of the difference in Gabe's and Matt's notation
    alpha = config["alpha"]
    curvature = config["curvature"]
    torsion = config["torsion"]
    iota = config["iota"]
    helicity = -config["helicity"] # Gabe's definition is clockwise
    B0 = config["B0"]

    # Read the parameters for the curvature and torsion models, only accept if this simple model is used
    params = config.get("{ζ, κs[1], τc[0], τc[1]}", None)
    if params is not None:
        flag_params = False
    else:
        flag_params = True


    #######################
    # CREATE CLASS OBJECT #
    #######################
    stel = Struct()

    # Define attributes
    stel.B0 = B0[:-1]; stel.B0_end =  B0[-1]; 
    stel.nphi = nphi; stel.nfp = nfp
    stel.varphi = varphi[:-1]; stel.varphi_end = varphi[-1]; stel.d_l_d_varphi = d_l_d_varphi[:-1]; 
    stel.abs_G0_over_B0 = d_l_d_varphi[:-1]; stel.d = d[:-1]; stel.d_end = d[-1] 
    stel.G0 = np.mean(d_l_d_varphi*B0); 
    stel.sigma = sigma[:-1]; stel.sigma_end = sigma[-1];
    stel.Bbar = 1 # np.mean(stel.B0)
    stel.sG = 1
    stel.d_bar = d_over_curvature[:-1]; stel.d_bar_end = d_over_curvature[-1]; 
    stel.alpha = alpha[:-1]; stel.alpha_no_buffer = iota*varphi[:-1];
    stel.curvature = curvature[:-1]
    stel.torsion = torsion[:-1]; 
    stel.iota = iota
    stel.helicity = helicity
    stel.iotaN = stel.iota + stel.helicity * stel.nfp
    stel.order = 'r2'
    stel.params = params
    stel.flag_params = flag_params
    
    return stel, 0

def construct_qic(stel_in, model = False, smooth = False, no_alpha = False, verbose = False, **kwargs):
    """
    Construct the QIC class object.
    Parameters:
    ----------
        stel_in: class object
            Class object with the configuration parameters as attributes.
        model: bool, optional
            Whether to use the model for the curvature and torsion. Default is False, and will only run if the model flag is True. Need to improve for more general use.
        smooth: bool, optional
            Whether to smooth the B0 and d_bar inputs. Default is False.
        no_alpha: bool, optional
            Whether to skip the alpha_tilde input. Default is False.
        verbose: bool, optional
            Whether to print verbose output. Default is False.
        **kwargs: additional arguments to pass to the QIC class.
    Returns:
    -------
        stel: QIC class object.
            QIC class object.
    """
    ##############
    # DEFINE ELL #
    ##############
    # Calculate the length along the curve
    ell = integrate.cumtrapz(np.append(stel_in.d_l_d_varphi,stel_in.d_l_d_varphi[0]), np.append(stel_in.varphi, stel_in.varphi[0]+2*np.pi/stel_in.nfp),\
                            initial = 0.0)
    stel_in.L_in = ell[-1]
    stel_in.ell = ell[:-1]
    nphi = len(stel_in.ell)
    if "nphi" in kwargs:
        raise ValueError("nphi is not a valid argument. Use stel_in.nphi instead.")

    varphi = stel_in.varphi

    #####################
    # DEFINE AXIS SHAPE #
    #####################
    if model and stel_in.flag_params:
        # Read the parameters for the curvature and torsion models
        params = stel_in.params
        nfp = stel_in.nfp
        # Calculate the curvature and torsion
        curvature = 0.5*(1+np.cos(nfp*stel_in.ell))*np.sin(0.5*nfp*stel_in.ell)*np.sin(nfp*stel_in.ell)*params[1]
        torsion = params[2] + params[3]*np.cos(nfp*stel_in.ell)
        # Define the functions for the curvature and torsion
        curvature_func = lambda x: 0.5*(1+np.cos(x/stel_in.L_in*2*np.pi))*np.sin(0.5*x/stel_in.L_in*2*np.pi)*np.sin(x/stel_in.L_in*2*np.pi)*params[1]
        torsion_func = lambda x: params[2] + params[3]*np.cos(x/stel_in.L_in*2*np.pi)

        # Construct inputs for the QIC class
        curvature = {"type": 'grid', "input_value": curvature, "function_ell": curvature_func}
        torsion = {"type": 'grid', "input_value": torsion, "function_ell": torsion_func}
        ell = {"type": 'grid', "input_value": stel_in.ell}
    else:
        # Define the inputs for the QIC class
        curvature = {"type": 'grid', "input_value": stel_in.curvature}
        torsion = {"type": 'grid', "input_value": stel_in.torsion}
        ell = {"type": 'grid', "input_value": ell}

    ##############
    # B0 & d_bar #
    ##############
    if smooth:
        #####################
        # SMOOTH THE INPUTS #
        #####################
        def smooth_fourier(data, nfp, n_harm, phi_out):
            """
            Smooth data using truncated Fourier series. 
            Parameters:
            -----------
                data: array_like
                    Input data to be smoothed.
                nfp: int
                    Number of field periods.
                n_harm: int
                    Number of harmonics to keep.
                phi_out: array_like
                    Output grid.
            Returns:
            --------
                smoothed_data: array_like
                    Smoothed data.
            """
            # Input grid
            nphi = len(data)
            phi_in = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)

            # Harmonic components of axis position
            data_harm = np.zeros(int(n_harm + 1))
            data_harm[0] = np.sum(data) / nphi            
            factor = 2 / nphi
            smoothed_data = np.full(len(phi_out), fill_value = data_harm[0])

            for n in range(1, n_harm+1):
                angle = - n * nfp * phi_in
                cosangle = np.cos(angle)
                factor2 = factor
                # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
                # if n == 0: factor2 = factor2 / 2
                data_harm[n] = np.sum(data * cosangle * factor2)
                smoothed_data += data_harm[n] * np.cos(n * nfp * phi_out)

            return smoothed_data
        # Define the underresolved grid to perform smoothing on
        phi_under = np.linspace(0, 1.0, 99, endpoint=False) * 2*np.pi/stel_in.nfp
        
        # Smooth B0
        sp = make_interp_spline(np.append(stel_in.varphi, stel_in.varphi_end), np.append(stel_in.B0, stel_in.B0_end), k=3)
        smooth_B0 = smooth_fourier(sp(phi_under), stel_in.nfp, 4, varphi)
        stel_in.B0 = smooth_B0
        B0 = {"type": 'grid', "input_value": stel_in.B0}

        # Smooth d_bar
        sp = make_interp_spline(np.append(stel_in.varphi, stel_in.varphi_end), np.append(stel_in.d_bar, stel_in.d_bar_end), k=3)
        smooth_dbar = smooth_fourier(sp(phi_under), stel_in.nfp, 4, varphi)
        stel_in.d_bar = smooth_dbar
        d_over_curvature = {"type": 'grid', "input_value": stel_in.d_bar}

    else:
        ###################
        # READ THE INPUTS #
        ###################
        # B0
        B0 = {"type": 'grid', "input_value": stel_in.B0}

        # d_bar
        d_over_curvature = {"type": 'grid', "input_value": stel_in.d_bar}

    ################
    # OTHER INPUTS #
    ################
    # Alpha
    alpha_tilde = None if no_alpha else {"type": 'grid', "input_value": stel_in.alpha + varphi*stel_in.helicity*stel_in.nfp}

    # Buffer type
    omn_buffer = {"omn_method": 'simple-fourier', 
                "k_buffer": 3}

    # Second order shaping : 'minimal' shaping choice
    X2c = {"type": 'scalar', "input_value": 0}
    X2s = {"type": 'scalar', "input_value": 0}

    # Other inputs
    d = None
    sigma0 = 0.0
    helicity = stel_in.helicity
    alpha_tilde = alpha_tilde
    L = stel_in.L_in
    frenet = True
    omn = True
    order = 'r2'
    phi_shift = 0
    diff_finite = None
    Raxis = None
    Zaxis = None
    nfp = stel_in.nfp
    p2 = 0
    I2 = 0

    ####################
    # VARIABLES TO SET #
    ####################
    vars = ["Raxis", "Zaxis", "B0", "d", "d_over_curvature", "X2c", "X2s", "omn_buffer", "sigma0", "nfp", "p2", "I2", "order", "omn", "helicity",\
            "alpha_tilde", "varphi", "L", "ell", "torsion", "curvature", "frenet", "nphi", "phi_shift", "diff_finite"]

    # Put all input parameters together 
    for var in vars:
        if var in kwargs:
            # External variable definitions have priority
            if verbose:
                print("WARNING! The arguments passed overwrite some of the default values used in the construction of the called equilibrium:...." + f"{var}")
        else:
            kwargs[var] = eval(var)
            
    return kwargs





