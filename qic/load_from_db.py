"""
Scripts to load QI configs from the database.
"""
import ijson
import numpy as np
from pathlib import Path
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import make_interp_spline

class Struct():
    pass

# Location of the database files
_DATA_FOLDER_PATH = "/home/../mnt/d/Research/Stellerator/Datsshare_sync/DB/Paper_scripts/"
_DATAFRAME_PATH = _DATA_FOLDER_PATH + "Database/dataframe.pkl"

@classmethod
def from_db(cls, config_name, db_path = _DATA_FOLDER_PATH, reshape = True, help = False, online = False, **kwargs):
    """
    Load a configuration from the built-in database of configurations or online repository. If reshape is True, the mag_well_reshape function will be applied to the loaded configuration. Note that the config name should be in the format ``"Nx_xxx_xxxx"``, where ``Nx`` is the number of field periods, ``xxx`` and ``xxxx`` are the identifiers for the database file and configuration ID, respectively.

    Parameters
    ----------
    config_name: str
        The name of the configuration, in the format ``"Nx_xxx_xxxx"``.
    db_path: str, optional
        The path to the database files. Default is ``_DATA_FOLDER_PATH``.
    reshape: bool, optional
        Whether to apply the ``mag_well_reshape`` function to the loaded configuration. Default is ``True``.
    help: bool, optional
        Whether to show some selected configurations from the paper in the database.
    online: bool, optional
        Whether to load the configuration from the online database. Default is ``False``, and will only run if the online flag is ``True``.
    **kwargs:
        Additional arguments to pass to the ``load_config_from_db`` function.

    Returns
    -------
    stel: QIC class object
        QIC class object corresponding to the loaded configuration.
    """
    # If help, print configurations from DB paper in /qic_path/configs/configs_paper_db.json

    if help:
        # Find the path to the configs_paper_db.json file
        configs_paper_db_path = Path(__file__).parent / "configs" / "configs_paper_db.json"
        if not configs_paper_db_path.is_file():
            raise FileNotFoundError(f"The file {configs_paper_db_path} does not exist.")

        # Load the configuration list
        import json
        with open(configs_paper_db_path, "r") as f:
            configs_paper_db = json.load(f)

        # Elegant and structured printing
        print("\n" + "="*70)
        print("Database configurations from the paper (format: 'Nx_xxx_xxx')")
        print("="*70)
        for case, case_data in configs_paper_db.items():
            print(f"\n{case.center(68, '-')}")
            desc = case_data.get("Description", "")
            print(f"Description: {desc}")
            configs = case_data.get("Configs", [])
            if configs:
                print("Configs:")
                # Print configs in columns, 4 per row
                col_width = max(len(cfg) for cfg in configs) + 2
                n_cols = 4
                for i in range(0, len(configs), n_cols):
                    row = configs[i:i+n_cols]
                    print("  " + "".join(cfg.ljust(col_width) for cfg in row))
            print("-"*68)
        print("="*70 + "\n")
        return

    kwargs = load_config_from_db(config_name, db_path=db_path, online = online, **kwargs)
    stel = cls(**kwargs)

    if reshape:
        cls.mag_well_reshape(stel)
        
    # add the name of the configuration as an attribute to the object
    stel.config_name = config_name

    return stel

##################
# DATABASE FILES #
##################
def load_info_from_db(config_name, dataframe_path = _DATAFRAME_PATH):
    """
    Load the info of a specific configuration from database ``.pkl`` providing the name of the configuration. The ``config_name`` should be in the format ``"Nx_xxx_xxxx"``, where ``Nx`` is the number of field periods, ``xxx`` and ``xxxx`` are the identifiers for the database file and configuration ID, respectively.

    Parameters
    ----------
    config_name: str
        The name of the configuration, in the format ``"Nx_xxx_xxxx"``.
    dataframe_path: str, optional
        The path to the dataframe ``.pkl`` file. Default is ``_DATAFRAME_PATH``.

    Returns
    -------
    info: dict
        Dictionary containing the info of the configuration.
    """
    # Undo the config name to obtain the nfp, db_file and config_id
    _, db_file, config_id = undo_config_name(config_name, path_header="/home/IPP-HGW/rodre/Documents/qi_database/")

    # Check if the pickle file exists and is a pkl file
    import os
    pkl_file = dataframe_path
    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"The file {pkl_file} does not exist.")
    if not pkl_file.endswith(".pkl"):
        raise ValueError("The file must be a pickle file.")
    
    # Load data
    import pickle
    with open(pkl_file, "rb") as file:
        df = pickle.load(file)

    # Filter the dataframe for the target configuration
    target_row = df[(df["FilePath"] == db_file) & (df["ID"] == config_id + 1)] # config_id is 0-indexed 

    if target_row.empty:
        raise ValueError(f"No configuration found with db_file {db_file} and config_id {config_id - 1} in the dataframe.")
    
    # Make the info dictionary
    info = target_row.to_dict(orient="records")[0]

    return info

def load_config_from_db(config_name, db_path = _DATA_FOLDER_PATH,  verbose = False, no_alpha = False, solve_geo = True, online = False, **kwargs):
    """
    Load a specific configuration from database providing the name of the configuration and construct the corresponding QIC object. The ``config_name`` should be in the format ``"Nx_xxx_xxxx"``, where ``Nx`` is the number of field periods, ``xxx`` and ``xxxx`` are the identifiers for the database file and configuration ID, respectively.

    Parameters
    ----------
        config_name: str
            The name of the configuration, in the format ``"Nx_xxx_xxxx"``.
        db_path: str, optional
            The path to the database files. Default is ``_DATA_FOLDER_PATH``.
        verbose: bool, optional
            Whether to print verbose output. Default is ``False``.
        no_alpha: bool, optional
            Whether to use the alpha_tilde input or resolve for it. Default is ``False``.
        solve_geo: bool, optional
            Whether to construct the axis in R3. Default is ``True``.
        online: bool, optional
            Whether to load the configuration from the online database. Default is ``False``, and will only run if the online flag is ``True``. 
        **kwargs: 
            additional arguments to pass to the QIC class.

    Returns
    -------
        stel_qic: QIC class object.
            QIC class object corresponding to the loaded configuration.
    """
    if online:
        config = load_config_id_file_from_db_online(config_name)

    else:
        # Undo the config name to obtain the nfp, db_file and config_id
        _, db_file, config_id = undo_config_name(config_name, path_header=db_path)

        # Load the configuration from the database file
        config = load_config_id_file_from_db(db_file, config_id)

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

def undo_config_name(config_name, path_header = _DATA_FOLDER_PATH):
    """
    Undo the name of the configuration to obtain the ``nfp``, ``db_file`` and ``config_id``.
    Parameters
    ----------
    config_name : str
        The name of the configuration, in the format ``"Nx_xxx_xxxx"``.
    path_header : str, optional
        The path header for the database files. Default is "/home/IPP-HGW/rodre/Documents/qi_database/".

    Returns
    -------
    nfp : int
        The number of field periods of the configuration.
    db_file : str
        The name of the database file, which contains the configuration ID.
    config_id : int
        The ID of the configuration in the database, 0-indexed. Note that the ID in the database is 1-indexed, so the returned config_id is the database ID minus 1.
    """
    parts = config_name.split("_")
    nfp = int(parts[0][1:])
    sweep_id = int(parts[1])
    db_file = f"{path_header}data/N{nfp}/QI-Database-N{nfp}-m-0.5-kappa23-Broad-Sweep-{sweep_id:03d}.json"
    config_id = int(parts[2])

    return nfp, db_file, config_id

def do_config_name(db_file, config_id, path_header = _DATA_FOLDER_PATH):
    """
    Do a config name from the nfp, db_file and config_id.
    Parameters
    ----------
    db_file : str
        The name of the database file, which contains the configuration ID.
    config_id : int
        The ID of the configuration in the database, 0-indexed. Note that the ID in the database is 1-indexed, so the input config_id should be the database ID minus 1.
    path_header : str, optional
        The path header for the database files. Default is _DATA_FOLDER_PATH.

    Returns
    -------
    config_name : str
        The name of the configuration, in the format ``"Nx_xxx_xxxx"``.
    """
    # Extract the identifier for the database file from the db_file path
    db_file_identifier = int(Path(db_file).stem.split("-")[-1].split(".")[0])

    # Extract the nfp from the db_file path
    nfp = int(Path(db_file).stem.split("-")[2][1:])
    
    # Construct the config name
    config_name = f"N{nfp}_{db_file_identifier:03d}_{config_id:04d}"
    
    return config_name


def load_config_id_file_from_db(db_file, config_id): 
    """
    Load a specific configuration from database file and return the data.

    Parameters
    ----------
        db_file: str
            Path to the database file in JSON format.
        config_id: int
            Config ID to load from the database (0-indexed). Note that the ID in the database is 1-indexed, so the input config_id should be the database ID minus 1.

    Returns
    -------
        config: dict
            Dictionary containing the data from the JSON object.
    """
    # Check if the file exists
    if not Path(db_file).is_file():
        raise FileNotFoundError(f"The database file {db_file} does not exist. May have to use the correct _DATA_FOLDER_PATH or check the config name and undo_config_name function.")

    # Open the file and parse it incrementally
    with open(db_file, 'rb') as file:
        # Create a generator for the objects in the JSON file
        objects = ijson.items(file, 'item', use_float=True)
        
        # Loop over the items and check for the target configurationIndex
        for current_index, obj in enumerate(objects):
            if current_index == config_id:
                config = {nam: np.array(val) for nam, val in obj}
                break
    return config


def download_from_database(nfp, sweep_id, list_id, skip_if_exists = True):
    """
    Download a specific configuration file from the online database and return the local path to the downloaded file.

    Parameters
    ----------
    nfp: int
        Number of field periods of the configuration.
    sweep_id: int
        Identifier for the database file (sweep), which contains the configuration ID.
    list_id: int
        The list ID of the configuration in the database file (sweep), 0-indexed. 
    skip_if_exists: bool, optional
        Whether to skip the download if the file already exists in the local temp directory. Default is True.
        
    Returns
    -------
    local_path: str
        The local path to the downloaded configuration file.
    """
    import requests
    import shutil

    # Name of downloaded file:
    filename = f"N{nfp}_{sweep_id:03d}_{list_id:04d}.json"
    temp_dir = Path("temp_db")
    temp_dir.mkdir(parents=True, exist_ok=True)
    local_path = temp_dir / filename

    # Check if the file already exists and skip download if specified
    if skip_if_exists and local_path.is_file():
        print(f'file "{local_path}" already exists, skipping download.')
        return str(local_path)
    else:
        # Address for download
        url = f"https://s3.nexus.mpcdf.mpg.de/public-pyqic-database/data/N{nfp}/sweep{sweep_id:03d}/{filename}"

        with requests.get(url, stream=True) as response, open(local_path, "wb") as file:
            if not response.ok:
                raise RuntimeError(
                    f"Failed to download configuration nfp={nfp}, sweep_id={sweep_id}, list_id={list_id} from url={url}: {response.status_code} {response.reason}"
                )
            shutil.copyfileobj(response.raw, file)
            print(f'file "{local_path}" sucessfully downloaded from {url}') 
        
        return str(local_path)

def load_config_id_file_from_db_online(config_name):
    """
    Load a specific configuration from online database and return the data.

    Parameters
    ----------
        config_name : str
            The name of the configuration, in the format ``"Nx_xxx_xxxx"``.

    Returns
    -------
        config: dict
            Dictionary containing the data from the JSON object.
    """
    # Undo the config name to obtain the nfp, db_file and config_id
    parts = config_name.split("_")
    nfp = int(parts[0][1:])
    sweep_id = int(parts[1])
    config_id = int(parts[2])

    # Download the file from the online database if not exist in local temp directory and load the configuration
    filename = download_from_database(nfp, sweep_id, config_id, skip_if_exists=True)

    # Open the file and load the single JSON object
    import json
    with open(filename, 'r') as file:
        obj = json.load(file)
    config = {nam: np.array(val) for nam, val in obj.items()}
    
    return config
    

def repath_file(file_path, new_base_path=_DATA_FOLDER_PATH):
    """
    Repath file from processed database to original database, new address. It will change the path 
    ``/home/IPP-HGW/rodre/Documents/qi_database/data/...``
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
    # Define the old path
    old_base_path = "/home/IPP-HGW/rodre/Documents/qi_database/"

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
    Returns the argument necessary to initialise a QIC class object.

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
    ell = cumtrapz(np.append(stel_in.d_l_d_varphi,stel_in.d_l_d_varphi[0]), np.append(stel_in.varphi, stel_in.varphi[0]+2*np.pi/stel_in.nfp),\
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