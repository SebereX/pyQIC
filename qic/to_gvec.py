"""
This module contains the routines to output a
near-axis boundary to a GVEC input file using the gframe.
"""
import logging
import numpy as np
from pathlib import Path

def to_gvec(
    stel,
    r: float = 0.1,
    ntheta : int = 41,
    nzeta : int = 81,
    outpath : str = "to_gframe",
    output_prefix : str = None,
    kwargs_gframe: dict = dict(
        tolerance_clean_surface=1e-6,
        tolerance_output=2e-6,
        impose_stell_symmetry=True,
        atol_field_periodicity=1e-5,
        cutoff_gframe=15,
        ),
    mpol : int  = None,
    ntor : int = None,
    verbose: bool = False
    ):
    """
    Export the boundary surface of the given pyQIC configuration into gvec compatible files for parameters and G-frame.
    
    Parameters
    ----------
    stel : Qic 
        pyQIC configuration object
    r : float, optional
        radius at which the pyQIC near-axis expansion is evaluated, yielding the boundary surface, default is ``0.1``
    ntheta : int, optional
        number of poloidal points to evaluate the pyQIC surface, default is ``41``
    nzeta : int, optional
        number of toroidal points per field period to evaluate the pyQIC surface, default is ``81``
    outpath : str, optional
        output path for the files. Will be overwritten if existing. Should have a meaningful name, default is ``"to_gframe"``
    output_prefix: str, optional
        prefix for output files. Default is None, which then uses either ``stel.config_name`` if available, otherwise ``"mycase"``.
    kwargs_gframe: dict, optional
        additional parameters passed to the ``gvec.gframe.construct_gframe_from_surface`` function.
        Default keyword arguments set tolerances and enforce stellarator symmetry.
    mpol : int, optional
        maximum poloidal mode number for the gframe output. Default is None, which means the mode number is chosen to meet the default gframe tolerances.
    ntor : int, optional
        maximum toroidal mode number for the gframe output. Default is None, which means the mode number is chosen to meet the default gframe tolerances.
    verbose: bool, optional
        if True, print additional information during the gframe construction, default is False.

    Return
    ------
    params : dict
        dictionary with gvec parameters, also written to parameterfile
    dict_gframe : dict
        dictionary with the gframe data, also written to the gframe file.
    """
    # Check if gvec is installed
    try:
        import gvec
    except ImportError:
        raise ImportError("gvec is not installed. Please install gvec to use the to_gvec function.")

    # logger setup for printing output:
    gvec.util.logging_setup()
    logger = logging.getLogger("pyQIC_to_gframe")
    if verbose:
        logger.setLevel("INFO")

    # In-memory log handler to capture log messages
    import io
    class ListHandler(logging.Handler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.log_records = []
        def emit(self, record):
            msg = self.format(record)
            self.log_records.append(msg)

    list_handler = ListHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    list_handler.setFormatter(formatter)
    logger.addHandler(list_handler)

    # Get the surface in Cartesian coordinates
    theta = np.linspace(0,2*np.pi,ntheta,endpoint=False) 
    varphi = np.linspace(0,2*np.pi,nzeta*stel.nfp,endpoint=False) 
    X_2D, Y_2D, Z_2D, _, _ = stel.get_boundary_cartesians(
        r=r, 
        theta=theta,
        varphi = varphi
        )
    
    if stel.nfp==1:
        # Rotate by pi/2 around z-axis, for stellarator-symmetry definition in GVEC (xhat~cos, yhat~sin,z~sin)
        X_2D, Y_2D = -Y_2D,X_2D
    
    # Put surface data into array of shape (ntheta, nzeta, 3) for gvec
    xyz = np.stack((X_2D.T,Y_2D.T,Z_2D.T),axis=-1)

    # Directory setup
    if outpath is None:
        outpath = Path.cwd()
    else:
        outpath = Path(outpath)

        if not outpath.exists():
            outpath.mkdir()
            logger.info(f"created output directory {outpath}")
        else:
            logger.info(f"output directory '{outpath}' is overwritten")
    # prefix setup
    if output_prefix is None:
        if hasattr(stel,"config_name"):
            output_prefix = stel.config_name
        else:
            output_prefix = "mycase"
    
    # write the surface data to a netCDF file and construct the gframe files
    with gvec.util.chdir(outpath):
        gvec.scripts.quasr.save_xyz(
            xyz,stel.nfp, 
            output_prefix+'_xyz_surface.nc'
            )
        logger.info(f"pyQIC surface data file written to {outpath / (output_prefix+'_xyz_surface.nc')}")

        dict_params, dict_gframe = gvec.gframe.construct_gframe_from_surface(
            xyz,
            stel.nfp, 
            output_prefix,
            theta0=theta[0],
            zeta0=varphi[0],
            format="toml",
            logger=logger,
            **kwargs_gframe
            )
        
        ## Further modify the toml parameters file as needed ##

        # Adjust mpol and ntor mode numbers
        if mpol is not None and mpol < dict_params['X1_mn_max'][0]:
            dict_params['X1_mn_max'][0] = mpol
            dict_params['X2_mn_max'][0] = mpol
            dict_params['LA_mn_max'][0] = mpol

        if ntor is not None and ntor < dict_params['X1_mn_max'][1]:
            dict_params['X1_mn_max'][1] = ntor
            dict_params['X2_mn_max'][1] = ntor
            dict_params['LA_mn_max'][1] = ntor
        
        # Toroidal current profile over normalized toroidal flux s=rho^2
        mu0 = 4 * np.pi * 1e-7
        curtor = 2 * np.pi / mu0 * stel.I2 * r * r
        dict_params["I_tor"] = {
            "type": "polynomial",
            "coefs": [0.0,1.0], # Itor(s) = scale*(coef0 +coef1*s)
            "scale": curtor, # total toroidal current, in Ampere
        }

        # Pressure profile over normalized toroidal fluxs=rho^2
        pscale = - stel.p2 * r * r
        dict_params["pres"] = {
            "type": "polynomial",
            "coefs": [1.0,-1.0], # p(s) = scale*(coef0 +coef1*s)
            "scale": pscale,    # pscale should be positive, in pascal
        }

        # Set toroidal flux
        dict_params["phiedge"] = np.pi * r * r * stel.spsi * stel.Bbar

        # Save into toml file
        from gvec.util import write_parameters
        write_parameters(dict_params, f"{dict_params['ProjectName']}-parameters.toml")

    return dict_params,dict_gframe

def plot_cross_sections_gframe(dict_gframe, n_cross_sections=7, tolerance = 1e-5):
    """
    Plot cross sections of the G-frame surface for a quick check of the output.

    Parameters
    ----------
    dict_gframe : dict
        dictionary containing the gframe data, as returned by the to_gvec function.
    n_cross_sections : int, optional
        number of cross sections to plot, default is 7
    tolerance : float, optional
        tolerance for the gframe construction, default is 1e-5.

    Returns 
    -------
    fig : matplotlib.figure.Figure
        figure containing the cross section plots.
    """
    # Check if gvec is installed
    try:
        import gvec
    except ImportError:
        raise ImportError("gvec is not installed. Please install gvec to use the plot_cross_sections_gframe function.")

    # Construct surface in G-frame
    surf = gvec.gframe.to_surface(dict_gframe,ntheta=81,nzeta=80,tolerance=tolerance)

    # Construct surface in cylindrical coordinates
    surf_RZ = gvec.gframe.to_RZ(surf["xyz"],surf["nfp"],ntheta=81,nzeta=80,tolerance=tolerance)

    # Make cross section plot
    nz_in = surf["X1"].shape[1]
    step = int(nz_in // (n_cross_sections + 1))
    fig = gvec.gframe.plot_cross_section_comparison(surf,surf_RZ,step=step,halfperiod=True)

    return fig