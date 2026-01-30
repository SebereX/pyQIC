"""
This module contains the routines to output a
near-axis boundary to a VMEC input file
"""
from datetime import datetime
import numpy as np
from .Frenet_to_cylindrical import Frenet_to_cylindrical
from .Frenet_to_normal_frame import Frenet_to_normal_frame
from .util import mu0, to_Fourier
from .obtain_current_version import get_qic_info
from multiprocessing import Manager, Pool
from scipy.optimize import fsolve
import os
from tqdm import tqdm
import netCDF4


def make_gvec_frenet_inputs(self, r, nfp, mpol, ntor, ntor_axis):
    """
    Create GVEC Frenet inputs for near-axis expansion.
    
    Parameters
    ----------
    r : float
        Expansion parameter
    nfp : int
        Number of field periods
    mpol : int
        Maximum poloidal mode number
    ntor : int
        Maximum toroidal mode number
    ntor_axis : int
        Maximum toroidal mode number for axis
    xo_phi : callable
        Axis position function
    bo_phi : callable
        Binormal vector function
    no_phi : callable
        Normal vector function
    X1 : callable
        First-order X displacement function
    Y1 : callable
        First-order Y displacement function
    
    Returns
    -------
    dict
        Dictionary containing GVEC input data
    """
    #########
    # GRIDS #
    #########
    # Grid shift factor
    gs = 0.5
    
    # Grid dimensions
    N_theta = 2 * mpol + 1
    N_zeta = 2 * ntor + 1
    N_zeta_axis = 2 * ntor_axis + 1
        
    # Zeta grid for axis (extended to full 2*pi)
    zeta_extended = np.linspace(0, 2 * np.pi, nfp * N_zeta_axis, endpoint=False) + (gs * 2 * np.pi) / (nfp * N_zeta_axis)

    ############################
    # NORMAL FRAME COORDINATES #
    ############################
    # Obtain surface in normal frame
    X_2D, Y_2D, phi0_2D, phi_2D, (theta, phi) = self.Frenet_to_normal_frame(r=r, ntor=ntor, mpol=mpol, parallel=True, return_theta_phi=True)

    ##########################
    # G-FRAME TRANSFORMATION #
    ##########################
    # Gamma twist angle
    m = self.helicity
    gamma_2D = -m * nfp * phi_2D

    # Rotated G-frame (X',Y')
    X_G_2D = -Y_2D * np.cos(gamma_2D) + X_2D * np.sin(gamma_2D)
    Y_G_2D = X_2D * np.cos(gamma_2D) + Y_2D * np.sin(gamma_2D)

    ####################
    # AXIS INFORMATION #
    ####################
    # Evaluate the axis
    x_axis = self.x0_cart_spline(zeta_extended)
    y_axis = self.y0_cart_spline(zeta_extended)
    z_axis = self.z0_cart_spline(zeta_extended)

    # Put together into array (3, N)
    axis_xyz = np.array([x_axis, y_axis, z_axis])

    # Normal vector in untwisted frame
    gamma = -m * nfp * zeta_extended
    normal_cart = np.array([self.normal_x_cart_spline(zeta_extended),
                            self.normal_y_cart_spline(zeta_extended),
                            self.normal_z_cart_spline(zeta_extended)])
    binormal_cart = np.array([self.binormal_x_cart_spline(zeta_extended),
                              self.binormal_y_cart_spline(zeta_extended),
                              self.binormal_z_cart_spline(zeta_extended)])
    axis_Nxyz = (-binormal_cart * np.cos(gamma) + normal_cart * np.sin(gamma))
    axis_Bxyz = (normal_cart * np.cos(gamma) + binormal_cart * np.sin(gamma))  
    
    ##############################
    # OUTPUT DICTIONARY FOR GVEC #
    ##############################
    """
    The dictionary has the following structure:
    {
            'NFP': int,
            'VERSION': int,
            'axis': {
                'n_max': int,
                'nzeta': int,
                'zeta': 1D array,
                'xyz': 2D array (3, N),
                'Nxyz': 2D array (3, N),
                'Bxyz': 2D array (3, N)
            },
            'boundary': {
                'm_max': int,
                'n_max': int,
                'lasym': int,
                'ntheta': int,
                'nzeta': int,
                'theta': 1D array,
                'zeta': 1D array,
                'X': 2D array (ntheta, nzeta),
                'Y': 2D array (ntheta, nzeta)
            }
        }
    """

    # Return dictionary with GVEC inputs
    GVEC_data = {
        "NFP": nfp,
        "VERSION": 310,
        'axis': {
                'n_max': ntor_axis,
                'nzeta': N_zeta_axis,
                'zeta': zeta_extended,
                'xyz': axis_xyz,
                'Nxyz': axis_Nxyz,
                'Bxyz': axis_Bxyz
            },
        'boundary': {
                'm_max': mpol,
                'n_max': ntor,
                'lasym': 0,
                'ntheta': N_theta,
                'nzeta': N_zeta,
                'theta': theta,
                'zeta': phi,
                'X': X_G_2D,
                'Y': Y_G_2D
            }
    }

    return GVEC_data

def to_gvec(self, filename, r, nfp, mpol, ntor, ntor_axis, overwrite=False, parallel=True, verbose=False):
    """
    Create a GVEC netCDF file from near-axis expansion data.
    
    Parameters
    ----------
    filename : str
        Path to the output netCDF file.
    r : float
        Expansion parameter
    nfp : int
        Number of field periods
    mpol : int
        Maximum poloidal mode number
    ntor : int
        Maximum toroidal mode number
    ntor_axis : int
        Maximum toroidal mode number for axis
    overwrite : bool, optional
        If True, overwrite existing file. Default is False.
    parallel : bool, optional
        If True, use parallel processing. Default is True.
    verbose : bool, optional
        If True, display progress bar. Default is False.
    
    Returns
    -------
    str
        Path to the created file.
    """
    # Create GVEC input data dictionary
    GVEC_data = self.make_gvec_frenet_inputs(r, nfp, mpol, ntor, ntor_axis)
    
    # Write to netCDF file
    write_nc_GVEC(filename, GVEC_data, overwrite=overwrite)
    
    return filename

def write_nc_GVEC(filename, data_dict, overwrite=False):
    """
    Create a GVEC netCDF file from Python data.

    Parameters
    ----------
    filename : str
        Path to the output netCDF file.
    data_dict : dict
        Dictionary containing the data to write. Structure:
        {
            'NFP': int,
            'VERSION': int,
            'axis': {
                'n_max': int,
                'nzeta': int,
                'zeta': 1D array,
                'xyz': 2D array (3, N),
                'Nxyz': 2D array (3, N),
                'Bxyz': 2D array (3, N)
            },
            'boundary': {
                'm_max': int,
                'n_max': int,
                'lasym': int,
                'ntheta': int,
                'nzeta': int,
                'theta': 1D array,
                'zeta': 1D array,
                'X': 2D array (ntheta, nzeta),
                'Y': 2D array (ntheta, nzeta)
            }
        }
    overwrite : bool, optional
        If True, overwrite existing file. Default is False.

    Returns
    -------
    str
        Path to the created file.
    
    Example
    -------
    >>> data = {
    ...     'NFP': 2,
    ...     'VERSION': 310,
    ...     'axis': {
    ...         'n_max': 100,
    ...         'nzeta': 201,
    ...         'zeta': np.linspace(0, np.pi, 201),
    ...         'xyz': np.random.randn(3, 402),
    ...         'Nxyz': np.random.randn(3, 402),
    ...         'Bxyz': np.random.randn(3, 402)
    ...     },
    ...     'boundary': {
    ...         'm_max': 4,
    ...         'n_max': 10,
    ...         'lasym': 0,
    ...         'ntheta': 9,
    ...         'nzeta': 21,
    ...         'theta': np.linspace(0, 2*np.pi, 9),
    ...         'zeta': np.linspace(0, np.pi, 21),
    ...         'X': np.random.randn(9, 21),
    ...         'Y': np.random.randn(9, 21)
    ...     }
    ... }
    >>> write_nc_GVEC('output.nc', data)
    """
    mode = 'w' if overwrite else 'w'
    
    # Create the netCDF file
    with netCDF4.Dataset(filename, mode, format='NETCDF4') as nc:
        
        # Write root-level variables
        var_nfp = nc.createVariable('NFP', 'i8')
        var_nfp[:] = data_dict['NFP']
        
        var_version = nc.createVariable('VERSION', 'i8')
        var_version[:] = data_dict['VERSION']
        
        # Create and populate 'axis' group
        if 'axis' in data_dict:
            axis_grp = nc.createGroup('axis')
            axis_data = data_dict['axis']
            
            # Create scalar variables
            var_n_max = axis_grp.createVariable('n_max', 'i8')
            var_n_max[:] = axis_data['n_max']
            
            var_nzeta = axis_grp.createVariable('nzeta', 'i8')
            var_nzeta[:] = axis_data['nzeta']
            
            # Create zeta array
            zeta_arr = axis_data['zeta']
            dim_zeta = axis_grp.createDimension('zeta(:)_dimension_1', len(zeta_arr))
            var_zeta = axis_grp.createVariable('zeta(:)', 'f8', ('zeta(:)_dimension_1',))
            var_zeta[:] = zeta_arr
            
            # Create xyz array
            xyz_arr = axis_data['xyz']
            dim_xyz_1 = axis_grp.createDimension('xyz(::)_dimension_1', xyz_arr.shape[0])
            dim_xyz_2 = axis_grp.createDimension('xyz(::)_dimension_2', xyz_arr.shape[1])
            var_xyz = axis_grp.createVariable('xyz(::)', 'f8', ('xyz(::)_dimension_1', 'xyz(::)_dimension_2'))
            var_xyz[:, :] = xyz_arr
            
            # Create Nxyz array
            nxyz_arr = axis_data['Nxyz']
            dim_nxyz_1 = axis_grp.createDimension('Nxyz(::)_dimension_1', nxyz_arr.shape[0])
            dim_nxyz_2 = axis_grp.createDimension('Nxyz(::)_dimension_2', nxyz_arr.shape[1])
            var_nxyz = axis_grp.createVariable('Nxyz(::)', 'f8', ('Nxyz(::)_dimension_1', 'Nxyz(::)_dimension_2'))
            var_nxyz[:, :] = nxyz_arr
            
            # Create Bxyz array
            bxyz_arr = axis_data['Bxyz']
            dim_bxyz_1 = axis_grp.createDimension('Bxyz(::)_dimension_1', bxyz_arr.shape[0])
            dim_bxyz_2 = axis_grp.createDimension('Bxyz(::)_dimension_2', bxyz_arr.shape[1])
            var_bxyz = axis_grp.createVariable('Bxyz(::)', 'f8', ('Bxyz(::)_dimension_1', 'Bxyz(::)_dimension_2'))
            var_bxyz[:, :] = bxyz_arr
        
        # Create and populate 'boundary' group
        if 'boundary' in data_dict:
            boundary_grp = nc.createGroup('boundary')
            boundary_data = data_dict['boundary']
            
            # Create scalar variables
            var_m_max = boundary_grp.createVariable('m_max', 'i8')
            var_m_max[:] = boundary_data['m_max']
            
            var_n_max = boundary_grp.createVariable('n_max', 'i8')
            var_n_max[:] = boundary_data['n_max']
            
            var_lasym = boundary_grp.createVariable('lasym', 'i8')
            var_lasym[:] = boundary_data['lasym']
            
            var_ntheta = boundary_grp.createVariable('ntheta', 'i8')
            var_ntheta[:] = boundary_data['ntheta']
            
            var_nzeta = boundary_grp.createVariable('nzeta', 'i8')
            var_nzeta[:] = boundary_data['nzeta']
            
            # Create theta array
            theta_arr = boundary_data['theta']
            dim_theta = boundary_grp.createDimension('theta(:)_dimension_1', len(theta_arr))
            var_theta = boundary_grp.createVariable('theta(:)', 'f8', ('theta(:)_dimension_1',))
            var_theta[:] = theta_arr
            
            # Create zeta array
            zeta_arr = boundary_data['zeta']
            dim_zeta = boundary_grp.createDimension('zeta(:)_dimension_1', len(zeta_arr))
            var_zeta = boundary_grp.createVariable('zeta(:)', 'f8', ('zeta(:)_dimension_1',))
            var_zeta[:] = zeta_arr
            
            # Create X array
            x_arr = boundary_data['X']
            dim_x_1 = boundary_grp.createDimension('X(::)_dimension_1', x_arr.shape[0])
            dim_x_2 = boundary_grp.createDimension('X(::)_dimension_2', x_arr.shape[1])
            var_x = boundary_grp.createVariable('X(::)', 'f8', ('X(::)_dimension_1', 'X(::)_dimension_2'))
            var_x[:, :] = x_arr
            
            # Create Y array
            y_arr = boundary_data['Y']
            dim_y_1 = boundary_grp.createDimension('Y(::)_dimension_1', y_arr.shape[0])
            dim_y_2 = boundary_grp.createDimension('Y(::)_dimension_2', y_arr.shape[1])
            var_y = boundary_grp.createVariable('Y(::)', 'f8', ('Y(::)_dimension_1', 'Y(::)_dimension_2'))
            var_y[:, :] = y_arr
    
    print(f"Successfully created netCDF file: {filename}")
    return filename

def check_nc_GVEC(filename, verbose=True):
    """
    Validate a GVEC netCDF file to ensure it has the correct structure and format.

    Parameters
    ----------
    filename : str
        Path to the netCDF file to validate.
    verbose : bool, optional
        If True, print detailed validation messages. Default is True.

    Returns
    -------
    dict
        Dictionary with validation results:
        {
            'valid': bool,
            'errors': list of str,
            'warnings': list of str,
            'info': dict with file information
        }
    """
    errors = []
    warnings = []
    info = {}
    
    try:
        nc = netCDF4.Dataset(filename, 'r')
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Cannot open file: {e}"],
            'warnings': [],
            'info': {}
        }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATING GVEC NETCDF FILE: {filename}")
        print(f"{'='*60}\n")
    
    # Check file format
    info['format'] = nc.file_format
    info['data_model'] = nc.data_model
    if nc.file_format != 'NETCDF4':
        warnings.append(f"File format is {nc.file_format}, expected NETCDF4")
    
    # Check root-level variables
    if verbose:
        print("Checking root-level variables...")
    
    required_root_vars = ['NFP', 'VERSION']
    for var_name in required_root_vars:
        flag_error = False
        if var_name not in nc.variables:
            errors.append(f"Missing required root variable: {var_name}")
        else:
            var = nc.variables[var_name]
            # Check type (should be integer)
            if not np.issubdtype(var.dtype, np.integer):
                errors.append(f"Variable {var_name} has type {var.dtype}, expected integer")
                flag_error = True
            # Check shape (should be scalar)
            if var.shape != ():
                errors.append(f"Variable {var_name} has shape {var.shape}, expected scalar ()")
                flag_error = True
            # Store value
            info[var_name] = int(var[:])
            if verbose and not flag_error:
                print(f"  ✓ {var_name}: {info[var_name]}")
            elif verbose:
                print(f"  ✗ {var_name} has issues")
    
    # Check required groups
    if verbose:
        print("\nChecking groups...")
    
    required_groups = ['axis', 'boundary']
    for group_name in required_groups:
        flag_error = False
        if group_name not in nc.groups:
            errors.append(f"Missing required group: {group_name}")
            flag_error = True
        else:
            if verbose:
                if not flag_error:
                    print(f"  ✓ Group '{group_name}' exists")
                else:
                    print(f"  ✗ Group '{group_name}' has issues")

    
    # Validate 'axis' group
    if 'axis' in nc.groups:
        if verbose:
            print("\nValidating 'axis' group...")
        axis_grp = nc.groups['axis']
        info['axis'] = {}
        
        # Check scalar variables
        axis_scalars = {
            'n_max': ('int', ()),
            'nzeta': ('int', ())
        }
        
        for var_name, (expected_type, expected_shape) in axis_scalars.items():
            flag_error = False
            if var_name not in axis_grp.variables:
                errors.append(f"Missing variable in axis group: {var_name}")
                flag_error = True
            else:
                var = axis_grp.variables[var_name]
                if expected_type == 'int' and not np.issubdtype(var.dtype, np.integer):
                    errors.append(f"axis/{var_name} has type {var.dtype}, expected integer")
                    flag_error = True
                if var.shape != expected_shape:
                    errors.append(f"axis/{var_name} has shape {var.shape}, expected {expected_shape}")
                    flag_error = True
                info['axis'][var_name] = int(var[:])
                if verbose:
                    if not flag_error:
                        print(f"  ✓ {var_name}: {info['axis'][var_name]}")
                    else:
                        print(f"  ✗ {var_name} has issues")
        
        # Check array variables
        axis_arrays = ['zeta(:)', 'xyz(::)', 'Nxyz(::)', 'Bxyz(::)']
        
        for var_name in axis_arrays:
            flag_error = False
            if var_name not in axis_grp.variables:
                errors.append(f"Missing variable in axis group: {var_name}")
                flag_error = True
            else:
                var = axis_grp.variables[var_name]
                if not np.issubdtype(var.dtype, np.floating):
                    warnings.append(f"axis/{var_name} has type {var.dtype}, expected float")
                # Check specific shapes
                if var_name == 'zeta(:)':
                    if len(var.shape) != 1:
                        errors.append(f"axis/zeta(:) should be 1D, got shape {var.shape}")
                        flag_error = True
                    else:
                        nzeta = var.shape[0]
                        info['axis']['zeta_size'] = nzeta
                        if 'nzeta' in info['axis'] and nzeta != info['axis']['nzeta']:
                            errors.append(f"axis/zeta(:) size {nzeta} doesn't match nzeta={info['axis']['nzeta']}")
                            flag_error = True
                        if verbose:
                            if not flag_error:
                                print(f"  ✓ zeta(:): shape {var.shape}")
                            else:
                                print(f"  ✗ zeta(:) has issues")
                
                elif var_name in ['xyz(::)', 'Nxyz(::)', 'Bxyz(::)']:
                    if len(var.shape) != 2:
                        errors.append(f"axis/{var_name} should be 2D, got shape {var.shape}")
                        flag_error = True
                    elif var.shape[0] != 3:
                        errors.append(f"axis/{var_name} first dimension should be 3, got {var.shape[0]}")
                        flag_error = True
                    else:
                        if verbose:
                            if not flag_error:
                                print(f"  ✓ {var_name}: shape {var.shape}")
                            else:
                                print(f"  ✗ {var_name} has issues")
                        # Check consistency across arrays
                        if 'axis_array_size' not in info['axis']:
                            info['axis']['axis_array_size'] = var.shape[1]
                        elif info['axis']['axis_array_size'] != var.shape[1]:
                            errors.append(f"axis/{var_name} second dimension {var.shape[1]} doesn't match other arrays")
                            flag_error = True
    # Validate 'boundary' group
    if 'boundary' in nc.groups:
        if verbose:
            print("\nValidating 'boundary' group...")
        boundary_grp = nc.groups['boundary']
        info['boundary'] = {}
        
        # Check scalar variables
        boundary_scalars = {
            'm_max': ('int', ()),
            'n_max': ('int', ()),
            'lasym': ('int', ()),
            'ntheta': ('int', ()),
            'nzeta': ('int', ())
        }
        
        for var_name, (expected_type, expected_shape) in boundary_scalars.items():
            flag_error = False
            if var_name not in boundary_grp.variables:
                errors.append(f"Missing variable in boundary group: {var_name}")
                flag_error = True
            else:
                var = boundary_grp.variables[var_name]
                if expected_type == 'int' and not np.issubdtype(var.dtype, np.integer):
                    errors.append(f"boundary/{var_name} has type {var.dtype}, expected integer")
                    flag_error = True
                if var.shape != expected_shape:
                    errors.append(f"boundary/{var_name} has shape {var.shape}, expected {expected_shape}")
                    flag_error = True
                info['boundary'][var_name] = int(var[:])
                if verbose:
                    if not flag_error:
                        print(f"  ✓ {var_name}: {info['boundary'][var_name]}")
                    else:
                        print(f"  ✗ {var_name} has issues")
        
        # Check array variables
        boundary_arrays = {
            'theta(:)': 1,
            'zeta(:)': 1,
            'X(::)': 2,
            'Y(::)': 2
        }
        
        for var_name, expected_ndim in boundary_arrays.items():
            flag_error = False
            if var_name not in boundary_grp.variables:
                errors.append(f"Missing variable in boundary group: {var_name}")
                flag_error = True
            else:
                var = boundary_grp.variables[var_name]
                if not np.issubdtype(var.dtype, np.floating):
                    warnings.append(f"boundary/{var_name} has type {var.dtype}, expected float")
                
                if len(var.shape) != expected_ndim:
                    errors.append(f"boundary/{var_name} should be {expected_ndim}D, got shape {var.shape}")
                    flag_error = True
                else:
                    # Check specific shapes
                    if var_name == 'theta(:)':
                        ntheta = var.shape[0]
                        info['boundary']['theta_size'] = ntheta
                        if 'ntheta' in info['boundary'] and ntheta != info['boundary']['ntheta']:
                            errors.append(f"boundary/theta(:) size {ntheta} doesn't match ntheta={info['boundary']['ntheta']}")
                            flag_error = True
                        if verbose:
                            if not flag_error:
                                print(f"  ✓ theta(:): shape {var.shape}")
                            else:
                                print(f"  ✗ theta(:) has issues")
                    
                    elif var_name == 'zeta(:)':
                        nzeta = var.shape[0]
                        info['boundary']['zeta_size'] = nzeta
                        if 'nzeta' in info['boundary'] and nzeta != info['boundary']['nzeta']:
                            errors.append(f"boundary/zeta(:) size {nzeta} doesn't match nzeta={info['boundary']['nzeta']}")
                            flag_error = True
                        if verbose:
                            if not flag_error:
                                print(f"  ✓ zeta(:): shape {var.shape}")
                            else:
                                print(f"  ✗ zeta(:) has issues")
                    
                    elif var_name in ['X(::)', 'Y(::)']:
                        expected_shape = (info['boundary'].get('ntheta', 0), 
                                        info['boundary'].get('nzeta', 0))
                        if 'ntheta' in info['boundary'] and 'nzeta' in info['boundary']:
                            if var.shape != expected_shape:
                                errors.append(f"boundary/{var_name} shape {var.shape} doesn't match (ntheta, nzeta)={expected_shape}")
                                flag_error = True
                        if verbose:
                            if not flag_error:
                                print(f"  ✓ {var_name}: shape {var.shape}")
                            else:
                                print(f"  ✗ {var_name} has issues")
    
    nc.close()
    
    # Summary
    valid = len(errors) == 0
    
    if verbose:
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        if valid:
            print("✓ File is VALID")
        else:
            print("✗ File has ERRORS")
        
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
        
        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        print()
    
    return {
        'valid': valid,
        'errors': errors,
        'warnings': warnings,
        'info': info
    }

def read_nc_GVEC(filename):
    """
    Read GVEC input netCDF file and extract all information.

    Parameters
    ----------
    filename : str
        Path to the netCDF file.

    Returns
    -------
    netCDF4.Dataset
        The opened netCDF dataset with all groups and variables.
    """
    data = netCDF4.Dataset(filename, 'r')

    print(f"Reading GVEC netCDF file: {filename}\n")
    print(f"File format: {data.file_format}")
    print(f"Data model: {data.data_model}")
    
    # Print all information recursively
    print_group_info(data, "root")
    
    return data

def print_group_info(group, group_name="root", indent=0):
    """
    Recursively print all information about a netCDF group.
    
    Parameters
    ----------
    group : netCDF4.Group or netCDF4.Dataset
        The group to inspect
    group_name : str
        Name of the current group
    indent : int
        Indentation level for nested groups
    """
    prefix = "  " * indent
    print(f"\n{prefix}{'='*60}")
    print(f"{prefix}GROUP: {group_name}")
    print(f"{prefix}{'='*60}")
    
    # Print global attributes
    if group.ncattrs():
        print(f"{prefix}Global Attributes:")
        for attr in group.ncattrs():
            print(f"{prefix}  {attr}: {getattr(group, attr)}")
    
    # Print dimensions
    if group.dimensions:
        print(f"\n{prefix}Dimensions:")
        for dim_name, dim in group.dimensions.items():
            unlimited = " (unlimited)" if dim.isunlimited() else ""
            print(f"{prefix}  {dim_name}: {len(dim)}{unlimited}")
    
    # Print variables
    if group.variables:
        print(f"\n{prefix}Variables:")
        for var_name, var in group.variables.items():
            print(f"{prefix}  {var_name}:")
            print(f"{prefix}    - Type: {var.dtype}")
            print(f"{prefix}    - Dimensions: {var.dimensions}")
            print(f"{prefix}    - Shape: {var.shape}")
            
            # Print variable attributes
            if var.ncattrs():
                print(f"{prefix}    - Attributes:")
                for attr in var.ncattrs():
                    print(f"{prefix}      * {attr}: {getattr(var, attr)}")
            
            # Print actual data (if not too large)
            try:
                data = var[:]
                if data.size <= 100:  # Only print if reasonably small
                    print(f"{prefix}    - Data:\n{prefix}      {data}")
                else:
                    print(f"{prefix}    - Data: [array with {data.size} elements]")
                    print(f"{prefix}      Min: {np.min(data)}, Max: {np.max(data)}, Mean: {np.mean(data)}")
            except Exception as e:
                print(f"{prefix}    - Data: Unable to read ({e})")
    
    # Recursively print subgroups
    if group.groups:
        print(f"\n{prefix}Subgroups:")
        for subgroup_name, subgroup in group.groups.items():
            print_group_info(subgroup, subgroup_name, indent + 1)

