import os
import sys
import shutil
import numpy as np
from qic import Qic
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
from simsopt.mhd import Vmec, Boozer

def test_to_vmec_benchmark(num = 1):
    ##########
    # INPUTS #
    ##########
    # Folder name
    lead_name_array = ["QI_NFP2_Katia_std_buffer/", "QI_NFP2_Katia_smooth_buffer/", "QI_NFP3_Katia_smooth_buffer_beta_shape/", \
                        # "QI_NFP3_half_period_better_varphi_right_axis_extra/", \
                        "QI_opt_NFP3_Katia_smooth_buffer_hres/"]

    # NAE field name
    name_array = ["QI NFP2 Katia", "QI NFP2 Katia smooth", "QI NFP3 Katia smooth beta shape", \
                # "QI NFP3 half period better", \
                "QI NFP3 Katia opt"]

    # Scan of aspect ratios to consider
    N_r = 8
    r_array = np.logspace(-2.5,-0.75,N_r)

    ##################################
    # CHOOSE num RANDOM CASES TO RUN #
    ##################################
    # Choose random configs
    ind_array = np.random.choice(len(name_array), size=num, replace=True)

    # Unique configs chosen
    ind_array_unique = np.unique(ind_array)

    # Choose random r
    ind_r_array = np.random.choice(np.arange(N_r), size=num, replace=True)

    # Tolerance to run VMEC with
    tol_exp = 18

    #######
    # RUN #
    #######
    for ind_config in ind_array_unique:
        # Get the name and lead name for this configuration
        name = name_array[ind_config]
        lead_name = lead_name_array[ind_config]

        # Construct near-axis field
        stel = make_nae_stellarator(name, nphi = 501 if name == "QI NFP3 half period better" else 2001, order = 'r3')

        # Parameters for vmec
        params={'mpol':10 if name == "QI NFP3 Katia opt" else 8, 'ntor': 35 if name == "QI NFP3 Katia opt" else 30, 'ns_array': [25, 51, 75, 101], 'delt': 0.2,
                                                "niter_array":[4000,10000,20000,int(4e6)],
                                                "lasym": False if name == "QI NFP3 Katia opt" else True
                                                }
        tol = 10**(-tol_exp)
        params['ftol_array'] = [tol*100, tol*100, tol*50, tol]

        # Run over the r for which ind_array has the current configuration index
        r_indices = np.where(ind_array == ind_config)[0]
        ind_r_array_config = np.unique(ind_r_array[r_indices])
        r_array_config = np.unique(r_array[ind_r_array_config])

        print(f"Running VMEC benchmark for configuration {name} for {lead_name} with aspect ratio scan {r_array_config}...")


        for jr, r in zip(ind_r_array_config, r_array_config):
            # Use the correct subfolder for both output and comparison
            if (name == "QI NFP2 Katia" or name == "QI NFP2 Katia smooth") and jr == 7:
                params['lasym'] = False
            subfolder = lead_name + f'A_{jr}'
            filename_out = create_vmec_inputs(stel, r=r, params=params, name='', outpath='./vmec_benchmark/')
            filename_compare = os.path.join('vmec_benchmark', subfolder, 'input.temp')
            
            # Compare the output file with the expected file
            data_out = read_vmec_inputs(filename_out)
            data_compare = read_vmec_inputs(filename_compare)
            compare_dict(data_out, data_compare, r=r, verbose = False)

            # Clean up the output file
            print(f"Comparison successful for r={r}")

def create_vmec_inputs(stel, 
                        r=0.1, 
                        name='', 
                        params={'mpol': 13, 'ntor': 51, 
                                'ns_array': [25, 51, 75, 101], 
                                'ftol_array':[1e-15, 1e-15, 1e-15, 1e-15], 
                                "niter_array":[4000,10000,20000,40000], 
                                "delt": 0.9, 
                                "lasym": True},
                        outpath = None):
    ## Function that creates the VMEC inputs in the right folder ##

    # Define paths
    if outpath is not None:
        OUT_DIR = os.path.join(outpath, name)
    else:
        this_path = Path(__file__).parent.resolve()
        OUT_DIR=os.path.join(this_path,name)

    ## COMPLETE PARAMS VMEC
    if not 'mpol' in params.keys():
        params["mpol"] = 6
    if not 'ntor' in params.keys():
        params["ntor"] = 16
    if not 'ns_array' in params.keys():
        params["ns_array"] = [25, 51, 101]
    if not 'ftol_array' in params.keys():
        params["ftol_array"] = [1e-14, 1e-14, 1e-14]
    if not 'niter_array' in params.keys():
        params["niter_array"] = [4000, 10000, 40000]
    if not 'lasym' in params.keys():
        params["lasym"] = True

    ## CONFIGURATION 
    if not stel:
        print('*')
        r_order="r1"

        name_stel = 'QI NFP2 r2'
        qsc = Qic.from_paper(name_stel,order="r1")

    vmec_input = os.path.join(OUT_DIR,f'input.temp')

    ## CREATE VMEC INPUT FILE
    stel.to_vmec(vmec_input, r=r, ntorMax=100, params=params, ntheta = 50)

    return vmec_input

def make_nae_stellarator(name, **kwargs):
    # Reference buffer consideration dictionary
    default_buffer_dict = {"omn_method": 'buffer', 
                           "k_buffer": 1, 
                           "p_buffer": 2, 
                           "delta": np.pi/5}

    if name == "QI NFP2 Katia":
        ########################
        # ORIGINAL KATIA NFP 2 # (from [Camacho-Mata, Plunk, Jorge (2022)])
        ######################## 
        # Axis
        Raxis = {"type": 'fourier',
                    "input_value": {"cos": [ 1.0,0.0,-1/17 ],
                                    "sin":  [ 0.0,0.0,0.0 ]}}
        Zaxis = {"type": 'fourier',
                    "input_value": {"cos": [ 0.0,0.0,0.0 ],
                                    "sin": [ 0.0,0.8/2.04,0.01/2.04 ]}}
        # Magnetic field
        B0 = {"type": 'fourier',
                    "input_value": {"cos": [1.0, 0.15], "sin": [0.0, 0.0]}}
        
        # First order field
        d = None
        d_over_curvature = {"type": 'scalar', "input_value": 0.73}

        # Second order shaping : 'minimal' shaping choice
        X2c = {"type": 'scalar', "input_value": 0}
        X2s = {"type": 'scalar', "input_value": 0}
        
        # Buffer region details : use the standard form of [Camacho et al. (2022)]
        omn_buffer = default_buffer_dict.copy()
        omn_buffer["omn_method"] = 'non-zone'
        omn_buffer["delta"] = 0.0
        omn_buffer["k_buffer"] = 2

        # Additional properties
        sigma0 = 0.0
        nfp     = 2
        p2      = 0.0
        order = 'r3'
        omn = True

        vars = ["Raxis", "Zaxis", "B0", "d", "d_over_curvature", "X2c", "X2s", "omn_buffer", "sigma0", "nfp", "p2", "order", "omn"]

    elif name == "QI NFP2 Katia smooth":
        ####################################
        # ORIGINAL KATIA NFP 2 w/ SMOOTH α #
        ####################################
        # Axis
        Raxis = {"type": 'fourier',
                    "input_value": {"cos": [ 1.0,0.0,-1/17 ],
                                    "sin":  [ 0.0,0.0,0.0 ]}}
        Zaxis = {"type": 'fourier',
                    "input_value": {"cos": [ 0.0,0.0,0.0 ],
                                    "sin": [ 0.0,0.8/2.04,0.01/2.04 ]}}
        # Magnetic field
        B0 = {"type": 'fourier',
                    "input_value": {"cos": [1.0, 0.15], "sin": [0.0, 0.0]}}
        
        # First order field
        d = None
        d_over_curvature = {"type": 'scalar', "input_value": 0.73}

        # Second order shaping : 'minimal' shaping choice
        X2c = {"type": 'scalar', "input_value": 0}
        X2s = {"type": 'scalar', "input_value": 0}
        
        # Buffer region details : use the standard form of [Camacho et al. (2022)]
        omn_buffer = {"omn_method": 'simple-fourier', 
                       "k_buffer": 5}

        # Additional properties
        sigma0 = 0.0
        nfp     = 2
        p2      = 0.0
        order = 'r3'
        omn = True

        vars = ["Raxis", "Zaxis", "B0", "d", "d_over_curvature", "X2c", "X2s", "omn_buffer", "sigma0", "nfp", "p2", "order", "omn"]
    
    elif name == "QI NFP3 Katia smooth beta shape":
        #############################################################
        # ORIGINAL KATIA NFP 3 w/ smooth α and added I2, p2, and X2 # (original was from [Camacho-Mata, Plunk, Jorge (2022)])
        #############################################################
        # Axis
        Raxis = {"type": 'fourier',
                     "input_value": {"cos": [ 1.0,  9.075485257221899e-02, -2.058279495912439e-02, -1.106766494783158e-02, -1.644390251809640e-03 ],
                                     "sin":  [ 0.0,0.0,0.0,0.0,0.0 ]}}
        Zaxis = {"type": 'fourier',
                    "input_value": {"cos": [ 0.0,0.0,0.0,0.0 ],
                                    "sin": [ 0.0,0.36,0.02,0.01 ]}}
        # Magnetic field
        B0 = {"type": 'fourier', "input_value": {"cos": [1.0, 0.25], "sin": []}}
        
        # First order field
        d = None
        d_over_curvature = {"type": 'scalar', "input_value": 0.73}

        # Second order shaping : 'minimal' shaping choice
        X2c = {"type": 'fourier', "input_value": {"cos": [ 0.0, 0.0, 0.0], 
                                          "sin": [ 0.0,0.1,-0.6,0.1]}}
        X2s = {"type": 'fourier', "input_value": {"cos": [ 0.1,0.0,-0.1,0.6], 
                                                "sin": [ 0.0,0.0,0.0]}}
        
        # Buffer region details : use the standard form of [Camacho et al. (2022)]
        omn_buffer = {"omn_method": 'simple-fourier', 
                       "k_buffer": 5}

        # Additional properties
        sigma0 = 0.0
        nfp = 3
        I2 = -0.9
        p2 = -600000.
        order = 'r3'
        omn = True
        

        vars = ["Raxis", "Zaxis", "B0", "d", "d_over_curvature", "X2c", "X2s", "omn_buffer", "sigma0", "nfp", "p2", "I2", "order", "omn"]

    elif name == "QI NFP3 Katia smooth":
        ####################################
        # ORIGINAL KATIA NFP 3 w/ smooth α #
        ####################################
        # Axis
        Raxis = {"type": 'fourier',
                    "input_value": {"cos": [ 1.0,  9.075485257221899e-02, -2.058279495912439e-02, -1.106766494783158e-02, -1.644390251809640e-03 ],
                                    "sin":  [ 0.0,0.0,0.0,0.0,0.0 ]}}
        Zaxis = {"type": 'fourier',
                    "input_value": {"cos": [ 0.0,0.0,0.0,0.0 ],
                                    "sin": [ 0.0,0.36,0.02,0.01 ]}}
        # Magnetic field
        B0 = {"type": 'fourier', "input_value": {"cos": [1.0, 0.25], "sin": []}}
        
        # First order field
        d = None
        d_over_curvature = {"type": 'scalar', "input_value": 0.73}

        # Second order shaping : 'minimal' shaping choice
        X2c = {"type": 'scalar', "input_value": 0}
        X2s = {"type": 'scalar', "input_value": 0}
        
        # Buffer region details : use the standard form of [Camacho et al. (2022)]
        omn_buffer = {"omn_method": 'simple-fourier', 
                    "k_buffer": 5}

        # Additional properties
        sigma0 = 0.0
        nfp     = 3
        p2      = 0.0
        order = 'r3'
        omn = True

        vars = ["Raxis", "Zaxis", "B0", "d", "d_over_curvature", "X2c", "X2s", "omn_buffer", "sigma0", "nfp", "p2", "order", "omn"]
        
    elif name == "QI NFP3 Katia opt":
        ##################################
        # 2nd order QI optimised example #
        ##################################
        # Create the equilibrium starting off QI NFP3 Katia smooth, so focus on things to change : i.e. Z and d which were used as degrees of freedom in the optimisation of the configuration

        Zaxis = {"type": 'fourier',
                 "input_value": {"cos": [ 0.0,0.0,0.0,0.0 ],
                                 "sin": [ 0.0,0.39789044636334636,0.02440303995940328,0.0029241265517590473]}}

        d_over_curvature = {"type": "fourier",
                            "input_value": {"cos": [0.6422328799604891,-7.912645531583918e-05,0.0003468186279177299,0.0011403008263701614,0.0036629896518737427,-0.002302780748370314,-9.801584810662561e-05],
                                            "sin": []}}
        # Call smooth QI NFP3 Katia
        stel = make_nae_stellarator("QI NFP3 Katia smooth", Zaxis = Zaxis, d_over_curvature = d_over_curvature, **kwargs)
        
        return stel
        
    elif name == "QI NFP3 half period better":  
        #########################
        # Half helicity example #
        #########################
        ## Read half helicity configuration from JSON : there may be some minor differences due to finite precission in which it was saved ##
        import json 
        file = './vmec_benchmark/half_nae.json'

        # Recursive function to convert strings "None", "True", "False" to Python None, True, False
        def convert_values(data):
            if isinstance(data, dict):
                return {k: convert_values(v) for k, v in data.items()}
            elif isinstance(data, list):
                return np.array([convert_values(item) for item in data])
            elif data == "None":
                return None
            elif data == "True":
                return True
            elif data == "False":
                return False
            else:
                return data
            
        with open(file) as json_file:
            inputs = json.load(json_file)
            inputs = convert_values(inputs)

        vars_kwarg = kwargs.keys()
        for var in vars_kwarg:
            if var in inputs:
                inputs.pop(var)


        # Curvature and torsion as functions of ell : how they were constructed 
        params = [1.99509005, -8.072526812612956, 1.341749970978372, -0.133333048678782]
        curvature_func = lambda x: 0.5*(1+np.cos(x/inputs["L"]*2*np.pi))*np.sin(0.5*x/inputs["L"]*2*np.pi)*np.sin(x/inputs["L"]*2*np.pi)*params[1]
        torsion_func = lambda x: params[2] + params[3]*np.cos(x/inputs["L"]*2*np.pi)

        inputs["curvature"]["function_ell"] = curvature_func
        inputs["torsion"]["function_ell"] = torsion_func

        # Run NAE solve
        stel = Qic(**inputs, **kwargs) 
                  
        return stel

    else: 
        raise NameError('Unrecognised name of nae configuration!')
    
    # Put all input parameters together 
    for var in vars:
        if var in kwargs:
            print("WARNING! The arguments passed overwrite some of the default values used in the construction of the called equilibrium:...." + f"{var}")
        else:
            kwargs[var] = eval(var)

    # Create near-axis stellarator
    stel = Qic(**kwargs)
    return stel 

def read_vmec_inputs(filename):
    """
    Reads a VMEC input file and loads the information into a dictionary.
    Only supports files in the format generated by pyQIC's to_vmec.
    """
    import re

    data = {}
    with open(filename, 'r') as f:
        lines = f.readlines()

    keyval_pattern = re.compile(r'^\s*([A-Z_]+)\s*=\s*([^\n]+)')
    fourier_pattern = re.compile(r'([A-Z]{3})\(([-\d]+),([\d]+)\)\s*=\s*([eE\d\+\-\.]+)')


    for line in lines:
        # Skip comments and empty lines
        if line.strip().startswith('!') or not line.strip():
            continue

        # Parse Fourier boundary coefficients
        m = fourier_pattern.findall(line)
        if m:
            for arr, n, mnum, val in m:
                arr_key = arr
                n = int(n)
                mnum = int(mnum)
                val = float(val)
                if arr_key not in data:
                    data[arr_key] = {}
                data[arr_key][(n, mnum)] = val
            continue

        # Parse key-value pairs
        m = keyval_pattern.match(line)
        if m:
            key, val = m.groups()
            val = val.strip().strip(',')
            # Fortran allows both comma and space separated arrays
            # Replace commas with spaces, then split on whitespace
            val_clean = val.replace(',', ' ')
            arr = []
            for x in val_clean.split():
                try:
                    arr.append(float(x))
                except ValueError:
                    arr.append(x)
            if len(arr) > 1:
                # If all elements are floats, keep as float array
                if all(isinstance(x, float) for x in arr):
                    data[key] = arr
                else:
                    data[key] = arr  # fallback: mixed types, keep as is
            else:
                # Scalar
                sval = arr[0] if arr else ''
                try:
                    if isinstance(sval, float):
                        data[key] = sval
                    elif '.' in sval or 'e' in sval or 'E' in sval:
                        data[key] = float(sval)
                    else:
                        data[key] = int(sval)
                except Exception:
                    data[key] = sval.strip("'")
            continue

    # Check last line in file is '/'
    if lines[-1].strip() != '/':
        raise ValueError("VMEC input file should end with a '/' character.")

    return data

def compare_dict(data_out, data_compare, skip = None, r=None, verbose = True):
    import numpy as np
    for key in data_compare:
        if skip and key in skip:
            continue
        if verbose:
            print(f"Comparing key: {key}")
        if key not in data_out:
            raise KeyError(f"Key {key} not found in output data")
        val_out = data_out[key]
        val_compare = data_compare[key]
        if verbose:
            print(f"Output value: {val_out}")
            print(f"Compare value: {val_compare}")
        # Handle nested dicts (e.g. Fourier coefficients)
        if isinstance(val_out, dict) and isinstance(val_compare, dict):
            for subkey in val_compare:
                if subkey not in val_out:
                    raise KeyError(f"Subkey {subkey} not found in output data for key {key}")
                v_out = val_out[subkey]
                v_cmp = val_compare[subkey]
                if isinstance(v_out, (float, int)) and isinstance(v_cmp, (float, int)):
                    assert np.isclose(v_out, v_cmp, rtol=1e-5, atol=1e-6), f"Mismatch in {key}({subkey}) for r={r}: {v_out} vs {v_cmp}"
                else:
                    assert v_out == v_cmp, f"Mismatch in {key}({subkey}) for r={r}: {v_out} vs {v_cmp}"
        # Handle arrays
        elif isinstance(val_out, (list, np.ndarray)) and isinstance(val_compare, (list, np.ndarray)):
            assert np.allclose(val_out, val_compare, rtol=1e-5, atol=1e-6), f"Mismatch in {key} for r={r}: {val_out} vs {val_compare}"
        # Handle scalars
        elif isinstance(val_out, (float, int)) and isinstance(val_compare, (float, int)):
            assert np.isclose(val_out, val_compare, rtol=1e-5, atol=1e-6), f"Mismatch in {key} for r={r}: {val_out} vs {val_compare}"
        # Fallback to direct equality for strings/other
        else:
            assert val_out == val_compare, f"Mismatch in {key} for r={r}: {val_out} vs {val_compare}"

def test_read_vmec_inputs():
    # Read the input file using the function
    data = read_vmec_inputs('./vmec_benchmark/QI_NFP2_Katia_smooth_buffer/A_0/input.temp')

    # Print the results
    for key, value in data.items():
        print(f"{key}: {value}")

    
if __name__ == "__main__":
    test_to_vmec_benchmark()