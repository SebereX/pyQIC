#!/usr/bin/env python3

"""
Structure of input variables, to distinguish between array of Fourier coefficients, constant value, 
or spline is provided
"""
import logging
import numpy as np
from qic.fourier_interpolation import fourier_interpolation
from scipy.interpolate import CubicSpline as spline
from scipy.interpolate import make_interp_spline
from .util import Struct

logger = logging.getLogger(__name__)

def evaluate_input_on_grid(self, input_quantity, grid, derivative = None, periodic = True):
    """
    Evaluate an input quantity on a grid, depending on its type. The input must be a dictionary with 
     "input_value" and "type" (scalar, fourier array, grid array, spline) as attributes. Evaluation 
     evaluates the inputs on a grid, which becomes "value_grid". If derivative is called, it only returns
     the value but does not store it.
     If grid length is different from the input when grid, if length difference is +-1, then resample with
     a simple cubic interpolation.
    """
    if derivative == None:
        if input_quantity["type"] == 'scalar':
            # The input in this case should look like:
            # input_quantity = {"type": 'scalar', "input_value": 1.52}
            size_grid = len(grid)
            input_quantity["value_grid"] = np.ones(size_grid) * input_quantity["input_value"]
        elif input_quantity["type"] == 'fourier':
            # The input in this case corresponds to Fourier components:
            # input_quantity = {"type": 'fourier', "input_value": {"cos": [1.0, -0.1], "sin": [0.0, 0.05]}}
            Fourier_cos = np.array(input_quantity["input_value"]["cos"])
            Fourier_sin = np.array(input_quantity["input_value"]["sin"])
            input_quantity["value_grid"] = sum([Fourier_cos[j]*np.cos(self.nfp*j*grid) for j in range(len(Fourier_cos))])
            input_quantity["value_grid"] += sum([Fourier_sin[j]*np.sin(self.nfp*j*grid) for j in range(len(Fourier_sin))])

        elif input_quantity["type"] == 'grid':
            # The input in this case corresponds to an array in a uniform:
            # input_quantity = {"type": 'grid', "input_value": [1.0, -0.1]}
            diff_length = len(input_quantity["input_value"]) - len(grid)
            if diff_length:
                # Works well when the grid points are included in the original domain and a difference of 1 in lengths
                # This is meant to deal with the situation when we change nphi to be odd 
                if diff_length == 1:
                    if periodic:
                        # Compute spline
                        sp = fourier_interpolation(input_quantity["input_value"], grid * self.nfp)
                        # Evaluate on new grid
                        input_quantity["value_grid"] = sp
                    else: 
                        # Grid of the array provided
                        x_ref = np.linspace(0, 1, len(input_quantity["input_value"]), endpoint = False) * 2*np.pi/self.nfp
                        # Compute spline
                        sp = make_interp_spline(x_ref, input_quantity["input_value"], k=5)
                        # Evaluate on new grid
                        input_quantity["value_grid"] = sp(grid)    
                else:
                    raise ValueError("The input array does not match the length of the grid for an input_type = 'grid'. It is possible that the input nphi was not odd and it was changed by the code.")
            else:
                input_quantity["value_grid"] = input_quantity["input_value"]

        elif input_quantity["type"] == 'spline':
            # The input in this case corresponds to an array representing a spline:
            # input_quantity = {"type": 'spline', "input_value": [0.0, 0.3, 0.5]}
            N_points = len(input_quantity["input_value"])
            # Equally distributed collocation points in the period
            x_in = np.linspace(0,1,N_points)*np.pi/self.nfp
            # Make periodic spline, so extend domain
            x_in_periodic = np.append(x_in, 2*np.pi/self.nfp-x_in[-2::-1])
            y_in_periodic = np.append(input_quantity["input_value"], input_quantity["input_value"][-2::-1])
            # The order of the spline is important, as we are going to take derivatives with respect to phi
            spline_input = make_interp_spline(x_in_periodic, y_in_periodic, bc_type = 'periodic', k = 7) 
            input_quantity["spline"] = spline_input
            input_quantity["value_grid"] = spline_input(grid)
        elif input_quantity["type"] == 'function':
            # Evaluate the function on the grid
            input_quantity["value_grid"] = input_quantity["function"](grid)
        else:
            raise ValueError('Unrecognised input type!')
        
        return input_quantity["value_grid"]
    elif isinstance(derivative, int) and derivative > 0:
        if input_quantity["type"] == 'scalar':
            size_grid = len(grid)
            der_value = np.zeros(size_grid)
        elif input_quantity["type"] == 'fourier':
            Fourier_cos = np.array(input_quantity["input_value"]["cos"])
            Fourier_sin = np.array(input_quantity["input_value"]["sin"])
            if np.mod(derivative, 2) == 0:
                der_value = sum([Fourier_cos[j]*(self.nfp*j)**derivative*(-1)**(derivative/2)*\
                                                    np.cos(self.nfp*j*grid) for j in range(len(Fourier_cos))])
                der_value += sum([Fourier_sin[j]*(self.nfp*j)**derivative*(-1)**(derivative/2)*\
                                                     np.sin(self.nfp*j*grid) for j in range(len(Fourier_sin))])
            else:
                der_value = sum([Fourier_cos[j]*(self.nfp*j)**derivative*(-1)**((derivative+1)/2)*\
                                                    np.sin(self.nfp*j*grid) for j in range(len(Fourier_cos))])
                der_value += sum([Fourier_sin[j]*(self.nfp*j)**derivative*(-1)**((derivative-1)/2)*\
                                                     np.cos(self.nfp*j*grid) for j in range(len(Fourier_sin))])
        elif input_quantity["type"] == 'spline':
            # Derivative of spline
            spline_der = input_quantity["spline"].derivative(n = derivative)
            der_value = spline_der(grid)
        else:
            raise KeyError('Input is not of the type for derivative evaluation.')
        return der_value
    else:
        raise ValueError('Derivative input provided is not valid.')
                
        
