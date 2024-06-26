import numpy as np
from qic.spectral_diff_matrix import spectral_diff_matrix, spectral_diff_matrix_extended, finite_difference_matrix
# Example usage

N = 20  # Number of grid points
x = np.linspace(0, 2*np.pi, N, endpoint=False)

# Create a spectral differentiation matrix
Dspec = spectral_diff_matrix(N)

# Create a spectral differentiation matrix for half period functions
Dspec_ext = spectral_diff_matrix_extended(N)

# Create a second-order differentiation matrix
order = 2
D2 = finite_difference_matrix(N, order=2)
D2[:order,-order:] = -D2[:order,-order:]
D2[-order:,:order] = -D2[-order:,:order]

# Create a fourth-order differentiation matrix
order = 4
D4 = finite_difference_matrix(N, order=4)
D4[:order,-order:] = -D4[:order,-order:]
D4[-order:,:order] = -D4[-order:,:order]

# Create a sixth-order differentiation matrix
order = 6
D6 = finite_difference_matrix(N, order=6)
D6[:order,-order:] = -D6[:order,-order:]
D6[-order:,:order] = -D6[-order:,:order]

# Example function
f = np.cos(x/2)*np.cos(x)

# Apply the differentiation matrices
f_prime_spec = np.dot(Dspec, f)
f_prime_spec_ext = np.dot(Dspec_ext, f)
f_prime_2nd = np.dot(D2, f)
f_prime_4th = np.dot(D4, f)
f_prime_6th = np.dot(D6, f)

# Exact derivative
f_prime_exact = -0.5*np.sin(x/2)*np.cos(x)-np.cos(x/2)*np.sin(x)

# Plotting the results for comparison
import matplotlib.pyplot as plt

# plt.plot(x, f_prime_spec-f_prime_exact, label='Spectral Derivative')
plt.plot(x, f_prime_spec_ext-f_prime_exact, label='Spectral Ext Derivative')
plt.plot(x, f_prime_2nd-f_prime_exact, label='2nd Order Derivative')
plt.plot(x, f_prime_4th-f_prime_exact, label='4th Order Derivative')
plt.plot(x, f_prime_6th-f_prime_exact, label='6th Order Derivative')
plt.legend()
plt.title('Error in Finite Difference Derivatives for Different Orders')
plt.show()

