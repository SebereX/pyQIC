import numpy as np
from qic.spectral_diff_matrix import spectral_diff_matrix, finite_difference_matrix
# Example usage

N = 1280  # Number of grid points
x = np.linspace(0, 2*np.pi, N, endpoint=False)

# Create a spectral differentiation matrix
Dspec = spectral_diff_matrix(N)

# Create a second-order differentiation matrix
D2 = finite_difference_matrix(N, order=2)

# Create a fourth-order differentiation matrix
D4 = finite_difference_matrix(N, order=4)

# Create a sixth-order differentiation matrix
D6 = finite_difference_matrix(N, order=6)

# Example function
f = np.cos(x)

# Apply the differentiation matrices
f_prime_spec = np.dot(Dspec, f)
f_prime_2nd = np.dot(D2, f)
f_prime_4th = np.dot(D4, f)
f_prime_6th = np.dot(D6, f)

# Exact derivative
f_prime_exact = -np.sin(x)

# Plotting the results for comparison
import matplotlib.pyplot as plt

plt.plot(x, f_prime_spec-f_prime_exact, label='Spectral Derivative')
plt.plot(x, f_prime_2nd-f_prime_exact, label='2nd Order Derivative')
plt.plot(x, f_prime_4th-f_prime_exact, label='4th Order Derivative')
plt.plot(x, f_prime_6th-f_prime_exact, label='6th Order Derivative')
plt.legend()
plt.title('Error in Finite Difference Derivatives for Different Orders')
plt.show()

