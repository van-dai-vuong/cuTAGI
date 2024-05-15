# CDF activation function
import numpy as np
from math import *
class cdf_activate(object):
    def __init__(self, x, Sx):
        self.x = x
        self.Sx = Sx

    def get_jacobian(self, index):
        # Compute the jacobian of the index-th unit
        return self.norm_pdf(self.x[index], 0, 1)

    def activate(self, activate_index):
        # Get jacobian
        jacobian = self.get_jacobian(activate_index)

        # Compute moments
        self.activated_unit_mu = self.norm_cdf(self.x[activate_index],0,1)
        self.activated_unit_var = jacobian**2 * self.Sx[activate_index][activate_index]
        self.activated_unit_cov = jacobian * self.Sx[activate_index]

        # Augment the activated unit to the gaussian_variable
        activated_gaussian_variable = {'mu': np.array([]), 'var': np.array([])}
        activated_gaussian_variable['mu'] = np.vstack((self.x, self.activated_unit_mu))
        activated_gaussian_variable['var'] = self.block_diag(self.Sx, self.activated_unit_var)
        # Fill the n*1 cov in the last row and column (excluding the (n+1,n+1) element) of the (n+1)*(n+1) var
        activated_gaussian_variable['var'][-1][:-1] = self.activated_unit_cov
        activated_gaussian_variable['var'][:-1][:, -1:] = self.activated_unit_cov.reshape(-1,1)

        # Update the gaussian_variable
        self.x = activated_gaussian_variable['mu']
        self.Sx = activated_gaussian_variable['var']

        # Swap the activated unit from the last index to the activate_index
        self.swap_elements(activate_index-1, -1)

        # Remove the last element
        self.remove_element(-1)

        return self.x, self.Sx

    def norm_cdf(self, x, mu, sigma):
        return 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))

    def norm_pdf(self, x, mu, sigma):
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def block_diag(self, block1, block2):
        out = np.zeros((block1.shape[0] + 1, block1.shape[1] + 1))
        # Determine the shape of the blocks
        block1_shape = block1.shape
        block2_shape = block2.shape
        # Fill the block diagonal matrix
        out[:block1_shape[0], :block1_shape[1]] = block1
        out[block1_shape[0], block1_shape[1]] = block2
        return out

    def remove_element(self, idx):
        # Remove element idx from the variable
        self.x = np.delete(self.x, idx, axis=0)
        self.Sx = np.delete(self.Sx, idx, axis=0)
        self.Sx = np.delete(self.Sx, idx, axis=1)
        pass

    def swap_elements(self, idx1, idx2):
        # Swap elements idx1 and idx2
        self.x[[idx1, idx2]] = self.x[[idx2, idx1]]
        self.Sx[[idx1, idx2]] = self.Sx[[idx2, idx1]]
        self.Sx[:, [idx1, idx2]] = self.Sx[:, [idx2, idx1]]
        pass