# GMA utils
import numpy as np

class GMA(object):
    # Gaussian Multiplicative Approximation
    def __init__(self, x, Sx):
        # variable is a dictionary with keys 'mu' and 'var'
        self.x = x
        self.Sx = Sx
        self.results = []

    def multiplicate_elements(self, idx1, idx2):
        # Compute the product of two Gaussian elements of the variable
        mu_idx12 = self.x[idx1] * self.x[idx2] + \
                   self.Sx[idx1][idx2]
        var_idx12 = self.Sx[idx1][idx1] * self.Sx[idx2][idx2] + \
                    self.Sx[idx1][idx2]**2 + \
                    2 * self.x[idx1] * self.x[idx2] * self.Sx[idx1][idx2] + \
                    self.Sx[idx2][idx2] * self.x[idx1]**2 + \
                    self.Sx[idx1][idx1] * self.x[idx2]**2
        new_mu = np.vstack((self.x, mu_idx12))
        new_var = self.block_diag(self.Sx, var_idx12)

        # Compute the covariance
        num_hidden_states = len(self.x)
        for i in range(num_hidden_states):
            cov_i_idx12 = self.Sx[i][idx1] * self.x[idx2] + \
                          self.Sx[i][idx2] * self.x[idx1]
            new_var[i][-1] = cov_i_idx12
            new_var[-1][i] = cov_i_idx12
        self.x = new_mu
        self.Sx = new_var
        pass

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

    def get_results(self):
        return self.x, self.Sx