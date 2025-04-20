import copy

import numpy as np


class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim > len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is greater than the number of dimensions in Z")
        
        # TODO: Implement forward pass
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        self.Z = copy.deepcopy(Z)
        Z_exp = np.exp(Z)
        self.A = Z_exp / np.sum(Z_exp, axis=self.dim, keepdims=True)
        # raise NotImplementedError
        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # TODO: Implement backward pass
        
        # Get the shape of the input
        A_shape = self.A.shape
        # Find the dimension along which softmax was applied
        C = A_shape[self.dim]

        # move the dim to the end
        self.A = np.moveaxis(self.A, self.dim, -1)
        dLdA = np.moveaxis(dLdA, self.dim, -1)
        dLdA_shape = dLdA.shape
           
        # Reshape input to 2D
        if len(A_shape) > 2:
            A_new_shape, dLdA_new_shape = 1, 1
            for d in A_shape[:-1]:
                A_new_shape *= d
            for d in dLdA_shape[:-1]:
                dLdA_new_shape *= d
            self.A = self.A.reshape(A_new_shape, self.A.shape[-1])
            dLdA = dLdA.reshape(dLdA_new_shape, dLdA.shape[-1])

        # operate backward as a 2d array
        N = len(self.A)  # TODO
        C_ = len(self.A[0])  # TODO

        dLdZ = np.zeros(shape=(N, C_))  # TODO

        # Fill dLdZ one data point (row) at a time.
        for i in range(N):
            J = np.zeros(shape=(C_, C_))  # TODO

            for m in range(C_):
                for n in range(C_):
                    p = self.A[i][m] * (1 - self.A[i][m]) if m == n else -self.A[i][m] * self.A[i][n]  # TODO
                    J[m, n] = p

            dLdZ[i, :] = np.matmul(dLdA[i], J)  # TODO

        # Reshape back to original dimensions if necessary
        if len(A_shape) > 2:
            # Restore shapes to original
            self.A = self.A.reshape(A_shape)
            dLdZ = dLdZ.reshape(A_shape)
            self.A = np.moveaxis(self.A, -1, self.dim)
            dLdZ = np.moveaxis(dLdZ, -1, self.dim)

        # raise NotImplementedError
        return dLdZ
 

    