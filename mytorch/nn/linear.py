import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        """
        Initialize the weights and biases with zeros
        W shape: (out_features, in_features)
        b shape: (out_features,)  # Changed from (out_features, 1) to match PyTorch
        """
        # DO NOT MODIFY
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros(out_features)


    def init_weights(self, W, b):
        """
        Initialize the weights and biases with the given values.
        """
        # DO NOT MODIFY
        self.W = W
        self.b = b

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (*, in_features)
        :return: Output Z with shape (*, out_features)
        
        Handles arbitrary batch dimensions like PyTorch
        """
        # TODO: Implement forward pass
        
        # Store input for backward pass
        self.A = A

        input_shape = A.shape
        output_dim = self.W.shape[0]
        new_input_shape = 1
        for d in input_shape[:-1]:
            new_input_shape *= d
        A = A.reshape(new_input_shape, input_shape[-1])

        Z = np.matmul(A, self.W.transpose()) + self.b

        output_shape = input_shape[:-1] + (output_dim,)

        Z = Z.reshape(output_shape)
        
        # raise NotImplementedError
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (*, out_features)
        :return: Gradient of loss wrt input A (*, in_features)
        """
        # TODO: Implement backward pass
        A_shape = self.A.shape
        dLdZ_shape = dLdZ.shape
        new_A_shape, new_dldz_shape = 1, 1
        for d in A_shape[:-1]:
            new_A_shape *= d
        for d in dLdZ_shape[:-1]:
            new_dldz_shape *= d

        A_reshaped = self.A.reshape(new_A_shape, A_shape[-1])
        dLdZ_reshaped = dLdZ.reshape(new_dldz_shape, dLdZ.shape[-1])

        dldA = np.matmul(dLdZ_reshaped, self.W)
        dldA = dldA.reshape(A_shape)

        dldW = np.matmul(dLdZ_reshaped.transpose(), A_reshaped)
        dldW = dldW.reshape(self.W.shape)

        dldb = np.sum(dLdZ_reshaped, axis=0)

        # Compute gradients (refer to the equations in the writeup)
        self.dLdA = dldA
        self.dLdW = dldW
        self.dLdb = dldb
        # self.dLdA = NotImplementedError
        
        # Return gradient of loss wrt input
        # raise NotImplementedError
        return dldA
