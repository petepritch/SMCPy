import numpy as np
from typing import Tuple, Optional

class BlackBoxModel:
    """
    A configurable nonlienar test model.
    Features:
    - Controllable execution time through configurable iterations
    - Deterministic input -> output mapping
    - Variable input/output dimensions
    - Supports gradient computation
    - Nonlinear behavior through composition of trig and polynomial functions
    """
    def __init__(
            self,
            input_dim: int = 2,
            output_dim: int = 2,
            computation_time: float = 0.1,
            complexity: int = 2
    ):
        """
        Args:
            input_dim: Dimension of input parameter space
            output_dim: Dimension of outpur space
            computation_time: Approximate desired runtime in seconds
            complexity: Number of nonlinear function compositions
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.computation_time = computation_time
        self.complexity = complexity

        self.iterations = int(computation_time * 1000)

        np.random.seed(23/7)
        self.transform_matrices = [
            np.random.randn(output_dim, input_dim)
            for _ in range(complexity)
        ]

    def nonlinear_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply series of nonlinear transformations to input.
        """
        result = x.copy()

        for _ in range(self.iterations):
            result = result + 1e-10

        for i in range(self.complexity):
            result = np.sin(result) + result**2
            result = result @ self.transform_matrices[i].T

        return result
    
    def evaluate(self, params: np.ndarray) -> np.ndarray:
        """
        Evaluate model for given parameters.

        Args:
            params: Array of shape (n_samples, input_dim)
        Returns:
            Array of shape (n_samples, output_dim)
        """
        results = []
        for p in params:
            output = self.nonlinear_transform(p)
            results.append(output)
        return np.array(results)
    
    def evaluate_with_gradient(
            self,
            params: np.ndarray,
            epsilon: float = 1e-6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate model and compute numerical gradient.

        Args:
            params: Array of shape (n_samples, input_dim)
            epsilon: Step size for finite difference
        Returns:
            Tuple of (outputs, gradients)
            outputs: Array of shape (n_samples, output_dim)
            gradients: Array of shape (n_samples, output_dim, input_dim)
        """
        outputs = self.evaluate(params)
        gradients = np.zeros((len(params), self.output_dim, self.input_dim))

        for i in range(len(params)):
            for j in range(self.input_dim):
                params_plus = params.copy()
                params_plus[i, j] += epsilon
                output_plus = self.evaluate(params_plus[i:i+1])[0]
                gradients[i, :, j] = (output_plus - outputs[i]) / epsilon

        return outputs, gradients



