import ufl


class GrowthTensor:
    """Growth tensor for transversely isotropic growth."""

    def __init__(self, theta, eta, a0):
        """Initialize growth tensor for transversely isotropic growth.

        Args:
            theta (float): Growth rate in the direction orthogonal to the fibers
            eta (float): Growth rate in the direction of the fibers
            a0 (ndarray or Function): Fiber direction
        """
        self.theta = theta
        self.eta = eta
        self.a0 = a0

    def tensor(self):
        """Construct the growth tensor"""
        dim = len(self.a0)
        return self.theta * ufl.Identity(dim) + (self.eta - self.theta) * ufl.outer(
            self.a0, self.a0
        )

    def inverse(self):
        """Construct the inverse of the growth tensor"""
        dim = len(self.a0)
        # For this simple model, the inverse is known analytically
        return 1 / self.theta * ufl.Identity(dim) + (
            1 / self.eta - 1 / self.theta
        ) * ufl.outer(self.a0, self.a0)
