import numpy as np

class Optimizer:

    def __init__(self, model):
        self.model = model
        self.iter = 0

    def step(self, X, y, activation_functions):
        """
        Performs a single step of the gradient descent
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        raise NotImplementedError("Method not yet implemented.")

    def converged(self):
        """

        :return: True if the gradient descent iteration ended
        """
        raise NotImplementedError("Method not yet implemented.")

    def optimize_full_batch(self, X, y, activation_functions):
        """
        Runs the optimization processing all the data at each step
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        while not self.converged():
            self.step(X, y, activation_functions)
            self.iter += 1


class GradientDescent(Optimizer):

    def __init__(self, model, alpha=0.0005, num_iters=1000, min_cost=0, min_theta_diff=0, **options):
        super(GradientDescent, self).__init__(model)
        self.options = options
        self.alpha = alpha
        self.num_iters = num_iters
        self.min_cost = min_cost
        self.min_theta_diff = min_theta_diff
        self.cost = np.Inf

    def step(self, X, y, activation_functions):
        """
        Performs a single step of the gradient descent
        :param X: input data as row vectors
        :param y: vector of the expected outputs
        :return:
        """
        grad, A = self.model.grad(X, y, activation_functions)
        self.cost = self.model.loss(X, y, activation_functions)
        # Weight update
        thetas = np.copy(self.model.thetas)
        for layer in np.arange(0, len(self.model.thetas)):
            thetas[layer] = self.model.thetas[layer] - self.alpha * np.dot(A[layer].T, grad[layer])
        self.model.update(thetas, self.cost)

    def converged(self):
        """

        :return: True if the gradient descent iteration ended
        """
        # num of iterations
        if self.iter >= self.num_iters:
            return True
        # # minimal cost
        # if len(self.model.cost_history) > 0:
        #     if self.model.cost_history[-1] <= self.min_cost:
        #         return True
        # minimal difference between thetas - euclidean distance
        if len(self.model.theta_history) > 1:
            if np.linalg.norm(self.model.theta_history[-1] - self.model.theta_history[-2]) <= self.min_theta_diff:
                return True
        return False

