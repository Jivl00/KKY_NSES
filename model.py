import numpy as np

class NeuralNetwork:
    def __init__(self, x_size, num_classes, hidden_layers):
        self.cost_history = []
        self.theta_history = []
        self.input_size = x_size # velikost vstupni vrstvy
        self.output_size = num_classes # velikost vystupni vrstvy
        self.hidden_layers = hidden_layers # pole poctu neuronu v jednotlivych skrytych vrstvach

        # inicializace vah a biasu
        self.thetas = [] # matice vah + jeden vektor biasu pro kazdou vrstvu
        self.thetas.append(np.random.randn(self.input_size, self.hidden_layers[0]))
        for i in range(len(self.hidden_layers) - 1):
            self.thetas.append(np.random.randn(self.hidden_layers[i], self.hidden_layers[i + 1]))
        self.thetas.append(np.random.randn(self.hidden_layers[-1], self.output_size))
    def predict(self, X, activation_functions):
        """
        Predicts the output for given input X using the current weights and biases of the neural network.

        Parameters:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).
        activation_functions (list): List of activation functions for each layer.

        Returns:
        numpy.ndarray: Predicted output of shape (n_samples, n_classes).
        """
        # forward propagation
        A = X
        for i in range(len(self.thetas)):
            Z = np.dot(A, self.thetas[i])
            A = activation_functions[i](Z)

        return A
        
    def loss(self, X, y, activation_functions):
        """
        Computes the loss function of a neural network.

        Parameters:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).
        y (numpy.ndarray): Expected outputs of shape (n_samples, n_classes).
        activation_functions (list): List of activation functions for each layer.

        Returns:
        float: Loss value.
        """
        # TODO categorical cross entropy?
        return np.dot((self.predict(X, activation_functions) - y).T, (self.predict(X, activation_functions) - y)) / (2 * len(y))

    def grad(self, X, y, activation_functions):
        """
        Backpropagation algorithm except for the weights update.

        Parameters:
        X (numpy.ndarray): Input data of shape (n_samples, n_features).
        y (numpy.ndarray): Expected outputs of shape (n_samples, n_classes).
        activation_functions (list): List of activation functions for each layer.

        Returns:
        list: List of gradients for each layer.
        """
        # The first activation is a special case, it's just the input feature vector itself
        A = [X]
        # Forward pass - loop over the layers of the network
        for i in range(len(self.thetas)):
            A.append(activation_functions[i](np.dot(A[i], self.thetas[i])))

        # Backward pass
        # The last layer is a special case
        error = [A[-1] - y]
        D = [error * activation_functions[-1](A[-1], derivative=True)]

        # Chain Rule
        for layer in np.arange(len(A)-2, 0, -1):
            delta = np.dot(D[-1], self.thetas[layer].T)
            delta *= activation_functions[layer](A[layer], derivative=True)
            D.append(delta)

        # Reverse the deltas
        D.reverse()

        return D, A


    def update(self, thetas, cost):
        """
        Updates the weights and biases of the neural network.

        Parameters:
        theta (list): List of parameters theta.
        cost (float): Cost value.
        """
        self.thetas = thetas
        self.theta_history.append(np.copy(self.thetas))
        self.cost_history.append(cost)


def sigmoid(Z, derivative=False):
    """
    Computes the sigmoid function.

    Parameters:
    Z (numpy.ndarray): Input data.

    Returns:
    numpy.ndarray: Output data.
    """
    if derivative:
        return sigmoid(Z) * (1 - sigmoid(Z))
    return 1 / (1 + np.exp(-Z))

def relu(Z, derivative=False):
    """
    Computes the relu function.

    Parameters:
    Z (numpy.ndarray): Input data.

    Returns:
    numpy.ndarray: Output data.
    """
    if derivative:
        return np.where(Z > 0, 1, 0)
    return np.maximum(0, Z)




