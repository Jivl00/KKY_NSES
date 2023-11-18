import numpy as np
from visualization import live_plot


# TODO early stopping

class DNNClassifier(object):

    def __init__(self, layers_dims, activations):
        """

        :param layers_dims: array of the dimensions of each layer in the network (including the input layer)
        :param activations: array of the activation functions for each layer (excluding the input layer),
                            each activation function is a static method of the class
                            supported activation functions: sigmoid, relu, tanh, softmax, linear, step, sgn
                            Note: softmax is only supported in the output layer
        """
        assert len(layers_dims) == len(
            activations) + 1, "Number of layers must be equal to the number of activations + 1."
        self.layers_dims = layers_dims  # numbers of units in each layer of the network
        self.num_layers = len(layers_dims)  # number of layers in the network
        self.activations = activations  # activation functions for each layer
        self.parameters = {}  # parameters of the model - weights and biases
        self.cost_history = []  # history of the cost function
        self.caches = []  # cache of the linear and activation values for each layer

        np.random.seed(1)
        # Initialize parameters - weights and biases
        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(
                layers_dims[l - 1])
            self.parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    def forward_propagation(self, X):
        """
        Forward propagation AL = g(ZL) = g(WL * g(WL-1 * ... * g(W1 * X + b1) + bL-1) + bL)
        :param X: input data - shape (input size, number of examples)
        :return: AL - last post-activation value (output of the model)
        """
        A = X  # A0 = X
        self.caches = []
        for l in range(1, self.num_layers):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = W.dot(A_prev) + b  # Z = W * A_prev + b
            A = self.activations[l - 1](Z)  # A = g(Z)
            cache = ((A_prev, W, b), Z)
            self.caches.append(cache)

        return A

    def cost(self, AL, Y):  # TODO: MSE
        """
        Compute the cost function
        :param AL: output of the model
        :param Y: true labels vector - shape (output size, number of examples)
        :return: cost
        """

        if self.activations[-1] == DNNClassifier.softmax:
            # categorical cross-entropy
            return -np.mean(np.sum(Y * np.log(AL), axis=0))
        else:
            # binary cross-entropy
            return np.mean(np.sum(-Y * np.log(AL) - (1 - Y) * np.log(1 - AL), axis=0))

    def backward_propagation(self, AL, Y):
        """
        Backward propagation
        :param AL: output of the model
        :param Y: true labels vector - shape (output size, number of examples)
        :return: grads - dictionary with the gradients
        """
        grads = {}
        L = self.num_layers - 1  # number of layers

        # Initialize the backpropagation
        # for Softmax the derivative of the cost function is dL/dZ = AL - Y
        if self.activations[-1] == DNNClassifier.softmax:
            dA = AL - Y
        else:
            dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # TODO: MSE

        for l in reversed(range(L)):
            current_cache = self.caches[l]
            linear_cache, Z = current_cache

            A_prev, W, b = linear_cache
            m = A_prev.shape[1]

            if self.activations[-1] == DNNClassifier.softmax:
                dZ = dA
            else:
                dZ = self.activations[l](Z, derivative=True) * dA  # dL/dZ = dL/dA * dA/dZ

            # TODO: MSE
            dW = 1. / m * np.dot(dZ, A_prev.T)  # dL/dW = dL/dA * dA/dZ * dZ/dW
            db = 1. / m * np.sum(dZ, axis=1, keepdims=True)  # dL/db = dL/dA * dA/dZ * dZ/db
            dA_prev = np.dot(W.T, dZ)  # dL/dA_prev = dL/dA * dA/dZ * (dZ/dA_prev = W)

            grads["dA" + str(l)] = dA_prev
            grads["dW" + str(l + 1)] = dW
            grads["db" + str(l + 1)] = db

            dA = grads["dA" + str(l)]

        return grads

    def update_parameters(self, grads, learning_rate):
        """
        Update parameters using gradient descent
        :param grads: dictionary with the gradients
        """
        for l in range(self.num_layers - 1):
            self.parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    def train(self, X, Y, learning_rate=0.01, epochs=100, batch_size=1, print_cost=False, plot_cost=True):
        """
        Train the model
        :param X: input data - shape (input size, number of examples)
        :param Y: true labels vector - shape (output size, number of examples) - one-hot encoded
        :param learning_rate: learning rate of the gradient descent update rule
        :param epochs: number of epochs of the optimization loop
        :param batch_size: size of a mini batch
        :param print_cost: whether to print the cost every epoch
        :param plot_cost: whether to plot the cost every epoch - live plot
        :return: cost_history - list of costs per epoch
        """
        print("Training model...")
        self.cost_history = []
        assert epochs > 0, "Epochs must be greater than 0."
        assert batch_size > 0, "Batch size must be greater than 0."
        assert learning_rate > 0, "Learning rate must be greater than 0."
        assert X.shape[1] == Y.shape[1], "Number of examples must be equal in X and Y."
        assert X.shape[0] == self.layers_dims[0], "Input shape must be equal to the number of features."
        assert Y.shape[0] == self.layers_dims[-1], "Output shape must be equal to the number of classes."

        for i in range(0, epochs):
            permutation = list(np.random.permutation(X.shape[1]))  # shuffle the examples
            X = X[:, permutation]
            Y = Y[:, permutation]
            epoch_cost = []
            for batch in range(0, X.shape[1], batch_size):
                X_batch = X[:, batch:batch + batch_size]  # get a mini batch
                Y_batch = Y[:, batch:batch + batch_size]  # get a mini batch
                AL = self.forward_propagation(X_batch)
                cost = self.cost(AL, Y_batch)
                epoch_cost.append(cost)
                grads = self.backward_propagation(AL, Y_batch)
                self.update_parameters(grads, learning_rate)
            self.cost_history.append(np.mean(epoch_cost))
            if plot_cost:
                live_plot(self.cost_history, title='Cost')
            if print_cost:
                print("Cost after epoch {}: {}".format(i, np.squeeze(cost)))

        return self.cost_history

    def predict(self, X):
        """
        Make predictions using the trained model
        :return: predictions vector - shape (1, number of examples)
        """
        assert X.shape[0] == self.layers_dims[0], "Input shape must be equal to the number of features."

        # Forward propagation
        preds = self.forward_propagation(X)
        y_pred = np.argmax(preds,
                           axis=0) + 1  # get the index of the max value in each column, +1 to get the class (indexing starts from 0)
        return y_pred

    def evaluate(self, X, Y, confusion_matrix=False):
        """
        Evaluate the model
        :param X: input data - shape (input size, number of examples)
        :param Y: true labels vector - shape (1, number of examples) - NOT one-hot encoded
        :param confusion_matrix: whether to return the confusion matrix
        :return: accuracy, confusion_matrix
        """
        assert X.shape[1] == Y.shape[1], "Number of examples must be equal in X and Y."
        assert X.shape[0] == self.layers_dims[0], "Input shape must be equal to the number of features."

        y_pred = self.predict(X) - 1  # -1 to get the index of the class (indexing starts from 0)
        Y = np.array(Y, dtype=int)[0] - 1  # -1 to get the index of the class (indexing starts from 0)
        accuracy = np.sum(y_pred == Y) / Y.shape[0]  # number of correct predictions / number of examples
        if confusion_matrix:
            K = len(np.unique(Y))  # Number of classes
            confusion_matrix = np.bincount(Y * K + y_pred).reshape((K, K))  # confusion matrix

            return accuracy, confusion_matrix
        return accuracy

    @staticmethod
    def sigmoid(Z, derivative=False):
        """
        Sigmoid activation function
        :param Z: input - can be a scalar or an array
        :param derivative: whether to return the derivative of the activation function
        :return: sigmoid(Z) or sigmoid'(Z)
        """
        if derivative:
            return DNNClassifier.sigmoid(Z) * (1 - DNNClassifier.sigmoid(Z))
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sgn(Z, derivative=False):
        """
        Sign activation function
        :param Z: input - can be a scalar or an array
        :param derivative: whether to return the derivative of the activation function
        :return: sign(Z) or 1
        """
        if derivative:
            return 1
        return np.sign(Z)

    @staticmethod
    def linear(Z, derivative=False):
        """
        Linear activation function
        :param Z: input - can be a scalar or an array
        :param derivative: whether to return the derivative of the activation function
        :return: Z or 1
        """
        if derivative:
            return 1
        return Z

    @staticmethod
    def step(Z, derivative=False):
        """
        Step activation function
        :param Z: input - can be a scalar or an array
        :param derivative: whether to return the derivative of the activation function
        :return: 1 if Z > 0, 0 otherwise
        """
        if derivative:
            return 0
        return np.where(Z > 0, 1, 0)

    @staticmethod
    def relu(Z, derivative=False):
        """
        ReLU activation function
        :param Z: input - can be a scalar or an array
        :param derivative: whether to return the derivative of the activation function
        :return: relu(Z) or relu'(Z)
        """
        if derivative:
            return np.where(Z > 0, 1, 0)
        return np.maximum(0, Z)

    @staticmethod
    def tanh(Z, derivative=False):
        """
        Tanh activation function
        :param Z: input - can be a scalar or an array
        :param derivative: whether to return the derivative of the activation function
        :return: tanh(Z) or tanh'(Z)
        """
        if derivative:
            return 1 - np.tanh(Z) ** 2
        return np.tanh(Z)

    @staticmethod
    def softmax(Z, derivative=False):
        """
        Softmax activation function
        :param Z: input - array
        :param derivative: not supported
        :return: softmax(Z)
        """
        if derivative:
            raise NotImplementedError("Softmax is only supported in the output layer.")
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)

    @staticmethod
    def to_one_hot(y, num_classes):
        """
        Transform labels vector to one-hot encoded matrix
        :param y: labels vector - shape (1, number of examples)
        :param num_classes: number of classes
        :return: one-hot encoded matrix - shape (number of classes, number of examples)
        """
        y = y.astype(int)
        one_hot = np.zeros((num_classes, y.shape[0]), dtype=int)
        for i in range(y.shape[0]):
            one_hot[y[i] - 1, i] = 1
        return one_hot


class OneVsAllClassifier(object):
    def __init__(self, layers_dims, activations, num_classes):
        """
        One-vs-all classifier for multi-class classification
        :param layers_dims: array of the dimensions of each layer in the network (including the input layer)
        :param activations: array of the activation functions for each layer (excluding the input layer),
                            each activation function is a static method of the DNNClassifier class
                            supported activation functions: sigmoid, relu, tanh, softmax, linear, step, sgn
                            Note: softmax is only supported in the output layer
        :param num_classes: number of classifiers to train
        """
        self.layers_dims = layers_dims
        self.activations = activations
        self.num_classes = num_classes
        self.classifiers = []

        for i in range(num_classes):
            self.classifiers.append(DNNClassifier(layers_dims, activations))  # create a classifier for each class

    def train(self, X, Y, learning_rate=0.01, epochs=100, batch_size=1, print_cost=False):
        """
        Train the models for each class - one-vs-all, each classifier is trained to predict one class vs. all the others
        :param X: input data - shape (input size, number of examples)
        :param Y: true labels vector - shape (1, number of examples) - NOT one-hot encoded
        :param learning_rate: learning rate of the gradient descent update rule
        :param epochs: number of epochs of the optimization loop
        :param batch_size: size of a mini batch
        :param print_cost: whether to print the cost every epoch
        """
        for i in range(self.num_classes):
            print("Training classifier {}...".format(i))
            y_train_binary = np.array([1 if Y[j] == i + 1 else 0 for j in range(
                len(Y))])  # transform the labels to binary - 1 if the class is i, 0 otherwise
            y_train_binary = y_train_binary.reshape(1, y_train_binary.shape[0]) # reshape to (1, number of examples)
            self.classifiers[i].train(X, y_train_binary, learning_rate, epochs, batch_size, print_cost=print_cost,
                                      plot_cost=False)

    def predict(self, X):
        """
        Make predictions using the trained models
        :param X: input data - shape (input size, number of examples)
        :return: predictions vector - shape (1, number of examples)
        """
        predictions = [None] * self.num_classes
        for i in range(self.num_classes):
            predictions[i] = self.classifiers[i].forward_propagation(X)[0]

        return [np.argmax([predictions[j][i] for j in range(self.num_classes)]) + 1 for i in range(len(predictions[0]))]
        # get the index of the max value in each column, +1 to get the class (indexing starts from 0)

    def evaluate(self, X, Y, confusion_matrix=False):
        """
        Evaluate the models - one-vs-all
        :param X: input data - shape (input size, number of examples)
        :param Y: true labels vector - shape (1, number of examples) - NOT one-hot encoded
        :param confusion_matrix: whether to return the confusion matrix
        :return: accuracy, confusion_matrix
        """
        y_pred = np.array(self.predict(X))
        Y = np.array(Y, dtype=int)[0]
        accuracy = np.sum(y_pred == Y) / Y.shape[0]
        if confusion_matrix:
            K = len(np.unique(Y))
            y_pred = y_pred - 1
            Y = Y - 1
            confusion_matrix = np.bincount(Y * K + y_pred).reshape((K, K))
            return accuracy, confusion_matrix
        return accuracy
