import numpy as np
from visualization import live_plot

# TODO early stopping

class DNNClassifier(object):

    def __init__(self, layers_dims, activations):
        assert len(layers_dims) == len(activations) + 1, "Number of layers must be equal to the number of activations + 1."
        self.layers_dims = layers_dims
        self.num_layers = len(layers_dims)
        self.activations = activations
        self.parameters = {}
        self.cost_history = []
        self.caches = []

        np.random.seed(1)
        for l in range(1, self.num_layers):
            self.parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) / np.sqrt(
                layers_dims[l - 1])
            self.parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

    def forward_propagation(self, X):
        A = X
        self.caches = []
        for l in range(1, self.num_layers):
            A_prev = A
            W = self.parameters['W' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = W.dot(A_prev) + b
            A = self.activations[l - 1](Z)
            cache = ((A_prev, W, b), Z)
            self.caches.append(cache)

        return A

    def cost(self, AL, Y):  # TODO: MSE

        if self.activations[-1] == DNNClassifier.softmax:
            # categorical cross-entropy
            return -np.mean(np.sum(Y * np.log(AL), axis=0))
        else:
            # binary cross-entropy
            return np.mean(np.sum(-Y * np.log(AL) - (1 - Y) * np.log(1 - AL), axis=0))

    def backward_propagation(self, AL, Y):
        grads = {}
        L = self.num_layers - 1  # number of layers

        # Initialize the backpropagation
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
        for l in range(self.num_layers - 1):
            self.parameters["W" + str(l + 1)] -= learning_rate * grads["dW" + str(l + 1)]
            self.parameters["b" + str(l + 1)] -= learning_rate * grads["db" + str(l + 1)]

    def train(self, X, Y, learning_rate=0.01, epochs=100, batch_size=1, print_cost=False, plot_cost=True):
        print("Training model...")
        self.cost_history = []
        assert epochs > 0, "Epochs must be greater than 0."
        assert batch_size > 0, "Batch size must be greater than 0."
        assert learning_rate > 0, "Learning rate must be greater than 0."
        assert X.shape[1] == Y.shape[1], "Number of examples must be equal in X and Y."
        assert X.shape[0] == self.layers_dims[0], "Input shape must be equal to the number of features."
        assert Y.shape[0] == self.layers_dims[-1], "Output shape must be equal to the number of classes."
        for i in range(0, epochs):
            permutation = list(np.random.permutation(X.shape[1]))
            X = X[:, permutation]
            Y = Y[:, permutation]
            epoch_cost = []
            for batch in range(0, X.shape[1], batch_size):
                X_batch = X[:, batch:batch + batch_size]
                Y_batch = Y[:, batch:batch + batch_size]
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
        assert X.shape[0] == self.layers_dims[0], "Input shape must be equal to the number of features."

        # Forward propagation
        preds = self.forward_propagation(X)
        y_pred = np.argmax(preds, axis=0) + 1
        return y_pred

    def evaluate(self, X, Y, confusion_matrix=False):
        assert X.shape[1] == Y.shape[1], "Number of examples must be equal in X and Y."
        assert X.shape[0] == self.layers_dims[0], "Input shape must be equal to the number of features."

        y_pred = self.predict(X) - 1
        Y = np.array(Y, dtype=int)[0] - 1
        accuracy = np.sum(y_pred == Y) / Y.shape[0]
        if confusion_matrix:
            K = len(np.unique(Y))  # Number of classes
            confusion_matrix = np.bincount(Y * K + y_pred).reshape((K, K))

            return accuracy, confusion_matrix
        return accuracy

    @staticmethod
    def sigmoid(Z, derivative=False):
        if derivative:
            return DNNClassifier.sigmoid(Z) * (1 - DNNClassifier.sigmoid(Z))
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def sgn(Z, derivative=False):
        if derivative:
            return 1
        return np.sign(Z)

    @staticmethod
    def linear(Z, derivative=False):
        if derivative:
            return 1
        return Z

    @staticmethod
    def step(Z, derivative=False):
        if derivative:
            return 0
        return np.where(Z > 0, 1, 0)

    @staticmethod
    def relu(Z, derivative=False):
        if derivative:
            return np.where(Z > 0, 1, 0)
        return np.maximum(0, Z)

    @staticmethod
    def tanh(Z, derivative=False):
        if derivative:
            return 1 - np.tanh(Z)**2
        return np.tanh(Z)

    @staticmethod
    def softmax(Z, derivative=False):
        if derivative:
            raise NotImplementedError("Softmax is only supported in the output layer.")
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)


class OneVsAllClassifier(object):
    def __init__(self, layers_dims, activations, num_classes):
        self.layers_dims = layers_dims
        self.activations = activations
        self.num_classes = num_classes
        self.classifiers = []

        for i in range(num_classes):
            self.classifiers.append(DNNClassifier(layers_dims, activations))

    def train(self, X, Y, learning_rate=0.01, epochs=100, batch_size=1, print_cost=False):
        '''

        :param X:
        :param Y: raw data, not one-hot encoded
        :param learning_rate:
        :param epochs:
        :param batch_size:
        :param print_cost:
        :return:
        '''
        for i in range(self.num_classes):
            print("Training classifier {}...".format(i))
            y_train_binary = np.array([1 if Y[j] == i+1 else 0 for j in range(len(Y))])
            y_train_binary = y_train_binary.reshape(1, y_train_binary.shape[0])
            self.classifiers[i].train(X, y_train_binary, learning_rate, epochs, batch_size, print_cost=print_cost,plot_cost=False)

    def predict(self, X):
        predictions = [None] * self.num_classes
        for i in range(self.num_classes):
            predictions[i] = self.classifiers[i].forward_propagation(X)[0]

        return [np.argmax([predictions[j][i] for j in range(self.num_classes)])+1 for i in range(len(predictions[0]))]

    def evaluate(self, X, Y, confusion_matrix=False):
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
