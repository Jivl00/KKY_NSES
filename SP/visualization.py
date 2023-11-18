import seaborn as sns
import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt

cmap = 'rainbow' # color map


def plot_data(data, title='Data'):
    """
    Plots the data as a scatter plot.

    :param data: numpy array of shape (m, 3)
    :param title: title of the plot
    :return: None
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], cmap=cmap, c=(data[:, 2]), edgecolors='k')
    plt.title(title)
    plt.grid('on')
    plt.show()


def plot_decision_boundary(data, nn, title='Decision Boundary'):
    """
    Plots the decision boundary of the neural network nn.

    :param data: numpy array of shape (m, 3)
    :param nn: neural network object, must have a predict method
    :param title: title of the plot
    :return: None
    """
    sns.set_style('whitegrid')
    dt = data[:, :-1]
    x_min, x_max = dt[:, 0].min() - 1, dt[:, 0].max() + 1
    y_min, y_max = dt[:, 1].min() - 1, dt[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    inp = np.c_[xx.ravel(), yy.ravel()].T
    boundary = nn.predict(inp)
    # Put the result into a color plot
    boundary = np.array(boundary)
    boundary = boundary.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, boundary, alpha=0.4, cmap=cmap)
    plt.scatter(data[:, 0], data[:, 1], cmap=cmap, c=(data[:, 2]), edgecolors='k')
    plt.title(title)
    plt.show()

def plot_confusion_matrix(cm):
    """
    Plots the confusion matrix as a heatmap.

    :param cm: confusion matrix
    :return: None
    """
    sns.set_style('whitegrid')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap=cmap, alpha=0.4, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def live_plot(cost, figsize=(7,5), title=''):
    """
    Plots the cost as a function of epochs. The plot evolves as the function is called.

    :param cost: list of costs
    :param figsize: size of the figure
    :param title: title of the plot
    :return: None
    """
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    plt.plot(cost, label='cost')
    plt.title(title)
    plt.grid(True)
    plt.xlabel('epoch')
    plt.legend(loc='center left') # the plot evolves to the right
    plt.show()