import numpy as np
import matplotlib.pyplot as plt
import reprlib
from time import time


class ANNRegularGDBackProp:

    """
    A Feed-forward Artificial Neural Network with regular
    Batch/Stochastic Gradient Descend optimization by Backpropagation
    """

    def __init__(self, x_shape, y_shape, hidden_layers_dim, random_state=0):
        """
        Creates the desired ANN Architecture
        :param x_shape: input data's shape
        :param y_shape: target's shape (One-Hot encode more multiple classes)
        :param hidden_layers_dim: list containing the number of neurons of each Hidden Layer
        example: [S1-S2-...SM-1] (S0 is det. by x_shape and SM by y_shape)
        :param random_state: random seed
        """

        # Structure
        self.n_inputs = x_shape[1]
        # Accounts for flat array as target
        if len(y_shape) == 1:
            self.n_neurons_output = y_shape[0]
        else:
            self.n_neurons_output = y_shape[1]
        self.hidden_layers_dim = hidden_layers_dim
        structure = [self.n_inputs] + self.hidden_layers_dim + [self.n_neurons_output]
        pattern = "%s-"+"%s-"*len(self.hidden_layers_dim)+"%s"
        self.structure = pattern % tuple(structure)

        # Other attributes
        self.random_seed = random_state
        self.target_shape = y_shape  # To be used on self.predict()
        # We will use these attributes later 
        self.W = None  # We want to initialize the parameters 
        self.b = None  # everytime self.fit() is called
        self.layers_transfer_func = None
        self.perf_index = None
        # MSE at each epoch
        self.perf_index = None
        self.perf_index_val = None  # In case we want to do early stopping
        self.layers_transfer_func = None  # To be used in self.predict()

    def __repr__(self):
        hid_layers_dim = reprlib.repr(self.hidden_layers_dim)
        return 'ANNRegularSGDBackProp({}, {}, {})'.format(self.n_inputs,
                                                          self.n_neurons_output,
                                                          hid_layers_dim)

    def fit(self, x_train, y_train, layers_transfer_func, layers_transfer_func_der,
            n_epochs=1000, alpha=0.01, print_time=True, batch=False, perf_index="MSE",
            early_stop=False, stop=False, stop_crit=5, x_val=None, y_val=None):
        """
        Trains the initialized ANN using regular backpropagation gradient descent
        :param x_train: training set's features
        :param y_train: training set's target (needs some reshaping if multiclass, see demo at
        (https://github.com/PedroUria/DL/tree/master/feed-forward-ann/Demo-ANNRegularGDBackProp.ipynb)
        :param layers_transfer_func: list containing the transfer function at each layer
        (including Output Layer, so it should have one more element than hidden_layers_dim)
        :param layers_transfer_func_der: derivatives of layers_transfer_func
        :param n_epochs: number of training iterations
        :param alpha: learning rate
        :param print_time: if True, prints out the time it took to train
        :param batch: if True, do batch GD, if False, do Stochastic
        :param perf_index: loss function, admits "MSE" and "Cross-Entropy"
        :param early_stop: if True, do Early Stopping fitting strategy to prevent overfitting
        ;param stop: if True, actually stop the training process
        :param stop_crit: number of previous iterations on which the validation error
        must be monotonically increasing for the learning process to stop 
        :param x_val: validation set's features
        :param y_val: validation set's target
        :return: None (or best epoch if early_stop=True)
        just updates self.W, self.b and computes self.perf_index 
        """

        start = time()
        self.perf_index_str = perf_index  # For plotting title purposes
        # Randomly initializes parameters
        self.W, self.b = [], []
        np.random.seed(self.random_seed)
        current_n_input = self.n_inputs  
        # If the target is a flat array, we need to reshape it
        if len(y_train) == 1:
            y_train = y_train.reshape(-1, 1)
        # Loop to initialize weights and biases for the hidden layers
        for dim in self.hidden_layers_dim:
            self.W.append(np.random.uniform(-0.5, 0.5, (dim, current_n_input)))
            self.b.append(np.random.uniform(-0.5, 0.5, (dim, 1)))
            current_n_input = dim
        # Initializes weights and biases for the Output layer
        self.W.append(np.random.uniform(-0.5, 0.5, (self.n_neurons_output, dim)))
        self.b.append(np.random.uniform(-0.5, 0.5, (self.n_neurons_output, 1)))

        # MSE at each epoch
        self.perf_index = np.array([])
        if early_stop:
            self.perf_index_val = np.array([])
        self.layers_transfer_func = layers_transfer_func  # To be used in self.predict()

        if early_stop:
            stopped = False  # To stop the training if stop=True and monotonically incr condition
            n_samples_val = x_val.shape[0]  # To stop appending errors to self.perf_index_val

        # Training loop
        for epoch in range(n_epochs):
            # Array of the errors for each sample
            error = np.empty((x_train.shape[0], self.n_neurons_output))
            if early_stop:
                # Array of the validation errors for each sample
                val_error = np.empty((n_samples_val, self.n_neurons_output))
            if batch:
                # To store the whole update on the weights across all samples
                delta_W = [0] * len(self.W)
                delta_b = [0] * len(self.W)
            # Updating parameters for each sample
            for index, sample in enumerate(x_train):

                # Propagates the input forward
                n, a = [], []
                if early_stop:
                    if index < n_samples_val:
                        n_val, a_val = [], []
                # Reshapes input sample and assigns to a[0]
                a.append(sample.reshape(self.n_inputs, 1))
                if early_stop:
                    if index < n_samples_val:  # Omits when we do not have any more validation data
                        a_val.append(x_val[index].reshape(self.n_inputs, 1))
                # Hidden Layers
                for m in range(len(self.W)):
                    n.append(np.dot(self.W[m], a[m]) + self.b[m])
                    a.append(self.layers_transfer_func[m](n[m]))
                    if early_stop:
                        if index < n_samples_val:
                            n_val.append(np.dot(self.W[m], a_val[m]) + self.b[m])
                            a_val.append(self.layers_transfer_func[m](n_val[m]))
                # Gets the error (a[-1] is self.n_neurons_output x 1, so we need some reshaping)
                if perf_index == "MSE":
                    e = y_train[index].reshape(self.n_neurons_output, 1) - a[-1]
                    error[index] = e[:, 0]
                    if early_stop:
                        if index < n_samples_val:
                            e_val = y_val[index].reshape(self.n_neurons_output, 1) - a_val[-1]
                            val_error[index] = e_val[:, 0]
                elif perf_index == "Cross-Entropy":
                    y_train_sample_re = y_train[index].reshape(self.n_neurons_output, 1)
                    e = -np.dot(y_train_sample_re.T, np.log(a[-1])) - np.dot((1-y_train_sample_re).T, np.log(1-a[-1]))
                    error[index] = e[:, 0]
                    if early_stop:
                        y_val_sample_re = y_train[index].reshape(self.n_neurons_output, 1)
                        if index < n_samples_val:
                            e_val = -np.dot(y_val_sample_re.T, np.log(a[-1])) - np.dot((1-y_val_sample_re).T, np.log(1-a[-1]))
                            val_error[index] = e_val[:, 0]
                else:
                    return "Only MSE and Cross-Entropy shall be used as Performance Index"

                # Backpropagates the sensitivities
                F_der, s = [], []
                F_der.append(np.diag(layers_transfer_func_der[-1](n[-1]).ravel()))
                if perf_index == "MSE":
                    s.append(-2 * np.dot(F_der[-1], e))
                if perf_index == "Cross-Entropy":  # The Output Layer's senstivity is different
                    s.append(layers_transfer_func_der[-1](n[-1])*((1-y_train_sample_re)/(1-a[-1])- y_train_sample_re/a[-1]))
                for m in range(len(n) - 1)[::-1]:  # We are going backwards
                    F_der.append(np.diag(layers_transfer_func_der[m](n[m]).ravel()))
                    s.append(np.dot(F_der[-1], np.dot(self.W[m + 1].T, s[-1])))

                # Updates the weights and biases
                if batch:
                    for m in range(len(self.W)):
                        delta_W[m] += np.dot(s[::-1][m], a[m].T)
                        delta_b[m] += s[::-1][m]
                else:
                    for m in range(len(self.W)):
                        self.W[m] += -alpha * np.dot(s[::-1][m], a[m].T)
                        self.b[m] += -alpha * s[::-1][m]

                if early_stop:
                    if stop and len(self.perf_index_val) > 10:  # If we have at least gone through 10 iterations
                        copy_perf_index_val_stop_crit = self.perf_index_val[-stop_crit:].copy()
                        copy_perf_index_val_stop_crit.sort()
                        equal = self.perf_index_val[-stop_crit:] == copy_perf_index_val_stop_crit
                        if equal.sum() == stop_crit:  # Monotonically increasing condition
                            stopped = True
                            break
            
            if early_stop and stopped:
                break

            if batch:
                for m in range(len(self.W)):
                    self.W[m] += -alpha * delta_W[m]/x_train.shape[0]
                    self.b[m] += -alpha * delta_b[m]/x_train.shape[0]

            if perf_index == "MSE":
                power = 2
            else:
                power = 1
            self.perf_index = np.append(self.perf_index,
                                      np.sum(error**power, axis=0).reshape(self.n_neurons_output, 1).sum()/error.shape[0])
                                      
            if early_stop:
                self.perf_index_val = np.append(self.perf_index_val,
                                      np.sum(val_error**power, axis=0).reshape(self.n_neurons_output,
                                                                               1).sum()/val_error.shape[0])
        if print_time:
            print("The training process took", round(time() - start, 2), "seconds")
        if early_stop:
            return epoch

    def predict(self, x_test):
        """
        Predicts the desired target after calling self.fit()
        :param x_test: data used to predict (n_columns must be equal to self.n_inputs)
        :return: predicted target
        """

        output = np.array([])
        for sample in x_test:
            n, a = [], []
            # Propagates the input forward
            a.append(sample.reshape(self.n_inputs, 1))
            for m in range(len(self.W)):
                n.append(np.dot(self.W[m], a[m]) + self.b[m])
                a.append(self.layers_transfer_func[m](n[m]))
            output = np.append(output, a[-1])
            
        # When returning, uses the shape of the testing target, if there is one
        if len(self.target_shape) == 3:
            output_shape = (x_test.shape[0], self.target_shape[1], self.target_shape[2])
        else:
            output_shape = self.target_shape
        return output.reshape(output_shape)

    def score(self, x_test, y_test, classif=True, print_wrong=False):
        """
        Scores the model on x_test, t_test after calling self.fit() on x_train, y_train
        :param x_test: data to score the model on
        :param y_test: idem
        :param classif: True if we have a classification problem
        :param print_wrong: if True, the wrong predictions will be printed
        :return: scoring metric
        """
        if classif:
            count_wrong = 0
            if len(self.target_shape) == 3:  # Multiple classes need to be passed after One-Hot encoding them
                pred = self.predict(x_test)
                for i in range(y_test.shape[0]):
                    if y_test[i][np.argmax(pred[i])][0] != 1:  # For multiple classes, chooses
                        count_wrong += 1                       # biggest prob as predicted class
                        if print_wrong:
                            print("\nsample " + str(i) + "\n")
                            for j in range(y_test.shape[1]):
                                print("real:", y_test[i, j],
                                      "pred:", np.round(pred[i, j], 3)
                                      )
                return round(1 - count_wrong/y_test.shape[0], 3)
            else:
                pred = self.predict(x_test)
                for i in range(y_test.shape[0]):
                    if abs(y_test[i][0] - pred[i][0]) >= 0.5:  # For two classes, this criterion is enough
                        count_wrong += 1
                        if print_wrong:
                            print("real:", y_test[i][0], "pred:", pred[i][0])
                return round(1 - count_wrong/y_test.shape[0], 3)
        else:
            return (self.predict(x_test) - y_test)**2/x_test.shape[0]

    def viz_train_error(self, plot_val=False):
        """
        Plots the Performance Index along the training loop
        self.fit() must have been called previously
        :param plot_val: if self.fit(early_stop=True) was called
        and plot_val=True, also plots the validation error
        :return: None
        """
        plt.title(self.perf_index_str + " along the training process")
        plt.ylabel(self.perf_index_str)
        plt.xlabel("epoch")
        plt.plot(range(len(self.perf_index)), self.perf_index, label="Training " + self.perf_index_str)
        if plot_val:
            plt.plot(range(len(self.perf_index_val)), self.perf_index_val, label="Validation " + self.perf_index_str)
            plt.legend()
        plt.show()


# Defines some transfer functions and their derivatives

def logsigmoid(n):
    return 1/(1 + np.exp(-n))


def logsigmoid_der(n):
    return (1 - 1/(1 + np.exp(-n))) * 1/(1 + np.exp(-n))


def purelin(n):
    return n


def purelin_der(n):
    return np.ones(n.shape)


@np.vectorize
def poslin(n):
    return n if n >= 0 else 0


@np.vectorize
def poslin_der(n):
    if n > 0:
        return 1
    elif n < 0:
        return 0
    else:
        return None


# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
def stablesoftmax(n):
    """Compute the softmax of vector x in a numerically stable way."""
    shiftn = n - np.max(n)
    exps = np.exp(shiftn)
    return exps / np.sum(exps)


def softmax_der(n):
    return stablesoftmax(n) * (1 - stablesoftmax(n))


# Short Demo 
# More detailed at:
# (https://github.com/PedroUria/DL/tree/master/feed_forward_ann/Demo-ANNRegularGDBackProp.ipynb)
# NOTE: This takes approx 36 seconds on a MacBook Pro
# 2.6 GHz Intel Core i5 Processor, 8 GB 1600 MHz DDR3 Memory

if __name__ == '__main__':

    def f_to_approx(p):
        return np.exp(-abs(p)) * np.sin(np.pi * p)

    # Creates the interval in which to approximate the function
    p = np.linspace(-2, 2).reshape(-1, 1)
    # Creates 1-10-1 ANN Architecture (the parameters aren't initialized until calling the .fit() method)
    nn = ANNRegularGDBackProp(p.shape, f_to_approx(p).shape, [10], random_state=0)
    # Trains the ANN, using logsigmoid on the Hidden Layer and purelin on the Outer Layer
    nn.fit(p, f_to_approx(p), [logsigmoid, purelin], [logsigmoid_der, purelin_der], n_epochs=10000, alpha=0.07)
    # Visualizes the MSE along the training process
    nn.viz_train_error()
    # Predicts the values of p
    pred = nn.predict(p)
    # Shows the results
    plt.plot(p, f_to_approx(p), label="real function")
    plt.plot(p, pred, linestyle="dashed", label="approximated function")
    plt.legend()
    plt.show()




