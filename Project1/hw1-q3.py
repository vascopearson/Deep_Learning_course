# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q3.1a
        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:
            self.W[y_i, :] += x_i
            self.W[y_hat, :] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q3.1b
        label_scores = self.W.dot(x_i)[:, None]

        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1

        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))

        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None, :]

# Q3.2b
class MLP(object):
    
    def __init__(self, n_classes, n_features, hidden_size):
        
        self.W = [np.random.normal(0.1,0.01,(hidden_size, n_features)),np.random.normal(0.1,0.01,(n_classes, hidden_size))]
        self.B = [np.zeros(hidden_size),np.zeros(n_classes)]
        

    def predict(self, X):
        
        # forward
        predicted_labels = []
        for x in X:
            num_layers = len(self.W)
            hiddens = []
            z = []
            for i in range(num_layers):
                h = x if i == 0 else hiddens[i-1]
                z.append(self.W[i].dot(h) + self.B[i])
                if i < num_layers-1:
                    hiddens.append(np.maximum(0, z[i]))
            output = z[-1]
            output -= output.max()

            # compute loss
            probs = np.exp(output) / np.sum(np.exp(output),keepdims=True)
            
            # predict
            y_hat = np.argmax(probs)
            predicted_labels.append(y_hat)
        predicted_labels = np.array(predicted_labels)
        return predicted_labels
        
        

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        
        total_loss = 0
        for x, y_i in zip(X, y):
            
            # forward
            num_layers = len(self.W)
            hiddens = []
            z = []
            for i in range(num_layers):
                h = x if i == 0 else hiddens[i-1]
                z.append(self.W[i].dot(h) + self.B[i])
                if i < num_layers-1:
                    hiddens.append(np.maximum(0, z[i]))
            output = z[-1]
            output -= output.max()

            # compute loss
            probs = np.exp(output) / np.sum(np.exp(output),keepdims=True)
            loss = -np.log(probs[y_i])
            total_loss += loss

            # backward
            num_layers = len(self.W)

            grad_z = probs
            grad_z[y_i] -= 1
            grad_weights = []
            grad_biases = []
            for i in range(num_layers-1, -1, -1):
                h = x if i == 0 else hiddens[i-1]
                grad_weights.append(np.outer(h, grad_z))
                grad_biases.append(grad_z)
                if i != 0:
                    grad_h = self.W[i].T.dot(grad_z)
                    grad_z = grad_h * np.where(z[0] <= 0, 0, 1)
            grad_weights.reverse()
            grad_biases.reverse()

            # update parameters
            num_layers = len(self.W)
            for i in range(num_layers):
                self.W[i] -= learning_rate*grad_weights[i].T
                self.B[i] -= learning_rate*grad_biases[i]


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()

    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()

