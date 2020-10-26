import sys
import numpy as np
import random
from scipy import stats


def main():
    # training examples file name
    train_x_file_name = sys.argv[1]
    # training labels file name
    train_y_file_name = sys.argv[2]
    # testing examples file name
    test_x_file_name = sys.argv[3]
    if len(sys.argv) < 4:
        print("invalid number of arguments to main")
        exit(0)
    # apply the learning algorithms on the data
    apply_learning_algorithms(train_x_file_name, train_y_file_name, test_x_file_name)


def apply_learning_algorithms(train_x_file_name, train_y_file_name, test_x_file_name):
    train_x = read_examples_file(train_x_file_name)
    train_y = read_labels_file(train_y_file_name)
    # test_different_hyper_parameters(train_x, train_y)
    combined = list(zip(train_x, train_y))
    random.seed(24)
    # shuffle the data
    random.shuffle(combined)
    train_x[:], train_y[:] = zip(*combined)
    train_x = np.asarray(train_x)
    train_y = np.asarray(train_y)
    # set aside a small part from the data for the validation set
    num_of_instances = len(train_x)
    m = int(num_of_instances * 0.8)
    validation_set_x = train_x[0: (num_of_instances - m)]
    train_x = train_x[(num_of_instances - m): num_of_instances]
    validation_set_y = train_y[0: (num_of_instances - m)]
    train_y = train_y[(num_of_instances - m): num_of_instances]
    w_perceptron = perceptron(train_x, train_y, validation_set_x, validation_set_y, 30, 0.1)
    w_svm = svm(train_x, train_y, validation_set_x, validation_set_y, 10, 0.1, 0.1)
    w_passive_aggressive = passive_aggressive(train_x, train_y, validation_set_x, validation_set_y, 30)
    # testing phase
    test_x = read_examples_file(test_x_file_name)
    print_test_results(test_x, w_perceptron, w_svm, w_passive_aggressive)


def read_examples_file(file_name):
    examples_features = []
    file = open(file_name, "r")
    for line in file:
        current_example_features = []
        split_line = line.split(",")
        # convert the categorial feature to a numerical one
        if split_line[0] == "M":
            current_example_features += [1.0, 0.0, 0.0]
        elif split_line[0] == "F":
            current_example_features += [0.0, 1.0, 0.0]
        elif split_line[0] == "I":
            current_example_features += [0.0, 0.0, 1.0]
        # add the rest of the features
        for i in range(1, len(split_line)):
            current_example_features.append(float(split_line[i]))
        examples_features.append(current_example_features)
    return examples_features


def read_labels_file(file_name):
    training_labels = []
    file = open(file_name, "r")
    for line in file:
        split_line = line.split(".")
        training_labels.append(int(split_line[0]))
    return training_labels


# The function applies perceptron learning algorithm
def perceptron(train_x, train_y, validation_set_x, validation_set_y, epochs, eta):
    d = len(train_x[0])
    w = np.zeros((3, d))
    w = np.asarray(w)
    np.random.seed(24)
    for e in range(0, epochs):
        for x, y in zip(train_x, train_y):
            # predict
            y_hat = np.argmax(np.dot(w, x))
            # update rules
            if y != y_hat:
                w[y, :] = w[y, :] + eta * x
                w[y_hat, :] = w[y_hat, :] - eta * x
            eta *= 0.9999
    # calculate the error rate
    err = error_rate(validation_set_x, validation_set_y, w)
    return w


def error_rate(validation_set_x, validation_set_y, w):
    num_of_mistakes = 0
    for t in range(0, len(validation_set_x)):
        y_hat = np.argmax(np.dot(w, validation_set_x[t]))
        if validation_set_y[t] != y_hat:
            num_of_mistakes = num_of_mistakes + 1
    return float(num_of_mistakes) / len(validation_set_x)


# min max normalization method
def min_max_normalization(train_x):
    train_x = np.transpose(train_x)
    new_min = 0
    new_max = 1
    new_max_min_difference = new_max - new_min
    for i in range(0, np.size(train_x, 0)):
        # at first the min and the max values are the first element in the row
        min_val = max_val = train_x[i][0]
        for j in range(0, np.size(train_x, 1)):
            if train_x[i][j] < min_val:
                min_val = train_x[i][j]
            if train_x[i][j] > max_val:
                max_val = train_x[i][j]
        max_min_difference = max_val - min_val
        for j in range(0, np.size(train_x, 1)):
            train_x[i][j] = ((train_x[i][j] - min_val) / max_min_difference) * new_max_min_difference + new_min
    return np.transpose(train_x)


# z score normalization method
def z_score_normalization(train_x):
    train_x = stats.mstats.zscore(train_x, axis=1, ddof=1)
    return train_x


# The function applies svm learning algorithm
def svm(train_x, train_y, validation_set_x, validation_set_y, epochs, eta, lambda_value):
    d = len(train_x[0])
    w = np.zeros((3, d))
    w = np.asarray(w)
    np.random.seed(24)
    for e in range(0, epochs):
        indexes = np.arange(train_x.shape[0])
        np.random.shuffle(indexes)
        train_x = train_x[indexes]
        train_y = train_y[indexes]
        for x, y in zip(train_x, train_y):
            eta_lambda_shortcut = (1 - eta * lambda_value)
            # predict
            y_hat = np.argmax(np.dot(w, x))
            # update rules
            if y != y_hat:
                w[y, :] = (w[y, :] * eta_lambda_shortcut) + eta * x
                w[y_hat, :] = (w[y_hat, :] * eta_lambda_shortcut) - eta * x
                j = 3 - y - y_hat
                w[j, :] = eta_lambda_shortcut * w[j, :]
                eta *= 0.999
            else:
                eta *= 0.999
                for i in range(len(w)):
                    if i != y:
                        w[i] *= eta_lambda_shortcut
    err = error_rate(validation_set_x, validation_set_y, w)
    return w


# The function applies passive aggressive learning algorithm
def passive_aggressive(train_x, train_y, validation_set_x, validation_set_y, epochs):
    len_class = len(train_x[0])
    w = np.zeros((3, len_class))
    rand = np.arange(len(train_y))
    np.random.seed(9001)
    np.random.shuffle(rand)
    train_x = train_x[rand]
    train_y = train_y[rand]
    indexes = np.arange(0, train_x.shape[0])
    for e in range(0, epochs):
        for k in range(0, train_x.shape[0]):
            # choose one -k of random example from all of the train_x.shape[0]
            i = indexes[k]
            vector_xw = np.dot(w, train_x[i])
            # prediction the model find prediction y
            y_hat = np.argmax(vector_xw)
            # label of current sample
            y = int(train_y[i])
            # if the prediction of the model is wrong, update the weights matrix
            if y_hat != y:
                # compute tau- be careful about dive zero
                norm_x = np.linalg.norm(train_x[i], ord=2)
                if norm_x == 0:
                    tau = 0.01
                else:
                    loss = max(0, 1 - vector_xw[y] + vector_xw[y_hat])
                    tau = loss / (2 * norm_x * norm_x)
                # multi class of passive aggressive update rule
                w[y_hat] = w[y_hat] - tau * train_x[i]
                w[y] = w[y] + tau * train_x[i]
    # calculate the error rate
    err = error_rate(validation_set_x, validation_set_y, w)
    return w


def print_test_results(test_x, w_perceptron, w_svm, w_passive_aggressive):
    for t in range(0, len(test_x)):
        y_hat_perceptron = np.argmax(np.dot(w_perceptron, test_x[t]))
        y_hat_svm = np.argmax(np.dot(w_svm, test_x[t]))
        y_hat_passive_aggressive = np.argmax(np.dot(w_passive_aggressive, test_x[t]))
        print("perceptron: ", str(y_hat_perceptron), ", svm: ", y_hat_svm, ", pa: ", y_hat_passive_aggressive, sep='')


def check_test_err(w_perceptron, w_svm, w_passive_aggressive, test_x, test_y):
    print("\n")
    print("test perceptron", error_rate(test_x, test_y, w_perceptron))
    print("test svm", error_rate(test_x, test_y, w_svm))
    print("test pa", error_rate(test_x, test_y, w_passive_aggressive))


def test_different_hyper_parameters (train_x, train_y):
    combined = list(zip(train_x, train_y))
    # shuffle the data
    random.shuffle(combined)
    train_x[:], train_y[:] = zip(*combined)
    train_x = np.asarray(train_x)
    z_score_normalization(train_x)
    train_y = np.asarray(train_y)
    # set aside a small part from the data for the validation set
    num_of_instances = len(train_x)
    m = int(num_of_instances * 0.8)
    validation_set_x = train_x[0: (num_of_instances - m)]
    train_x = train_x[(num_of_instances - m): num_of_instances]
    validation_set_y = train_y[0: (num_of_instances - m)]
    train_y = train_y[(num_of_instances - m): num_of_instances]
    # find the optimal epochs number
    perceptron(train_x, train_y, validation_set_x, validation_set_y, 22, 0.1)
    svm(train_x, train_y, validation_set_x, validation_set_y, 25, 0.01, 0.01)
    for i in range(0, 100):
        passive_aggressive(train_x, train_y, validation_set_x, validation_set_y, 20, i)


main()
