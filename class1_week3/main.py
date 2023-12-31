import numpy as np
import sklearn
from matplotlib import pyplot as plt
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from testCases_v2 import layer_sizes_test_case, initialize_parameters_test_case, forward_propagation_test_case, \
    compute_cost_test_case, backward_propagation_test_case, update_parameters_test_case, nn_model_test_case, predict_test_case

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    """
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1= np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) *  0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    """
    Z1 = np.dot(parameters["W1"], X) + parameters["b1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["W2"], A1) + parameters["b2"]
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1,"Z2": Z2,"A2": A2}
    return A2, cache


def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    """
    lost = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    sum = np.sum(lost)
    cost = - sum / A2.shape[1]
    # cost = float(np.squeeze(lost))
    assert (isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.
    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    dZ2 = cache["A2"] - Y
    dW2 = (1/m) * np.dot(dZ2, cache["A1"].T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(parameters["W2"].T, dZ2) * (1 - np.power(cache["A1"], 2))
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW2": dW2, "db2": db2, "dW1": dW1, "db1": db1}
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using the gradient descent update rule given above
    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients
    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    W1 = parameters["W1"] - learning_rate * grads["dW1"]
    b1 = parameters["b1"] - learning_rate * grads["db1"]
    W2 = parameters["W2"] - learning_rate * grads["dW2"]
    b2 = parameters["b2"] - learning_rate * grads["db2"]
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X, Y, n_h, learning_rate, num_iterations = 10000, print_cost=False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    A2, _ = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions

if __name__ == '__main__':

    """
    X, Y = load_planar_dataset()
    # 1. Visualize the planar data:
    # plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    # plt.show()
    """

    """
    # 2. Visualize the data shape:
    shape_X = X.shape
    shape_Y = Y.shape
    m = (X.size) / shape_X[0]  # training set size
    print('The shape of X is: ' + str(shape_X))
    print('The shape of Y is: ' + str(shape_Y))
    print('I have m = %d training examples!' % (m))
    """

    """
    # 3. Try logistic regression:
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T)
    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title("Logistic Regression")
    plt.show()
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float(
        (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
          '% ' + "(percentage of correctly labelled datapoints)")
    """

    # 4. Try NN:
    # test1
    # X_assess, Y_assess = layer_sizes_test_case()
    # (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
    # print("The size of the input layer is: n_x = " + str(n_x))
    # print("The size of the hidden layer is: n_h = " + str(n_h))
    # print("The size of the output layer is: n_y = " + str(n_y))

    # test2
    # n_x, n_h, n_y = initialize_parameters_test_case()
    # parameters = initialize_parameters(n_x, n_h, n_y)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    # test3
    # X_assess, parameters = forward_propagation_test_case()
    # A2, cache = forward_propagation(X_assess, parameters)
    # Note: we use the mean here just to make sure that your output matches ours.
    # print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

    # test4
    # A2, Y_assess, parameters = compute_cost_test_case()
    # print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

    # test5
    # parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
    #
    # grads = backward_propagation(parameters, cache, X_assess, Y_assess)
    # print("dW1 = " + str(grads["dW1"]))
    # print("db1 = " + str(grads["db1"]))
    # print("dW2 = " + str(grads["dW2"]))
    # print("db2 = " + str(grads["db2"]))

    # test6
    # parameters, grads = update_parameters_test_case()
    # parameters = update_parameters(parameters, grads, 1.2)
    #
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    # test7: whole model
    # X_assess, Y_assess = nn_model_test_case()
    # parameters = nn_model(X_assess, Y_assess, 4, 1.02, num_iterations=10000, print_cost=True)
    # print("W1 = " + str(parameters["W1"]))
    # print("b1 = " + str(parameters["b1"]))
    # print("W2 = " + str(parameters["W2"]))
    # print("b2 = " + str(parameters["b2"]))

    # test8: whole model
    # parameters, X_assess = predict_test_case()
    # predictions = predict(parameters, X_assess)
    # print("predictions mean = " + str(np.mean(predictions)))

    # test9: planar data
    # Build a model with a 4-dimensional hidden layer
    X, Y = load_planar_dataset()
    # parameters = nn_model(X, Y, 4, 1.2, num_iterations=10000, print_cost=True)
    # Plot the decision boundary
    # plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    # plt.title("Decision Boundary for hidden layer size " + str(4))
    # plt.show()
    #
    # predictions = predict(parameters, X)
    # print('Accuracy: %d' % float(
    #     (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

    # test10: tuning hidden layer size
    # plt.figure(figsize=(16, 32))
    # hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    # for i, n_h in enumerate(hidden_layer_sizes):
    #     plt.subplot(5, 2, i + 1)
    #     plt.title('Hidden Layer of size %d' % n_h)
    #     parameters = nn_model(X, Y, n_h, 1.2, num_iterations=5000)
    #     plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    #     plt.show()
    #     predictions = predict(parameters, X)
    #     accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    #     print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

    # test11: test other datasets
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}
    dataset = "noisy_moons"
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    # make blobs binary
    if dataset == "blobs":
        Y = Y % 2
    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.show()

    # train and predict
    parameters = nn_model(X, Y, 4, 1.2, num_iterations=10000, print_cost=True)
    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()
    predictions = predict(parameters, X)
    print('Accuracy: %d' % float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')