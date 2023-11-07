import pickle
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
# from dnn_app_utils_v3 import *
from dnn_app_utils_v3 import load_data, predict, print_mislabeled_images
from main import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    np.random.seed(1)
    parameters = initialize_parameters(layers_dims[0], layers_dims[1], layers_dims[2])
    costs = []
    # two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    for i in range(num_iterations):
        A1, cache1 = linear_activation_forward(X, parameters["W1"], parameters["b1"], "relu")
        A2, cache2 = linear_activation_forward(A1, parameters["W2"], parameters["b2"], "sigmoid")
        cost = compute_cost(A2, Y)
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")
        grads = {"dW1":dW1, "db1":db1, "dW2":dW2, "db2":db2 }
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    Arguments:
    X -- data, numpy array of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(1)
    parameters = initialize_parameters_deep(layers_dims)
    costs = []
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

if __name__ == '__main__':
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

    # Example of a picture
    # index = 25
    # plt.imshow(train_x_orig[index])
    # plt.show()
    # print("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")

    # Explore your dataset
    # m_train = train_x_orig.shape[0]
    # num_px = train_x_orig.shape[1]
    # m_test = test_x_orig.shape[0]
    # print ("Number of training examples: " + str(m_train))
    # print ("Number of testing examples: " + str(m_test))
    # print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    # print ("train_x_orig shape: " + str(train_x_orig.shape))
    # print ("train_y shape: " + str(train_y.shape))
    # print ("test_x_orig shape: " + str(test_x_orig.shape))
    # print ("test_y shape: " + str(test_y.shape))

    # Reshape the training and test examples
    # The "-1" makes reshape flatten the remaining dimensions
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255
    # print("train_x's shape: " + str(train_x.shape))
    # print("test_x's shape: " + str(test_x.shape))

    # 2-layer model
    ### CONSTANTS DEFINING THE MODEL ####
    # n_x = train_x.shape[0] # num_px * num_px * 3
    # n_h = 7
    # n_y = 1
    # layers_dims = (n_x, n_h, n_y)
    #
    # parameters = two_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
    # predictions_train = predict(train_x, train_y, parameters)
    # predictions_test = predict(test_x, test_y, parameters)


    # 4-layer model
    ### CONSTANTS ###
    # layers_dims = [12288, 20, 7, 5, 1]
    # parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
    # pred_train = predict(train_x, train_y, parameters)
    # pred_test = predict(test_x, test_y, parameters)
    # # For compatibility we use open(filename, 'wb') for non-text files and open(filename, 'w') for text files
    # with open("parameters.pickle", 'wb') as fileToBeWritten:
    #     pickle.dump(parameters, fileToBeWritten)

    with open('parameters.pickle', 'rb') as fileToBeRead:
        parameters = pickle.load(fileToBeRead)

    # Analysis
    # print_mislabeled_images(classes, test_x, test_y, pred_test)


    # test own image
    num_px = 64
    fileImage = Image.open("../images/my_image.jpg").convert("RGB").resize([num_px, num_px], Image.ANTIALIAS)
    my_label_y = [1]  # the true class of your image (1 -> cat, 0 -> non-cat)
    image = np.array(fileImage)
    my_image = image.reshape(num_px * num_px * 3, 1)
    my_image = my_image / 255.
    my_predicted_image = predict(my_image, my_label_y, parameters)
    plt.imshow(image)
    plt.show()
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[
        int(np.squeeze(my_predicted_image)),].decode("utf-8") + "\" picture.")
