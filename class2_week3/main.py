import os
import subprocess
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable
import time


def normalize(image):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, )
    and normalize its components.
    Arguments
    image - Tensor.
    Returns:
    result -- Transformed tensor
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1, ])   # 转为一维向量（一纵列）
    return image

def linear_function():
    """
    Implements a linear function:
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- Y = WX + b
    """
    np.random.seed(1)
    X = tf.constant(np.random.randn(3, 1), name = "X")
    W = tf.constant(np.random.randn(4, 3), name="W")
    b = tf.constant(np.random.randn(4, 1), name="f")
    result = tf.add(tf.matmul(W, X), b)
    return result


def sigmoid(z):
    """
    Computes the sigmoid of z
    Arguments:
    z -- input value, scalar or vector
    Returns:
    a -- (tf.float32) the sigmoid of z
    """
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a


def one_hot_matrix(label, depth=6):
    """
    Computes the one hot encoding for a single label
    Arguments:
        label --  (int) Categorical labels
        depth --  (int) Number of different classes that label can take
    Returns:
         one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """
    label = tf.one_hot(label, depth, axis=0)
    one_hot = tf.reshape(label, [depth, ])
    return one_hot

def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    parameters = {}
    parameters["W1"] = tf.Variable(initializer(shape=(25, 12288)))
    parameters["b1"] = tf.Variable(initializer(shape=(25, 1)))
    parameters["W2"] = tf.Variable(initializer(shape=(12, 25)))
    parameters["b2"] = tf.Variable(initializer(shape=(12, 1)))
    parameters["W3"] = tf.Variable(initializer(shape=(6, 12)))
    parameters["b3"] = tf.Variable(initializer(shape=(6, 1)))
    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    Z1 = tf.add(tf.linalg.matmul(parameters["W1"], X), parameters["b1"])
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.add(tf.linalg.matmul(parameters["W2"], A1), parameters["b2"])
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.add(tf.linalg.matmul(parameters["W3"], A2), parameters["b3"])
    return Z3


def compute_cost(logits, labels):
    """
    Computes the cost
    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- "true" labels vector, same shape as Z3
    Returns:
    cost - Tensor of the cost function
    """
    cost = tf.reduce_sum(
        tf.keras.metrics.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=True))
    cost = tf.reduce_mean(cost)

    # loss = tf.keras.losses.categorical_crossentropy(labels, logits)
    # cost = tf.reduce_mean(tf.reduce_sum(loss))
    return cost


def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    costs = []  # To keep track of the cost
    train_acc = []
    test_acc = []

    parameters = initialize_parameters()
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()

    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))

    m = dataset.cardinality().numpy()
    # prevent a memory bottleneck that can occur when reading from disk
    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)

    for epoch in range(num_epochs):
        epoch_cost = 0.
        # We need to reset object to start measuring from 0 the accuracy each epoch
        train_accuracy.reset_states()
        for (minibatch_X, minibatch_Y) in minibatches:
            # GradientTape can moving backwards through the graph recorded and compute derivatives
            with tf.GradientTape() as tape:
                # 1. predict
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                # 2. loss
                minibatch_cost = compute_cost(Z3, tf.transpose(minibatch_Y))
            # We acumulate the accuracy of all the batches
            train_accuracy.update_state(tf.transpose(Z3), minibatch_Y)

            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost
        epoch_cost /= m

        # Print the cost every 10 epochs
        if print_cost == True and epoch % 10 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("Train accuracy:", train_accuracy.result())

            # We evaluate the test set every 10 epochs to avoid computational overhead
            for (minibatch_X, minibatch_Y) in test_minibatches:
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                test_accuracy.update_state(tf.transpose(Z3), minibatch_Y)
            print("Test_accuracy:", test_accuracy.result())

            costs.append(epoch_cost)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()

    return parameters, costs, train_acc, test_acc

if __name__ == "__main__":
    # print(tf.__version__)

    # train_dataset = h5py.File('../datasets/train_signs.h5', "r")
    # test_dataset = h5py.File('../datasets/test_signs.h5', "r")
    # x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    # y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])
    # x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    # y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])
    # print(type(x_train))

    # inspect the shape and element
    # print(x_train.element_spec)
    # print(next(iter(x_train)))
    #
    # six different class label
    # unique_labels = set()   # set()可以存储各种元素，但不允许重复元素
    # for element in y_train:
    #     unique_labels.add(element.numpy())
    # print(unique_labels)

    # inspect images
    # images_iter = iter(x_train)
    # labels_iter = iter(y_train)
    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     ax = plt.subplot(5, 5, i + 1)
    #     plt.imshow(next(images_iter).numpy().astype("uint8"))
    #     plt.title(next(labels_iter).numpy().astype("uint8"))
    #     plt.axis("off")

    # use map method to apply the function
    # new_train = x_train.map(normalize)
    # new_test = x_test.map(normalize)
    # print(new_train.element_spec)
    # print(next(iter(new_train)))

    # linear function
    # result = linear_function()
    # print(result)
    # assert type(result) == EagerTensor, "Use the TensorFlow API"
    # assert np.allclose(result, [[-2.15657382], [2.95891446], [-1.08926781], [-0.84538042]]), "Error"
    # print("\033[92mAll test passed")

    # sigmoid
    # result = sigmoid(-1)
    # print("type: " + str(type(result)))
    # print("dtype: " + str(result.dtype))
    # print("sigmoid(-1) = " + str(result))
    # print("sigmoid(0) = " + str(sigmoid(0.0)))
    # print("sigmoid(12) = " + str(sigmoid(12)))
    #
    # def sigmoid_test(target):
    #     result = target(0)
    #     assert (type(result) == EagerTensor)
    #     assert (result.dtype == tf.float32)
    #     assert sigmoid(0) == 0.5, "Error"
    #     assert sigmoid(-1) == 0.26894143, "Error"
    #     assert sigmoid(12) == 0.99999386, "Error"
    #     print("\033[92mAll test passed")
    # sigmoid_test(sigmoid)

    # one_hot
    # def one_hot_matrix_test(target):
    #     label = tf.constant(1)
    #     depth = 4
    #     result = target(label, depth)
    #     print("Test 1:", result)
    #     assert result.shape[0] == depth, "Use the parameter depth"
    #     assert np.allclose(result, [0., 1., 0., 0.]), "Wrong output. Use tf.one_hot"
    #     label_2 = [2]
    #     result = target(label_2, depth)
    #     print("Test 2:", result)
    #     assert result.shape[0] == depth, "Use the parameter depth"
    #     assert np.allclose(result, [0., 0., 1., 0.]), "Wrong output. Use tf.reshape as instructed"
    #     print("\033[92mAll test passed")
    # one_hot_matrix_test(one_hot_matrix)
    #
    # new_y_test = y_test.map(one_hot_matrix)
    # new_y_train = y_train.map(one_hot_matrix)
    # print(next(iter(new_y_test)))

    # initialize_parameters
    # def initialize_parameters_test(target):
    #     parameters = target()
    #     values = {"W1": (25, 12288),
    #               "b1": (25, 1),
    #               "W2": (12, 25),
    #               "b2": (12, 1),
    #               "W3": (6, 12),
    #               "b3": (6, 1)}
    #     for key in parameters:
    #         print(f"{key} shape: {tuple(parameters[key].shape)}")
    #         assert type(parameters[key]) == ResourceVariable, "All parameter must be created using tf.Variable"
    #         assert tuple(parameters[key].shape) == values[key], f"{key}: wrong shape"
    #         assert np.abs(np.mean(parameters[key].numpy())) < 0.5, f"{key}: Use the GlorotNormal initializer"
    #         assert np.std(parameters[key].numpy()) > 0 and np.std(
    #             parameters[key].numpy()) < 1, f"{key}: Use the GlorotNormal initializer"
    #     print("\033[92mAll test passed")
    # initialize_parameters_test(initialize_parameters)
    #

    # build NN
    # parameters = initialize_parameters()
    # def forward_propagation_test(target, examples):
    #     minibatches = examples.batch(2)
    #     for minibatch in minibatches:
    #         forward_pass = target(tf.transpose(minibatch), parameters)
    #         print(forward_pass)
    #         assert type(forward_pass) == EagerTensor, "Your output is not a tensor"
    #         assert forward_pass.shape == (6, 2), "Last layer must use W3 and b3"
    #         assert np.allclose(forward_pass,
    #                            [[-0.13430887, 0.14086473],
    #                             [0.21588647, -0.02582335],
    #                             [0.7059658, 0.6484556],
    #                             [-1.1260961, -0.9329492],
    #                             [-0.20181894, -0.3382722],
    #                             [0.9558965, 0.94167566]]), "Output does not match"
    #         break
    #     print("\033[92mAll test passed")
    # forward_propagation_test(forward_propagation, new_train)

    # compute cost
    # def compute_cost_test(target, Y):
    #     pred = tf.constant([[2.4048107, 5.0334096],
    #                         [-0.7921977, -4.1523376],
    #                         [0.9447198, -0.46802214],
    #                         [1.158121, 3.9810789],
    #                         [4.768706, 2.3220146],
    #                         [6.1481323, 3.909829]])
    #     minibatches = Y.batch(2)
    #     for minibatch in minibatches:
    #         result = target(pred, tf.transpose(minibatch))
    #         break
    #     #result = target(pred, Y)
    #     print(result)
    #     assert (type(result) == EagerTensor), "Use the TensorFlow API"
    #     assert (np.abs(result - (
    #                 0.25361037 + 0.5566767) / 2.0) < 1e-7), "Test does not match. Did you get the mean of your cost functions?"
    #     print("\033[92mAll test passed")
    #
    # compute_cost_test(compute_cost, new_y_train)

    # Train the Model
    parameters, costs, train_acc, test_acc = model(new_train, new_y_train, new_test, new_y_test, num_epochs=100)
    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))
    plt.show()

    # Plot the train accuracy
    plt.plot(np.squeeze(train_acc))
    plt.ylabel('Train Accuracy')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))
    # Plot the test accuracy
    plt.plot(np.squeeze(test_acc))
    plt.ylabel('Test Accuracy')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(0.0001))
    plt.show()