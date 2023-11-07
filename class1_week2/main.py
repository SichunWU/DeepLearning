import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

def pre_processing():
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    # Example of a picture
    """
    index = 18
    plt.imshow(train_set_x_orig[index])
    plt.show()
    print("y = " + str(train_set_y[0, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
        "utf-8") + "' picture.")
    """
    # train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3)
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    # 假设 X.shape 为 (3, 64, 64, 3)
    # X.reshape(X.shape[0], -1) 将 X 重新排列成一个二维数组，第一个维度保持不变（即图片的数量），第二个维度使用 -1 表示自动计算，
    # 以使数据能够完全展平。结果是一个形状为 (3, 64*64*3) 的二维数组，其中每一行代表一张展平后的图片。
    # .T 表示转置，即将行和列互换。每一列代表一个样本，而每一行代表一个特征。得到一个形状为 (64*64*3, 3) 的二维数组。
    train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

    # 上面两行等同于:
    train_set_x_flatten = train_set_x_orig.reshape(
        train_set_x_orig.shape[1] * train_set_x_orig.shape[2] * train_set_x_orig.shape[3], train_set_x_orig.shape[0])
    test_set_x_flatten = test_set_x_orig.reshape(
        test_set_x_orig.shape[1] * test_set_x_orig.shape[2] * test_set_x_orig.shape[3], test_set_x_orig.shape[0])

    # standardize
    train_set_x = train_set_x_flatten / 255.
    test_set_x = test_set_x_flatten / 255.

    # Another method:
    # Assuming train_set_x_flatten is your dataset with shape (12288, 209)
    # Step 1: Compute mean and standard deviation
    # mean = np.mean(train_set_x_flatten, axis=1, keepdims=True)
    # std = np.std(train_set_x_flatten, axis=1, keepdims=True)
    # Handle zero standard deviation by replacing it with a small constant (e.g., 1e-8)
    # std[std < 1e-8] = 1e-8
    # Step 2: Standardize the dataset
    # train_set_x = (train_set_x_flatten - mean) / std

    """
    print("Number of training examples: m_train = " + str(m_train))
    print("Number of testing examples: m_test = " + str(m_test))
    print("Height/Width of each image: num_px = " + str(num_px))
    print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print("train_set_x_orig shape: " + str(train_set_x_orig.shape))
    print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    print("train_set_y shape: " + str(train_set_y.shape))
    print("test_set_x_orig shape: " + str(test_set_x_orig.shape))
    print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    print("test_set_y shape: " + str(test_set_y.shape))
    print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))
    """
    return train_set_x, train_set_y, test_set_x, test_set_y, num_px, classes

def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w,b

def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)     # forward propagation: activate
    cost = np.sum(Y * np.log(A) + (1-Y)* np.log(1-A)) / -m
    dw = np.dot(X, (A-Y).T) / m         # backword propagation: gradients
    db = np.sum(A - Y) / m

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        w -= learning_rate * grads["dw"]
        b -= learning_rate * grads["db"]

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w":w, "b":b}

    return params, grads, costs

def predict(w, b, X):
    Y_prediction = np.zeros((1, X.shape[1]))    # X.shape[1] examples
    w = w.reshape(X.shape[0], 1)                # X.shape[0] vectors
    A = sigmoid(np.dot(w.T, X) + b)

    #### WORKING SOLUTION
    # for i in range(A.shape[1]):
    #   Y_prediction[0, i] = 1 if A[0,i] >=0.5 else 0

    Y_prediction = (A >= 0.5) * 1.0
    assert (Y_prediction.shape == (1, X.shape[1]))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    Y_prediction_train = predict(params["w"], params["b"], X_train)
    Y_prediction_test = predict(params["w"], params["b"], X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": params["w"],
         "b": params["b"],
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """
    # 0. Test
    # a = np.array([[2,3,4,5],
    #     [3,3,3,3],
    #     [1,1,1,1]])
    # print(a)
    # print(a[0:2, 1])
    # print(np.sum(a, axis=0))
    # print(np.sum(a, axis=1))
    """

    """
    # 1. pre-processing
    train_set_x, train_set_y, test_set_x, test_set_y = pre_processing()
    """

    """
    # 2. Sigmoid function
    # print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))
    """

    """
    # 3. initialize w and b
    dim = 2
    w, b = initialize_with_zeros(dim)
    print("w = " + str(w))
    print("b = " + str(b))
    """

    """
    # 4. forward and backward propagate
    w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
    grads, cost = propagate(w, b, X, Y)
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    print("cost = " + str(cost))
    """

    """
    # 5. optimize w and b
    w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
    params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
    print("w = " + str(params["w"]))
    print("b = " + str(params["b"]))
    print("dw = " + str(grads["dw"]))
    print("db = " + str(grads["db"]))
    plt.plot(costs)
    #plt.show()
    """

    """
    # 6. predict
    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    print("predictions = " + str(predict(w, b, X)))
    """


    # 7. model
    train_set_x, train_set_y, test_set_x, test_set_y, num_px, classes = pre_processing()
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005,
              print_cost=False)
    """
    # wrongly classified, y = 1 but y^ = 0
    # index = 25
    # plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
    # plt.show()
    # print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[
    #     int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")

    # Plot learning curve (with costs)
    # costs = np.squeeze(d['costs'])
    # plt.plot(costs)
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(d["learning_rate"]))
    # plt.show()

    # tuning learning rate, prevent overshooting or learning slowly
    print('\n' + "-------------------------------------------------------" + '\n')
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
                               print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')
    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()
    """

    # We preprocess the image to fit your algorithm.
    my_image = "my_image.jpg"
    fname = "images/" + my_image
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = image / 255.0
    image = cv2.resize(image, (num_px, num_px))
    image_reshape = image.reshape((1, num_px * num_px * 3)).T

    my_predicted_image = predict(d["w"], d["b"], image_reshape)
    cv2.imshow("Resized Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[
        int(np.squeeze(my_predicted_image))].decode("utf-8") + "\" picture.")


