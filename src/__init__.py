import numpy as np
import time


def compute_error(b, m, data):
    e_total = 0
    for i in range(0, data.shape[0]):
        x = data[i, 0]
        y = data[i, 1]
        e_total += (y - (m * x + b)) ** 2
    return e_total / float(data.shape[0])


def compute_error_matrix(b, m, data):
    x, y = np.split(data, 2, axis=1)
    e_total = np.sum((y - (m * x + b)) ** 2)/float(data.shape[0])
    return e_total


def step_gradient(b, m, data, learning_rate):
    b_der = 0
    m_der = 0
    N = data.shape[0]
    for i in range(0, N):
        x = data[i, 0]
        y = data[i, 1]
        # We are plotting the error for our equation
        # So the equation isnt y = mx + b
        # The equation is Mean Square Error
        # Which is MSE = (1/N) SUM( algorithms prediction for x - trainings value )**2
        # MSE = (1/N) SUM( y_p - y_a )**2
        # MSE = (1/N) SUM( mx+b - y )**2
        b_der += (2/N) * (m * x + b - y)
        m_der += ((2 * x)/N) * (m * x + b - y)
    b_new = b - (learning_rate * b_der)
    m_new = m - (learning_rate * m_der)
    return b_new, m_new


def step_gradient_matrix(b, m, data, learning_rate):
    N = data.shape[0]
    x, y = np.split(data, 2, axis=1)
    dFdB = (2/N)*(m * x + b - y)
    dFdM = ((2 * x)/N) * (m * x + b - y)

    dFdB_sum = np.sum(dFdB)
    dFdM_sum = np.sum(dFdM)

    b_new = b - (learning_rate * dFdB_sum)
    m_new = m - (learning_rate * dFdM_sum)
    return b_new, m_new


def gradient_descent_runner(data, b_i, m_i, learning_rate, n):
    b = b_i
    m = m_i

    for i in range(0, n):
        b, m = step_gradient(b, m, data, learning_rate)
    return b, m


def gradient_descent_runner_matrix(data, b_i, m_i, learning_rate, n):
    b = b_i
    m = m_i

    for i in range(0, n):
        b, m = step_gradient_matrix(b, m, data, learning_rate)
    return b, m


def run():
    DATA_LOCATION = "C:/Users/Greg/IdeaProjects/GradientDescent/src/data.csv"
    data = np.genfromtxt(DATA_LOCATION, delimiter=",")

    learning_rate = 0.0001
    b_i = 0
    m_i = 0
    n = 1000
    e_i = compute_error(b_i, m_i, data)
    print("Initial b: {0}, m: {1}, error: {2}".format(b_i, m_i, e_i))
    print("Running...")
    b, m = gradient_descent_runner(data, b_i, m_i, learning_rate, n)
    e = compute_error(b, m, data)
    print("Final b: {0}, m: {1}, error: {2}".format(b, m, e))


def runNumpy():
    DATA_LOCATION = "C:/Users/Greg/IdeaProjects/GradientDescent/src/data.csv"
    data = np.genfromtxt(DATA_LOCATION, delimiter=",")

    learning_rate = 0.0001
    b_i = 0
    m_i = 0
    n = 1000
    e_i = compute_error_matrix(b_i, m_i, data)
    print("Initial b: {0}, m: {1}, error: {2}".format(b_i, m_i, e_i))
    print("Running...")
    b, m = gradient_descent_runner_matrix(data, b_i, m_i, learning_rate, n)
    e = compute_error_matrix(b, m, data)
    print("Final b: {0}, m: {1}, error: {2}".format(b, m, e))

np_start = time.time()
run()
np_end = time.time()
print("Basic %.3f s" % (np_end - np_start))

print("--------------------------------")

np_start = time.time()
runNumpy()
np_end = time.time()
print("Basic %.3f s" % (np_end - np_start))