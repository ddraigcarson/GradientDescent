import numpy as np

import time

# INCOMPLETE

DATA_LOCATION = "C:/Users/Greg/IdeaProjects/GradientDescent/src/data.csv"
data_points = np.genfromtxt(DATA_LOCATION, delimiter=",")


def total_error(b, m):
    x, y = np.split(data_points, 2, axis=1)
    e_total = np.sum((y - (m * x + b))**2)/float(data_points.shape[0])
    return e_total


def compute_jacobian(potential_solution, h=1e-5):
    # First principle derivative is dy/dx = (f(x+h) - f(x))/h where h->0
    # We are trying to find the minimum of the function, where dy/dx = 0 (minima)
    # For simplicity I will only solve the 2 feature problem
    jacobian = np.zeros(len(potential_solution))
    b = potential_solution[0]
    m = potential_solution[1]
    jacobian[0] = (total_error(b + h, m) - total_error(b, m))/h
    jacobian[1] = (total_error(b, m + h) - total_error(b, m))/h
    return jacobian


def compute_hessian(potential_solution, h=1e-5):
    hessian = np.zeros(len(potential_solution))
    b = potential_solution[0]
    m = potential_solution[1]
    hessian[0] = (compute_jacobian([b + h, m]) - compute_jacobian([b, m]))/h
    hessian[1] = (compute_jacobian([b, m + h]) - compute_jacobian([b, m]))/h
    return hessian


def run_2nd_order():
    print("Starting second order newton optimization")
    initial_solution = [0., 1.]
    iterations = 10000
    solutions = np.zeros((iterations, len(initial_solution)))
    solutions[0] = initial_solution
    final_solution = None

    for i in range(iterations):
        jacobian = compute_jacobian(solutions[i])


np_start = time.time()
run_2nd_order()
np_end = time.time()
print("Basic %.3f s" % (np_end - np_start))

