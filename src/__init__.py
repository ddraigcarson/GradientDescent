import numpy as np
from scipy.optimize import newton

import time


def f(x):
    return 6*x**5 - 5*x**4 - 4*x**3 + 3*x**2


def df(x):
    return 30*x**4 - 20*x*3 - 12*x**2 + 6*x


def dx(f, x):
    return abs(0-f(x))


def newtons_method(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - f(x0)/df(x0)
        delta = dx(f, x0)
    return x0, f(x0)


def test_with_scipy(f, df, x0s, e):
    print("SCIPY TEST")
    for x0 in x0s:
        x = newton(f, x0, df, tol=1E-3)
        print(x, f(x))


def run():
    print("Running...")
    error_tolerance = 1e-10
    x0s = [0, 0.5, 1]
    for x0 in x0s:
        print(newtons_method(f, df, x0, error_tolerance))
    test_with_scipy(f, df, x0s, error_tolerance)


def run_numpy():
    print("Running Numpy...")
    # not much point, at most I can do multiple x0 values at the same time :(


np_start = time.time()
run()
np_end = time.time()
print("Basic %.3f s" % (np_end - np_start))

print("--------------------------------")

np_start = time.time()
run_numpy()
np_end = time.time()
print("Numpy %.3f s" % (np_end - np_start))