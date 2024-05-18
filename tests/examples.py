import numpy as np

def quadratic_example_1(evaluate_hessian=False):
    f = lambda x: np.dot(x, x)
    g = lambda x: 2 * x
    h = None
    
    if evaluate_hessian:
        h = lambda x: 2 * np.eye(len(x))
    
    return f, g, h

def quadratic_example_2(evaluate_hessian=False):
    Q = np.array([[1, 0], [0, 100]])

    f = lambda x: x.T.dot(Q).dot(x)
    g = lambda x: 2 * Q.dot(x)
    h = None

    if evaluate_hessian:
        h = lambda x: 2 * Q

    return f, g, h

def quadratic_example_3(evaluate_hessian=False):
    Q = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    R = np.array([[100, 0], [0, 1]])
    QRT = Q.T.dot(R).dot(Q)

    f = lambda x: x.T.dot(QRT).dot(x)
    g = lambda x:  2 * QRT.dot(x)
    h = None

    if evaluate_hessian:
        h = lambda x: 2 * QRT

    return f, g, h

def rosenbrock_function(evaluate_hessian=False):
    f = lambda x: 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    g = lambda x: np.array([-400 * x[0] * x[1] + 400 * (x[0] ** 3) + 2 * x[0] - 2, 200 * x[1] - 200 * (x[0] ** 2)])
    h = None

    if evaluate_hessian:
        h = lambda x: np.array([[-400 * x[1] + 1200 * (x[0] ** 2) + 2, -400 * x[0]], [-400 * x[0], 200]])

    return f, g, h

def linear_function(evaluate_hessian=False):
    a = np.array([2, -3])
    f = lambda x: a.T.dot(x)
    g = lambda _: a
    h = None

    return f, g, h

def triangle_function(evaluate_hessian=False):
    f = lambda x: np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1)
    g = lambda x: np.array([np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1),
                            3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)])
    h = None

    if evaluate_hessian:
        h = lambda x: np.array([[np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1),
                                 3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)],
                                 [3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1),
                                  9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)]])
    
    return f, g, h
