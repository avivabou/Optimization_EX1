import unittest
import numpy as np
import examples
from src.unconstrained_min import UnconstrainedMinimization
from src.utils import plot_contour_lines, plot_function_values

class TestUnconstrainedMinimization(unittest.TestCase):
    def _run_test(self, example, x0, title,  x_lim = [-2.0,2.0], y_lim=[-2.0,2.0]):
        f, grad_f, hessian_f = example(True)
        
        gradient_descent = UnconstrainedMinimization(f, grad_f, method='gradient_descent')
        gradient_descent.minimize(x0)

        newton = UnconstrainedMinimization(f, grad_f, hessian_f=hessian_f, method='newton')
        newton.minimize(x0)

        plot_contour_lines(f,
                           title,
                           [(gradient_descent.path, 'Gradient Descent'), (newton.path, 'Newton')],
                           x_lim,
                           y_lim)

        plot_function_values([(gradient_descent.evaluations, 'Gradient Descent'), (newton.evaluations, 'Newton')], title=f'Function values per Iteration of {title}')

    def test_quadratic_example_1(self):
        self._run_test(examples.quadratic_example_1, np.array([1.0, 1.0]), title="Quadratic Example 1 Contour Plot")

    def test_quadratic_example_2(self):
        self._run_test(examples.quadratic_example_2, np.array([1.0, 1.0]), title="Quadratic Example 2 Contour Plot")

    def test_quadratic_example_3(self):
        self._run_test(examples.quadratic_example_3, np.array([1.0, 1.0]), title="Quadratic Example 3 Contour Plot")

    def test_rosenbrock_function(self):
        self._run_test(examples.rosenbrock_function, np.array([-1.0, 2.0]), title="Rosenbrock Function Contour Plot", x_lim=[-2.0,2.0], y_lim=[-2.0,5.0])

    def test_linear_function(self):
        self._run_test(examples.linear_function, np.array([-100.0, 2.0]), title="Linear Function Contour Plot")

    def test_triangle_function(self):
       self._run_test(examples.triangle_function, np.array([2.0, 2.0]), title="Triangle Function Contour Plot")

if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
