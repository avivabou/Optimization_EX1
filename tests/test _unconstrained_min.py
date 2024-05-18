import unittest
import numpy as np
import examples
from src.unconstrained_min import UnconstrainedMinimization
from src.utils import plot_contour_lines, plot_function_values

class TestUnconstrainedMinimization(unittest.TestCase):
    def _run_test(self, example, title, x0 = np.array([1.0,1.0]), x_lim = [-2.0,2.0], y_lim=[-2.0,2.0], max_iter=100):
        f, grad_f, hessian_f = example(True)
        
        gradient_descent = UnconstrainedMinimization(f, grad_f, method='gradient_descent', max_iter=max_iter)
        gd_iterations, gd_location, gd_is_min = gradient_descent.minimize(x0)

        newton = UnconstrainedMinimization(f, grad_f, hessian_f=hessian_f, method='newton')
        n_iterations, n_location, n_is_min = newton.minimize(x0)

        plot_contour_lines(f,
                           title,
                           [(gradient_descent.path, 'Gradient Descent'), (newton.path, 'Newton')],
                           x_lim,
                           y_lim)

        plot_function_values([(gradient_descent.evaluations, 'Gradient Descent'), (newton.evaluations, 'Newton')], title=title)
        
        print()
        print("TEST:", title)
        print("="*(len(title)+5))
        print()
        print("  Method   |     Gradient Descent     |     Newton ")
        print("-----------------------------------------------------")
        print("Iterations | {:<24} | {:<10}".format(gd_iterations, n_iterations))
        print("Last X     | {:<23}  | {:<20}".format(str(gd_location), str(n_location)))
        print("Is Minimum | {:<24} | {:<10}".format(str(gd_is_min), str(n_is_min)))
        print()

    def test_quadratic_example_1(self):
        self._run_test(examples.quadratic_example_1, title="Quadratic Example 1")

    def test_quadratic_example_2(self):
        self._run_test(examples.quadratic_example_2, title="Quadratic Example 2")

    def test_quadratic_example_3(self):
        self._run_test(examples.quadratic_example_3, title="Quadratic Example 3")

    def test_rosenbrock_function(self):
        self._run_test(examples.rosenbrock_function, x0=np.array([-1.0,2.0]), title="Rosenbrock Function", x_lim=[-2.0,2.0], y_lim=[-2.0,5.0], max_iter=10000)

    def test_linear_function(self):
        self._run_test(examples.linear_function, title="Linear Function")

    def test_triangle_function(self):
       self._run_test(examples.triangle_function, title="Triangle Function")

if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit:
        pass
