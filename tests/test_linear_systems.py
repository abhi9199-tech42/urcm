from urcm.core.logic_gates import LinearSystem, QuadraticOpt

def test_solve_linear_system_2x2():
    A = [[2.0, 1.0],[1.0, 3.0]]
    b = [5.0, 7.0]
    x = LinearSystem.solve(A, b)
    assert x is not None
    assert abs(2.0*x[0] + 1.0*x[1] - 5.0) < 1e-6
    assert abs(1.0*x[0] + 3.0*x[1] - 7.0) < 1e-6

def test_quadratic_opt_diag():
    x = QuadraticOpt.minimize_diag([2.0, 4.0], [6.0, -8.0])
    assert x == [-3.0, 2.0]
