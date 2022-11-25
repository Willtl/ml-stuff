from bayes_opt import BayesianOptimization
from numpy import cos
from numpy import e
from numpy import exp
from numpy import pi
from numpy import sqrt
from scipy.optimize import differential_evolution


def ackley_bo(x, y):
    return -(-20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(
        0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20)


def ackley_de(v):
    x, y = v
    return -20.0 * exp(-0.2 * sqrt(0.5 * (x ** 2 + y ** 2))) - exp(
        0.5 * (cos(2 * pi * x) + cos(2 * pi * y))) + e + 20


def bo():
    optimizer = BayesianOptimization(
        f=ackley_bo,
        pbounds={'x': (-5, 5), 'y': (-5, 5)},
        random_state=1,
        verbose=2
    )

    optimizer.maximize(
        init_points=5,
        n_iter=50,
    )

    print(optimizer.max)


def de():
    bounds = [[-5.0, 5.0], [-5.0, 5.0]]

    result = differential_evolution(ackley_de, bounds, workers=6)

    print(result)
    print(ackley_de(result['x']))


def main():
    bo()
    de()


if __name__ == '__main__':
    main()
