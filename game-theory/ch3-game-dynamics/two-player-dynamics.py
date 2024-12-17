import numpy as np

def strategies_overtime(game, init_strategies = None):
    strategies = init_strategies if init_strategies is not None else [random_probability_distribution(2), random_probability_distribution(2)]

    xs, ys = game.asymmetric_replicator_dynamics(
    x0=np.array(strategies[0]), y0=np.array(strategies[0]))

    return xs, ys


def random_probability_distribution(N):
    # Generate N random numbers
    random_values = np.random.rand(N)
    # Normalize the values so they sum to 1
    probabilities = random_values / np.sum(random_values)
    return probabilities