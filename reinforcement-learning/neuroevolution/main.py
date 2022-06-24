import torch

from network import MLP
from differential_evolution import DifferentialEvolution as DE

torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)


def main():
    de = DE(MLP)
    model_name, best_fitness = de.optimize()
    de.test_model(model_name)
    quit()


if __name__ == '__main__':
    main()
