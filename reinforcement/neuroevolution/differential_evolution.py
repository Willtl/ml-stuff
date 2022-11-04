import copy
import multiprocessing
import random
import time

import torch
from joblib import Parallel, delayed

import env.game as gm
import env.graphics as env 

torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)


class DifferentialEvolution:
    def __init__(self, model):
        self.model = model
        self.cpu = torch.device("cpu")
        self.gpu = torch.device("cuda")
        self.device = self.cpu
        self.numb_parameters = 0
        self.number_individuals = multiprocessing.cpu_count()
        self.iterations = 500
        self.diff_coeff = 1.1
        self.crossover_rate = 0.1
        self.pop = []
        self.pop_t1 = []
        self.fitness = torch.Tensor(self.number_individuals)
        self.fitness_t1 = torch.Tensor(self.number_individuals)
        self.init_population()

    def init_population(self):
        with torch.no_grad():
            for param in self.model().parameters():
                if len(param.shape) == 2:
                    self.numb_parameters += param.shape[0] * param.shape[1]
                else:
                    self.numb_parameters += param.shape[0]

            for i in range(self.number_individuals):
                self.pop.append(self.model().to(self.device))
                self.pop_t1.append(self.model().to(self.device))
                self.pop[i].disable_grad()
                self.pop_t1[i].disable_grad()
                self.fitness[i] = -100
                self.fitness_t1[i] = -100

    def optimize(self):
        with torch.no_grad():
            start = time.time()
            for iteration in range(self.iterations):
                if (iteration + 1) % 10 == 0:
                    print(f"Iteration: {iteration}, fitness: {self.fitness.data}, time: {time.time() - start}")
                    start = time.time()
                for i in range(self.number_individuals):
                    # if random.random() <= 0.5:
                    #     a, b, c = self.select_three_rand(i)
                    # else:
                    #     a, b, c = self.select_best_three(i)
                    a, b, c = self.select_three_rand(i)
                    R = random.randint(0, self.numb_parameters - 1)
                    self.update_weights(a, b, c, R, i)
                self.fitness_t1 = self.evaluate()
                for i in range(self.number_individuals):
                    if self.fitness[i] <= self.fitness_t1[i]:
                        self.fitness[i] = self.fitness_t1[i]
                        self.pop[i].load_state_dict(copy.deepcopy(self.pop_t1[i].state_dict()))
                    else:
                        self.pop_t1[i].load_state_dict(copy.deepcopy(self.pop[i].state_dict()))

        best = self.get_best()
        model_name = f'model_score{int(self.fitness[best])}'
        best_fitness = self.fitness[best]
        torch.save(self.pop[best].state_dict(), f'neural-evo-models/{model_name}')
        return model_name, best_fitness

    def evaluate(self):
        fitness = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(self.run_game)(i) for i in range(self.number_individuals))
        return fitness

    def select_best_three(self, index):
        best = torch.argmax(self.fitness)
        index1 = random.randint(0, self.number_individuals - 1)
        while index1 == best:
            index1 = random.randint(0, self.number_individuals - 1)
        index2 = random.randint(0, self.number_individuals - 1)
        while index2 == best and index2 == index1:
            index2 = random.randint(0, self.number_individuals - 1)
        return best, index1, index2

    def select_three_rand(self, index):
        indices = [i for i in range(self.number_individuals) if i != index]
        random.shuffle(indices)
        return indices[0], indices[1], indices[2]

    def update_weights(self, a, b, c, R, child):
        param_counter = 0
        for name, param in self.pop_t1[child].named_parameters():
            if len(param.shape) == 2:
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        if random.random() <= self.crossover_rate or R == param_counter:
                            va = self.pop[a].state_dict()[name].data[i][j]
                            vb = self.pop[b].state_dict()[name].data[i][j]
                            vc = self.pop[c].state_dict()[name].data[i][j]
                            param.data[i][j] = va + (self.diff_coeff * (vb - vc))
            if len(param.shape) == 1:
                for i in range(param.shape[0]):
                    if random.random() <= self.crossover_rate or R == param_counter:
                        va = self.pop[a].state_dict()[name].data[i]
                        vb = self.pop[b].state_dict()[name].data[i]
                        vc = self.pop[c].state_dict()[name].data[i]
                        param.data[i] = va + (self.diff_coeff * (vb - vc))
            param_counter += 1

    # Evaluate only with game logic
    def run_game(self, id):
        fitness = 0
        game_instance = gm.Game()
        row_count = float(gm.ROW_COUNT)
        col_count = float(gm.COLUMN_COUNT)
        game_state = gm.Game.get_state(game_instance.reset(), row_count, col_count)

        for i in range(300):
            input = torch.tensor([game_state]).to(self.device)
            output = self.pop_t1[id](input)
            action = torch.argmax(output[0])
            new_state, reward = game_instance.step(int(action) + 1)
            if reward == 'wall' or reward == 'dead':
                break
            fitness += reward
            game_state = gm.Game.get_state(new_state, row_count, col_count)

        return fitness

    def run_game_graphic(self, id):
        index = torch.argmax(self.fitness)
        fitness = 0
        game_instance = env.Pygame()
        row_count = float(game_instance.row_count)
        col_count = float(game_instance.col_count)
        game_state = gm.Game.get_state(game_instance.game.reset(), row_count, col_count)
        for i in range(50):
            input = torch.tensor([game_state])
            output = self.pop_t1[id](input)
            action = torch.argmax(output[0])
            new_state, reward = game_instance.game.step(int(action) + 1)
            if reward == 'wall' or reward == 'dead':
                break
            fitness += reward
            game_state = gm.Game.get_state(new_state, row_count, col_count)
            if id == index.item():
                game_instance.pump()
                game_instance.render()
                time.sleep(0.05)
        game_instance.quit_game()
        del game_instance
        return fitness

    def get_best(self):
        return torch.argmax(self.fitness)

    def test_model(self, model_name):
        with torch.no_grad():
            game_instance = env.Pygame()
            model = self.model()
            model.load_state_dict(torch.load('neural-evo-models/' + model_name))
            row_count = float(game_instance.row_count)
            col_count = float(game_instance.col_count)
            state_t = gm.Game.get_state(game_instance.game.reset(), row_count, col_count)
            for t in range(1000):
                input = torch.tensor([state_t])
                output = model(input)
                action = torch.argmax(output[0])
                new_state, reward = game_instance.game.step(int(action) + 1)
                state_t1 = gm.Game.get_state(new_state, row_count, col_count)
                if reward == 'wall' or reward == 'dead':
                    print(f'Dead by {reward}')
                    break
                state_t = state_t1
                # Render
                game_instance.pump()
                game_instance.render()
                time.sleep(0.01)
