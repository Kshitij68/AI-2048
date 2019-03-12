import numpy as np
import pygame
import time
from utils import get_logger

from repos.my_2048 import Game


class Bot:

    #TODO: Try out a CNN

    def __init__(self,name,network=None):
        self.name = name
        self.mutation_chance = 0.03
        self.network = network
        self.logger = get_logger("Bot")
        self.mapping = {0: pygame.K_UP,
                        1: pygame.K_LEFT,
                        2: pygame.K_DOWN,
                        3: pygame.K_RIGHT}

    def create_nn(self):
        n_inputs = 16
        n_nodes_h1 = 32
        # n_nodes_h2 = 4
        # n_nodes_h3 = 4
        n_classes = 4

        hidden_layer_1 = {'weights': np.random.random((n_inputs, n_nodes_h1)),
                               'biases': np.random.random(n_nodes_h1)}
        # hidden_layer_2 = {'weights': np.random.random((n_nodes_h1, n_nodes_h2)),
        #                        'biases': np.random.random(n_nodes_h2)}
        # hidden_layer_3 = {'weights': np.random.random((n_nodes_h2, n_nodes_h3)),
        #                        'biases': np.random.random(n_nodes_h3)}
        hidden_layer_4 = {'weights': np.random.random((n_nodes_h1, n_classes)),
                               'biases': np.random.random(n_classes)}

        if self.network is None:
            self.network = list()

        self.network.append(hidden_layer_1)
        # self.network.append(hidden_layer_2)
        # self.network.append(hidden_layer_3)
        self.network.append(hidden_layer_4)

    def predict_nn(self,data):
        if self.network is None:
            raise ValueError('The Neural Network model has not been initialized. Use create_nn() to initialize')

        l1 = np.add(np.dot(data,self.network[0]['weights']), self.network[0]['biases'])
        l1 = np.array([max(value,0) for value in l1])

        # l2 = np.add(np.dot(l1,self.network[1]['weights']), self.network[1]['biases'])
        # l2 = np.array([max(value,0) for value in l2])
        #
        # l3 = np.add(np.dot(l2,self.network[2]['weights']), self.network[2]['biases'])
        # l3 = np.array([max(value,0) for value in l3])

        output = np.add(np.dot(l1,self.network[1]['weights']), self.network[1]['biases'])
        output = np.array([max(value,0) for value in output])
        output = [[index,value] for index,value in enumerate(output)]
        output.sort(key=lambda x: x[1],reverse=True)
        return [self.mapping[value[0]] for value in output]

    def mutate_nn(self):

        for a,layer in enumerate(self.network):
            for b,row in enumerate(layer['weights']):
                    for c,col in enumerate(row):
                        if self.mutation_chance >= np.random.random():
                            self.network[a]['weights'][b][c] = np.random.random()
            for b,element in enumerate(layer['biases']):
                if self.mutation_chance >= np.random.random():
                    self.network[a]['biases'][b] = np.random.random()


class Optimizer:

    def __init__(self,n_generations, n_children):
        self.n_children = n_children
        self.logger = get_logger("Optimizer")
        self.bots = list()
        for i in range(self.n_children):
            bot = Bot('Bot Number' + str(i+1))
            bot.create_nn()
            self.bots.append(bot)
        self.high_score = 0
        self.n_generations = n_generations
        self.survival_rate = 0.1

        self.new_children = int(self.survival_rate * (1 - self.survival_rate) * self.n_children)


    @staticmethod
    def get_flat_tiles(tiles):
        return tiles.transpose().flatten()

    def get_high_scores(self):
        # It is not working properly
        bot_scores = list()
        for index,child in enumerate(self.bots):
            if index == 0:
                game = Game(show=True)
            else:
                time.sleep(100)
                game = Game(show=False)
            while True:
                time.sleep(0.1)
                input_matrix = self.get_flat_tiles(game.tiles)
                prediction = self.bots[index].predict_nn(input_matrix)
                output = game.run(prediction[0])
                if output is not None:
                    if output[0] == 300:
                        output = game.run(prediction[1])
                        if output is not None:
                            if output[1] == 300:
                                output = game.run(prediction[2])
                                if output is not None:
                                    if output[2] == 300:
                                        output = game.run(prediction[3])
                                        if output is not None:
                                            bot_scores.append([index, game.total_points, output[1]])
                                            break
                                    else:
                                        bot_scores.append([index, game.total_points, output[1]])
                                        break
                            else:
                                bot_scores.append([index, game.total_points, output[1]])
                                break

                    else:
                        bot_scores.append([index,game.total_points,output[1]])
                        break
        bot_scores.sort(key=lambda x: x[1],reverse=True)
        return bot_scores

    def main(self):
        for gen in range(self.n_generations):
            self.logger.info('Running generation number {} with total {} bots'.format(gen,len(self.bots)))
            scores = self.get_high_scores()[:int(self.survival_rate*self.n_children)]
            survived_bots_indexes = set([value[0] for value in scores])
            self.logger.info("The highest score obtained is: {}".format(scores[0][1]))
            new_bots = list()
            child_index = 0
            for index,bot in enumerate(self.bots):
                if index in survived_bots_indexes:
                    new_bots.append(bot)
                    bot_nn = bot.network
                    for _ in range(self.new_children):
                        new_bot = Bot(name='Bot Number '+str(child_index+1),network=bot_nn)
                        new_bot.mutate_nn()
                        new_bots.append(new_bot)
                        child_index += 1
            self.bots = new_bots.copy()
        return None

if __name__== "__main__":
    optimize = Optimizer(n_generations=1000,n_children=100)
    optimize.main()