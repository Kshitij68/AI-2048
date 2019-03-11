import time
import numpy as np
import os

from utils import get_logger
logger = get_logger("2048")

class Bot:


    #TODO: Try out a convulational neural network optimizer

    def __init__(self,name):
        self.name = name
        self.mutation_chance = 0.03
        self.network = None
        self.logger = get_logger("Bot")


    def create_nn(self):
        n_inputs = 16
        n_nodes_h1 = 32
        n_nodes_h2 = 48
        n_nodes_h3 = 32
        n_classes = 4

        hidden_layer_1 = {'weights': np.random.random((n_inputs, n_nodes_h1)),
                               'biases': np.random.random(n_nodes_h1)}
        hidden_layer_2 = {'weights': np.random.random((n_nodes_h1, n_nodes_h2)),
                               'biases': np.random.random(n_nodes_h2)}
        hidden_layer_3 = {'weights': np.random.random((n_nodes_h2, n_nodes_h3)),
                               'biases': np.random.random(n_nodes_h3)}
        hidden_layer_4 = {'weights': np.random.random((n_nodes_h3, n_classes)),
                               'biases': np.random.random(n_classes)}

        if self.network is None:
            self.network = list()

        self.network.append(hidden_layer_1)
        self.network.append(hidden_layer_2)
        self.network.append(hidden_layer_3)
        self.network.append(hidden_layer_4)

    def predict_nn(self,data):
        if self.network is None:
            raise ValueError('The Neural Network model has not been initialized. Use create_nn() to initialize')

        l1 = np.add(np.dot(data,self.network[0]['weights']), self.network[0]['biases'])
        l1 = np.array([max(value,0) for value in l1])

        l2 = np.add(np.dot(l1,self.network[1]['weights']), self.network[1]['biases'])
        l2 = np.array([max(value,0) for value in l2])

        l3 = np.add(np.dot(l2,self.network[2]['weights']), self.network[2]['biases'])
        l3 = np.array([max(value,0) for value in l3])

        output = np.add(np.dot(l3,self.network[3]['weights']), self.network[3]['biases'])
        output = np.array([max(value,0) for value in output])

        return output

    def mutate_nn(self):

        for layer in self.network:
            for value in layer['weights']:
                for i in value:
                    if self.mutation_chance >= np.random.random():
                        i = np.random.random()
            for value in layer['biases']:
                for i in value:
                    if self.mutation_chance >= np.random.random():
                        i = np.random.random()

# class Game:
#
#     #TODO: The tiles are initialized to zero. This needs to be changed I guess
#     def __init__(self):
#         self.matrix = np.zeros((4, 4))
#
#     def is_game_over(self,move):
#         if move == 'up'
#
#         elif move == 'down':
#
#         elif move == 'left':
#
#         else:
#
#     def update_matrix(self,move):

class Optimizer:

    def __init__(self,n_generations, n_children):
        self.n_children = n_children
        self.bots = [Bot('Bot Number'+str(i+1)) for i in range(n_children)]
        self.n_generations = n_generations


    def main(self):
        for gen in range(self.n_generations):
            logger.info('Running generation number {}'.format(gen))
