import random
import numpy as np
from ga_2048.utils import get_logger
import pygame


class NNBot:

    def __init__(self, name, matrix=None, disable_logs = True):
        self.name = name
        self.mutation_chance = 0.03
        self.network = matrix
        if self.network is None:
            self.network = self.create_nn()
        else:
            self.network = matrix
        self.network = matrix
        self.logger = get_logger("NN Bot")
        self.logger.disabled = disable_logs
        self.mapping = {0: pygame.K_UP,
                        1: pygame.K_LEFT,
                        2: pygame.K_DOWN,
                        3: pygame.K_RIGHT}

    def create_nn(self):
        # TODO: Read up research on creating NN architecture
        n_inputs = 16
        n_nodes_h1 = 32
        n_nodes_h2 = 4
        n_nodes_h3 = 4
        n_classes = 4

        hidden_layer_1 = {'weights': np.random.random((n_inputs, n_nodes_h1)),
                               'biases': np.random.random(n_nodes_h1)}
        hidden_layer_2 = {'weights': np.random.random((n_nodes_h1, n_nodes_h2)),
                               'biases': np.random.random(n_nodes_h2)}
        hidden_layer_3 = {'weights': np.random.random((n_nodes_h2, n_nodes_h3)),
                               'biases': np.random.random(n_nodes_h3)}
        hidden_layer_4 = {'weights': np.random.random((n_nodes_h3, n_classes)),
                               'biases': np.random.random(n_classes)}

        network = list()

        network.append(hidden_layer_1)
        network.append(hidden_layer_2)
        network.append(hidden_layer_3)
        network.append(hidden_layer_4)
        return network

    def predict(self, data):
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
        output = [[index,value] for index,value in enumerate(output)]
        output.sort(key=lambda x: x[1],reverse=True)
        return [self.mapping[value[0]] for value in output]

    def mutate(self):

        for a,layer in enumerate(self.network):
            for b,row in enumerate(layer['weights']):
                    for c,col in enumerate(row):
                        if self.mutation_chance >= np.random.random():
                            self.network[a]['weights'][b][c] = random.uniform(-1,1)
            for b,element in enumerate(layer['biases']):
                if self.mutation_chance >= np.random.random():
                    self.network[a]['biases'][b] = np.random.random()
