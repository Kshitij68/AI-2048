import numpy as np
import random
import pygame
import tensorflow as tf
from ga_2048.utils import get_logger

# TODO: Add Asserts and test cases


class CNNBot:

    def __init__(self, name, matrix=None, disable_logs=True,mutation = 0.3,convolutions = 1,seed=None):
        self.seed = seed
        self.name = name
        self.mutation = mutation
        self.convolutions = convolutions
        self.matrix = matrix
        self.logger = get_logger("CNN Bot")
        self.logger.disabled = disable_logs
        self.mapping = {0: pygame.K_UP,
                        1: pygame.K_LEFT,
                        2: pygame.K_DOWN,
                        3: pygame.K_RIGHT}

        self.W1 = None
        self.Z1 = None
        self.A1 = None
        self.P1 = None
        self.P2 = None

    @staticmethod
    def create_placeholders(a, b, c, d):
        return tf.placeholder(tf.float32, [a, b, c, d])

    def generate_random_matrix(self):
        matrix = np.zeros((2,2))
        for a,row in enumerate(matrix):
            for b,col in enumerate(matrix):
                if self.seed:
                    random.seed(self.seed+a+b)
                matrix[a][b] = random.uniform(-1, 1)
        return matrix.reshape(2,2,1,1)

    def parameter_initialization(self):
        if self.convolutions == 2:
            if self.matrix is None:
                matrix_1 = self.generate_random_matrix()
                matrix_2 = self.generate_random_matrix()
                self.matrix = [matrix_1,matrix_2]
            else:
                matrix_1 = self.matrix[0]
                matrix_2 = self.matrix[1]

            W1 = tf.constant(matrix_1, dtype='float32')

            W2 = tf.constant(matrix_2, dtype='float32')

            return {"W1": W1,'W2':W2}

        elif self.convolutions == 1:

            if self.matrix is None:
                matrix_1 = self.generate_random_matrix()
                self.matrix = [matrix_1]
            else:
                matrix_1 = self.matrix[0]

            W1 = tf.constant(matrix_1, dtype='float32')

            return {"W1": W1}

    def forward_propogation(self, X, network):

        self.W1 = network['W1']

        self.Z1 = tf.nn.conv2d(X, self.W1, strides=[1, 1, 1, 1], padding='VALID')

        self.A1 = tf.nn.relu(self.Z1)

        self.P1 = tf.nn.max_pool(self.A1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')

        self.P2 = tf.contrib.layers.flatten(self.P1)

        return self.P2[0]

    def predict(self, input_matrix):
        """

        :param input_matrix: A numpy matrix of shape (4,4)
        :return:
        """
        input_matrix = input_matrix.reshape(1, 4, 4, 1)
        tf.reset_default_graph()

        with tf.Session() as sess:
            X = self.create_placeholders(1, 4, 4, 1)
            parameters = self.parameter_initialization()
            graph = self.forward_propogation(X,parameters)
            output = sess.run(graph, {X: input_matrix})
        output = [[index,value] for index,value in enumerate(output)]
        output.sort(key=lambda x: x[1], reverse=True)
        return [self.mapping[value[0]] for value in output]

    def mutate(self):
        for a,convolution in enumerate(self.matrix):
            for b,row in enumerate(convolution):
                for c,col in enumerate(row):
                    if self.mutation >= np.random.random():
                        self.logger.info("Mutating")
                        self.matrix[a][b][c][0][0] = random.uniform(-1,1)

    def show_matrix(self):
        if self.convolutions == 1:
            return [
                self.matrix[0][0][0][0][0],
                self.matrix[0][0][1][0][0],
                self.matrix[0][1][0][0][0],
                self.matrix[0][1][1][0][0],
            ]
        else:
            raise NotImplementedError()
