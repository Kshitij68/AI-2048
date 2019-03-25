from ga_2048.utils import get_logger
from ga_2048.CNNBot import CNNBot
from ga_2048.NNBot import NNBot
from ga_2048.simulator import Game


# TODO: Add Asserts and test cases
# TODO: The CNN has to become much faster to train fast
# TODO: The model is broken for Neural Networks. Fix to make both work


class Optimizer:

    def __init__(self, n_generations, n_children, convolutions=1, network='CNN',
                 disable_game_logs=True, disable_optimizer_logs=True, disable_model_logs=True):

        self.disable_game_logs = disable_game_logs
        self.disable_optimizer_logs = disable_optimizer_logs
        self.disable_model_logs = disable_model_logs

        self.network = network
        self.convolutions = convolutions

        if n_children < 10:
            raise ValueError("Please provide number of children to be atleast 10")
        self.n_children = n_children

        self.logger = get_logger("Optimizer")
        self.logger.disabled = self.disable_optimizer_logs

        self.bots = list()
        for i in range(self.n_children):
            bot = self.create_bot(name='bot number' + str(i))
            self.bots.append(bot)

        self.high_score = 0

        self.n_generations = n_generations

        self.survival_rate = 0.2
        self.new_children = int(
            (self.n_children - (self.survival_rate * self.n_children)) / (self.survival_rate * self.n_children))
        self.logger.info("Will be creating {} children per bot".format(self.new_children))

    def create_bot(self, name, parameters=None):
        if self.network == 'NN':
            return NNBot(name=name, disable_logs=self.disable_model_logs, parameters=parameters)
        elif self.network == 'CNN':
            return CNNBot(name=name, disable_logs=self.disable_model_logs, parameters=parameters)
        else:
            raise ValueError("The current model only supports NN and CNN. Please pass 'NN' for Neural Network "
                             "architecture and 'CNN' for Convolution Neural Network based architecture ")

    def get_high_scores(self):
        bot_scores = list()
        for index, child in enumerate(self.bots):
            game = Game(show=False, disable_logs=self.disable_game_logs)
            while True:
                prediction = self.bots[index].predict(game.tiles.transpose())
                output = game.run(prediction[0])
                if output is not None:
                    if output[0] == 300:
                        output = game.run(prediction[1])
                        if output is not None:
                            if output[0] == 300:
                                output = game.run(prediction[2])
                                if output is not None:
                                    if output[0] == 300:
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
                        bot_scores.append([index, game.total_points, output[1]])
                        break
        bot_scores.sort(key=lambda x: x[1], reverse=True)
        return bot_scores

    def create_new_bots(self, indexes):
        indexes_params = [self.bots[i].parameters.copy() for i in indexes]
        bot_index = 0
        for param in indexes_params:
            self.bots[bot_index] = self.create_bot(name='Bot', parameters=param.copy())
            bot_index += 1
            for j in range(self.new_children):
                self.bots[bot_index] = self.create_bot(name='Bot', parameters=param.copy())
                self.bots[bot_index].mutate()
                bot_index += 1

    def main(self):
        for gen in range(self.n_generations):
            self.logger.info('Running generation number {} with total {} bots'.format(gen, len(self.bots)))
            scores = self.get_high_scores()[:int(self.survival_rate * self.n_children)]
            survived_bots_indexes = set([value[0] for value in scores])
            self.logger.info("The highest score obtained is: {}".format(scores[0][1]))
            self.create_new_bots(indexes=survived_bots_indexes)


if __name__ == "__main__":
    optimize = Optimizer(n_generations=1000, n_children=10, convolutions=1, network='CNN', disable_game_logs=True,
                         disable_optimizer_logs=False, disable_model_logs=True)
    optimize.main()
