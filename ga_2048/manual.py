from ga_2048.utils import get_logger
from ga_2048.CNNBot import CNNBot
from ga_2048.NNBot import NNBot
from ga_2048.my_2048 import Game

# TODO: Add Asserts and test cases
# TODO: The CNN has to become much faster to train fast
# TODO: The model is broken for Neural Networks. Fix to make both work

class Optimizer:

    def __init__(self,n_generations, n_children,convolutions=1,network='CNN'):
        if network == 'NN':
            self.model = NNBot
        else:
            self.model = CNNBot
        self.convolutions = convolutions
        if n_children < 10:
            raise ValueError("Please provide number of children to be atleast 10")
        self.n_children = n_children
        self.logger = get_logger("Optimizer")
        self.bots = list()
        for i in range(self.n_children):
            bot = self.model('Bot Number' + str(i + 1))
            self.bots.append(bot)
        self.high_score = 0
        self.n_generations = n_generations
        self.survival_rate = 0.2
        self.new_children = int((self.n_children - (self.survival_rate*self.n_children))/(self.survival_rate*self.n_children))
        self.logger.info("Will be creating {} children per bot".format(self.new_children))

    def get_high_scores(self):
        bot_scores = list()
        disable_logs = True
        for index,child in enumerate(self.bots):
            game = Game(show = False, disable_logs=disable_logs)
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
            self.logger.info("Scores are {}".format(scores))
            for index,bot in enumerate(self.bots):
                if index in survived_bots_indexes:
                    new_bots.append(bot)
                    if self.convolutions == 1:
                        bot_matrix = bot.matrix[0]
                    else:
                        raise NotImplementedError()
                        # TODO: This method is jugaddoo. Get the elements and then remake the numpy matrix to avoid pass by reference
                    for _ in range(self.new_children):
                        new_bot = self.model(name='Bot Number ' + str(child_index + 1),matrix=[bot_matrix.copy()])
                        new_bot.mutate()
                        new_bots.append(new_bot)
                        child_index += 1
            self.bots = new_bots.copy()
        return None


if __name__== "__main__":
    optimize = Optimizer(n_generations=1000,n_children=10,convolutions=1,network='CNN')
    optimize.main()
