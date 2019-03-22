from ga_2048.simulator import Game
import unittest
import numpy as np


class TestGame(unittest.TestCase):

    # def setUpClass(cls):
    #     pass
    #
    # def setUp(self):
    #     pass

    def test_check_game_status(self):
        game = Game()
        game.tiles = np.array([
            [8, 16, 64, 8],
            [2, 4, 16, 64],
            [4, 2, 16, 64],
            [2, 2, 8, 2]
        ])
        self.assertEqual(game.check_game_status(),True)

