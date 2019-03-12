import numpy as np
import pygame
import random
import sys
from repos.colours import Colors
from pygame.locals import QUIT,KEYDOWN
from utils import get_logger


class Game:

    def __init__(self,loaded_game = False,show = True, bot=True):
        self.show = show
        self.bot = bot
        self.total_points = 0
        self.board_size = 4
        self.logger = get_logger("GAME")
        self.previous_tile = None

        pygame.init()
        if not bot and not show:
            raise ValueError("A human cannot operate without the visual screen")
        if self.show:
            self.surface = pygame.display.set_mode((500, 500), 1, 32)
            pygame.display.set_caption("2048")
            self.myfont = pygame.font.SysFont("monospace", 25)
            self.scorefont = pygame.font.SysFont("monospace", 50)

        if not loaded_game:
            self.tiles = np.zeros((4, 4),dtype=int)
        else:
            if not isinstance(loaded_game,(list,np.ndarray)):
                raise TypeError("Please input a list of numpy array")
            self.tiles = np.array(loaded_game)

        self.place_random_tile()
        self.place_random_tile()

        if self.show:
            self.show_matrix()
        self.undo = list()

    def show_matrix(self):
        self.surface.fill(Colors.BLACK)

        for i in range(0,self.board_size):
            for j in range(0,self.board_size):
                pygame.draw.rect(self.surface, Colors().getcolour(self.tiles[i][j]),
                                 (i * (400 / self.board_size), j * (400 / self.board_size) + 100, 400 / self.board_size,
                                  400 / self.board_size))

                label = self.myfont.render(str(self.tiles[i][j]), 1, (255, 255, 255))
                label2 = self.scorefont.render("Score:" + str(self.total_points), 1, (255, 255, 255))

                self.surface.blit(label, (i * (400 / self.board_size) + 30, j * (400 / self.board_size) + 130))
                self.surface.blit(label2, (10, 20))

    def show_game_over(self):

        self.surface.fill(Colors.BLACK)

        label = self.scorefont.render("Game Over!", 1, (255, 255, 255))
        label2 = self.scorefont.render("Score:" + str(self.total_points), 1, (255, 255, 255))
        label3 = self.myfont.render("Press r to restart!", 1, (255, 255, 255))

        self.surface.blit(label, (50, 100))
        self.surface.blit(label2, (50, 200))
        self.surface.blit(label3, (50, 300))

    @staticmethod
    def get_rotations(k):
        if k == pygame.K_UP:
            return 0
        elif k == pygame.K_DOWN:
            return 2
        elif k == pygame.K_LEFT:
            return 1
        elif k == pygame.K_RIGHT:
            return 3

    def rotate_matrix(self):
        for i in range(0, int(self.board_size / 2)):
            for k in range(i, self.board_size - i - 1):
                temp1 = self.tiles[i][k]
                temp2 = self.tiles[self.board_size - 1 - k][i]
                temp3 = self.tiles[self.board_size - 1 - i][self.board_size - 1 - k]
                temp4 = self.tiles[k][self.board_size - 1 - i]

                self.tiles[self.board_size - 1 - k][i] = temp1
                self.tiles[self.board_size - 1 - i][self.board_size - 1 - k] = temp2
                self.tiles[k][self.board_size - 1 - i] = temp3
                self.tiles[i][k] = temp4

    @staticmethod
    def is_arrow(k):
        return k in [pygame.K_UP,pygame.K_DOWN,pygame.K_LEFT,pygame.K_RIGHT]

    def move_tiles(self):
        for i in range(self.board_size):
            for j in range(self.board_size-1):
                while self.tiles[i][j] == 0 and sum(self.tiles[i][j:]) > 0:
                    for k in range(j,self.board_size-1):
                        self.tiles[i][k] = self.tiles[i][k+1]
                    self.tiles[i][self.board_size-1] = 0

    def merge_tiles(self):
        for i in range(0, self.board_size):
            for k in range(0, self.board_size - 1):
                if self.tiles[i][k] == self.tiles[i][k + 1] and self.tiles[i][k] != 0:
                    self.tiles[i][k] = self.tiles[i][k] * 2
                    self.tiles[i][k + 1] = 0
                    self.total_points += self.tiles[i][k]
                    self.move_tiles()

    def place_random_tile(self):
        # TODO: Always 2 is being added. Probabilistically 4 should also be added
        positions = list()
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.tiles[i][j] == 0:
                    positions.append((i,j))
        tile_x,tile_y = random.choice(positions)
        self.tiles[tile_x][tile_y] = 2

    def check_game_status(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.tiles[i][j] == 0:
                    return True

                try:
                    if self.tiles[i][j] == self.tiles[i][j + 1]:
                        return True
                except IndexError:
                    pass

                try:
                    if self.tiles[i][j] == self.tiles[i+1][j]:
                        return True
                except IndexError:
                    pass
        return False

    def human_player(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if self.check_game_status():
                    if event.type == KEYDOWN:
                        if self.is_arrow(event.key):
                            rotations = self.get_rotations(event.key)

                            for i in range(rotations):
                                self.rotate_matrix()

                            self.move_tiles()
                            self.merge_tiles()

                            for j in range(4-rotations):
                                self.rotate_matrix()

                            if self.previous_tile is None or (self.previous_tile is not None and not np.array_equal(self.previous_tile,self.tiles)):
                                try:
                                    self.place_random_tile()
                                    self.previous_tile = self.tiles.copy()
                                except IndexError:
                                    pass

                            if self.show:
                                self.show_matrix()

                            self.logger.info("Total score is {}".format(self.total_points))

                else:
                    return self.total_points
            pygame.display.update()

    def bot_simulation(self,key):
        rotations = self.get_rotations(key[0])

        for i in range(rotations):
            self.rotate_matrix()

        self.move_tiles()
        self.merge_tiles()

        for j in range(4-rotations):
            self.rotate_matrix()

        if self.show:
            self.show_matrix()
            pygame.display.update()

        if self.previous_tile is None or (
                self.previous_tile is not None and not np.array_equal(self.previous_tile, self.tiles)):
            try:
                self.place_random_tile()
                self.previous_tile = self.tiles.copy()
            except IndexError:
                return 300, np.amax(self.tiles)
        else:
            return 300, np.amax(self.tiles)
        if self.check_game_status():
            return None
        else:
            return 500, np.amax(self.tiles)

    def run(self,*args):
        if self.bot:
            return self.bot_simulation(key = args)
        else:
            self.human_player()


class HumanBot:

    def __init__(self):
        self.mapping = {0: pygame.K_UP, 1: pygame.K_LEFT, 2: pygame.K_DOWN, 3: pygame.K_RIGHT}

    def run(self):
        game = Game(show=False, bot=True)
        for i in range(10):
            print(game.tiles.transpose())
            direction = int(input())
            try:
                direction = self.mapping[direction]
            except KeyError:
                raise ValueError("Please enter 0 for up, 1 for left, 2 for down, 3 for right")
            game.run(direction)


# if __name__== "__main__":
# #    Game(show=True,bot=False).run()
#     HumanBot().run()