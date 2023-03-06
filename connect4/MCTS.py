import random
import sys
import math
import time
from copy import deepcopy

import numpy as np
import pygame



ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

PLAYER_WIN = 1
AI_WIN = 2
DRAW = 3

WINDOW_LENGTH = 4

EXPLORATION = math.sqrt(2)

BLUE = (0,0,255)
BLACK = (0,0,0)
RED =  (255,0,0)
YELLOW = (255,255,0)

class ConnectState:
    def __init__(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))
        self.to_play = -1

    def create_board(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT))
        return self.board

    def drop_piece(self, row, col, piece):
        # Drop a piece in the specified row and column
        self.board[row][col] = piece
        return self.board

    def is_valid_location(self, col):
        # Check if the top row in the specified column is empty
        return self.board[ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        # Find the lowest empty row in the specified column
        for r in range(ROW_COUNT):
            if self.board[r][col] == 0:
                return r
        return -1

    def move(self, col):
        row = self.get_next_open_row(col)
        piece = AI_PIECE if self.to_play == AI else PLAYER_PIECE
        self.board = self.drop_piece(row, col, piece)
        self.to_play = AI if self.to_play == PLAYER else PLAYER


    def winning_move(self, piece):
        # Check if the specified piece has won the game
        # Check horizontally
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece and self.board[r][c + 2] == piece and self.board[r][c + 3] == piece:
                    return True
        # Check vertically
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece and self.board[r + 2][c] == piece and self.board[r + 3][c] == piece:
                    return True
        # Check diagonally (positive slope)
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and self.board[r + 2][c + 2] == piece and self.board[r + 3][c + 3] == piece:
                    return True
        # Check diagonally (negative slope)
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if self.board[r][c] == piece and self.board[r - 1][c + 1] == piece and self.board[r - 2][c + 2] == piece and self.board[r - 3][c + 3] == piece:
                    return True
        return False

    def is_terminal_node(self):  # Game over
        return self.winning_move(PLAYER_PIECE) or self.winning_move(AI_PIECE) or len(self.get_valid_locations()) == 0

    def get_valid_locations(self):  # Get all the columns that could drop pieces
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if self.is_valid_location(col):
                valid_locations.append(col)
        return valid_locations

    def get_outcome(self):
        if self.winning_move(PLAYER_PIECE):
            return PLAYER_WIN
        elif self.winning_move(AI_PIECE):
            return AI_WIN
        else:
            return DRAW


class Node:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.N = 0      #visit time
        self.Q = 0      #rewards
        self.children = {}
        self.outcome = -1

    def add_children(self, children: dict) -> None:
        for child in children:
            self.children[child.move] = child


    def value(self, explore: float = math.sqrt(2)):
        if self.N == 0:
            return 0 if explore == 0 else float('inf')
        else:
            return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)


class MCTS:
    def __init__(self, state=ConnectState()):
        self.root_state = deepcopy(state)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    def select_node(self) -> tuple:
        node = self.root
        state = deepcopy(self.root_state)

        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value()).value()
            max_nodes = [n for n in children if n.value() == max_value]
            #max_nodes = {n.move: n for n in children if n.value() == max_value}


            node = random.choice(max_nodes)
            state.move(node.move)

            if node.N == 0:
                return node, state

        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.move(node.move)

        return node, state

    def expand(self, parent: Node, state: ConnectState) -> bool:
        if state.is_terminal_node():
            return False

        children = [Node(move, parent) for move in state.get_valid_locations()]
        parent.add_children(children)

        return True

    def roll_out(self, state: ConnectState) -> int:
        while not state.is_terminal_node():
            state.move(random.choice(state.get_valid_locations()))

        return state.get_outcome()

    def back_propagate(self, node: Node, turn: int, outcome: int) -> None:
        # For the current player, not the next player
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            if outcome == DRAW:
                reward = 0
            else:
                reward = 1 - reward

    def search(self, time_limit: int):
        start_time = time.process_time()

        num_rollouts = 0
        while time.process_time() - start_time < time_limit:
            node, state = self.select_node()
            outcome = self.roll_out(state)
            self.back_propagate(node, state.to_play, outcome)
            num_rollouts += 1

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts

    def best_move(self):
        if self.root_state.is_terminal_node():
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)
        print("max_value:" + str(max_value))
        print(best_child.N)

        return best_child.move

    def move(self, move):
        if move in self.root.children:
            self.root_state.move(move)
            self.root = self.root.children[move]
            return

        self.root_state.move(move)
        self.root = Node(None, None)

    def statistics(self) -> tuple:
        return self.num_rollouts, self.run_time



def print_board(board):
    print(np.flip(board, 0))




def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c*SQUARESIZE, r*SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (int(c*SQUARESIZE + SQUARESIZE/2), int(r*SQUARESIZE + SQUARESIZE + SQUARESIZE/2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, RED, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, YELLOW, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    pygame.display.update()








#board = create_board()
connect_state = ConnectState()
mcts = MCTS(connect_state)
#print(board)

game_over = False


pygame.init()

SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)
draw_board(connect_state.board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)

turn = random.randint(PLAYER,AI)
#turn = PLAYER
connect_state.to_play = turn

while not game_over:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEMOTION:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            posx = event.pos[0]
            if turn == PLAYER:
                pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)   #球跟随鼠标移动
            pygame.display.update()

        if event.type == pygame.MOUSEBUTTONDOWN:
            pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx/SQUARESIZE) )
                #col = int(input("Player 1 make your selection (0-6): "))

                if connect_state.is_valid_location(col):
                    #row = connect_state.get_next_open_row(col)
                    connect_state.move(col)
                    mcts.move(col)



                    if connect_state.winning_move(PLAYER_PIECE):
                        label = myfont.render("Player 1 win", True, RED)
                        screen.blit(label, (40,10))
                        print("Player 1 win")
                        game_over = True

                    turn += 1
                    turn = turn % 2

                    print_board(connect_state.board)
                    draw_board(connect_state.board)


    if turn == AI and not game_over:
        mcts.search(1)
        num_rollouts, run_time = mcts.statistics()
        print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
        col = mcts.best_move()
        #col = random.randint(0,COLUMN_COUNT-1)
        # col = int(input("Player 2 make your selection (0-6): "))
        if connect_state.is_valid_location(col):
            #pygame.time.wait(500)
            #row = connect_state.get_next_open_row(col)
            connect_state.move(col)
            mcts.move(col)


            if connect_state.winning_move(AI_PIECE):
                label = myfont.render("Player 2 win", True, YELLOW)
                screen.blit(label, (40, 10))
                print("Player 2 win")
                game_over = True

            print_board(connect_state.board)
            draw_board(connect_state.board)

            turn += 1
            turn = turn % 2


    if(game_over):
        pygame.time.wait(3000)
