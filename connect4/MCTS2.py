import random
import sys
import math
import time
from copy import deepcopy

import numpy as np
import pygame

from CalculateTime import get_valid_location, winning_move, is_valid_location, get_next_open_row, drop_piece, \
    print_board

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
        self.board = create_board()
        self.to_play = PLAYER
        self.height = [ROW_COUNT - 1] * COLUMN_COUNT
        self.last_played = []

    def move(self, col):
        self.board[self.height[col]][col] = self.to_play
        self.last_played = [self.height[col], col]
        self.height[col] -= 1
        self.to_play = AI if self.to_play == PLAYER else PLAYER

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece
        return self.board

    def is_valid_location(self, col):
        return self.board[ROW_COUNT - 1][col] == 0

    def get_next_open_row(self, col):
        for r in range(ROW_COUNT):
            if self.board[r][col] == 0:
                return r

    def print_board(self):
        print(np.flip(self.board, 0))

    def winning_move(self, piece):
        # check horizontal
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT):
                if self.board[r][c] == piece and self.board[r][c + 1] == piece and self.board[r][c + 2] == piece and self.board[r][
                    c + 3] == piece:
                    return True

        # check vertical
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c] == piece and self.board[r + 2][c] == piece and self.board[r + 3][
                    c] == piece:
                    return True

        # check positively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(ROW_COUNT - 3):
                if self.board[r][c] == piece and self.board[r + 1][c + 1] == piece and self.board[r + 2][c + 2] == piece and \
                        self.board[r + 3][c + 3] == piece:
                    return True

        # check negatively sloped diaganols
        for c in range(COLUMN_COUNT - 3):
            for r in range(3, ROW_COUNT):
                if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and \
                        board[r - 3][c + 3] == piece:
                    return True

    def is_ternimal_node(self):  # game over
        return winning_move(self, PLAYER_PIECE) or winning_move(self, AI_PIECE) or len(get_valid_location(self)) == 0

    def get_valid_location(self):  # get all the cols that could drop pieces
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if is_valid_location(self, col):
                valid_locations.append(col)
        return valid_locations

    def get_outcome(self):
        if winning_move(self, PLAYER_PIECE):
            return PLAYER_WIN
        if winning_move(self, AI_PIECE):
            return AI_WIN
        if len(get_valid_location(self)) == 0:
            return DRAW


def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board




class Node:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.N = 0
        self.Q = 0
        self.children = {}
        self.outcome = -1

    def add_children(self, children: dict) -> None:
        for child in children:
            self.children[child.move] = child

    def value(self, explore: float = EXPLORATION):
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

            node = random.choice(max_nodes)
            state.move(node.move)

            if node.N == 0:
                return node, state

        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.move(node.move)

        return node, state

    def expand(self, parent: Node, state: ConnectState) -> bool:
        if state.is_ternimal_node():
            return False

        children = [Node(move, parent) for move in state.get_valid_location()]
        parent.add_children(children)

        return True

    def roll_out(self, state: ConnectState) -> int:
        while not state.is_ternimal_node():
            state.move(random.choice(get_valid_location(board)))

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
        if self.root_state.game_over():
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)

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






state = ConnectState()
mcts = MCTS(state)
#print(board)

game_over = False
turn = 0

pygame.init()

SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)
draw_board(state.board)
pygame.display.update()

myfont = pygame.font.SysFont("monospace", 75)

turn = random.randint(PLAYER,AI)

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
            0
            if turn == PLAYER:
                posx = event.pos[0]
                col = int(math.floor(posx/SQUARESIZE) )
                #col = int(input("Player 1 make your selection (0-6): "))

                if state.is_valid_location(col):
                    row = state.get_next_open_row(col)
                    board = state.drop_piece(row, col, PLAYER_PIECE)
                    mcts = MCTS(board)

                    if state.winning_move(PLAYER_PIECE):
                        label = myfont.render("Player 1 win", 1, RED)
                        screen.blit(label, (40,10))
                        print("Player 1 win")
                        game_over = True

                    turn += 1
                    turn = turn % 2

                    state.print_board()
                    draw_board(state.board)


    if turn == AI and not game_over:
        #col, minimax_score = minimax_ab(board, 3, True, -math.inf, math.inf)
        #col, minimax_score = minimax(board,3,True)
        mcts.search(15)
        num_rollouts, run_time = mcts.statistics()
        col = mcts.best_move()
        #col = pick_last_move(board, AI_PIECE)
        #col = random.randint(0,COLUMN_COUNT-1)
        # col = int(input("Player 2 make your selection (0-6): "))
        if state.is_valid_location(col):
            pygame.time.wait(500)
            row = state.get_next_open_row(col)
            board = state.drop_piece(row, col, AI_PIECE)
            mcts = MCTS(board)

            if state.winning_move(AI_PIECE):
                label = myfont.render("Player 2 win", 1, YELLOW)
                screen.blit(label, (40, 10))
                print("Player 2 win")
                game_over = True

            state.print_board()
            draw_board(state.board)

            turn += 1
            turn = turn % 2

    if(game_over):
        pygame.time.wait(3000)
