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

def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board

def drop_piece(board, row, col, piece):
    board[row][col] = piece
    return board

def is_valid_location(board, col):
    return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
    return -1

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    #check horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True

    #check vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True

    #check positively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True

    #check negatively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True




def is_ternimal_node(board):    #game over
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_location(board))==0

def get_valid_location(board):   #get all the cols that could drop pieces
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def get_outcome(board):
    if winning_move(board, PLAYER_PIECE):
        return PLAYER_WIN
    if winning_move(board, AI_PIECE):
        return  AI_WIN
    if len(get_valid_location(board))==0:
        return DRAW


class MCTS:
    class Node:
        def __init__(self, state, parent=None, action=None):
            self.state = deepcopy(state)
            self.parent = parent
            self.action = action
            self.children = []
            self.visits = 0
            self.wins = 0

        def is_fully_expanded(self):
            return len(self.get_untried_actions()) == 0

        def get_untried_actions(self):
            return [col for col in range(COLUMN_COUNT) if is_valid_location(self.state, col)]

        def get_best_child(self, c):
            best_score = -1
            best_child = None

            for child in self.children:
                if(child.visits != 0):
                    score = child.wins / child.visits + c * math.sqrt(2 * math.log(self.visits) / child.visits)
                else:
                    score = 0 if len(self.get_untried_actions()) == 0 else float('inf')
                if score > best_score:
                    best_score = score
                    best_child = child
            return best_child


    def __init__(self, state, c=1):
        self.root = self.Node(state)
        self.c = c

    def selection(self, node):
        while not is_ternimal_node(node.state):
            if not node.is_fully_expanded():
                return self.expansion(node)
            else:
                node = node.get_best_child(self.c)
        return node

    def expansion(self, node):
        untried_actions = node.get_untried_actions()
        action = random.choice(untried_actions)
        new_state = deepcopy(node.state)

        if(get_next_open_row(new_state, action) == -1) :
            pass

        drop_piece(new_state, get_next_open_row(new_state, action), action, AI_PIECE)
        new_node = self.Node(new_state, parent=node, action=action)
        node.children.append(new_node)
        return new_node

    def simulation(self, node):
        state = deepcopy(node.state)
        player = AI
        while not is_ternimal_node(state):
            valid_locations = get_valid_location(state)
            if len(valid_locations) == 0:
                break
            if player == PLAYER:
                action = random.choice(valid_locations)
                drop_piece(state, get_next_open_row(state, action), action, PLAYER_PIECE)
                player = AI
            else:
                col = self.selection(self.root).action
                drop_piece(state, get_next_open_row(state, col), col, AI_PIECE)
                player = PLAYER
        outcome = get_outcome(state)
        if outcome == AI_WIN:
            return 1
        elif outcome == PLAYER_WIN:
            return 0
        else:
            return 0.5

    def backpropagation(self, node, outcome):
        while node is not None:
            node.visits += 1
            node.wins += outcome
            node = node.parent

    def search(self, num_iterations):
        for i in range(num_iterations):
            node = self.selection(self.root)
            outcome = self.simulation(node)
            self.backpropagation(node, outcome)

    def best_move(self):
        best_child = self.root.get_best_child(1)
        return best_child.action






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






board = create_board()
mcts = MCTS(board)

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
draw_board(board)
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

                if is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    board = drop_piece(board, row, col, PLAYER_PIECE)
                    mcts = MCTS(board)


                    if winning_move(board, PLAYER_PIECE):
                        label = myfont.render("Player 1 win", 1, RED)
                        screen.blit(label, (40,10))
                        print("Player 1 win")
                        game_over = True

                    turn += 1
                    turn = turn % 2

                    print_board(board)
                    draw_board(board)


    if turn == AI and not game_over:
        mcts.search(1000)
        col = mcts.best_move()
        #col = random.randint(0,COLUMN_COUNT-1)
        # col = int(input("Player 2 make your selection (0-6): "))
        if is_valid_location(board, col):
            pygame.time.wait(500)
            row = get_next_open_row(board, col)
            board = drop_piece(board, row, col, AI_PIECE)
            mcts = MCTS(board)


            if winning_move(board, AI_PIECE):
                label = myfont.render("Player 2 win", 1, YELLOW)
                screen.blit(label, (40, 10))
                print("Player 2 win")
                game_over = True

            print_board(board)
            draw_board(board)

            turn += 1
            turn = turn % 2

    if(game_over):
        pygame.time.wait(3000)
