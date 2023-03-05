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
            # we prioritize nodes that are not explored
            return 0 if explore == 0 else float('inf')
        else:
            return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)

class MCTS:
    def __init__(self, board):
        self.root_board = deepcopy(board)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.to_play = PLAYER

    def select_node(self) -> tuple:
        node = self.root
        board = deepcopy(self.root_board)

        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value()).value()
            max_nodes = [n for n in children if n.value() == max_value]

            node = random.choice(max_nodes)
            drop_piece(board,get_next_open_row(board, node.move),node.move,AI_PIECE)

            if node.N == 0:
                return node, board

        if self.expand(node, board):
            node = random.choice(list(node.children.values()))
            drop_piece(board,get_next_open_row(board, node.move),node.move,AI_PIECE)

        return node, board

    def expand(self, parent: Node, board) -> bool:
        if is_ternimal_node(board):
            return False

        children = [Node(move, parent) for move in get_valid_location(board)]
        parent.add_children(children)

        return True

    def roll_out(self, board) -> int:
        state = deepcopy(board)
        player = PLAYER
        while not is_ternimal_node(state):
            valid_locations = get_valid_location(state)
            if len(valid_locations) == 0:
                break
            if player == PLAYER:
                action = random.choice(valid_locations)
                s1 = drop_piece(state, get_next_open_row(state, action), action, PLAYER_PIECE)
                player = AI
            else:
                col = random.choice(get_valid_location(s1))
                drop_piece(state, get_next_open_row(state, col), col, AI_PIECE)
                player = PLAYER
        outcome = get_outcome(state)
        if outcome == AI_WIN:
            return 1
        elif outcome == PLAYER_WIN:
            return -1
        else:
            return 0


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
            node, board = self.select_node()
            outcome = self.roll_out(board)
            self.back_propagate(node,  AI if self.to_play == PLAYER else PLAYER, outcome)
            num_rollouts += 1

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts

    def best_move(self):
        if is_ternimal_node(self.root_board):
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)

        return best_child.move



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
        #col, minimax_score = minimax_ab(board, 3, True, -math.inf, math.inf)
        #col, minimax_score = minimax(board,3,True)
        mcts.search(8)
        num_rollouts, run_time = mcts.statistics()
        col = mcts.best_move()
        #col = pick_last_move(board, AI_PIECE)
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
