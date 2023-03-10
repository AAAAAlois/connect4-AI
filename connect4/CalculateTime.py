import random
import sys
import math
import time

import numpy as np
import pygame

NUM_RUNS = 50

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

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


def evaluate_window(window, piece):
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE
    score = 0
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 40

    return score


def score_position(board,piece):
    score = 0

    #center
    center_array = [int(i) for i in list(board[:,COLUMN_COUNT//2])]
    center_count = center_array.count(piece)
    score += center_count * 6

    #horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    #vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:,c])]
        for r in range(ROW_COUNT -3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    #diagnol
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    #negative diagnol
    for r in range(ROW_COUNT-3):
        for c in range(COLUMN_COUNT-3):
            window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]  #row + 3: ?????????????????????????????????????????????row????????????
            score += evaluate_window(window, piece)

    return score


def is_ternimal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, AI_PIECE) or len(get_valid_location(board))==0


def minimax(board, depth, maximizingPlayer):
    valid_locations = get_valid_location(board)
    is_terminal = is_ternimal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return None, 100000
            elif winning_move(board, PLAYER_PIECE):
                return None, -100000
            else:
                return None, 0
        else: #depth = 0
            return None, score_position(board, AI_PIECE)

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, False)[1]
            if new_score > value:
                value = new_score
                column = col  #????????????best score???column

        return column, value
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, True)[1]
            if new_score < value:
                value = new_score
                column = col
        return column, value

def minimax_ab(board, depth, maximizingPlayer, alpha, beta):
    valid_locations = get_valid_location(board)
    is_terminal = is_ternimal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return None, 100000
            elif winning_move(board, PLAYER_PIECE):
                return None, -100000
            else:
                return None, 0
        else: #depth = 0
            return None, score_position(board, AI_PIECE)

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth-1, False)[1]
            if new_score > value:
                value = new_score
                column = col  #????????????best score???column
            alpha = max(alpha,value)
            if alpha>= beta:
                break
        return column, value

    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth-1, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta,value)
            if alpha >= beta:
                break
        return column, value

def get_valid_location(board):   #get all the cols that could drop pieces
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations

def pick_last_move(board, piece):

    valid_locations = get_valid_location(board)
    best_score = -10000
    best_col = random.choice(valid_locations)

    for col in valid_locations:
        row = get_next_open_row(board,col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)

        if score > best_score:
            best_score = score
            best_col = col

    return  best_col





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


def run_game():
    board = create_board()
    print_board(board)
    game_over = False
    turn = random.randint(PLAYER, AI)
    pygame.init()

    while not game_over:
        # Player turn
        if turn == PLAYER:
            col = random.randint(0,COLUMN_COUNT-1)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)

                if winning_move(board, PLAYER_PIECE):
                    print("PLAYER WINS!")
                    game_over = True

                turn = AI
                #print_board(board)

        # AI turn
        if turn == AI and not game_over:

            # Uncomment to let the AI think
            # time.sleep(0.5)

            col, minimax_score = minimax(board, 5, True)
            #col, minimax_score = minimax_ab(board, 5, True, -math.inf, math.inf)
            if is_valid_location(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, AI_PIECE)

                if winning_move(board, AI_PIECE):
                    print("AI WINS!")
                    game_over = True

                #print_board(board)

                turn = PLAYER

        if is_ternimal_node(board):
            print("GAME OVER")
            game_over = True

    return board

# ???????????????????????????
player_wins = 0
ai_wins = 0
total_time = 0

for i in range(NUM_RUNS):
    print(f"\nRound {i+1}\n")
    start_time = time.time()
    final_board = run_game()
    end_time = time.time()
    total_time += end_time - start_time

    # ??????????????????
    if winning_move(final_board, PLAYER_PIECE):
        player_wins += 1
    elif winning_move(final_board, AI_PIECE):
        ai_wins += 1

# ????????????
print(f"\nPlayer wins {player_wins} times, AI wins {ai_wins} times.")
print(f"Average time for each game: {total_time / NUM_RUNS:.2f} seconds.")
