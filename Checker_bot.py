import copy
import matplotlib.pyplot as plt
import numpy as np

# Constants
EMPTY = 0
WHITE = 1
BLACK = 2
KING_WHITE = 3
KING_BLACK = 4

# Board dimensions
ROWS = 8
COLS = 8

# Board representation
initial_board = [
    [0, 2, 0, 2, 0, 2, 0, 2],
    [2, 0, 2, 0, 2, 0, 2, 0],
    [0, 2, 0, 2, 0, 2, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 0]
]

# Move directions for white and black pieces
WHITE_DIRECTIONS = [(-1, -1), (-1, 1), (-2, -2), (-2, 2)]
BLACK_DIRECTIONS = [(1, -1), (1, 1), (2, -2), (2, 2)]

# Utility functions
def draw_board(board):
    plt.figure(figsize=(8, 8))
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    plt.xticks(np.arange(0.5, 8.5, 1), [])
    plt.yticks(np.arange(0.5, 8.5, 1), [])
    
    for i in range(ROWS):
        for j in range(COLS):
            if (i + j) % 2 == 0:
                plt.fill_between([j, j+1], i, i+1, color='#D2B48C')  # Biscuit color
            if board[i][j] == WHITE:
                plt.plot(j + 0.5, i + 0.5, 'ro', markersize=20)  # Red pieces
            elif board[i][j] == BLACK:
                plt.plot(j + 0.5, i + 0.5, 'bo', markersize=20)  # Blue pieces
            elif board[i][j] == KING_WHITE:
                plt.plot(j + 0.5, i + 0.5, 'r+', markersize=20)  # Red king
            elif board[i][j] == KING_BLACK:
                plt.plot(j + 0.5, i + 0.5, 'b+', markersize=20)  # Blue king
    
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.gca().invert_yaxis()
    plt.show()

def print_board(board):
    for row in board:
        print(' '.join(map(str, row)))

def is_valid_move(board, player, move):
    i, j, new_i, new_j = move
    if new_i < 0 or new_i >= ROWS or new_j < 0 or new_j >= COLS:
        return False
    if board[new_i][new_j] != EMPTY:
        return False
    
    if player == WHITE:
        directions = WHITE_DIRECTIONS
        opponent = BLACK
    else:
        directions = BLACK_DIRECTIONS
        opponent = WHITE
    
    # Check regular moves
    if (new_i - i, new_j - j) in directions[:2]:
        return True
    
    # Check jump moves
    for dir_i, dir_j in directions[2:]:
        mid_i, mid_j = i + dir_i // 2, j + dir_j // 2
        if 0 <= mid_i < ROWS and 0 <= mid_j < COLS and board[mid_i][mid_j] == opponent and (new_i - i, new_j - j) == (dir_i, dir_j):
            return True
    
    return False


def apply_move(board, move):
    i, j, new_i, new_j = move
    board[new_i][new_j] = board[i][j]
    
    # Check for promotion to king
    if board[new_i][new_j] == WHITE and new_i == 0:
        board[new_i][new_j] = KING_WHITE
    elif board[new_i][new_j] == BLACK and new_i == ROWS - 1:
        board[new_i][new_j] = KING_BLACK
    
    board[i][j] = EMPTY

def get_possible_moves(board, player):
    moves = []
    for i in range(ROWS):
        for j in range(COLS):
            if board[i][j] == player or board[i][j] == player + 2:
                directions = WHITE_DIRECTIONS if player == WHITE else BLACK_DIRECTIONS
                for dir_i, dir_j in directions:
                    new_i, new_j = i + dir_i, j + dir_j
                    if is_valid_move(board, player, (i, j, new_i, new_j)):
                        moves.append((i, j, new_i, new_j))
    return moves

def is_game_over(board):
    white_count = sum(row.count(WHITE) + row.count(KING_WHITE) for row in board)
    black_count = sum(row.count(BLACK) + row.count(KING_BLACK) for row in board)
    return white_count == 0 or black_count == 0

def evaluate_board(board):
    white_value = 0
    black_value = 0
    for i in range(ROWS):
        for j in range(COLS):
            if board[i][j] == WHITE:
                white_value += 1 + i * 0.1
            elif board[i][j] == KING_WHITE:
                white_value += 3
            elif board[i][j] == BLACK:
                black_value += 1 + (ROWS - 1 - i) * 0.1
            elif board[i][j] == KING_BLACK:
                black_value += 3
    
    return white_value - black_value

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or is_game_over(board):
        return evaluate_board(board)
    
    player = WHITE if maximizing_player else BLACK
    moves = get_possible_moves(board, player)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move)
            eval = minimax(new_board, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move)
            eval = minimax(new_board, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board, depth, player):
    moves = get_possible_moves(board, player)
    best_move = None
    if player == WHITE:
        max_eval = float('-inf')
        for move in moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move)
            eval = minimax(new_board, depth - 1, float('-inf'), float('inf'), False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
    else:
        min_eval = float('inf')
        for move in moves:
            new_board = copy.deepcopy(board)
            apply_move(new_board, move)
            eval = minimax(new_board, depth - 1, float('-inf'), float('inf'), True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
    return best_move

def player_move(board, player):
    moves = get_possible_moves(board, player)
    print("Available moves:")
    for i, move in enumerate(moves):
        print(f"{i + 1}: {move}")
    choice = int(input("Choose a move: ")) - 1
    return moves[choice]

def main():
    board = copy.deepcopy(initial_board)
    current_player = WHITE
    while not is_game_over(board):
        draw_board(board)
        if current_player == WHITE:
            print("White's turn")
            move = player_move(board, current_player)
        else:
            print("Black's turn")
            move = get_best_move(board, 3, BLACK)
        apply_move(board, move)
        current_player = BLACK if current_player == WHITE else WHITE
    draw_board(board)
    if evaluate_board(board) > 0:
        print("White wins!")
    elif evaluate_board(board) < 0:
        print("Black wins!")
    else:
        print("It's a draw!")

if __name__ == '__main__':
    main()