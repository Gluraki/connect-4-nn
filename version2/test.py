import numpy as np
from tensorflow import keras
import random

ROWS = 6
COLS = 7
MODEL_FILE = "version2-1000.keras"

def create_board():
    return [[' ' for _ in range(COLS)] for _ in range(ROWS)]

def drop_piece(board, col, player):
    for r in range(ROWS):
        if board[r][col] == ' ':
            board[r][col] = player
            return True
    return False

def available_moves(board):
    return [c for c in range(COLS) if board[ROWS - 1][c] == ' ']

def check_winner(board, player):
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == player for i in range(4)):
                return True
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == player for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            if all(board[r + i][c - i] == player for i in range(4)):
                return True
    return False

def is_draw(board):
    return len(available_moves(board)) == 0

def board_to_input(board, player):
    opponent = 'O' if player == 'X' else 'X'
    mapping = {player: 1, opponent: -1, ' ': 0}
    return np.array([mapping[cell] for row in board for cell in row], dtype=np.float32)

model = keras.models.load_model(MODEL_FILE)
print("Model loaded.")

def inspect(board, player):
    state = board_to_input(board, player)
    q = model.predict(state.reshape(1, -1), verbose=0)[0]

    print(f"\nQ-values for player {player}:")
    for c in range(COLS):
        print(f"Col {c+1}: {q[c]:+.4f}")
    print(f"Spread: {(q.max() - q.min()):.4f}")

print("\n=== EMPTY BOARD ===")
b = create_board()
inspect(b, 'X')

print("\n=== CENTER OPENING (X in column 4) ===")
b = create_board()
drop_piece(b, 3, 'X')
inspect(b, 'O')

print("\n=== X ABOUT TO WIN ===")
b = create_board()
drop_piece(b, 0, 'X')
drop_piece(b, 1, 'X')
drop_piece(b, 2, 'X')
inspect(b, 'X')

print("\n=== MUST BLOCK (O to move) ===")
inspect(b, 'O')

def play_ai_vs_random(games=20):
    wins = 0
    for _ in range(games):
        board = create_board()
        player = 'X'

        while True:
            if player == 'X':
                state = board_to_input(board, 'X')
                q = model.predict(state.reshape(1, -1), verbose=0)[0]
                move = max(available_moves(board), key=lambda c: q[c])
            else:
                move = random.choice(available_moves(board))

            drop_piece(board, move, player)

            if check_winner(board, player):
                if player == 'X':
                    wins += 1
                break
            if is_draw(board):
                break

            player = 'O' if player == 'X' else 'X'

    print(f"\nAI (X) wins vs random: {wins}/{games}")

play_ai_vs_random()
