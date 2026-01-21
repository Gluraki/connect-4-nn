import numpy as np
import random
from tensorflow import keras

ROWS = 6
COLS = 7
MODEL_FILE = "version1.keras"

def create_board():
    return [[' ' for _ in range(COLS)] for _ in range(ROWS)]

def print_board(board):
    for row in reversed(board):
        print(" | ".join(row))
    print("-" * 15)

def drop_piece(board, col, player):
    for row in range(ROWS):
        if board[row][col] == ' ':
            board[row][col] = player
            return True
    return False

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
    return all(board[ROWS - 1][c] != ' ' for c in range(COLS))

def board_to_input(board):
    data = []
    for row in board:
        for cell in row:
            if cell == 'X':
                data.append(1)
            elif cell == 'O':
                data.append(-1)
            else:
                data.append(0)
    return np.array(data, dtype=np.float32)

def create_model():
    model = keras.Sequential([
        keras.Input(shape=(42,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(7)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

model = create_model()

def choose_action_training(state, available_moves, epsilon=0.2):
    if random.random() < epsilon:
        return random.choice(available_moves)

    q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
    return max(available_moves, key=lambda c: q_values[c])

def choose_action_playing(state, available_moves):
    q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
    return max(available_moves, key=lambda c: q_values[c])

def train(episodes=20000):
    print("Training startet...")
    for ep in range(episodes):
        board = create_board()
        history = []
        player = 'X'

        while True:
            state = board_to_input(board)
            available = [c for c in range(COLS) if board[ROWS - 1][c] == ' ']

            history.append((state.copy(), available.copy(), player))
            move = choose_action_training(state, available)
            drop_piece(board, move, player)

            if check_winner(board, player):
                reward = 1 if player == 'X' else -1
                break

            if is_draw(board):
                reward = 0.5
                break

            player = 'O' if player == 'X' else 'X'

        for state, moves, p in history:
            target = model.predict(state.reshape(1, -1), verbose=0)[0]
            r = reward if p == player else -reward

            for m in moves:
                target[m] = target[m] * 0.9 + r * 0.1

            model.fit(
                state.reshape(1, -1),
                target.reshape(1, -1),
                epochs=1,
                verbose=0
            )

        if (ep + 1) % 500 == 0:
            print(f"Episode {ep + 1}")

    model.save(MODEL_FILE)
    print("Training abgeschlossen & Modell gespeichert.")

def play_game():
    global model
    model = keras.models.load_model(MODEL_FILE)

    board = create_board()

    while True:
        print_board(board)

        try:
            move = int(input("Dein Zug (1-7): ")) - 1
        except ValueError:
            print("Bitte Zahl 1–7 eingeben.")
            continue

        if move < 0 or move >= COLS or not drop_piece(board, move, 'X'):
            print("Ungültiger Zug!")
            continue

        if check_winner(board, 'X'):
            print_board(board)
            print("Du gewinnst!")
            break

        if is_draw(board):
            print("Unentschieden!")
            break

        state = board_to_input(board)
        available = [c for c in range(COLS) if board[ROWS - 1][c] == ' ']
        ai_move = choose_action_playing(state, available)
        drop_piece(board, ai_move, 'O')

        if check_winner(board, 'O'):
            print_board(board)
            print("KI gewinnt!")
            break

if __name__ == "__main__":
    train(10)
    play_game()
