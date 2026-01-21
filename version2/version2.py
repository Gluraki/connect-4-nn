import os
import random
import numpy as np
from tensorflow import keras
import copy

ROWS = 6
COLS = 7
MODEL_FILE = "version2-1000.keras"

GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 500

def create_board():
    return [[' ' for _ in range(COLS)] for _ in range(ROWS)]

def drop_piece(board, col, player):
    for r in range(ROWS):
        if board[r][col] == ' ':
            board[r][col] = player
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

def board_to_input(board, player):
    mapping = {player: 1, 'O' if player == 'X' else 'X': -1, ' ': 0}
    return np.array([mapping[cell] for row in board for cell in row], dtype=np.float32)

def opponent_can_win_next(board, opponent):
    for c in range(COLS):
        if board[ROWS - 1][c] == ' ':
            tmp = copy.deepcopy(board)
            drop_piece(tmp, c, opponent)
            if check_winner(tmp, opponent):
                return True
    return False

def create_model():
    model = keras.Sequential([
        keras.Input(shape=(42,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(COLS)
    ])
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    return model

def load_or_create_model():
    if os.path.exists(MODEL_FILE):
        print("Lade bestehendes Modell...")
        return keras.models.load_model(MODEL_FILE)
    print("Erstelle neues Modell...")
    return create_model()

model = load_or_create_model()
target_model = copy.deepcopy(model)

def choose_action(state, available_moves, epsilon):
    if random.random() < epsilon:
        return random.choice(available_moves)
    q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
    return max(available_moves, key=lambda c: q_values[c])

def train(episodes=1000):
    global model, target_model
    epsilon = EPSILON_START

    print("Training startet...")
    for ep in range(episodes):
        board = create_board()
        player = 'X'
        state = board_to_input(board, player)

        while True:
            available = [c for c in range(COLS) if board[ROWS - 1][c] == ' ']
            action = choose_action(state, available, epsilon)
            drop_piece(board, action, player)

            next_player = 'O' if player == 'X' else 'X'

            if check_winner(board, player):
                reward = 1.0
                done = True

            elif is_draw(board):
                reward = 0.0
                done = True

            else:
                if opponent_can_win_next(board, next_player):
                    reward = -1.0
                    done = True
                else:
                    reward = 0.0
                done = False
            
            next_state = board_to_input(board, next_player)

            q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
            q_next = target_model.predict(next_state.reshape(1, -1), verbose=0)[0]

            target = q_values.copy()
            if done:
                target[action] = reward
            else:
                target[action] = reward + GAMMA * np.max(q_next)

            target[action] = np.clip(target[action], -1.5, 1.5)

            model.fit(state.reshape(1, -1), target.reshape(1, -1), verbose=0)

            if done:
                break

            state = board_to_input(board, next_player)
            player = next_player

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if (ep + 1) % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())

        if (ep + 1) % 500 == 0:
            print(f"Episode {ep + 1} | epsilon={epsilon:.3f}")

    model.save(MODEL_FILE)
    print("Training abgeschlossen & Modell gespeichert.")

def inspect_q_values(board, player='X'):
    state = board_to_input(board, player)
    q_values = model.predict(state.reshape(1, -1), verbose=0)[0]
    print("\nQ-values:")
    for c in range(COLS):
        print(f"Column {c + 1}: {q_values[c]:+.3f}")
    print(f"Max Q: {np.max(q_values):+.3f}")
    print(f"Min Q: {np.min(q_values):+.3f}")
    print(f"Spread: {(np.max(q_values)-np.min(q_values)):.3f}")

def play_game():
    board = create_board()
    player = 'X'

    while True:
        for row in reversed(board):
            print(" | ".join(row))
        print("-" * 15)

        move = int(input("Dein Zug (1-7): ")) - 1
        if move < 0 or move >= COLS or not drop_piece(board, move, player):
            print("Ungültiger Zug.")
            continue

        if check_winner(board, player):
            print("Du gewinnst!")
            break
        if is_draw(board):
            print("Unentschieden!")
            break

        ai_player = 'O'
        state = board_to_input(board, ai_player)
        available = [c for c in range(COLS) if board[ROWS - 1][c] == ' ']
        ai_move = choose_action(state, available, epsilon=0.0)
        drop_piece(board, ai_move, ai_player)

        if check_winner(board, ai_player):
            print("KI gewinnt!")
            break

if __name__ == "__main__":
    #train(1000)
    inspect_q_values(create_board())
    play_game()
