import os
import random
import numpy as np
from tensorflow import keras

ROWS = 6
COLS = 7
MODEL_FILE = "version3-10000.keras"
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 10000
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
    opponent = 'O' if player == 'X' else 'X'
    mapping = {player: 1.0, opponent: -1.0, ' ': 0.0}
    arr = np.array([[mapping[cell] for cell in row] for row in board], dtype=np.float32)
    return arr.reshape((ROWS, COLS, 1))

class ReplayBuffer:
    def __init__(self, size=MEMORY_SIZE):
        self.buffer = []
        self.size = size

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

memory = ReplayBuffer()

def create_model():
    model = keras.Sequential([
        keras.Input(shape=(ROWS, COLS, 1)),
        keras.layers.Conv2D(64, (2, 2), activation='relu'),
        keras.layers.Conv2D(64, (2, 2), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(COLS)
    ])
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss='mse')
    return model

def load_or_create_model():
    if os.path.exists(MODEL_FILE):
        return keras.models.load_model(MODEL_FILE)
    return create_model()

model = load_or_create_model()
target_model = create_model()
target_model.set_weights(model.get_weights())

def choose_action(state, available_moves, epsilon):
    if random.random() < epsilon:
        return random.choice(available_moves)
    q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
    return max(available_moves, key=lambda c: q_values[c])

def opponent_can_win_next(board, opponent):
    for c in range(COLS):
        if board[ROWS - 1][c] == ' ':
            tmp = [row.copy() for row in board]
            drop_piece(tmp, c, opponent)
            if check_winner(tmp, opponent):
                return True
    return False


def train(episodes=5000):
    epsilon = EPSILON_START
    for ep in range(episodes):
        board = create_board()
        player = 'X'
        state = board_to_input(board, player)

        while True:
            available = [c for c in range(COLS) if board[ROWS - 1][c] == ' ']
            action = choose_action(state, available, epsilon)
            drop_piece(board, action, player)
            next_player = 'O' if player == 'X' else 'X'
            next_state = board_to_input(board, next_player)

            if check_winner(board, player):
                reward = 1.0
                done = True
            elif is_draw(board):
                reward = 0.0
                done = True
            elif opponent_can_win_next(board, next_player):
                reward = -1.0
                done = True
            else:
                reward = 0.0
                done = False

            memory.push(state, action, reward, next_state, done)

            batch = memory.sample()
            if batch:
                states = np.array([b[0] for b in batch])
                next_states = np.array([b[3] for b in batch])
                q_values = model.predict(states, verbose=0)
                q_next = target_model.predict(next_states, verbose=0)

                for i, (s, a, r, s_next, d) in enumerate(batch):
                    if d:
                        q_values[i][a] = r
                    else:
                        q_values[i][a] = r + GAMMA * np.max(q_next[i])

                model.fit(states, q_values, epochs=1, verbose=0)

            state = next_state
            player = next_player

            if done:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if (ep + 1) % TARGET_UPDATE_FREQ == 0:
            target_model.set_weights(model.get_weights())

        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1}, epsilon={epsilon:.3f}, memory={len(memory)}")

    model.save(MODEL_FILE)
    print("Training complete and model saved.")

if __name__ == "__main__":
    train(episodes=10000)
