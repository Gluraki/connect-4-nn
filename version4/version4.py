import numpy as np
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
import random
from collections import deque

ROWS = 6
COLS = 7
GAMMA = 0.95  
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE = 500

def create_board():
    return [[' ' for _ in range(COLS)] for _ in range(ROWS)]

def drop_piece(board, col, player):
    for r in range(ROWS):
        if board[r][col] == ' ':
            board[r][col] = player
            return r
    return -1

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
    mapping = {player: 1.0, opponent: -1.0, ' ': 0.0}
    arr = np.array([[mapping[cell] for cell in row] for row in board], dtype=np.float32)
    return arr.reshape((ROWS, COLS, 1))

def create_model():
    inputs = keras.Input(shape=(ROWS, COLS, 1))
    x = layers.Conv2D(64, (4, 4), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(64, (4, 4), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    
    outputs = layers.Dense(COLS, activation='linear')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse')
    return model

def get_reward(board, player, move_col, opponent):
    if check_winner(board, player):
        return 1.0
    for c in available_moves(board):
        tmp = [row[:] for row in board]
        drop_piece(tmp, c, opponent)
        if check_winner(tmp, opponent):
            return -0.8
    tmp = [row[:] for row in board]
    undo_row = -1
    for r in range(ROWS):
        if tmp[r][move_col] == player:
            undo_row = r
            tmp[r][move_col] = ' '
            break
    
    if undo_row >= 0:
        for c in available_moves(tmp):
            tmp2 = [row[:] for row in tmp]
            drop_piece(tmp2, c, opponent)
            if check_winner(tmp2, opponent):
                return 0.3
    if is_draw(board):
        return 0.0
    return -0.01

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def train_model(episodes=5000, save_file="connect4_trained.keras"):
    model = create_model()
    target_model = create_model()
    target_model.set_weights(model.get_weights())
    
    memory = ReplayMemory(MEMORY_SIZE)
    epsilon = EPSILON_START
    
    wins = 0
    losses = 0
    draws = 0
    
    for episode in range(episodes):
        board = create_board()
        player = 'X'
        opponent = 'O'
        
        states = []
        actions = []
        rewards = []
        
        while True:
            state = board_to_input(board, player)
            moves = available_moves(board)
            if random.random() < epsilon:
                action = random.choice(moves)
            else:
                q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
                q_masked = np.full(COLS, -np.inf)
                for m in moves:
                    q_masked[m] = q_values[m]
                action = np.argmax(q_masked)
            drop_piece(board, action, player)
            game_over = check_winner(board, player) or is_draw(board)
            reward = get_reward(board, player, action, opponent)
            
            next_state = board_to_input(board, opponent) if not game_over else None
            memory.push(state, action, reward, next_state, game_over)
            if game_over:
                if check_winner(board, player):
                    wins += 1
                elif is_draw(board):
                    draws += 1
                else:
                    losses += 1
                break
            player, opponent = opponent, player
        if len(memory) >= BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            
            states_batch = np.array([b[0] for b in batch])
            actions_batch = np.array([b[1] for b in batch])
            rewards_batch = np.array([b[2] for b in batch])
            next_states_batch = np.array([b[3] if b[3] is not None else np.zeros((ROWS, COLS, 1)) for b in batch])
            done_batch = np.array([b[4] for b in batch])
            current_q = model.predict(states_batch, verbose=0)
            next_q_online = model.predict(next_states_batch, verbose=0)
            next_q_target = target_model.predict(next_states_batch, verbose=0)
            
            targets = current_q.copy()
            
            for i in range(BATCH_SIZE):
                if done_batch[i]:
                    targets[i, actions_batch[i]] = rewards_batch[i]
                else:
                    best_action = np.argmax(next_q_online[i])
                    targets[i, actions_batch[i]] = rewards_batch[i] + GAMMA * next_q_target[i, best_action]
            
            model.fit(states_batch, targets, epochs=1, verbose=0)
        if episode % TARGET_UPDATE == 0:
            target_model.set_weights(model.get_weights())
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        
        if (episode + 1) % 500 == 0:
            win_rate = wins / (wins + losses + draws) if (wins + losses + draws) > 0 else 0
            print(f"Episode {episode + 1}/{episodes}")
            print(f"  Wins: {wins}, Losses: {losses}, Draws: {draws}")
            print(f"  Win Rate: {win_rate:.2%}, Epsilon: {epsilon:.4f}")
            wins = losses = draws = 0
    
    model.save(save_file)
    print(f"\nTraining complete! Model saved to {save_file}")
    return model

if __name__ == "__main__":
    print("Starting training...")
    print()
    model = train_model(episodes=10000, save_file="version4_test_10000.keras")
    print("\nTraining finished!")