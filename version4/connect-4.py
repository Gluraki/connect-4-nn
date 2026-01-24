import numpy as np
from tensorflow import keras

ROWS = 6
COLS = 7
MODEL_FILE = "version4_test_10000.keras"

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

model = keras.models.load_model(MODEL_FILE)

def choose_ai_move(board, player):
    state = board_to_input(board, player)
    q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
    for c in range(COLS):
        if board[ROWS - 1][c] != ' ':
            q_values[c] = -np.inf
    return int(np.argmax(q_values))


def print_board(board):
    for r in reversed(range(ROWS)):
        print('| ' + ' | '.join(board[r]) + ' |')
    print('-' * (COLS * 4 + 1))
    print('  ' + '   '.join(str(i+1) for i in range(COLS)))


def play_game():
    board = create_board()
    human_player = ''
    while human_player not in ['X', 'O']:
        human_player = input("Do you want to play as X or O? ").upper()
    ai_player = 'O' if human_player == 'X' else 'X'

    current_player = 'X'

    while True:
        print_board(board)
        if current_player == human_player:
            valid_move = False
            while not valid_move:
                try:
                    move = int(input("Your move (1-7): ")) - 1
                    if 0 <= move < COLS and drop_piece(board, move, human_player):
                        valid_move = True
                    else:
                        print("Invalid move, try again.")
                except ValueError:
                    print("Enter a number between 1 and 7.")
        else:
            print("AI is thinking...")
            move = choose_ai_move(board, ai_player)
            drop_piece(board, move, ai_player)

        if check_winner(board, current_player):
            print_board(board)
            if current_player == human_player:
                print("Congratulations! You win!")
            else:
                print("AI wins!")
            break
        if is_draw(board):
            print_board(board)
            print("It's a draw!")
            break

        current_player = 'O' if current_player == 'X' else 'X'

if __name__ == "__main__":
    play_game()
