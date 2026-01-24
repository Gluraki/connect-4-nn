import numpy as np
from tensorflow import keras
import random

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

def available_moves(board):
    return [c for c in range(COLS) if board[ROWS - 1][c] == ' ']

def check_winner(board, player):
    # Horizontal
    for r in range(ROWS):
        for c in range(COLS - 3):
            if all(board[r][c + i] == player for i in range(4)):
                return True
    # Vertical
    for c in range(COLS):
        for r in range(ROWS - 3):
            if all(board[r + i][c] == player for i in range(4)):
                return True
    # Diagonal \
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if all(board[r + i][c + i] == player for i in range(4)):
                return True
    # Diagonal /
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

def immediate_wins(board, player):
    """Find columns that result in immediate win"""
    wins = []
    for c in available_moves(board):
        tmp = [row[:] for row in board]
        drop_piece(tmp, c, player)
        if check_winner(tmp, player):
            wins.append(c)
    return wins

def blunders(board, player):
    """Find columns that allow opponent to win next turn"""
    opponent = 'O' if player == 'X' else 'X'
    bad = []
    for c in available_moves(board):
        tmp = [row[:] for row in board]
        drop_piece(tmp, c, player)
        for oc in available_moves(tmp):
            tmp2 = [row[:] for row in tmp]
            drop_piece(tmp2, oc, opponent)
            if check_winner(tmp2, opponent):
                bad.append(c)
                break
    return bad

def must_block(board, player):
    """Find columns where opponent can win (we must block)"""
    opponent = 'O' if player == 'X' else 'X'
    blocks = []
    for c in available_moves(board):
        tmp = [row[:] for row in board]
        drop_piece(tmp, c, opponent)
        if check_winner(tmp, opponent):
            blocks.append(c)
    return blocks

def print_board(board):
    """Pretty print the board"""
    print()
    for r in reversed(range(ROWS)):
        print('| ' + ' | '.join(board[r]) + ' |')
    print('-' * (COLS * 4 + 1))
    print('  ' + '   '.join(str(i+1) for i in range(COLS)))
    print()

# Load model
try:
    model = keras.models.load_model(MODEL_FILE)
    print(f"✓ Model loaded from {MODEL_FILE}\n")
except:
    print(f"✗ Could not load model from {MODEL_FILE}")
    print("Make sure you've trained the model first!")
    exit(1)

def inspect(board, player, title=""):
    """Inspect Q-values and identify strategic moves"""
    state = board_to_input(board, player)
    q = model.predict(state[np.newaxis, :], verbose=0)[0]
    
    wins = immediate_wins(board, player)
    bad = blunders(board, player)
    blocks = must_block(board, player)
    moves = available_moves(board)
    
    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
    
    print_board(board)
    print(f"Q-values for player {player}:")
    print("-" * 60)
    
    for c in range(COLS):
        if c not in moves:
            print(f"Col {c+1}: FULL")
        else:
            tags = []
            if c in wins:
                tags.append("⭐ WIN")
            if c in blocks:
                tags.append("🛡️ MUST BLOCK")
            if c in bad:
                tags.append("💀 BLUNDER")
            
            tag_str = "  " + " ".join(tags) if tags else ""
            print(f"Col {c+1}: {q[c]:+7.4f}{tag_str}")
    
    print("-" * 60)
    valid_q = [q[c] for c in moves]
    if valid_q:
        print(f"Best move: Col {np.argmax([q[c] if c in moves else -np.inf for c in range(COLS)]) + 1}")
        print(f"Q-value range: [{min(valid_q):+.4f}, {max(valid_q):+.4f}]")
        print(f"Spread: {max(valid_q) - min(valid_q):.4f}")
    
    # Quality checks
    issues = []
    if wins and max(q[c] for c in wins) < 0.5:
        issues.append("⚠️  Winning moves have low Q-values")
    if bad and max(q[c] for c in bad) > -0.3:
        issues.append("⚠️  Blunders not penalized enough")
    if blocks and max(q[c] for c in blocks) < max(valid_q) - 0.1:
        issues.append("⚠️  Not prioritizing blocks")
    
    if issues:
        print("\n❌ Issues detected:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✓ Model appears to understand this position")

# Test scenarios
print("="*60)
print("TESTING TRAINED MODEL")
print("="*60)

# Test 1: Empty board
b = create_board()
inspect(b, 'X', "TEST 1: Empty Board (Opening)")

# Test 2: Center opening
b = create_board()
drop_piece(b, 3, 'X')
inspect(b, 'O', "TEST 2: Center Opening Response")

# Test 3: Immediate win available
b = create_board()
drop_piece(b, 0, 'X')
drop_piece(b, 1, 'X')
drop_piece(b, 2, 'X')
inspect(b, 'X', "TEST 3: Win in Column 4 Available")

# Test 4: Must block opponent
b = create_board()
drop_piece(b, 0, 'X')
drop_piece(b, 1, 'X')
drop_piece(b, 2, 'X')
inspect(b, 'O', "TEST 4: Must Block or Lose")

# Test 5: Vertical threat
b = create_board()
drop_piece(b, 3, 'X')
drop_piece(b, 3, 'X')
drop_piece(b, 3, 'X')
inspect(b, 'X', "TEST 5: Vertical Win Available")

# Test 6: Diagonal setup
b = create_board()
drop_piece(b, 0, 'X')
drop_piece(b, 1, 'O')
drop_piece(b, 1, 'X')
drop_piece(b, 2, 'O')
drop_piece(b, 2, 'O')
drop_piece(b, 2, 'X')
inspect(b, 'X', "TEST 6: Diagonal Pattern")

# Test 7: Complex position with multiple threats
b = create_board()
drop_piece(b, 3, 'X')
drop_piece(b, 3, 'O')
drop_piece(b, 4, 'X')
drop_piece(b, 4, 'O')
drop_piece(b, 5, 'X')
drop_piece(b, 5, 'O')
inspect(b, 'X', "TEST 7: Multiple Threats")

print("\n" + "="*60)
print("AI VS RANDOM OPPONENT")
print("="*60)

def play_ai_vs_random(games=100, verbose=False):
    """Test AI against random opponent"""
    ai_wins = 0
    random_wins = 0
    draws = 0
    
    for game in range(games):
        board = create_board()
        player = 'X'
        
        while True:
            if player == 'X':  # AI
                state = board_to_input(board, 'X')
                q = model.predict(state[np.newaxis, :], verbose=0)[0]
                moves = available_moves(board)
                # Mask invalid moves
                q_masked = np.full(COLS, -np.inf)
                for m in moves:
                    q_masked[m] = q[m]
                move = np.argmax(q_masked)
            else:  # Random
                move = random.choice(available_moves(board))
            
            drop_piece(board, move, player)
            
            if check_winner(board, player):
                if player == 'X':
                    ai_wins += 1
                else:
                    random_wins += 1
                if verbose:
                    print(f"Game {game+1}: {player} wins")
                break
            
            if is_draw(board):
                draws += 1
                if verbose:
                    print(f"Game {game+1}: Draw")
                break
            
            player = 'O' if player == 'X' else 'X'
    
    print(f"\nResults over {games} games:")
    print(f"  AI Wins:     {ai_wins:3d} ({ai_wins/games*100:5.1f}%)")
    print(f"  Random Wins: {random_wins:3d} ({random_wins/games*100:5.1f}%)")
    print(f"  Draws:       {draws:3d} ({draws/games*100:5.1f}%)")
    
    if ai_wins / games >= 0.95:
        print("\n✓ Excellent! AI dominates random play")
    elif ai_wins / games >= 0.85:
        print("\n✓ Good! AI is strong against random play")
    elif ai_wins / games >= 0.70:
        print("\n⚠️  AI is decent but could be better")
    else:
        print("\n❌ AI needs more training")

play_ai_vs_random(games=100)

print("\n" + "="*60)
print("OVERALL ASSESSMENT")
print("="*60)

# Summary
print("""
A well-trained model should show:
  ✓ Winning moves: Q-value > 0.7
  ✓ Blocking moves: Q-value > 0.3
  ✓ Blunders: Q-value < -0.3
  ✓ Win rate vs random: > 90%
  ✓ Clear separation between good and bad moves
""")