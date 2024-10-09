import math

# Use these constants to fill in the game board
X = "X"
O = "O"
EMPTY = None


def start_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns which player (either X or O) who has the next turn on a board.

    In the initial game state, X gets the first move. Subsequently, the player alternates with each additional move.
    Any return value is acceptable if a terminal board is provided as input (i.e., the game is already over).
    """
    x_player = sum(row.count(X) for row in board)
    o_player = sum(row.count(O) for row in board)
    
    # X gets the first move
    return O if x_player > o_player else X


def actions(board):
    """
    Returns the set of all possible actions (i, j) available on the board.

    The actions function should return a set of all the possible actions that can be taken on a given board.
    Each action should be represented as a tuple (i, j) where i corresponds to the row of the move (0, 1, or 2)
    and j corresponds to the column of the move (also 0, 1, or 2).

    Possible moves are any cells on the board that do not already have an X or an O in them.

    Any return value is acceptable if a terminal board is provided as input.
    """
    # empty set to store the actions
    result = set()  
    
    # loop over each row and column
    for i in range(3):  
        for j in range(3):
            # check if the cell is empty
            if board[i][j] == EMPTY: 
                # add the empty cell's coordinates as an action
                result.add((i, j))  
    
    return result


def succ(board, action):
    """
    Returns the board that results from making move (i, j) on the board, without modifying the original board.

    If `action` is not a valid action for the board, you  should raise an exception.

    The returned board state should be the board that would result from taking the original input board, and letting
    the player whose turn it is make their move at the cell indicated by the input action.

    Importantly, the original board should be left unmodified. This means that simply updating a cell in `board` itself
    is not a correct implementation of this function. You’ll likely want to make a deep copy of the board first before
    making any changes.
    """
    # If `action` is not a valid action for the board, you  should raise an exception.
    if board[action[0]][action[1]] != EMPTY:
        raise Exception("Invalid action")
    
    # returned board state should be the board that would result from taking the original input board
    new_board = [row[:] for row in board]
    # the original board should be left unmodified
    new_board[action[0]][action[1]] = player(board)
    
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.

    - If the X player has won the game, the function should return X.
    - If the O player has won the game, the function should return O.
    - If there is no winner of the game (either because the game is in progress, or because it ended in a tie), the
      function should return None.

    You may assume that there will be at most one winner (that is, no board will ever have both players with
    three-in-a-row, since that would be an invalid board state).
    """
    for i in range(3):
        # check rows and columns
        if board[i][0] == board[i][1] == board[i][2] != EMPTY:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != EMPTY:
            return board[0][i]
    
    # check diagonals
    if board[0][0] == board[1][1] == board[2][2] != EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != EMPTY:
        return board[0][2]
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.

    If the game is over, either because someone has won the game or because all cells have been filled without anyone
    winning, the function should return True.

    Otherwise, the function should return False if the game is still in progress.
    """
    # Returns True if game is over
    if winner(board) is not None:
        return True
    
    filled = True  
    # loop to check if board is filled
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                filled = False  
                break  
        if not filled:
            break 
    
    return filled

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.

    You may assume utility will only be called on a board if terminal(board) is True.
    """
    win = winner(board)
    
    # Returns 1 if X has won the game
    if win == X:
        return 1
    # -1 if O has won
    elif win == O:
        return -1
    # 0 otherwise
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.

    The move returned should be the optimal action (i, j) that is one of the allowable actions on the board.

    If multiple moves are equally optimal, any of those moves is acceptable.

    If the board is a terminal board, the minimax function should return None.
    """
    if terminal(board):
        return None

    if player(board) == X:
        value, move = max_value(board)
        return move
    else:
        value, move = min_value(board)
        return move

def max_value(board):
    """
    Computes max value  of all possible actions for the player X.
    Returns a tuple of (utility value, action) for the best action.
    """
    if terminal(board):
        return utility(board), None
    
    v = float('-inf')
    best_action = None
    
    for action in actions(board):
        # Calculate the value of making a move at this action.
        result_value, _ = min_value(succ(board, action))
        
        # Update the best action if this action leads to a better value.
        if result_value > v:
            v = result_value
            best_action = action
    
    return v, best_action

def min_value(board):
    """
    Computes the minimum value of all possible actions for the player O.
    Returns a tuple of (utility value, action) for the best action.
    """
    if terminal(board):
        return utility(board), None
    
    v = float('inf')
    best_action = None
    
    for action in actions(board):
        # get max value of the opponent taking an action.
        result_value, _ = max_value(succ(board, action))
        
        # update the best action if not suboptimal 
        if result_value < v:
            v = result_value
            best_action = action
    
    return v, best_action