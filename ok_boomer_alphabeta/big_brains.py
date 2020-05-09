import ok_boomer_alphabeta.action_generator as actgen
import ok_boomer_alphabeta.game as game
import random
import logging
import math
from ok_boomer_alphabeta.util import *
from ok_boomer_alphabeta.constants import *

FORMAT = '%(asctime)s: %(levelname)s: %(message)s'

formatter = logging.Formatter(FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(filename='big-brains.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class Node:
    def __init__(self, board_dict, path_cost, action, parent):
        self.board_dict = board_dict
        self.path_cost = path_cost
        self.action = action
        self.parent = parent
        self.children = []
        self.heuristic = heuristic(self)
        self.f = self.heuristic + self.path_cost

    def __str__(self):
        return """
# State: {}
# Path Cost: {}
# Heuristic: {}
# Action: {}
# Parent: {}
# Children: {}
""".format(str(self.board_dict), str(self.path_cost), str(self.heuristic), str(self.action), hex(id(self.parent)), [hex(id(child)) for child in self.children])
    
    def __lt__(self, other):
        return self.f < other.f



def search(player):
    """ Returns an action for the given player. """

    actions = actgen.get_possible_actions(player.board, player.colour)
    
    for i in range(len(actions)):
        score = evaluate(player.colour, player.board)
        actions[i] = (actions[i], score)
    
    actions.sort(reverse=True, key=lambda pair: pair[1])
    
    # Trim the list
    trimmed_actions = [actions[0]]

    for i in range(1, len(actions)):
        if actions[i-1][1] == actions[i][1]:
            trimmed_actions.append(actions[i])
        else:
            break

    return trimmed_actions[random.randint(0, len(trimmed_actions)-1)][0].toTuple()

def evaluate(player_colour, board):
    """ Returns an evaluation value for a given action. """
    WEIGHT = 0.9    # Controls importance of removing pieces vs moving closer
    EPSILON = 1

    if player_colour == "black":
        new_state = {}
        new_state["white"] = board["black"]
        new_state["black"] = board["white"]

    before = count_tokens(board) 
    eval = (before[0]) - (before[1])    # Higher = More white removed
    distance_heuristic = 1/max(EPSILON, math.tanh(heuristic(board, player_colour))) # Higher = White is closer to black

    return math.tanh(eval)

def apply_action(player_colour, board, action):
    """ Applies an action to the board and returns the new board configuration. """

    if action.action == "BOOM":
        next_board = game.boom(action.source, board)
        
    if action.action == "MOVE":
        next_board = game.move(action.num, action.source, action.target, board, player_colour)
    return next_board

def count_tokens(board):
    """ Returns the number of white and black tokens on the board """
    count = [0, 0]

    for n, x, y in board["white"]:
        count[0] += n

    for n, x, y in board["black"]:
        count[1] += n
    
    return count


def heuristic(board, colour):
    # Best of n stack distance
    if len(board[colour]) > 0:
        best_stack = max([stack[N_TOKENS] for stack in board[colour]])
    else:
        best_stack = 1

    distances = []
    
    for stack in board[opponent(colour)]:
        distances.append(min_distance_from_stack(stack, board[colour]))
    
    w, b = count_tokens(board)
    count = w if colour=="white" else b

    return sum(distances)//max(1, best_stack)

def min_distance_from_stack(source, stack_list):
    # Minimum distance from a black stack to one of the white stacks
    min_distance = BOARD_SIZE*2
    for i in range(len(stack_list)):
        h_dist = manhattan_distance(source, stack_list[i]) - 1
        min_distance = min(min_distance, h_dist)

    return min_distance

def manhattan_distance(stack1, stack2):
    # Chess board distance as booming can take surrounding 9 squares
    return abs(stack1[X_POS]-stack2[X_POS]) + abs(stack1[Y_POS]-stack2[Y_POS])


def opponent(colour):
    if colour == "white":
        return "black"
    elif colour == "black":
        return "white"
    
    raise Exception("Invalid colour")

MIN = -1000
MAX = 1000

def is_game_over(board):
    n_white, n_black = count_tokens(board)
    return n_white == 0 or n_black == 0

def utility(board):
    n_white, n_black = count_tokens(board)
    if n_white == 0 and n_black == 0: return 0
    elif n_white == 0: return -1
    else: return 1

def minimax(board, depth, player_colour, alpha, beta):
    if is_game_over(board):
        return utility(board), None

    if depth == 5:
        evaluation = evaluate(player_colour, board)
        return evaluation, None
    
    if(player_colour=="white"):
        actions = actgen.get_possible_actions(board, "white")
       
        best = MIN
        best_action = None
        for action in actions:
            next_board = apply_action(player_colour, board, action)
        
            if action.action == "BOOM" and detect_suicide(board, next_board):
                continue

            score, _ = minimax(next_board, depth+1, "black", alpha, beta)
            if score > best:
                best = score
                best_action = action
            alpha = max(alpha, best)

            if alpha >= beta:
                #print("BROKEN")
                break
        #print("the best action for white is", best, best_action)

    else:
        actions = actgen.get_possible_actions(board, "black")
        best = MAX
        best_action = None
        for action in actions:
            #print(action)
            next_board = apply_action(player_colour, board, action)
            
            if action.action == "BOOM" and detect_suicide(board, next_board):
                continue

            score, _ = minimax(next_board, depth+1, "white", alpha, beta)
            
            if score < best:
                best = score
                best_action = action
            beta = min(beta, best)
            if beta <= alpha:
                #print("BROKEN")
                break
        #print("the best action for black is", best, best_action)
    
    if (depth==1):
        logger.debug(print_board(game.get_grid_format(board)))
        logger.debug("{colour} played {action} with an evaluation score: {score}".format(colour=player_colour, action=best_action, score=best))
        
    return best, best_action

#def evaluate_1(player_colour, board, action):
 
#print("this is minimax", minimax(b, 1, "white", -1000, 1000))

def detect_suicide(board, next_board):
    before_w, before_b = count_tokens(board)
    next_w, next_b = count_tokens(next_board)
    return (before_w != next_w) ^ (before_b != next_b)