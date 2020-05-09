import ok_boomer_tdleaf.action_generator as actgen
import ok_boomer_tdleaf.game as game
from ok_boomer_tdleaf.util import *
from ok_boomer_tdleaf.constants import *
import ok_boomer_tdleaf.calc_features as calc_features

import random
import logging
import math
import time
import numpy as np

FORMAT = '%(message)s'

formatter = logging.Formatter(FORMAT)

timestamp = time.time()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(filename='training/data.log'.format(timestamp), mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class TTable:
    def __init__(self):
        self.visited_states = {}

    def addState(self, board):
        State = self.dict_to_set(board)

        if State in self.visited_states.keys():
            self.visited_states[State] += 1
        else:
            self.visited_states[State] = 0

    def getCount(self, board):
        State = self.dict_to_set(board)
        if State in self.visited_states.keys():
            return self.visited_states[State]
        
        return 0

    def dict_to_set(self, board_dict):
        new_dict = set()
        for i in board_dict["white"]:
            new_dict.add(tuple([0] + i))

        for i in board_dict["black"]:
            new_dict.add(tuple([1] + i))
        
        return frozenset(new_dict)

def minimax_wrapper(board, depth, weights, player_colour, alpha, beta, depth_limit, ttable, nturns):
    # Play as white
    board = flip_board(board, player_colour)
    actions = actgen.get_possible_actions(board, "white")
    action_rank = []

    for action in actions:
        next_board = apply_action(player_colour, board, action)

        if action.action == "BOOM" and detect_suicide(board, next_board):
            continue
        
        score, _, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, ttable, nturns + 1)
        action_rank.append((score, action, best_leaf))
        
    action_rank.sort(reverse=True, key=lambda x: x[0])
    
    trimmed_actions = [action_rank[0]]

    for i in range(1, len(action_rank)):
        if action_rank[i-1][0] == action_rank[i][0]:
            trimmed_actions.append(action_rank[i])
        else:
            break
    
    #print(len(action_rank))
    #print(len(trimmed_actions))
    #if (len(action_rank) != len(trimmed_actions)):
    
    print(trimmed_actions)

    selected_action = action_rank[random.randint(0, len(trimmed_actions)-1)]
    best, best_action, best_leaf_state = selected_action

    print(best)
    
    #print(best_action)
    best_action = flip_action(best_action, player_colour)
    print(best_action)

    #logger.debug(print_board(game.get_grid_format(best_leaf_state)))
    feature_string = [str(x) for x in calc_features.get_features(best_leaf_state)]
    logger.debug("{},{}".format(best, ",".join(feature_string)))

    #evaluate_best_leaf(weights, best_leaf_state, ttable)
    
    return best_action

def minimax(board, depth, weights, player_colour, alpha, beta, depth_limit, ttable, nturns):
    MIN = -1000
    MAX = 1000

    best_leaf_state = board

    if terminal_test(board, ttable, nturns):
        print("Terminal test!")
        return utility(board, ttable, nturns), None, board

    if depth == depth_limit:
        evaluation = evaluate(weights, board) # returns the reward for the given weight
        return evaluation, None, board
    
    if(player_colour=="white"):
        actions = actgen.get_possible_actions(board, "white")
       
        best = MIN
        best_action = None
        
        for action in actions:
            next_board = apply_action(player_colour, board, action)

            if action.action == "BOOM" and detect_suicide(board, next_board):
                continue

            score, _, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, ttable, nturns + 1)
            if score > best:
                best = score
                best_action = action
                best_leaf_state = best_leaf
            alpha = max(alpha, best)

            if alpha >= beta:
                break

    else:
        actions = actgen.get_possible_actions(board, "black")
        best = MAX
        best_action = None
        for action in actions:
            next_board = apply_action(player_colour, board, action)

            if action.action == "BOOM" and detect_suicide(board, next_board):
                continue

            score, _, best_leaf = minimax(next_board, depth+1, weights, "white", alpha, beta, depth_limit, ttable, nturns + 1)
            
            if score < best:
                best = score
                best_action = action
                best_leaf_state = best_leaf
            beta = min(beta, best)
            if beta <= alpha:
                break
        
    return best, best_action, best_leaf_state

def evaluate(weights, state):
    """ Returns an evaluation value for a given action. """
    
    features = calc_features.get_features(state)
    eval = np.dot(weights, features)
    if len(state["black"]) == len(state["white"]) == 0 :
        reward = 0
    else: reward = math.tanh(eval)

    return reward

#def evaluate_best_leaf(weights, state, ttable):
#    # Ignores draws due to exceeding number of moves
#    if terminal_test(state, ttable, 0):
#        return utility(state, ttable, 0)
#    else:
#        return evaluate(weights, state)

def is_repeated(board, ttable):
    #Previously visited 3 times  - this time is the 4th
    if ttable.getCount(board) == 3:
        print("Draw!")
        return True

def terminal_test(board, ttable, nturns):
    n_white, n_black = count_tokens(board)
    if (n_white == 0 or 
    n_black == 0 or 
    is_repeated(board, ttable) or 
    nturns >= 250*2 or
    n_white == n_black == 1):
        print("Terminal state found")
        return True

def utility(board, ttable, nturns):
    n_white, n_black = count_tokens(board)
    
    if n_white == n_black == 1:
        print("Guaranteed draw")
        return -0.01
        
    if nturns >= 250*2:
        print("Draw by steps")
        return -0.01

    if is_repeated(board, ttable):
        print("Draw by repeated states")
        return -0.01
    
    if n_white == 0 and n_black == 0:
        print("Draw by mutual annihilation")
        return 0
    
    elif n_white == 0: 
        print("Opponent wins")
        return -1
    
    else: 
        print("I win")
        return 1

def apply_action(player_colour, board, action):
    """ Applies an action to the board and returns the new board configuration. """

    if action.action == "BOOM":
        next_board = game.boom(action.source, board)
        
    if action.action == "MOVE":
        next_board = game.move(action.num, action.source, action.target, board, player_colour)
    return next_board

def detect_suicide(board, next_board):
    before_w, before_b = count_tokens(board)
    next_w, next_b = count_tokens(next_board)
    return (before_w != next_w) ^ (before_b != next_b)

def flip_board(board, colour):
    "Flip board positions if its black"

    if colour == "black":
        new_state = {}
        new_state["black"] = [flip_row(stack) for stack in board["white"]]
        new_state["white"] = [flip_row(stack) for stack in board["black"]]
    else:
        new_state = board
    return new_state

def flip_row(stack):
    "Helper function for board flipper"
    return (stack[0], stack[1], 7 - stack[2])

def flip_action(action, colour):
    if colour == "black":
        action.source = (action.source[0], 7 - action.source[1])
        action.target = (action.target[0], 7 - action.target[1])
    return action

