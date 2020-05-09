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

    def contains(self, board):
        State = self.dict_to_set(board)
        if State in self.visited_states.keys():
            return True
        else: return False

    def get_info(self, board):
        state = self.dict_to_set(board)
        return self.visited_states[state]

    def addState(self, board, evaluation, depth, best_leaf):
        State = self.dict_to_set(board)
        self.visited_states[State] = (evaluation, depth, best_leaf)

    def getCount(self, board):
        State = self.dict_to_set(board)
        
        return 0

    def dict_to_set(self, board_dict):
        new_dict = set()
        for i in board_dict["white"]:
            new_dict.add(tuple([0] + i))

        for i in board_dict["black"]:
            new_dict.add(tuple([1] + i))
        
        return frozenset(new_dict)
    
    def clear(self):
        self.visited_states = {}

    def __str__(self):
        return str(self.visited_states.keys())

class HTable:
    def __init__(self):
        self.visited_states = {}

    def in_state(self, board):
        State = self.dict_to_set(board)
        if State in self.visited_states.keys():
            return True
        else: return False

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
    
    def clear(self):
        self.visited_states = {}


def iterative_depth_search(board, depth, weights, player_colour, alpha, beta, depth_limit, htable, nturns):
    ttable =TTable()

    board = flip_board(board, player_colour)
    
    for i in range(5,6):
        best_action = minimax_wrapper(board, depth, weights, "white", alpha, beta, i, htable, ttable, nturns)
        print(i)
    
    #best_action = minimax_wrapper(board, depth, weights, "white", alpha, beta, 4, ttable, nturns)
    logger.debug(str(ttable))
    best_action = flip_action(best_action, player_colour)
    
    return best_action

def select_random_action(action_rank):
    trimmed_actions = [action_rank[0]]
    #print(trimmed_actions)
    for i in range(1, len(action_rank)):
        if action_rank[i-1][0] == action_rank[i][0]:
            trimmed_actions.append(action_rank[i])
        else:
            break    
    selected_action = action_rank[random.randint(0, len(trimmed_actions)-1)]
    return selected_action

def minimax_wrapper(board, depth, weights, player_colour, alpha, beta, depth_limit, htable, ttable, nturns):
    # Play as white
    actions = actgen.get_possible_actions(board, "white")
    action_rank = []
    
    for action in actions:
        next_board = apply_action(player_colour, board, action)
        if action.action == "BOOM" and detect_suicide(board, next_board):
            continue
        """
        if ttable.contains(next_board):
            record_eval, record_depth, b_leaf = ttable.get_info(next_board)
        
            if depth + record_depth >= depth_limit:
                    score = record_eval
                    best_leaf = b_leaf
            else: score, _, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)            
        else:
            score, _, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)
        """
        score, _, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)
        alpha = max(alpha, score)
        action_rank.append((score, action, best_leaf))
    
    #print(actions)
    #print(action_rank)
    action_rank.sort(reverse=True, key=lambda x: x[0])
    best, best_action, best_leaf_state = select_random_action(action_rank)

    #logger.debug(print_board(game.get_grid_format(best_leaf_state)))
    feature_string = [str(x) for x in calc_features.get_features(best_leaf_state)]
    logger.debug("{},{}".format(best, ",".join(feature_string)))

    #evaluate_best_leaf(weights, best_leaf_state, ttable)
    
    return best_action


def minimax(board, depth, weights, player_colour, alpha, beta, depth_limit, htable, ttable, nturns):
    MIN = -1000
    MAX = 1000
    
    best_leaf_state = board

    if terminal_test(board, htable, nturns):
        #print("Terminal test!")
        return utility(board, htable, nturns), None, board

    if depth == depth_limit:
        #print("Depth limit {} reached".format(depth))
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
            """
            elif ttable.contains(next_board):
                #print("Maximiser: Found visited state")
                #check if depth limit 
                record_eval, record_depth, b_leaf = ttable.get_info(next_board)
                if depth + record_depth >= depth_limit:
                    score = record_eval
                    best_leaf = b_leaf
                else: score, _, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)
            else:
                score, _, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)
                ttable.addState(next_board, score, depth_limit-depth, best_leaf)
            """

            score, _, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)

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
            """
            elif ttable.contains(next_board):
                #print("Minimiser: Found visited state")
                #check if depth limit 
                record_eval, record_depth, b_leaf = ttable.get_info(next_board)
                if depth + record_depth >= depth_limit:
                    score = record_eval
                    best_leaf = b_leaf
                else: score, _, best_leaf = minimax(next_board, depth+1, weights, "white", alpha, beta, depth_limit, htable, ttable, nturns + 1)
            else:
                score, _, best_leaf = minimax(next_board, depth+1, weights, "white", alpha, beta, depth_limit, htable, ttable, nturns + 1)
                ttable.addState(next_board, score, depth_limit-depth, best_leaf)
            """
            score, _, best_leaf = minimax(next_board, depth+1, weights, "white", alpha, beta, depth_limit, htable, ttable, nturns + 1)

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
    WEIGHT = 0.9    # Controls importance of removing pieces vs moving closer
    EPSILON = 1

    before = count_tokens(state) 
    eval = (before[0]) - (before[1])    # Higher = More white removed
    distance_heuristic = 1/max(EPSILON, math.tanh(calc_features.heuristic(state, "white"))) # Higher = White is closer to black

    return math.tanh(eval)
    """
    features = calc_features.get_features(state)
    eval = np.dot(weights, features)
    if len(state["black"]) == len(state["white"]) == 0 :
        reward = 0
    else: reward = math.tanh(eval)
    """

    #return reward

def is_repeated(board, htable):
    #Previously visited 3 times  - this time is the 4th
    if htable.getCount(board) == 3:
        print("Draw!")
        return True

def terminal_test(board, htable, nturns):
    n_white, n_black = count_tokens(board)
    if (n_white == 0 or 
        n_black == 0 or 
        is_repeated(board, htable) or 
        nturns >= 250*2 or
        n_white == 1):
        print("Terminal state found")
        return True

def utility(board, htable, nturns):
    n_white, n_black = count_tokens(board)
    
    if n_white == 0:
        if n_black == 0:
            print("Draw by mutual annihilation")
            return -0.001
        else:
            print("Opponent wins")
            return -1

    elif n_white == 1:
        if n_black == 0:
            print("I win")
            return 1
        elif n_black == 1:
            print("Guaranteed draw")
            return -0.001
        else:
            return -1
    elif n_black == 0:
        print("I win")
        return 1
            
    if nturns >= 250*2:
        print("Draw by steps")
        return -0.001

    if is_repeated(board, htable):
        print("Draw by repeated states")
        return -0.001

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

