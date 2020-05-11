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

B_EXPLORED = 0
W_EXPLORED = 0

FORMAT = '%(message)s'

formatter = logging.Formatter(FORMAT)

timestamp = time.time()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(filename='training/data.log'.format(timestamp), mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def count_prune():
    return

class TTableEntry:
    """
    Structure for storing info about a given state
    """
    def __init__(self, evaluation, depth, best_leaf, best_action):
        self.evaluation = evaluation
        self.depth = depth
        self.best_leaf = best_leaf
        self.best_action = best_action
    
    def unpack(self):
        return (self.evaluation, self.depth, self.best_leaf, self.best_action)
    
    def __str__(self):
        return str("""
        Evaluation:     {}
        Depth:         {}
        Best_action:    {}
        """.format(self.evaluation, self.depth, self.best_action))

class TTable:
    """
    Transposition table, indexed by state
    """
    def __init__(self):
        self.visited_states = {}

    def contains(self, colour, board):
        State = self.dict_to_set(colour, board)
        try:
            x = self.visited_states[State]
            return True
        except:
            return False

    def get_info(self, colour, board):
        state = self.dict_to_set(colour, board)
        return self.visited_states[state]

    def addState(self, colour, board, evaluation, depth, best_leaf, best_action):
        if depth == 0: return
        if best_action == None: return
        if self.contains(colour, board) and depth <= self.get_info(colour, board).depth: return
        
        wasIn = self.contains(colour, board)
        
        #if wasIn:
        #    logger.debug("Old: " + str(self.get_info(colour, board)))
        
        State = self.dict_to_set(colour, board)
        self.visited_states[State] = TTableEntry(evaluation, depth, best_leaf, best_action)
        
        #if wasIn:
        #   logger.debug("New: " + str(self.get_info(colour, board)))
        
    def action_counter(self, colour, board):
        #logger.debug("Action looked up")
        return self.get_info(colour, board).best_action

    def actionLookup(self, colour, board):
        if self.contains(colour, board):
            return self.action_counter(colour, board)
        else:
            return None

    def clear(self):
        self.visited_states = {}        
    
    def dict_to_set(self, colour, board_dict):
        new_dict = set()
        if colour != "white":
            colour = "black"
        new_dict.add(colour)
        for i in board_dict["white"]:
            new_dict.add(tuple([0] + i))

        for i in board_dict["black"]:
            new_dict.add(tuple([1] + i))
        return frozenset(new_dict)

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


def iterative_depth_search(board, depth, weights, player_colour, alpha, beta, depth_limit, htable, ttable, nturns):

    board = flip_board(board, player_colour)
    player_colour = "white"

    max_depth = 4 #min(6, int(3 + (24-sum(count_tokens(board)))//8))
    print(max_depth)

    for i in range(2,max_depth):
        best_action = minimax_wrapper(board, depth, weights, player_colour, alpha, beta, i, htable, ttable, nturns)    
    
    best_action = flip_action(best_action, player_colour)

    global W_EXPLORED, B_EXPLORED    
    return best_action


""" Finds the best action for a given node by finding the best evaluation 
    1. Generates the actions for the root node. 
    2. Applies minimax on all actions 
    2.1 Checks if there are records
    2.1.1 Records are useful - Use records instead of minimax
    2.1.2 Records aren't 

    3a. If the records are useful use instead of minimax - 
    3b. If not useful, minimax needs to be applied
    4. If records don't exist, minimax """
def minimax_wrapper(board, depth, weights, player_colour, alpha, beta, depth_limit, htable, ttable, nturns):
    actions = actgen.get_possible_actions(board, player_colour)
    action_rank = []
    
    # check if this board is in table

    for action in actions:
        next_board = apply_action(player_colour, board, action)
        if action.action == "BOOM" and detect_suicide(board, next_board):
            continue

        """
        if ttable.contains("white", next_board):
            
            record_eval, record_depth, b_leaf, _ = ttable.get_info("white", next_board).unpack()
        
            if depth + record_depth >= depth_limit:
                score, best_leaf = record_eval, b_leaf
        
            else: 
                score, opp, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)
        else:
            score, opp, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)
        """    
        score, best_opp_action, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)
        alpha = max(score, alpha)
        action_rank.append((score, action, best_leaf))

        #print(score, action, best_opp_action)

    best, best_action, best_leaf_state = select_random_action(action_rank)
    #ttable.addState(player_colour, board, best, depth_limit-depth, best_leaf_state, best_action)
    
    #logger.debug(print_board(game.get_grid_format(best_leaf_state)))
    feature_string = [str(x) for x in calc_features.get_features(best_leaf_state)]
    #logger.debug("{},{}".format(best, ",".join(feature_string)))
    
    return best_action



def minimax(board, depth, weights, player_colour, alpha, beta, depth_limit, htable, ttable, nturns):
    MIN = -1000
    MAX = 1000
    global W_EXPLORED
    global B_EXPLORED
    
    if player_colour == "white": W_EXPLORED += 1
    else: B_EXPLORED += 1

    best_leaf_state = board

    if terminal_test(board, htable, nturns):
        return utility(board, htable, nturns), None, board

    if depth == depth_limit: 
        return evaluate(weights, board), None, board

    actions = actgen.get_possible_actions(board, player_colour)
    
    best_action = ttable.actionLookup(player_colour, board)
    if (best_action != None):
        actions.insert(0, best_action) 

    if(player_colour=="white"):
        
        best = MIN 

        for action in actions:
            next_board = apply_action(player_colour, board, action)

            if action.action == "BOOM" and detect_suicide(board, next_board):
                continue
            
            """ if ttable.contains(player_colour, next_board):
                record_eval, record_depth, b_leaf, _ = ttable.get_info(player_colour, next_board).unpack()
                if depth + record_depth >= depth_limit: score, best_leaf = record_eval, b_leaf
                else: 
                    score, a, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1)
            
            else:
            """
            score, a, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta, depth_limit, htable, ttable, nturns + 1) 

            if score > best: best, best_action, best_leaf_state = score, action, best_leaf

            alpha = max(alpha, best)

            if alpha >= beta:
                ttable.addState(player_colour, board, alpha, depth_limit-depth, best_leaf_state, best_action)
    
                return alpha, best_action, best_leaf_state
                break 
                

    if (player_colour=="black"):
        best = MAX

        for action in actions:
            next_board = apply_action(player_colour, board, action)

            if action.action == "BOOM" and detect_suicide(board, next_board):
                continue
            
            """ if ttable.contains(player_colour, next_board):
                
                record_eval, record_depth, record_leaf, _ = ttable.get_info(player_colour, next_board).unpack()
                
                if depth + record_depth >= depth_limit:
                    score, best_leaf = record_eval, record_leaf
            
                else:
                    score, a, best_leaf = minimax(next_board, depth+1, weights, "white", alpha, beta, depth_limit, htable, ttable, nturns + 1)
            else:"""

            score, a, best_leaf = minimax(next_board, depth+1, weights, "white", alpha, beta, depth_limit, htable, ttable, nturns + 1) 
            

            if score < best: best, best_action, best_leaf_state = score, action, best_leaf
            
            
            beta = min(beta, best)

            if beta <= alpha:
                ttable.addState(player_colour, board, beta, depth_limit-depth, best_leaf_state, best_action)
    
                return beta, best_action, best_leaf_state
                count_prune()
                break
    
    ttable.addState(player_colour, board, best, depth_limit-depth, best_leaf_state, best_action)
                 
    return best, best_action, best_leaf_state

def quiesce(board, depth, weights, player_colour, alpha, beta, depth_limit, htable, ttable, nturns):
    MIN = -1000
    MAX = 1000
    
    if terminal_test(board, htable, nturns):
        return utility(board, htable, nturns), None, board

    pass

def evaluate(weights, state):
    """ Returns an evaluation value for a given action."""
    
    WEIGHT = 0.9    # Controls importance of removing pieces vs moving closer
    EPSILON = 1

    before = count_tokens(state) 
    eval = ((before[0]) - (before[1]))/12   # Higher = More white removed
    #distance_heuristic = 1/max(EPSILON, math.tanh(calc_features.heuristic(state, "white"))) # Higher = White is closer to black

    return math.tanh(eval)
    
    """
    features = calc_features.get_features(state)
    eval = np.dot(weights, features)
    if len(state["black"]) == len(state["white"]) == 0 :
        reward = 0
    else: reward = math.tanh(eval)
    
    return reward"""
    

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
        #print("Terminal state found")
        return True

def utility(board, htable, nturns):
    n_white, n_black = count_tokens(board)
    
    if n_white == 0:
        if n_black == 0:
            #print("Draw by mutual annihilation")
            return -0.001
        else:
            #print("Opponent wins")
            return -1

    elif n_white == 1:
        if n_black == 0:
            #print("I win")
            return 1
        elif n_black == 1:
            #print("Guaranteed draw")
            return -0.001
        else:
            #print("Likely loss")
            return -1
    elif n_black == 0:
        #print("I win")
        return 1
            
    if nturns >= 250*2:
        #print("Draw by steps")
        return -0.001

    if is_repeated(board, htable):
        #print("Draw by repeated states")
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

""" Selects a random action from candidates of the same evaluation """
def select_random_action(action_rank):
    action_rank.sort(reverse=True, key=lambda x: x[0])
    trimmed_actions = [action_rank[0]]

    for i in range(1, len(action_rank)):
        if action_rank[i-1][0] == action_rank[i][0]:
            trimmed_actions.append(action_rank[i])
        else:
            break

    trimmed_actions.sort(reverse=True, key=lambda x: x[1].action=="BOOM")
    
    if trimmed_actions[0][1].action == "BOOM":
        selected_action = trimmed_actions[0]
    else:
        selected_action = trimmed_actions[0]
        #random.randint(0, len(trimmed_actions)-1)

    return selected_action