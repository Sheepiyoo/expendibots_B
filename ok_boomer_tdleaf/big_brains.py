import ok_boomer_tdleaf.action_generator as actgen
import ok_boomer_tdleaf.game as game
from ok_boomer_tdleaf.util import *
from ok_boomer_tdleaf.constants import *

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

def evaluate(weights, state):
    """ Returns an evaluation value for a given action. """
    features = get_features(state)
    eval = np.dot(weights, features)
    reward = math.tanh(eval)
    return reward

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

    distances.sort()

    return 1/max(1, distances[0])

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

def minimax(board, depth, weights, player_colour, alpha, beta):
    best_leaf_state = board
    if is_game_over(board):
        return utility(board), None, board

    if depth == 3:
        evaluation = evaluate(weights, board)
        return evaluation, None, board
    
    if(player_colour=="white"):
        actions = actgen.get_possible_actions(board, "white")
       
        best = MIN
        best_action = None
        
        for action in actions:
            next_board = apply_action(player_colour, board, action)
            score, _, best_leaf = minimax(next_board, depth+1, weights, "black", alpha, beta)
            if score > best:
                best = score
                best_action = action
                best_leaf_state = best_leaf
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
            score, _, best_leaf = minimax(next_board, depth+1, weights, "white", alpha, beta)
            
            if score < best:
                best = score
                best_action = action
                best_leaf_state = best_leaf
            beta = min(beta, best)
            if beta <= alpha:
                #print("BROKEN")
                break
        #print("the best action for black is", best, best_action)
    
    if (depth==1):
        #logger.debug(print_board(game.get_grid_format(best_leaf_state)))
        feature_string = [str(x) for x in get_features(best_leaf_state)]
        logger.debug("{},{}".format(best, ",".join(feature_string)))
        
    return best, best_action, best_leaf_state

def get_features(state):
    features = []

    # difference of tokens
    nw, nb = count_tokens(state)
    features.append(nw-nb)

    # difference of stacks  
    features.append(len(state["white"]) - len(state["black"]))
    
    # difference of chunks
    white_chunks = len(get_chunks({"white": state["white"]}))
    black_chunks = len(get_chunks({"black": state["black"]}))
    features.append(white_chunks-black_chunks)

    # difference of distances
    #features.append(heuristic(state, "white") - heuristic(state, "black"))

    return np.array(features)

def get_chunks(board_dict):
    chunks = [] 
    grid_format = game.get_grid_format(board_dict)
    grid_copy = grid_format.copy()

    for (x,y) in grid_copy:
        chunk = []
        chunk_recursive(x, y, grid_format, chunk)
        if len(chunk) != 0:
          chunks.append(chunk)
    return chunks


def chunk_recursive(x, y, grid_format, chunk):
    #Check bounds
    if not (0 <= x < 8 and 0 <= y < 8):
        return
    
    #If a token is present, explode!        
    if (x, y) in grid_format.keys():
        chunk.append([int(grid_format[(x,y)][1:]), x, y])
        del(grid_format[(x,y)])

        #Recursive explosion
        for i in range(-1,2):
            for j in range(-1, 2):
                chunk_recursive(x+i, y+j, grid_format, chunk)
    else:
      return
    return   