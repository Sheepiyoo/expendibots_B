"""
Module for all functions related to decision making
"""
from ok_boomer.game import *
from ok_boomer.util import *
import logging

FORMAT = '%(asctime)s: %(levelname)s: %(message)s'

formatter = logging.Formatter(FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(filename='action-gen.log', mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class Action:
    def __init__(self, action, num, source, target):
        self.action = action
        self.num = num
        self.source = source
        self.target = target

    def toTuple(self):
        if self.action == "BOOM":
            return (self.action, self.source)
        
        elif self.action == "MOVE":
            return (self.action, self.num, self.source, self.target)

        raise Exception("Invalid action name")
    
    def __repr__(self):
        return str(self.toTuple())

def get_possible_actions(board, player_colour):
    logger.debug(print_board(get_grid_format(board)))
    all_moves = []
    stacks = board[player_colour]
    #print(player_colour, stacks)
    for stack_from in stacks:
        all_moves += get_possible_actions_from_stack(stack_from, board, player_colour)
    
    all_moves.sort(key=lambda x: x.action=="MOVE")
    return all_moves

def detect_suicide(coordinate, board, player_colour):
    next_board = boom(coordinate, board)
    before_w, before_b = count_tokens(board)
    after_w, after_b = count_tokens(next_board)
    
    return (before_w == after_w) ^ (before_b == after_b)    # True if only white or black's tokens change, but not both

# Copied from big brains
def count_tokens(board):
    """ Returns the number of white and black tokens on the board """
    count = [0, 0]

    for n, x, y in board["white"]:
        count[0] += n

    for n, x, y in board["black"]:
        count[1] += n
    
    return count

# returns possible moves for a given stack
def get_possible_actions_from_stack(stack_from, board, player_colour):
    grid_board = get_grid_format(board)
    possible_actions = []
    x_pos, y_pos = stack_from[X_POS],  stack_from[Y_POS]
    

    if not (detect_suicide((x_pos, y_pos), board, player_colour)):
        possible_actions.append(Action("BOOM", 1, (x_pos, y_pos), (x_pos, y_pos)))
    else:
        logger.debug("{colour} BOOM at ({x}, {y}) is suicide".format(colour = player_colour, x = x_pos, y = y_pos))
    
    # for each possible stack of n tokens 
    for n in range(1, stack_from[N_TOKENS]+1):
        
        # for each possible position from given position
        for (x, y) in possible_positions(stack_from[X_POS], stack_from[Y_POS], n):

            # if a stack already exists on the board, add the stack
            if (x, y) in grid_board:
                if not is_opponent(grid_board[(x, y)], player_colour):
                    for i in range(1, stack_from[0]+1):
                        possible_actions.append(Action("MOVE", i, (x_pos, y_pos), (x, y)))
                
            else:
                for i in range(1, stack_from[0]+1):
                        possible_actions.append(Action("MOVE", i, (x_pos, y_pos), (x, y)))

    #print(possible_actions)
    return possible_actions

def is_opponent(colour_n, player_colour):
    player = colour_n[0]
    if player != player_colour[0]:
        return True
    return False

# returns a list of in-bound positions n spaces away from given x,y
def possible_positions(x, y, n):
    positions = []
    if y+n < BOARD_SIZE:
        positions.append((x, y+n))
    if x+n < BOARD_SIZE:
        positions.append((x+n, y))
    if (x-n) >= 0:
        positions.append((x-n, y))
    if y-n >= 0:
        positions.append((x, y-n))
    #print('the positions are', positions)
    return positions

