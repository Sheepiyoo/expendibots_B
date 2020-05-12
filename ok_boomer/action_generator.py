"""
Module for all functions related to decision making
"""
from ok_boomer.game import *
from ok_boomer.util import *
import logging
from collections import deque

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
    
    def __eq__(self, objects):
        if type(objects) != type(self): return False

        return (self.action == objects.action and
        self.num == objects.num and
        self.source == objects.source and
        self.target == objects.target)
    
    def __hash__(self):
        return hash(self.toTuple())



def get_possible_actions(board, player_colour):
    all_moves = []
    stacks = board[player_colour]
    stacks.sort(reverse=True, key=lambda stack: stack[N_TOKENS])
        
    for stack_from in stacks:
        all_moves += get_possible_actions_from_stack(stack_from, board, player_colour)
    
    all_moves.sort(reverse=True, key=lambda x: x.action == "BOOM")
    
    return all_moves

def get_possible_violent_actions(board, player_colour):
    
    all_moves = []
    stacks = board[player_colour]
    
    #Check for booms + yeets
    for stack in stacks:
        n_tokens = stack[N_TOKENS]
        x_pos, y_pos = stack[X_POS],  stack[Y_POS]
        all_moves.append(Action("BOOM", 1, (x_pos, y_pos), (x_pos, y_pos)))

        #if n_tokens >= 3:
        #    all_moves += get_possible_actions_from_stack(stack, board, player_colour)

    return all_moves

# returns possible moves for a given stack
def get_possible_actions_from_stack(stack_from, board, player_colour):
    grid_board = get_grid_format(board)

    """ for i in range(0, 8):
        for j in range(0, 8):
            if (i, j) not in grid_board:
                grid_board.append(((i,j), "empty"))
        for n in range(stack_from[N_TOKENS], 0, -1):
            poss_pos = [(0)] """
    possible_actions = deque()

    x_pos, y_pos = stack_from[X_POS],  stack_from[Y_POS]

    possible_actions.appendleft(Action("BOOM", 1, (x_pos, y_pos), (x_pos, y_pos)))
    
    for n in range(stack_from[N_TOKENS], 0, -1):   
        for (x, y) in possible_positions(stack_from[X_POS], stack_from[Y_POS], n):
            if (x, y) in grid_board:
                if not is_opponent(grid_board[(x, y)], player_colour):
                    for i in range(1, stack_from[0]+1):
                        possible_actions.appendleft(Action("MOVE", i, (x_pos, y_pos), (x, y)))
            else:
                for i in range(1, stack_from[0]+1):
                        possible_actions.append(Action("MOVE", i, (x_pos, y_pos), (x, y)))
    
    return list(possible_actions)

def is_opponent(colour_n, player_colour):
    player = colour_n[0]
    if player != player_colour[0]:
        return True
    #if colour_n == empty
    return False

def possible_positions(x, y, n):
    positions = []
    if y+n < BOARD_SIZE: #up
        positions.append((x, y+n))
    if x+n < BOARD_SIZE: #right
        positions.append((x+n, y))
    if (x-n) >= 0: #left
        positions.append((x-n, y))
    if y-n >= 0: # down
        positions.append((x, y-n))
    #print('the positions are', positions)
    return positions

# returns a list of in-bound positions n spaces away from given x,y
def possible_positions1(x, y, n, order):
    ordered_positions = []
    positions = {}
    if y+n < BOARD_SIZE: #up
        positions["up"] = (x, y+n)
    if x+n < BOARD_SIZE: #right
        positions["right"] = (x+n, y)
    if (x-n) >= 0: #left
        positions["left"] = (x-n, y)
    if y-n >= 0: # down
        positions["down"] = (x, y-n)
    #print('the positions are', positions)
    for direction in order:
        if direction[0] in positions.keys():
            ordered_positions.append(positions[direction[0]])
    return ordered_positions

# returns Left, Right, Up, Down, in order of which half has the most tokens of the colour given
def analyse_board(board, colour):
    order = {"up":0, "down":0, "left":0, "right":0}
    for stack in board[colour]:
        if stack[X_POS] <= BOARD_SIZE/2:
            order["left"] += stack[N_TOKENS]
        if stack[X_POS] > BOARD_SIZE/2:
            order["right"] += stack[N_TOKENS]
        if stack[Y_POS] <= BOARD_SIZE/2:
            order["down"] += stack[N_TOKENS]
        if stack[Y_POS] > BOARD_SIZE/2:
            order["up"] += stack[N_TOKENS]
    final_order = sorted(order.items(), key=lambda x: x[1], reverse=True)
    return final_order

