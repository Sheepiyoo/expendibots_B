"""
Module for all functions related to decision making
"""
from ok_boomer.game import *

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
    all_moves = []
    stacks = board[player_colour]
    #print(player_colour, stacks)
    for stack_from in stacks:
        all_moves += get_possible_actions_from_stack(stack_from, board, player_colour)
    
    return all_moves

# returns possible moves for a given stack
def get_possible_actions_from_stack(stack_from, board, player_colour):
    grid_board = get_grid_format(board)
    possible_actions = []
    x_pos, y_pos = stack_from[X_POS],  stack_from[Y_POS]
    possible_actions.append(Action("BOOM", 1, (x_pos, y_pos), (x_pos, y_pos)))
    
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

