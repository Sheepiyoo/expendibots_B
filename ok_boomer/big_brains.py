import ok_boomer.action_generator as actgen
import ok_boomer.game as game
import random
from ok_boomer.util import *
from ok_boomer.constants import *


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
        score = evaluate(player, actions[i])
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

def evaluate(player, action):
    """ Returns an evaluation value for a given action. """
    
    before = count_tokens(player.board)    
    next_board = apply_action(player, action)
    after = count_tokens(next_board)

    if player.colour == "white":
        eval = (before[1] - after[1]) - (before[0] - after[0])
    else:
        eval = (before[0] - after[0]) - (before[1] - after[1])

    return eval/max(1, heuristic(next_board, player.colour))


def apply_action(player, action):
    """ Applies an action to the board and returns the new board configuration. """

    if action.action == "BOOM":
        next_board = game.boom(action.source, player.board)
        
    if action.action == "MOVE":
        next_board = game.move(action.num, action.source, action.target, player.board, player.colour)
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
    
    return sum(distances)//best_stack

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
def minimax(node, depth, player_colour, values, alpha, beta):
    if depth == 3:
        return node.heuristic
    if(player_colour):
        best = MIN
        for child in node.children:
            node.heuristic = minimax(node, depth+1, False, values, alpha, beta)
            best = max(best, node.val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best

    else:
        best = MAX
        for child in node.children:
            node.heuristic = minimax(node, depth+1, True, values, alpha, beta)
            best = min(best, node.val)
            beta = min(beta, best)
            if alpha <= beta:
                break
        return best



