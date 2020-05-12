import ok_boomer_tdleaf_self_play.game as game
from ok_boomer_tdleaf_self_play.util import *
from ok_boomer_tdleaf_self_play.constants import *
import numpy as np
import math
from ok_boomer_tdleaf_self_play.action_generator import possible_positions

def get_features(state):
    features = []

    ###-------------------- difference of tokens--------------------###
    features.append(get_token_difference(state))
    
    nw, nb = get_token_number(state)

    features.append(nw)
    features.append(nb)

    ###-------------------- difference of stacks --------------------###
    features.append(get_stack_difference(state))

    ###-------------------- difference of chunks --------------------### 
    features.append(get_chunk_difference(state))
    
    ###-------------------- difference of ratio token --------------------### 
    features.append(get_token_ratio(state))

    ###-------------------- distance to opponent tokens --------------------### 
    features.append(distance_heuristic(state, "white"))

    ###-------------------- difference of neigbour count --------------------### 
    features.append(count_neighbours(state))

    ###-------------------- stack height differences --------------------### 
    features += stack_height_difference_counter(state)

    ###-------------------- stack height differences --------------------###
    features.append(1)

    ###-------------------- difference of targetable tokens --------------------### 
    #features.append(calculate_targetable_diff(state))

    ###-------------------- normalise features -------------------- ###
    #normalise(features)


    ###-------------------- average location of all tokens --------------------###
    # features.append(get_average_location)
    # I think these features were useless bc the weights got too small 
    #avg_white_x, avg_white_y = get_avg_loc(state, "white")
    #avg_black_x, avg_black_y = get_avg_loc(state, "black")
    #features.append((avg_white_x-avg_black_x)/8)
    #features.append((avg_white_y-avg_black_y)/8)

    ###-------------------- difference of distances --------------------### 
    #features.append(distance_heuristic(state, "white") - distance_heuristic(state, "black"))
    ###-------------------- difference of distances --------------------###
    #features.append(get_danger_score(state))

    ###-------------------- corner position --------------------### 
    # difference in corner positions
    # score_corners(grid_format)

    ###-------------------- difference of area covered --------------------### 
    # features.append(difference_of_area(state))

    #features.append((calc_dist_middle(state,"white")/48)-(calc_dist_middle(state,"black")/48))
    ###-------------------- difference in edge positions --------------------### 
    ###-------------------- difference in centre positions --------------------### 
    ###-------------------- number in opponent half --------------------### 
    
    #print(len(features))
    return np.array(features)

def normalise(features):
    total = 0
    normalised = []
    for feature in features:
        total += feature
    for feature in features:
        normalised.append(feature/total)
    return normalised

def get_token_number(state):
    nw, nb = count_tokens(state)
    return nw/12, nb/12



def get_token_difference(state):
    nw, nb = count_tokens(state)
    #features.append(nw/12)
    #features.append(nb/12)
    return (nw-nb)/12

def get_stack_difference(state):
    return (len(state["white"]) - len(state["black"]))/12

def get_chunk_difference(state):
    white_chunks = len(get_chunks({"white": state["white"]}))
    black_chunks = len(get_chunks({"black": state["black"]}))
    return (white_chunks-black_chunks)/12
    
def get_token_ratio(state):
    nw, nb = count_tokens(state)
    return (nw+1)/(nb+1)

def distance_heuristic(board, colour):
    """ Distance heuristic """
    
    # Best of n stack distance
    if len(board[colour]) > 0:
        best_stack = max([stack[N_TOKENS] for stack in board[colour]])
    else:
        best_stack = 1

    distances = []
    
    for stack in board[opponent(colour)]:
        distances.append(min_distance_from_stack(stack, board[colour]))
    
    w, b = count_tokens(board)

    if len(distances) == 0: return 0

    return math.tanh(1/sum(distances))


def count_neighbours(state):
    grid = game.get_grid_format(state)
    coordinates = grid.keys()
    neighW = 0
    neighB = 0

    for position in coordinates:
        x, y = position
        isWhite = (grid[position][0] == "w")

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                point = (x+dx, y+dy)
                if point in coordinates:
                    if isWhite and grid[point][0] != "w":
                        neighW += int(grid[point][1:])
                    elif not isWhite and grid[point][0] != "b":
                        neighB += int(grid[point][1:])

    return (neighW - neighB)/12

def stack_height_difference_counter(state):
    wStacks = state["white"]
    bStacks = state["black"]

    wCounter = [0, 0, 0]
    bCounter = [0, 0, 0]

    for stack in wStacks:
        #if stack[N_TOKENS] >= 5:
        #    wCounter[3] += 1
        if stack[N_TOKENS] == 4:
            wCounter[2] += 1
        elif stack[N_TOKENS] == 3:
            wCounter[1] += 1
        elif stack[N_TOKENS] == 2:
            wCounter[0] += 1   

    for stack in bStacks:
        #if stack[N_TOKENS] >= 5:
        #    bCounter[3] += 1
        if stack[N_TOKENS] == 4:
            bCounter[2] += 1
        elif stack[N_TOKENS] == 3:
            bCounter[1] += 1
        elif stack[N_TOKENS] == 2:
            bCounter[0] += 1

    diff = [(wCounter[i] - bCounter[i])/12 for i in range(len(wCounter))]
    
    return diff

def calculate_targetable_diff(state):
    grid = game.get_grid_format(state)
    
    bStacks = state["black"]
    wStacks = state["white"]
    bCounter = wCounter = 0

    for stack in bStacks:
        bCounter += count_targets("black", state, stack)
    
    for stack in wStacks:
        wCounter += count_targets("white", state, stack)

    return (wCounter - bCounter)/12


#################################################################################
def opponent_colour(colour):
    if colour == "white":
        return "black"
    return "white"

def count_targets(colour, state, stack):
    grid = game.get_grid_format({opponent_colour(colour):state[opponent_colour(colour)]})
    area = get_coverage(stack)
   
    count = 0

    for point in grid.keys():
        if point in area:
            count += int(grid[point][1:])

    return count

def get_coverage(stack):
    area = set()
    n, x, y = stack

    for i in range(max(0, x - n - 1), min(x + n + 1, 7) + 1):
        for j in range(max(0, y - 1), min(y + 1, 7) + 1):
            area.add((i, j))

    for j in range(max(0, y - n - 1), min(y + n + 1, 7) + 1):
        for i in range(max(0, x - 1), min(x + 1, 7) + 1):
            area.add((i, j))

    return area

def difference_of_area(state):
    return (calc_spread(state, "white") - calc_spread(state, "black"))/32

#calculate the distance from the middle
def calc_dist_middle(state, colour):
    spread = 0
    for stack in state[colour]:
        spread += manhattan_distance(stack, [0, 3.5, 3.5])
    return spread

def get_avg_loc(state, colour):
    x, y = 0, 0
    for stack in state[colour]:
        x += stack[1]*stack[0]
        y += stack[2]*stack[0]
    x = x/len(state[colour])
    y = y/len(state[colour])
    return x, y


"""
def score_corners(grid_state):
    corners = [(0, 0), (0, 7), (7, 7), (0, 0)]
    print(grid_state)
    for point in corners:
        pass
"""

def calc_spread(state, colour):
    x_positions = []
    y_positions = []
    if len(state[colour]) == 0:
        return 0
    for stack in state[colour]:
        x_positions.append(stack[1])
        y_positions.append(stack[2])
    x_positions.sort()
    y_positions.sort()
    #print(x_positions)
    x_spread = x_positions[-1] - x_positions[0]
    y_spread = y_positions[-1] - y_positions[0]
    return x_spread*y_spread


def calc_white_half(state):
    count = 0
    for stack in state["white"]:
        if stack[Y_POS] <= 3:
            count += stack[N_TOKENS]

    for stack in state["black"]:
        if stack[Y_POS] <= 3:
            count -= stack[N_TOKENS]

    return count

def calc_black_half(state):
    count = 0
    for stack in state["white"]:
        if stack[Y_POS] >= 4:
            count += stack[N_TOKENS]

    for stack in state["black"]:
        if stack[Y_POS] >= 4:
            count -= stack[N_TOKENS]

    return count

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

def is_low_risk(board):
    if((len(board["white"])+len(board["black"])>20) and (abs(len(board["white"])-len(board["black"])<2))):
        return True
    return False

# normalise the weights !!!
# does the chunkc calculation outweight the benefit of having a larger depth



def distance_heuristic2(board, colour):
    # Best of n stack distance
    if len(board[colour]) > 0:
        best_stack = max([stack[N_TOKENS] for stack in board[colour]])
    else:
        best_stack = 1

    distances = 0
    
    for stack in board[opponent(colour)]:
        distances += min_distance_from_stack(stack, board[colour])
    

    return 1/distances

def get_danger_score(board):
    score = 0
    for stack in board["white"]:
        score += len(possible_positions(stack[1], stack[2], 1))
    return score/96
        

def min_distance_from_stack(source, stack_list):
    # Minimum distance from a black stack to one of the white stacks
    min_distance = BOARD_SIZE*2
    for i in range(len(stack_list)):
        h_dist = manhattan_distance(source, stack_list[i]) 
        min_distance = min(min_distance, h_dist)

    return min_distance

def manhattan_distance(stack1, stack2):
    # Chess board distance as booming can take surrounding 9 squares
    return abs(stack1[X_POS]-stack2[X_POS]) + abs(stack1[Y_POS]-stack2[Y_POS])