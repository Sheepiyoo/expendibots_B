import ok_boomer_tdleaf.game as game
from ok_boomer_tdleaf.util import *
from ok_boomer_tdleaf.constants import *
import numpy as np

def get_features(state):
    features = []

    grid_format = game.get_grid_format(state)

    ###-------------------- difference of tokens--------------------###
    nw, nb = count_tokens(state)
    features.append(nw/12)
    features.append(nb/12)
    features.append((nw-nb)/12)




    ###-------------------- difference of stacks --------------------###
    #features.append((len(state["white"]) - len(state["black"]))/12)
    
    ###-------------------- average location of all tokens --------------------###
    # I think these features were useless bc the weights got too small 
    #avg_white_x, avg_white_y = get_avg_loc(state, "white")
    #avg_black_x, avg_black_y = get_avg_loc(state, "black")
    #features.append((avg_white_x-avg_black_x)/8)
    #features.append((avg_white_y-avg_black_y)/8)



    ###-------------------- difference of chunks --------------------### 
    white_chunks = len(get_chunks({"white": state["white"]}))
    black_chunks = len(get_chunks({"black": state["black"]}))
    features.append((white_chunks-black_chunks)/12)

    ###-------------------- difference of distances --------------------### 
    features.append(heuristic(state, "white") - heuristic(state, "black"))

    ###-------------------- ratio of white:black tokens --------------------### 
    features.append((nw+1)/(nb+1))

    ###-------------------- corner position --------------------### 
    # difference in corner positions
    # score_corners(grid_format)


    ###-------------------- difference of area covered --------------------### 
    # difference in amount of area covered
    # features.append((calc_spread(state, "white") - calc_spread(state, "black"))/32)

    #features.append((calc_dist_middle(state,"white")/48)-(calc_dist_middle(state,"black")/48))
    ###-------------------- difference in edge positions --------------------### 
    ###-------------------- difference in centre positions --------------------### 
    ###-------------------- number in opponent half --------------------### 


    return np.array(features)

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

def heuristic(board, colour):
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
    count = w if colour=="white" else b

    distances.sort()

    if len(distances) == 0: return 0

    return 1/max(1, distances[0])

def heuristic2(board, colour):
    # Best of n stack distance
    if len(board[colour]) > 0:
        best_stack = max([stack[N_TOKENS] for stack in board[colour]])
    else:
        best_stack = 1

    distances = 0
    
    for stack in board[opponent(colour)]:
        distances += min_distance_from_stack(stack, board[colour])
    

    return 1/distances
        

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