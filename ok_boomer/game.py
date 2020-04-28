# Converts from (x,y) as keys to "white", "black" as keys
# {(x, y): "wn"} --> {"white": [n, x, y]}
from ok_boomer.constants import *
from ok_boomer.util import *


def get_token_format(grid_dict):
    token_format = {"white":[], "black": []}
    for coordinate in grid_dict.keys():
        x, y = coordinate
        player = grid_dict[coordinate][0]
        n = grid_dict[coordinate][1:]

        if player == 'w':
            token_format["white"].append([int(n), x, y])
        elif player == 'b':
            token_format["black"].append([int(n), x, y])

    return token_format

# Converts from "white", "black" as keys to (x,y) as keys 
# {"white": [n, x, y]} --> {(x, y): "wn"}       
def get_grid_format(board_dict):
    grid_format = {}
    for player in board_dict.keys():
        for stack in board_dict[player]:
            grid_format[(stack[1], stack[2])] = ''.join([player[0], str(stack[0])])
    return grid_format



def move(n, from_pos, to_pos, board_dict, colour):
    grid_list = get_grid_format(board_dict)

    stack_from = grid_list[(from_pos[0], from_pos[1])], from_pos[0], from_pos[1]
    stack_to = n, to_pos[0], from_pos[1]

    grid_list[from_pos] = colour[0] + str(stack_from[N_TOKENS] - stack_to[N_TOKENS])

    # if we moved all tokens from a square, clear it from the dictionary representation
    if (int(grid_list[from_pos][1])) == 0:
        del(grid_list[from_pos])

    # if the position is already occupied
    if (to_pos) in grid_list:

        total = int(grid_list[to_pos][1]) + stack_to[N_TOKENS]
        grid_list[to_pos] = colour[0] + str(total)
        #print_move(stack_to[N_TOKENS], stack_from[X_POS], stack_from[Y_POS], stack_to[X_POS], stack_to[Y_POS])
            
    # if it's not occupied
    else:
        grid_list[to_pos] = colour[0] + str(stack_to[N_TOKENS])
        #print_move(stack_to[N_TOKENS], stack_from[X_POS], stack_from[Y_POS], stack_to[X_POS], stack_to[Y_POS])
    
    return get_token_format(grid_list)

# Preprocessing for boom
def boom(stack, board_dict):
    _, x, y = stack   
    grid_format = get_grid_format(board_dict)
    boom_recursive(x, y, grid_format)
    return get_token_format(grid_format)


def boom_recursive(x, y, grid_format):
    #Check bounds
    if not (0 <= x < 8 and 0 <= y < 8):
        return
    
    #If a token is present, explode!        
    if (x, y) in grid_format.keys():
        del(grid_format[(x,y)])
        
        #Debug line
        #print("# Removed token at" , x, y)

        #Recursive explosion
        for i in range(-1,2):
            for j in range(-1, 2):
                boom_recursive(x+i, y+j, grid_format)

    return