from ok_boomer_tdleaf.util import *
from ok_boomer_tdleaf.game import *

board = {'white': [[1, 0, 1], [1, 3, 1], [1, 4, 1], [1, 6, 1], [1, 7, 1], [1, 0, 0], [1, 1, 0], [1, 3, 0], [1, 4, 0], [1, 6, 0], [1, 7, 0], [1, 1, 3]], 'black': [[1, 0, 7], [1, 1, 7], [1, 3, 7], [1, 4, 7], [1, 6, 7], [1, 7, 7], [1, 1, 6], [1, 3, 6], [1, 4, 6], [1, 6, 6], [1, 7, 6], [1, 0, 5]]}
print(print_board(get_grid_format(board)))