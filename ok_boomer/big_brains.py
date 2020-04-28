import ok_boomer.action_generator as actgen
import random

def search(player):
    actions = actgen.get_possible_actions(player.board, player.colour)
    return actions[random.randint(0, len(actions)-1)].toTuple()