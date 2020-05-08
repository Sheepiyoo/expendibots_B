# coding: utf-8
from ok_boomer_tdleaf.constants import *
from ok_boomer_tdleaf.game import *
import ok_boomer_tdleaf.big_brains as bb
import ok_boomer_tdleaf.machine_learning as ml
import time
import numpy as np
import os


class ExamplePlayer:
    _BLACK_START_SQUARES = [(0,7), (1,7),   (3,7), (4,7),   (6,7), (7,7),
                        (0,6), (1,6),   (3,6), (4,6),   (6,6), (7,6)]
    _WHITE_START_SQUARES = [(0,1), (1,1),   (3,1), (4,1),   (6,1), (7,1),
                        (0,0), (1,0),   (3,0), (4,0),   (6,0), (7,0)]

    INITIAL_BOARD = {
        "white": [[1, x, y] for x,y in _WHITE_START_SQUARES],
        "black": [[1, x, y] for x,y in _BLACK_START_SQUARES]
    }

    WEIGHT_FILE = ml.WEIGHT_FILE
    N_FEATURES = 6 # No bias


    def __init__(self, colour):
        """
        This method is called once at the beginning of the game to initialise
        your player. You should use this opportunity to set up your own internal
        representation of the game state, and any other information about the 
        game state you would like to maintain for the duration of the game.

        The parameter colour will be a string representing the player your 
        program will play as (White or Black). The value will be one of the 
        strings "white" or "black" correspondingly.
        """
        # TODO: Set up state representation.
        
        self.board = self.INITIAL_BOARD
        self.colour = colour
        self.time_elapsed = 0
        self.depth_limit = 3

        # Generate random weights if no weight file present
        if not os.path.exists(self.WEIGHT_FILE):
            print("* Generating random weights")
            weights = np.random.uniform(low=-1.0, high = 1.0, size=(self.N_FEATURES, ))
            np.savetxt(self.WEIGHT_FILE, weights, delimiter=',')
            
        self.weights = np.loadtxt(open(self.WEIGHT_FILE, "rb"), delimiter=",")
        print(self.weights, self.N_FEATURES)
        assert (len(self.weights) == self.N_FEATURES)

    def action(self):
        """
        This method is called at the beginning of each of your turns to request 
        a choice of action from your program.

        Based on the current state of the game, your player should select and 
        return an allowed action to play on this turn. The action must be
        represented based on the spec's instructions for representing actions.
        """
        # TODO: Decide what action to take, and return it
        #an_action = bb.search(self)
        start = time.time()
        print(self.depth_limit)

        an_action = bb.minimax_wrapper(self.board, 1, self.weights, self.colour, -1000, 1000, self.depth_limit)

        print(an_action)
        self.time_elapsed += time.time() - start
        return an_action.toTuple()

#if bb.low_risk(self.board): an_action = bb.minimax_wrapper(self.board, 1, self.weights, self.colour, -1000, 1000, 1)
#else: an_action = bb.minimax_wrapper(self.board, 1, self.weights, self.colour, -1000, 1000, self.depth_limit)

    def update(self, colour, action):
        """
        This method is called at the end of every turn (including your player’s 
        turns) to inform your player about the most recent action. You should 
        use this opportunity to maintain your internal representation of the 
        game state and any other information about the game you are storing.

        The parameter colour will be a string representing the player whose turn
        it is (White or Black). The value will be one of the strings "white" or
        "black" correspondingly.

        The parameter action is a representation of the most recent action
        conforming to the spec's instructions for representing actions.

        You may assume that action will always correspond to an allowed action 
        for the player colour (your method does not need to validate the action
        against the game rules).
        """
        # TODO: Update state representation in response to action.
        curr_board = self.board
        if(action[0] == "BOOM"):
            next_board = boom(action[1], curr_board)
            self.board = next_board
        if(action[0] == "MOVE"):
            next_board = move(action[1], action[2], action[3], curr_board, colour)
            self.board = next_board

        # Linearly increase depth limit
        w, b = bb.count_tokens(self.board)
        #self.depth_limit = 6 - (w+b)//8

        """
        if self.colour == "white":
            self.depth_limit = 6 - w//3
        else:
            self.depth_limit = 6 - b//3
        """

