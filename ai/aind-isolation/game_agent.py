"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    #--- get the difference but multiply the opponents score by 1.5
    opponent = game.get_opponent(player)
    score = float (len(game.get_legal_moves(player)) - 1.5 * len(game.get_legal_moves(opponent)))
    
    return  score


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    #--- the lower the number of available moves for the opponent, the better the score
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    opponent = game.get_opponent(player)
    score = float (- len(game.get_legal_moves(opponent)))
    
    return score


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    #--- moves that are closer to the edge have a lower score
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    legal_moves = game.get_legal_moves()
    left = int (game.width * 0.2) - 1
    right = game.width - int (game.width * 0.2) - 1
    top = int (game.height * 0.2) - 1
    bottom = game.height - int (game.height * 0.2) - 1

    score = 0.0
    for row, col in legal_moves:
        if col <= left or right <= col or row <= top or bottom <= row:
            score += 0.5
        else:
            score += 1
    
    return score


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        return self.minimax_decision(game, depth)
    
    def minimax_decision(self, board, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        _, best_move = self.max_value(board, depth, 0, (-1, -1))
        
        return best_move
    
        
    def max_value(self, board, max_depth, curr_depth, move):
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = move
        
        utility = board.utility(self)
        if utility == 0:
            #--- there are moves left
            
            if curr_depth < max_depth:
                max_v = float("-inf")
                #--- find the best move based on what maximum value we can get
                for next_move in board.get_legal_moves():
                    next_board = board.forecast_move(next_move)
                    t, _ = self.min_value(next_board, max_depth, curr_depth + 1, next_move)
                    if max_v <= t:
                        #--- we have found a new maximum!
                        #--- update the maximum value and best move
                        max_v = t
                        best_move = next_move
            else:
                #--- we are at max_depth
                max_v = self.score(board, self)
        else:
            #--- there are no moves left
            #--- self has either lost or one for board
            max_v = utility
            
        return max_v, best_move
            
            
    def min_value(self, board, max_depth, curr_depth, move):
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = move

        utility = board.utility(self)
        if utility == 0:
            #--- there are moves left
            
            if curr_depth < max_depth:
                min_v = float("inf")
                #--- find the best move based on what minimum value we can get
                
                for next_move in board.get_legal_moves():
                    next_board = board.forecast_move(next_move)
                    t, _ = self.max_value(next_board, max_depth, curr_depth + 1, next_move)
                    if t <= min_v:
                        #--- new minimum found
                        #--- update minimum value and best_move
                        min_v = t
                        best_move = next_move
            else:
                #--- we are at max_depth
                #--- so just calculate the score
                min_v = self.score(board, self)
                
        else:
            #--- there are no moves left
            #--- self has either lost or one for board
            min_v = utility

        return min_v, best_move    

from collections import Counter    

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            
            best_moves = []
            self.search_depth = 3
            while True:
                best_moves.append(self.alphabeta(game, self.search_depth))
                self.search_depth += 1

        except SearchTimeout:
            
            if best_moves:
                best_move = Counter(best_moves).most_common(1)[0][0]
            
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move
    

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # TODO: finish this function!
        return self.alpha_beta_decision(game, depth, alpha, beta)
    
        
    def alpha_beta_decision(self, board, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        
        _, best_move = self.max_value(board, depth, 0, (-1, -1), alpha, beta)
        
        return best_move
    
        
    def max_value(self, board, max_depth, curr_depth, move, alpha, beta):
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = move
        
        utility = board.utility(self)
        if utility == 0:
            #--- there are moves left
            
            if curr_depth < max_depth:
                max_v = float("-inf")
                #--- find the best move based on what maximum value we can get
                for next_move in board.get_legal_moves():
                    next_board = board.forecast_move(next_move)
                    t, _ = self.min_value(next_board, max_depth, curr_depth + 1, next_move, alpha, beta)
                    if max_v <= t:
                        #--- we have found a new maximum!
                        #--- update the maximum value and best move
                        max_v = t
                        best_move = next_move
                        if beta <= max_v:
                            #--- upper limit exceeded
                            #--- no need searching the rest of the moves since they are all lower!
                            return max_v, best_move
                        alpha = max(alpha, max_v)
            else:
                #--- we are at max_depth
                max_v = self.score(board, self)
        else:
            #--- there are no moves left
            #--- self has either lost or won for board
            max_v = utility
            
        return max_v, best_move
            
            
    def min_value(self, board, max_depth, curr_depth, move, alpha, beta):
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_move = move

        utility = board.utility(self)
        if utility == 0:
            #--- there are moves left
            
            if curr_depth < max_depth:
                min_v = float("inf")
                #--- find the best move based on what minimum value we can get
                
                for next_move in board.get_legal_moves():
                    next_board = board.forecast_move(next_move)
                    t, _ = self.max_value(next_board, max_depth, curr_depth + 1, next_move, alpha, beta)
                    if t <= min_v:
                        #--- new minimum found
                        #--- update minimum value and best_move
                        min_v = t
                        best_move = next_move
                        if min_v <= alpha:
                            #--- lower limit exceeded
                            #--- no need searching the rest as they will all be greater!
                            return min_v, best_move
                        beta = min(beta, min_v)
            else:
                #--- we are at max_depth
                #--- so just calculate the score
                min_v = self.score(board, self)
                
        else:
            #--- there are no moves left
            #--- self has either lost or won for board
            min_v = utility

        return min_v, best_move        
        