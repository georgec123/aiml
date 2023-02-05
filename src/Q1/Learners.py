from TicTacToe import TicTacToe
from Players import Player
from functools import lru_cache

from typing import List
import numpy as np
import random


class StateDictX(dict):
    """
    Data structure to hold state and rewards.
    Abstracts updating formula

    {
        'state_key': [reward, num_times_seen]
    }

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def state_key(state: str):
        return ReinforcementTicTacToeLearner.get_state_key(state)

    def load_state(self, state: str):
        """
        Ensure the state exists in our dict.
        To load the state into the dict, we check to see if the game is won, lost, draw, or still in play.
        We set the default reward to 1, 0, 0, 0.5 resp.
        """
        state_key = self.state_key(state=state)

        if state_key in self:
            return

        # not in dict, need to prepopulate
        ttt_winner = TicTacToe._winner(state_key)
        board_full = TicTacToe.board_full(state_key)

        if ttt_winner == 'X':
            # this state is a winning state, reward is 1
            self[state_key] = [1, 0]
        elif ttt_winner == 'O' or board_full:
            # draw or other won
            self[state_key] = [0, 0]
        else:
            # game not over, default to 0.5
            self[state_key] = [0.5, 0]

        return

    def get_reward(self, state: str):
        """
        Return reward of current state. 
        Ensures state is initialised first.
        """
        state_key = self.state_key(state=state)
        self.load_state(state=state_key)

        return self[state_key][0]

    def get(self, state: str) :
        """
        Return reward and num_seen of current state. 
        Ensures state is initialised first.
        """

        state_key = self.state_key(state=state)
        self.load_state(state=state_key)

        return super().get(state_key)

    def update(self, old_state: str, new_state: str) -> None:
        """
        Update the previous state reward with the new state value

        :param old_state: Key of old state
        :param new_state: Key of new state

        """
        old_state_key = self.state_key(old_state)
        new_state_key_1 = self.state_key(new_state)

        v, seen_num = self.get(old_state_key)
        v_1, _ = self.get(new_state_key_1)

        seen_num += 1
        v_new = v + (v_1 - v)/(seen_num)

        self[old_state_key] = [v_new, seen_num]


class ReinforcementTicTacToeLearner:

    def __init__(self, n: int, epsilon: float, opponent: Player, player: str = 'X') -> None:
        """

        :param n: number of iterations to learn over
        :param epsilon: Probability to make a non greedy move
        :param opponent: Instance of Player (or child of) to play against
        :param player: Symbol to play with
        """
        self.state_dict = StateDictX()
        self.epsilon = epsilon
        self.n = n
        self.player = player
        self.oppoent = opponent

        return

    @staticmethod
    @lru_cache
    def get_state_key(state: str):
        """
        Given symetries in states of board, we return the state key from a list of different states
        """
        similar_states = TicTacToe().similar_states(curr_state=state)
        return sorted(similar_states)[0]

    def learn(self):
        wld = [None]*self.n

        for game_num in range(self.n):
            winner, played_first = self.learn_one_game()
            result = self.get_result(winner)
            wld[game_num] = [result, played_first]
        return wld

    def learn_one_move(self, game: TicTacToe):
        # we move
        if random.uniform(0, 1) < self.epsilon:
            game = self.random_move(board=game)
            did_greedy = False
        else:
            # move greedy and learn
            game = self.greedy_move(board=game)
            did_greedy = True

        return game, did_greedy

    def learn_one_game(self):

        game = TicTacToe()

        if random.randint(0,1):
            game, _ = self.learn_one_move(game)
            played_first = True
        else:
            played_first = False

        while not game.game_over() and not game.winner():
            old_state = game.str_state()

            # they move
            game = self.oppoent.move(game)

            if self.game_lost(game):
                self.state_dict.update(old_state, game.str_state())
                break

            # we move
            game, did_greedy = self.learn_one_move(game=game)

            if did_greedy:
                self.state_dict.update(old_state, game.str_state())

        return game.winner(), played_first

    def game_lost(self, board: TicTacToe):
        if board.winner() == self.player:
            return False
        elif board.game_over() or board.winner():
            return True
        else:
            return False

    def random_move(self, board: TicTacToe) -> TicTacToe:

        possible_moves = board.possible_moves()

        move = random.choice(possible_moves)
        board.add_move(player=self.player, index=move)

        return board

    def greedy_move(self, board: TicTacToe) -> TicTacToe:

        possible_moves = board.possible_moves()
        outcomes = [None]*len(possible_moves)

        for idx, move in enumerate(possible_moves):

            hyp_state = board.fake_move(player=self.player, index=move)
            hyp_key = self.get_state_key(''.join(hyp_state))

            outcomes[idx] = self.state_dict.get_reward(hyp_key)

        # if there are multiple optimal choices, we pick randomly from those
        best_outcome = np.max(outcomes)
        best_indexes = [idx for idx, outcome in enumerate(
            outcomes) if outcome == best_outcome]
        best_index = random.choice(best_indexes)
        best_move = possible_moves[best_index]

        board.add_move(player=self.player, index=best_move)

        return board

    def get_result(self, winner: str) -> str:
        """
        returns: result
        """
        # gameover
        if winner == self.player:
            # we won
            return 'w'
        elif winner is False:
            # game was a draw
            return 'd'
        else:
            # we lost
            return 'l'

    def play_one_game(self):
        game = TicTacToe()

        if random.randint(0,1):
            played_first = True
            game = self.greedy_move(game)
        else:
            played_first = False

        while not game.game_over() and not game.winner():

            # they move
            game = self.oppoent.move(game)

            if self.game_lost(game):
                break

            # we move
            game = self.greedy_move(game)

        return game.winner(), played_first

    def play_n_games(self, n):
        wld = [None]*n
        for game_num in range(n):
            winner, played_first = self.play_one_game()
            result = self.get_result(winner)
            wld[game_num] = [result, played_first]

        return wld
