from TicTacToe import TicTacToe
from Players import Player
from functools import lru_cache

import numpy as np
import random


class ReinforcementTicTacToeLearner:

    def __init__(self, n: int, epsilon: float, opponent: Player, player: str = 'X') -> None:

        self.state_dict = dict()
        self.epsilon = epsilon
        self.n = n
        self.player = player
        self.oppoent = opponent

        return

    @staticmethod
    @lru_cache
    def get_state_key(states):
        return sorted(states)[0]

    def decide_next_move(self, board: TicTacToe):

        possible_moves = board.possible_moves()

        if random.uniform(0, 1) < self.epsilon:
            # do random move
            return random.choice(possible_moves)

        # else do greedy move
        outcomes = [None]*len(possible_moves)

        for idx, move in enumerate(possible_moves):
            hyp_state = board.fake_move(player=self.player, index=move)
            hyp_key = self.get_state_key(
                board.similar_states(curr_state=hyp_state))

            if hyp_key in self.state_dict:
                outcomes[idx] = self.state_dict[hyp_key][0]
            elif board.winner(curr_state=hyp_key) == self.player:
                # next move would win
                outcomes[idx] = 1
            elif board.winner(curr_state=hyp_key):
                # next move would lose
                outcomes[idx] = 0
            elif board.moves == 8:
                # next move would end in draw
                outcomes[idx] = 0
            else:
                # not seen before, default to 0.5
                outcomes[idx] = 0.5

        # if there are multiple optimal choices, we pick randomly from those
        best_outcome = np.max(outcomes)
        best_indexes = [idx for idx, outcome in enumerate(
            outcomes) if outcome == best_outcome]
        best_index = random.choice(best_indexes)

        return possible_moves[best_index]

    def play_one_game(self):

        state_moves = []
        game = TicTacToe()

        if random.randint(0, 1):
            # other player goes first
            # print('2 going first')
            game = self.oppoent.move(game)
            played_first = False
        else:
            played_first = True

        p1_turn = True

        while not game.game_over() and not game.winner():

            if p1_turn:
                next_move = self.decide_next_move(board=game)
                state_moves.append(game.fake_move(
                    player=self.player, index=next_move))

                game.add_move(player=self.player, index=next_move)
            else:
                game = self.oppoent.move(game)

            p1_turn = not p1_turn

        return game.winner(), state_moves, played_first

    def get_result_and_increment(self, winner):
        """
        returns: (result, increment)
        """
        # gameover
        if winner == self.player:
            # we won
            return 'w', 1
        elif winner is False:
            # game was a draw
            return 'd', 0
        else:
            # we lost
            return 'l', 0

    def learn(self):

        wld = [None]*self.n
        """
        state_dict looks like 
        {'state1': [probability, num_wins, num]}
        """
        for game_num in range(0, self.n):

            winner, state_moves, played_first = self.play_one_game()
            result, increment = self.get_result_and_increment(winner)

            wld[game_num] = (played_first, result)

            for state in state_moves:
                str_state = ''.join(state)
                state_key = self.get_state_key(TicTacToe().similar_states(str_state))
                if state_key in self.state_dict:
                    prob, won, played = self.state_dict[state_key]
                    new_won = won + increment
                    new_played = played + 1

                    self.state_dict[state_key] = [
                        new_won/new_played, new_won, new_played]

                else:
                    self.state_dict[state_key] = [increment, increment, 1]

        return self.state_dict, wld

    def play_n_games(self, n):
        wld = [None]*n

        for game_num in range(n):
            winner, _, played_first = self.play_one_game()
            result, _ = self.get_result_and_increment(winner)
            wld[game_num] = (played_first, result)

        return wld
