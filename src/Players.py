from TicTacToe import TicTacToe
import random


class Player:
    def __init__(self, player='O') -> None:
        self.player = player
        self.other_player = 'O' if player == 'X' else 'X'
        return

class RandomPlayer(Player):
    def __init__(self, player='O') -> None:
        super().__init__(player)


    def move(self, board: TicTacToe):
        possible_moves = board.possible_moves()

        chosen_move = random.choice(possible_moves)

        board.add_move(player=self.player, index=chosen_move)
        return board


class RandomWinner(Player):
    def __init__(self, player='O') -> None:
        super().__init__(player)

    def move(self, board: TicTacToe):
        possible_moves = board.possible_moves()

        for move in possible_moves:
            hyp_state = board.fake_move(player=self.player, index=move)

            # if we can win with this move, play that move
            if board.winner(hyp_state) == self.player:
                board.add_move(player=self.player, index=move)
                return board
        
        move = random.choice(possible_moves)
        board.add_move(player=self.player, index=move)
        return board


class RandomWinnerBlocker(Player):
    def __init__(self, player='O') -> None:
        super().__init__(player)

    def move(self, board: TicTacToe):
        possible_moves = board.possible_moves()


        # see if we can win
        for move in possible_moves:
            hyp_state = board.fake_move(player=self.player, index=move)

            # if we can win with this move, play that move
            if board.winner(hyp_state) == self.player:
                board.add_move(player=self.player, index=move)
                return board
        
        # block if they can win
        for move in possible_moves:
            hyp_state = board.fake_move(player=self.other_player, index=move, verify=False)

            # if other player wins with this move, we block
            if board.winner(hyp_state) == self.other_player:
                board.add_move(player=self.player, index=move)
                return board

        move = random.choice(possible_moves)
        board.add_move(player=self.player, index=move)
        return board