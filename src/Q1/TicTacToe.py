from typing import List, Tuple, Union
import copy


class TicTacToe:
    players = {'O', 'X'}

    def __init__(self) -> None:

        self.last_turn = None

        self.state = [' ']*9
        self.moves = 0

    def str_state(self):

        return ''.join(self.state)

    def print_board(self):
        """
        Print current state of the board
        """
        for i in range(3):
            print(f"|{''.join(self.state[i*3:(i+1)*3:])}|")
        print('\n')

    def verify_move(self, player: str, index: int):
        """
        Verify if the move is a legal move via the criteria
        - It is that players turn
        - Player is currently playing the game
        - Space is not alreay taken

        :raises: ValueError
        """
        if player not in self.players:
            raise ValueError(f'Player must be in {self.players}, not {player}')

        if self.last_turn is not None and player == self.last_turn:
            raise ValueError(f"Not the turn of {player}, they went last go.")

        if self.state[index] != ' ':
            raise ValueError(f'Index {index} is already taken')

    def add_move(self, player: str, index: int):
        """
        Add a move to the board

        :param player: player making the move
        :param index: index of the move
        """
        self.verify_move(player, index)

        self.state[index] = player
        self.moves += 1
        self.last_turn = player

    def fake_move(self, player: str, index: int, verify: bool = True) -> List[str]:
        """
        Mimic a move without actually making it, return the state of the board after the move

        :param player: player making the move
        :param index: index of the move
        :param verify: verify the move is legal

        :return: state of the board after the move
        """
        if verify:
            self.verify_move(player, index)

        curr_state = copy.copy(self.state)
        curr_state[index] = player

        return curr_state

    def game_over(self) -> bool:
        """
        Checks to see if the game is over via all spaces being taken up

        :return: True if game is over, False otherwise
        """

        if self.moves == 9:
            return True
        return False

    @staticmethod
    def board_full(curr_state: str):
        """
        Checks to see if the board is full

        :param curr_state: state of the board
        :return: True if the board is full, False otherwise
        """

        if curr_state.count(' ') == 0:
            return True
        else:
            return False

    @staticmethod
    def _winner(curr_state: str) -> Union[str, bool]:
        """
        Checks to see if there is a winner

        :param curr_state: state of the board
        :return: player who won, False if no winner
        """

        def check_pos(pos, curr_state):
            first = curr_state[pos[0]]

            if first == " ":
                return False

            for i in pos[1:]:
                if first != curr_state[i]:
                    return False

            return True

        combs = [
            # right
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            # down
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            # diag
            [0, 4, 8],
            [2, 4, 6]
        ]

        for comb in combs:
            res = check_pos(pos=comb, curr_state=curr_state)
            if res:
                # winner in this combination, return player who is in that position
                return curr_state[comb[0]]
        return False

    def winner(self, curr_state=None):

        if curr_state is None:
            if self.moves < 5:
                # too little moves to have a winner
                return False

            curr_state = self.state

        return self._winner(curr_state)

    def similar_states(self, curr_state: str = None) -> Tuple[str]:
        """
        Returns a tuple of all the possible states that are similar to the current state (via symmetry)

        :param curr_state: state of the board
        :return: tuple of all the possible states
        """

        if curr_state is None:
            curr_state = self.state

        ss = [''.join(curr_state)]
        states = [
            '012345678',
            '210543876',
            '678345012',
            '036147258',
            '852741630',
            '630741852',
            '876543210',
            '258147036'
        ]

        for state in states:
            new_state = [curr_state[int(s)] for s in state]
            ss.append(''.join(new_state))

        return tuple(ss)

    def possible_moves(self, curr_state=None):
        """
        List of all possible moves

        :param curr_state: state of the board
        :return: list of possible moves
        """
        if curr_state is None:
            curr_state = self.state

        possible_moves = [idx for idx,
                          go in enumerate(curr_state) if go == ' ']

        return possible_moves


def main():
    # example of a tictactoe game
    ttt = TicTacToe()
    ttt.add_move('X', 1)
    ttt.add_move('O', 8)
    ttt.add_move('X', 4)
    ttt.add_move('O', 2)
    print(ttt.winner())

    ttt.add_move('X', 7)

    ttt.print_board()
    print(ttt.winner())


if __name__ == '__main__':
    main()
