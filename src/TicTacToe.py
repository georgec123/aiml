from typing import List, Tuple
import copy 

class TicTacToe:
    players = {'O', 'X'}

    def __init__(self) -> None:
            
        self.last_turn = None

        self.state = [' ']*9
        self.moves = 0

    def print_board(self):
        for i in range(3):
            print(f"|{''.join(self.state[i*3:(i+1)*3:])}|")
        print('\n')
    

    def verify_move(self, player, index):
        if player not in self.players:
            raise ValueError(f'Player must be in {self.players}, not {player}')

        if self.last_turn is not None and player == self.last_turn:
            raise ValueError(f"Not the turn of {player}, they went last go.")

        if self.state[index] != ' ':
            raise ValueError(f'Index {index} is already taken')

    def add_move(self, player: str, index: int):

        self.verify_move(player, index)

        self.state[index] = player
        self.moves += 1
        self.last_turn = player

    def fake_move(self, player, index, verify=True):
        if verify:
            self.verify_move(player, index)

        curr_state = copy.copy(self.state)
        curr_state[index] = player

        return curr_state

    def game_over(self):
        if self.moves == 9:
            return True
        return False


    @staticmethod
    def _winner(curr_state):
        
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
            [0,1,2],
            [3,4,5],
            [6,7,8],
            # down
            [0,3,6],
            [1,4,7],
            [2,5,8],
            # diag
            [0,4,8],
            [2,4,6]
        ]

        for comb in combs:
            res = check_pos(pos=comb, curr_state=curr_state)
            if res:
                # winner in this combination, return player who is in that position
                return curr_state[comb[0]]
        return False 


    def winner(self, curr_state=None):
            
        if curr_state is None:
            if self.moves<5: 
                # too little moves to have a winner
                return False

            curr_state = self.state

        return self._winner(curr_state)

    def similar_states(self, curr_state=None) -> Tuple[str]:
        if curr_state is None:
            curr_state = self.state

        ss = [''.join(curr_state)]
        states = [
            '210543876',
            '678345012',
            '036147258',
            '852741630'
        ]

        for state in states:
            new_state = [curr_state[int(s)] for s in state]
            ss.append(''.join(new_state))

        return tuple(ss)


    def possible_moves(self, curr_state=None):
        if curr_state is None:
            curr_state = self.state

        possible_moves = [idx for idx, go in enumerate(curr_state) if go==' ']

        return possible_moves


def main():
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