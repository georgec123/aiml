import numpy as np
from abc import ABC, abstractmethod


np.random.seed(3)


class Learner(ABC):

    def __init__(self, rounds: int, num_arms: int, epsilon: float, testbed: np.ndarray, testbed_means: np.ndarray) -> None:

        self.rounds = rounds
        self.num_arms = num_arms
        self.epsilon = epsilon

        self.Q = np.zeros(num_arms)
        self.N = np.zeros(num_arms)

        self.actions = np.zeros(rounds)
        self.rewards = np.zeros(rounds)
        self.regret = np.zeros(rounds)

        self.testbed = testbed
        self.testbed_means = testbed_means  # Used for calculating regret

        self.t: int = None

    @abstractmethod
    def choose_action(self):
        pass

    @abstractmethod
    def update(self, action, reward):
        pass


class GreedyLearner(Learner):

    def __init__(self, initial_value, *args, alpha=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.Q = np.ones(self.num_arms) * initial_value

    def update(self, action, reward):

        self.N[action] += 1

        if self.alpha:
            # Update rule for constant step size
            self.Q[action] += self.alpha*(reward - self.Q[action])
        else:
            # Update rule for sample average step size
            self.Q[action] += (1/self.N[action])*(reward - self.Q[action])

    def greedy_action(self):
        return np.argmax(self.Q)

    def non_greedy_action(self):
        return np.random.choice(self.num_arms)

    def choose_action(self):
        if np.random.uniform() < self.epsilon:
            return self.non_greedy_action()
        else:
            return self.greedy_action()

    def learn(self):

        for t in range(self.rounds):
            self.t = t
            action = self.choose_action()
            reward = self.testbed[t][action]

            self.update(action, reward)
            self.actions[t] = action
            self.rewards[t] = reward
            self.regret[t] = np.max(self.testbed_means[t]) - \
                self.testbed_means[t][action]


class GreedyUCBLearner(GreedyLearner):

    def __init__(self,  c, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c = c

    def greedy_action(self):
        Q_v = []

        for i in range(self.num_arms):
            # Check if we have ever chosen action i before.
            #  If not, then N[i] is 0 and we can't divide by zero

            if self.N[i] != 0:
                UB = self.c * np.sqrt(np.log(self.t)/self.N[i])
                Q_v.append(self.Q[i] + UB)
            else:
                Q_v.append(self.Q[i])

        A = np.argmax(Q_v)
        return A
