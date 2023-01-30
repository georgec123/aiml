import numpy as np

def greedy_k_arm_bandit(num_arms, initial_value, rounds):
    # Initialise action values Q and number of rewards received per action N
    Q = np.ones(num_arms) * initial_value
    N = np.zeros(num_arms)

    for num in range(rounds):
        # Define the rewards in R which are standard normal rvs
        R = np.random.normal(size=num_arms)

        # Choose A greedily
        A = np.argmax(Q)

        # Update our N and Q terms for the next round
        N[A] += 1
        Q[A] += (1/N[A])*(R[A] - Q[A])

    return [N, Q]

num_arms = 10
initial_value = 5
rounds = 1000
print(greedy_k_arm_bandit(num_arms, initial_value, rounds))
