import numpy as np
import scipy.stats as st

def epsilon_greedy_k_arm_bandit(num_arms, initial_value, epsilon, c, rounds):
    # Initialise action values, Q, and no. of rewards received per action, N.
    Q = np.ones(num_arms) * initial_value
    N = np.zeros(num_arms)

    # Initialise a count, t, which is needed for the upper bound and take the length of Q as L for simplicity
    t = 0
    L = len(Q)
    
    # Loop 'infinitely'
    for num in range(rounds):
        t += 1

        # Define the rewards in R which are standard normal rvs
        R = np.random.normal(size=num_arms)

        # Define a Bernoulli rv which indicates when we switch to a non-greedy action 
        X = st.bernoulli(1 - epsilon).rvs(1)
        
        # If X is one, take the greedy action with an upper bound defined as in the notes.
        # If X is zero, choose a random action
        if X == 1:
            Q_v = []
            for i in range(L):
                # Check if we have ever chosen action i before. If not, then N[i] is 0 and we can't divide by zero
                if N[i] != 0:
                    UB = c * np.sqrt(np.log(t)/N[i])
                    Q_v.append(Q[i] + UB)
                else:
                    Q_v.append(Q[i])
            A = np.argmax(Q_v)
        else:
            A = np.random.choice(range(L))
        
        # Update our N and Q terms for the next round
        N[A] += 1
        Q[A] += (1/N[A])*(R[A] - Q[A])
    
    return [N, Q]

num_arms = 10
initial_value = 0
epsilon = 0.1
c = 2
rounds = 1000
print(epsilon_greedy_k_arm_bandit(num_arms, initial_value, epsilon, c, rounds))
