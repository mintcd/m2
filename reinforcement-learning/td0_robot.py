import numpy as np

# TD(0) for estimating a policy 
# p. 120 Sutton & Barto

# Recycling robot parameters
alpha = 0.05
beta = 0.05
# rewards
rsearch = 4
rwait = 0

# Discount rate 
Gamma = 0.8
# learning rate: we use "al" instead of "alpha" since "alpha" is already defined
# as a probability in the MDP.
al = 0.01

# policy : random policy 
# search or wait with probability 1/2 if state is high
# search, wait or recharge with probability 1/3 if state is low

# Loop for each episode --> this is a continuing MDP, so no loop here

# State : we will use 0 for high battery and 1 for low battery
# Initialize S
S = np.random.randint(2)

# Initialize V function
V = np.zeros(2)

# Nt : max. number of steps
Nt = 1000

# Loop for each step of episode
for t in range(Nt):

    # Define the action according to the policy
    if S == 0:
        # action 0 search, 1 wait
        act = np.random.randint(2)
    else:
        # action 0 search, 1 wait, 2 recharge
        act = np.random.randint(3)

    # Take action and observe R, S'
    if S == 0:
        # State is high : 0
        if act == 0:
            X = np.random.rand()
            if X <= alpha:
                Sp = 0
                R = rsearch
                print("State : high, next State: high, search, reward:",R)
            else:
                Sp = 1
                R = rsearch
                print("State : high, next State: low, search, reward:",R)
        else:
            R = rwait
            Sp = 0
            print("State : high, next State: high, wait, reward:",R)
    else:
        # State is low :1
        if act == 0:
            X = np.random.rand()
            if X <= beta:
                Sp = 1
                R = rsearch
                print("State : low, next State: low, search, reward:",R)
            else:
                # no more battery
                Sp = 0
                R = -3
                print("State : low, next State: high, no battery ! , reward:",R)
        elif act == 1:
            R = rwait
            Sp = 1
            print("State : low, next State: low, wait , reward:",R)
        else:
            # Recharge
            Sp = 0
            R = 0
            print("State : low, next State: high, recharge , reward:",R)

    # Update V 
    V[S] += al*(R+Gamma*V[Sp]-V[S])

    # Update the state
    S = Sp




