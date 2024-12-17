import numpy as np

# Q-learning  
# p. 131 Sutton & Barto

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
al = 0.05

# Initialize Q
# always search !
Q = np.array([[1.0,0.0,0.0],[1.0,0.0,0.0]])

# Loop for each episode --> this is a continuing MDP, so no loop here

# State : we will use 0 for high battery and 1 for low battery
# Initialize S
S = np.random.randint(2)

# Initialize V function
V = np.zeros(2)

# Nt : max. number of steps
Nt = 10000

# Loop for each step of episode
for t in range(Nt):

    # Define the action according to the policy
    if S == 0:
        # action 0 search, 1 wait
        act = np.argmax(Q[0,:])
    else:
        # action 0 search, 1 wait, 2 recharge
        act = np.argmax(Q[1,:])

    # epsilon-greedy policy
    # with probability 0.1 we choose a random action
    if np.random.rand() < 0.1:
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

    # Update Q
    QSp = Q[Sp,:]
    Q[S,act] += al*(R+Gamma*QSp.max()-Q[S,act])

    # Update the state
    S = Sp




