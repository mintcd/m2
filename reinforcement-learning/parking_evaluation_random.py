import numpy as np

# number of parking places
Tmax = 10 

# probability free place
PR = 0.2

# Discount rate 
Gamma = 1
# No Discount : finite MDP

def mdp_proba(s2,s1,a):
    #  at starting point
    if s1 == 0:
        if s2 == 0:
            return 0
        elif s2 <= Tmax:
            return PR*((1-PR)**(s2-1))
        else:
            return (1-PR)**Tmax

    # if we park (a=1) then we go to state Tmax +1
    if a==1:
        if s2==Tmax+1:
            return 1
        else:
            return 0
    else:
        # it is not possible to go backward (s2>s1)
        if s2<=s1:
            return 0
        else:
            n = s2-s1
            if s2<=Tmax:
                return PR*((1-PR)**(n-1))
            elif s2 == Tmax+1:
                return (1-PR)**n

def mdp_reward(s2,s1,a):
    if s1 >=1 and s1<=Tmax and a==1:
        return s1
    else:
        return 0
   
# functions for bellman equation
# update of V(s) in algo. p.75
def bellman1(s2,s1,a, Vs2):
    P = mdp_proba(s2,s1,a)
    R = mdp_reward(s2,s1,a)
    return P*(R+Gamma*Vs2)

# policy : park as soon as possible 
# 1 : park, 0: do not park
pol0 = 1

# Policy evaluation 
# threshold theta
theta = 1e-5

# initialize V 
# 0 is the initial state
# 1-Tmax state where it is possible to park
# Tmax +1 terminal state
V = np.zeros(Tmax+2)

Test = True
# loop on the state
while Test:
    # Delta for the stop test
    Delta =  0
    k = 0
    for s in range(Tmax+2):
        #print("state : ",s)
        v = V[s]
        # bellman eq. fixed point
        b0 = [bellman1(kk,s, 1, V[kk]) for kk in range(Tmax+2)]
        b1 = [bellman1(kk,s, 0, V[kk]) for kk in range(Tmax+2)]
        # here we park with a probability of 1/2
        V[k] = 1/2*sum(b0)+1/2*sum(b1)
        
        dupdate = [Delta, abs(v-V[k])]
        Delta = max(dupdate)
        k += 1
    print(Test, Delta)
    Test = Delta >= theta


