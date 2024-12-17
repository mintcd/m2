from sympy import *

N=2
p = 0.2

k = symbols("k")

V0 = summation(p*((1-p)**(k-1))*k,(k,1,N))
print("Parking Problem with N=",N,", proba p=",p)
print("Expectation of the reward for policy park as soon as possible")
print("V0 = ",V0)
