import numpy as np
import matplotlib.pyplot as plt

# Create the dataset (X,Y)

# N : number of point
N = 100

# X random, uniform over [0,1]
X = np.random.rand(N)
# Y = sqrt(X) + noise
Y = np.sqrt(X) + np.random.randn(N)*X*0.1

# Vandermonde Matrix: 1, x , x^2 , x^3
VM = [[p**k for k in range(4)] for p in X]
VM = np.array(VM)

# objective function : least square fct
def objf(x):
    r = VM@x-Y
    v = 1/2*sum(r*r)
    # compute the gradient
    g = r@VM
    return v,g


# initialize the optimization variable
x0 = np.random.rand(4)

# numerical tolerance for the algorithm
tol =


plt.plot(X,Y,"b.")
plt.show()
