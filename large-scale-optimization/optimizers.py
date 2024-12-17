import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from LineSearch import line_search
from copy import deepcopy

class Optimizer:
  def __init__(self, dist):
    self.dist = dist
  def run(self):
    pass

  def numerical_gradient(self, f, x, dist):
    return (f(x + dist) - f(x - dist)) / (2 * dist)

class GradientDescent(Optimizer):
  def __init__(self, 
               input_size:int,
               f, 
               x0 = None,
               grad_f=None, 
               dist=1e-5,
               lr:float|Literal['wolfe']=0.01, 
               tol=1e-6,
               max_iter=int(1e6)):
    """
      f: objective function
      grad_f: gradient of the objective function. If not provided, it will be computed centrally with distance dist
      lr: learning rate
      max_iter: maximum number of iterations
    """
    super().__init__(dist)

    self.input_size = input_size
    self.f = f
    self.x0 = x0 if x0 is not None else np.random.rand(self.input_size)
    self.grad_f = grad_f if grad_f is not None else lambda x: self.numerical_gradient(f, x)
    self.init_lr = lr
    self.tol = tol
    self.max_iter = max_iter
  
  def run(self):
      x = self.x0  
      history = [x]
      for i in range(self.max_iter):
          grad = self.grad_f(x)

          lr = line_search(f=self.f, 
                           x=x, 
                           direction=-grad, 
                           grad_f=self.grad_f) if self.init_lr == 'wolfe' else self.init_lr
          x -= lr * grad
          history.append(deepcopy(x))

          if np.linalg.norm(grad) < self.tol:
              print(f"GD converged after {i} iterations.")
              break  # Converge early if tolerance is met
          if i == self.max_iter - 1:
              print(f"GD reached max iterations.")

      # if plot:
      #     plt.plot([self.f(x) for x in history])
      #     plt.xlabel('Iteration')
      #     plt.ylabel('$f(x)$')
      #     plt.show()

      return x, history
