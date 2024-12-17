from utils import gradient
import numpy as np

def line_search(f, x, direction, lr=1.0, grad_f = None, decrease_rate=0.5, lower=0.1, upper=0.9):
    lr = lr
    while not (sufficient_decrease(f=f, 
                                  x=x, 
                                  direction=direction, 
                                  lr=lr, 
                                  grad_f=grad_f, 
                                  threshold=lower) \
        and curvature(f=f, 
                         x=x, 
                         direction=direction, 
                         lr=lr, 
                         grad_f=grad_f, 
                         threshold=upper)):
        
        lr *= decrease_rate

    return lr


def sufficient_decrease(f, x, direction, lr, grad_f = None, threshold = 0.1):
    grad_fx = grad_f(x) if grad_f else gradient(f, x)
    return f(x + lr * direction) <= f(x) + threshold*lr*np.inner(grad_fx, direction)

def curvature(f, x, direction, lr, grad_f = None, threshold = 0.9):
    grad_fx = grad_f(x) if grad_f else gradient(f, x)
    grad_fxx = grad_f(x + lr*direction) if grad_f else gradient(f, x + lr*direction)

    return np.inner(grad_fxx, direction) >= threshold*np.inner(grad_fx, direction)
