import math
import numpy as np
import matplotlib.pyplot as plt

# f(x) = 3x^2 + 2x + 4sin(x)
def grad(x):
    return 6*x + 2 + 4*np.cos(x)

# f'(x) = 3x^2 + 2x + 4sin(x)
def cost(x):
    return 3*x**2 + 2*x + 4*np.sin(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad(x[-1])
        cost_value = cost(x_new)
        print(f"Iteration {it+1}: x = {x_new}, cost = {cost_value}")
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = myGD1(0.1, -5)  # x0 = -5
(x2, it2) = myGD1(0.1, 5)   # x0 = 5

print('Final result for x1: x = %f, cost = %f, after %d iterations' % (x1[-1], cost(x1[-1]), it1))
print('Final result for x2: x = %f, cost = %f, after %d iterations' % (x2[-1], cost(x2[-1]), it2))

plt.plot(np.linspace(-6, 6, 100), cost(np.linspace(-6, 6, 100)), 'b')
plt.plot(x1[-1], cost(x1[-1]), 'ro')
plt.title('Gradient Descent (x0 = -5)')
plt.show()

plt.plot(np.linspace(-6, 6, 100), cost(np.linspace(-6, 6, 100)), 'b')
plt.plot(x2[-1], cost(x2[-1]), 'ro')
plt.title('Gradient Descent (x0 = 5)')
plt.show()

(x3, it3) = myGD1(0.01, -1)

print('Final result for x3 (eta = 0.01, x0 = -1): x = %f, cost = %f, after %d iterations' % (x3[-1], cost(x3[-1]), it3))

plt.plot(np.linspace(-6, 6, 100), cost(np.linspace(-6, 6, 100)), 'b')
plt.plot(x3[-1], cost(x3[-1]), 'ro')
plt.title('Gradient Descent (x0 = -1, eta = 0.01)')
plt.show()