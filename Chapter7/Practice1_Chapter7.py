import numpy as np

class AddGate:
  def __init__(self):
    self.x = None
    self.y = None

  def forward(self, x, y):
    self.x = x
    self.y = y
    return x + y

  def backward(self, d_out):
    return d_out, d_out

class MultiplyGate:
  def __init__(self):
    self.x = None
    self.y = None
  def forward(self, x, y):
    self.x = x
    self.y = y
    return x * y
  def backward(self, d_out):
    dx = d_out * self.y
    dy = d_out * self.x
    return dx, dy

class PowerGate:
  def __init__(self,power):
    self.x = None
    self.power = power
  def forward(self, x):
    self.x = x
    return x ** self.power

  def backward(self, d_out):
    return d_out * self.power * (self.x ** (self.power - 1))

#initial values
w = 2.0
b = 8.0
x = -2.0
y = 2.0
# Forward propagation
multiply_gate1 = MultiplyGate()
multiply_gate2 = MultiplyGate()
add_gate1 = AddGate() # For w * x + b
add_gate2 = AddGate() # (a - y)
power_gate = PowerGate(2)

# Node 1: Compute c = w * x
c = multiply_gate1.forward(w, x)
# Node 2: Compute a = c + b
a = add_gate1.forward(c, b)
# Node 3: Compute d = a - y
d = add_gate2.forward(a, -y)
# Node 4: Compute e = d^2
e = power_gate.forward(d)
# Node 5: Compute J = 0.5 * e
J = multiply_gate2.forward(0.5, e)
print(f"Loss: {J}")
# Node 5
_, A = multiply_gate2.backward(1)
print("A = ", A)
# Node 4
B = power_gate.backward(A)
print("B = ", B)
# Node 3
C, _ = add_gate1.backward(B)
print("C = ", C)
# Node 2
D, E = add_gate1.backward(B)
print("D = ", D)
print("E = ", E)
# Node 1
F, _ = multiply_gate1.backward(D)
print("F = ", F)


     
Loss: 2.0
A =  0.5
B =  2.0
C =  2.0
D =  2.0
E =  2.0
F =  -4.0

import numpy as np
class SigmoidGate:
  def __init__(self):
    self.output = None
  def forward(self, z):
    self.output = 1 / (1 + np.exp(-z))
    return self.output
  def backward(self, d_out):
    sigmoid_derivative = self.output * (1 - self.output)
    return d_out * sigmoid_derivative
     
