import numpy as np
import matplotlib.pyplot as plt

# 1. Dữ liệu
X = np.array([0.245, 0.247, 0.285, 0.299, 0.327, 0.347, 0.356, 0.36, 0.363, 0.364, 
              0.398, 0.4, 0.409, 0.421, 0.432, 0.473, 0.509, 0.529, 0.561, 0.569, 
              0.594, 0.638, 0.656, 0.816, 0.853, 0.938, 1.036, 1.045])

y = np.array([0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 
              0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 
              1, 1, 1, 1, 1, 1, 1, 1])

# 2. Khởi tạo các tham số theta và tốc độ học
np.random.seed(0)
theta0 = np.random.rand()
theta1 = np.random.rand()
theta2 = np.random.rand()
learning_rate = 1e-4
iterations = 100
lambda_reg = 0.5

# 3. Hàm logistic
def logistic_function(z):
    return 1 / (1 + np.exp(-z))

# 4. Hàm dự đoán
def predict(X1, X2, theta0, theta1, theta2):
    z = theta0 + theta1 * X1 + theta2 * X2
    gz = logistic_function(z)
    return gz

# 5. Hàm chi phí
def cost_function(X1, X2, y_true, theta0, theta1, theta2):
    m = len(X)
    epsilon = 1e-15
    y_pred = predict(X1, X2, theta0, theta1, theta2)
    cost = -(1 / m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    return cost

# #Add L2 regularization term
# reg_term = (lambda_reg / (2 * m)) * (theta1**2)
# cost += reg_term
# return cost

# 6. Thuật toán Gradient Descent
def gradient_descent(X1, X2, y, theta0, theta1, theta2, learning_rate):
    m = len(X)
    gradient0 = (1 / m) * np.sum(predict(X, theta0, theta1) - y)
    gradient1 = (1 / m) * np.sum((predict(X, theta0, theta1) - y) * X1)
    gradient2 = (1 / m) * np.sum((predict(X, theta0, theta1, theta2) - y) * X2)
    
    new_theta0 = theta0 - learning_rate * gradient0
    new_theta1 = theta1 - learning_rate * gradient1
    new_theta2 = theta2 - learning_rate * gradient2
    
    return new_theta0, new_theta1, new_theta2

# 7. Huấn luyện mô hình
for i in range(iterations):
    theta0, theta1, theta2 = gradient_descent(X, y, theta0, theta1, theta2, learning_rate)
    cost = cost_function(X, y, theta0, theta1, theta2)
    print(f"Iteration {i + 1}: Cost = {cost}, theta0 = {theta0}, theta1 = {theta1}, theta2 = {theta2}")

# 8. Dự đoán kết quả sau huấn luyện
y_pred = predict(X, theta0, theta1, theta2)
print(f"Dự đoán: {y_pred}")

# 9. Vẽ đồ thị dự đoán
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Dữ liệu thực tế')
plt.plot(X, y_pred, color='red', label='Đường dự đoán')
plt.xlabel('Kích thước hạt cát (mm)')
plt.ylabel('Xác suất có mặt nhện')
plt.title('Logistic Regression - Xác suất dự đoán dựa trên kích thước hạt cát')
plt.legend()
plt.grid(True)
plt.show()