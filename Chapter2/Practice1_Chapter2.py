import numpy as np
import matplotlib.pyplot as plt

# Dữ liệu ban đầu (giờ học và điểm số)
hours = np.array([155, 180, 164, 162, 181, 182, 173, 190, 171, 170, 181, 182, 189, 184, 209, 210])
scores = np.array([51, 52, 54, 53, 55, 59, 61, 59, 63, 76, 64, 66, 69, 72, 70, 80])

# Chuẩn hóa dữ liệu (hours)
hours_norm = (hours - hours.mean()) / hours.std()

# Khởi tạo các tham số
m = 0  
b = 0  
learning_rate = 0.001  
iterations = 20000  

# Hàm tính chi phí (cost function)
def compute_cost(m, b, hours, scores):
    total_samples = len(scores)
    predictions = m * hours + b
    cost = (1 / (2 * total_samples)) * np.sum((predictions - scores) ** 2)
    return cost

# Hàm Gradient Descent
def gradient_descent(hours, scores, m, b, learning_rate, iterations):
    total_samples = len(scores)
    cost_history = []

    for i in range(iterations):
        # Dự đoán giá trị
        predictions = m * hours + b
        
        # Tính toán gradient cho m và b
        m_gradient = -(2 / total_samples) * np.sum(hours * (scores - predictions))
        b_gradient = -(2 / total_samples) * np.sum(scores - predictions)
        
        # Cập nhật m và b
        m = m - learning_rate * m_gradient
        b = b - learning_rate * b_gradient
        
        # Tính toán chi phí sau mỗi lần cập nhật
        cost = compute_cost(m, b, hours, scores)
        cost_history.append(cost)

    return m, b, cost_history

# Thực hiện Gradient Descent
m_optimal, b_optimal, cost_history = gradient_descent(hours_norm, scores, m, b, learning_rate, iterations)

# Vẽ biểu đồ chi phí theo số lần lặp
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function")
plt.show()

# Dự đoán điểm số từ giờ học (sau khi đã tối ưu hóa m và b)
predictions = m_optimal * hours_norm + b_optimal

# Vẽ biểu đồ dữ liệu gốc và đường hồi quy
plt.scatter(hours, scores, color='blue', label="Original Data")
plt.plot(hours, predictions, color='red', label="Fitted Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Scores")
plt.title("Linear Regression with Gradient Descent")
plt.legend()
plt.show()

# Dự đoán điểm số cho giờ học mới
new_hours = np.array([170])
new_hours_norm = (new_hours - hours.mean()) / hours.std()
predicted_scores = m_optimal * new_hours_norm + b_optimal
print("Predicted Scores for new hours:", predicted_scores)
