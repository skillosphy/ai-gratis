import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, accuracy_score, 
    precision_score, recall_score, f1_score, classification_report,
    mean_squared_error, r2_score
)

#lin
def predict_lin(w, x, b):
    return w * x + b

def compute_cost_lin(x, y, w, b):
    n = len(x) 
    y_hat = predict_lin(w, x, b)
    errors = y_hat - y
    return (1 / (2 * n)) * np.sum(errors**2)

def compute_gradient_lin(x, y, w, b):
    n = len(y)
    errors = predict_lin(w, x, b) - y
    dj_dw = (1 / n) * np.sum(errors * x)
    dj_db = (1 / n) * np.sum(errors)
    return dj_dw, dj_db

def grad_desc_lin(x, y, w_init, b_init, lr, iterations):
    w, b = w_init, b_init
    cost_history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient_lin(x, y, w, b)
        w = w - lr * dj_dw
        b = b - lr * dj_db
        cost_history.append(compute_cost_lin(x, y, w, b))
    return w, b, cost_history


#log
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict_proba(x, w, b):
    return sigmoid(w * x + b)

def predict_class(w, x, b, threshold=0.5):
    return (predict_proba(x, w, b) >= threshold).astype(int)

def compute_log_loss(x, y, w, b):
    n = len(y)
    y_hat = predict_proba(x, w, b)
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps) # Prevent log(0) crash
    return -(1 / n) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def compute_gradients_logistic(x, y, w, b):
    n = len(y)
    y_hat = predict_proba(x, w, b) 
    dj_dw = (1 / n) * np.sum((y_hat - y) * x)
    dj_db = (1 / n) * np.sum(y_hat - y)
    return dj_dw, dj_db

def logistic_grad_desc(x, y, w_init, b_init, lr, iterations):
    w, b = w_init, b_init
    cost_history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradients_logistic(x, y, w, b)
        w = w - lr * dj_dw
        b = b - lr * dj_db
        cost_history.append(compute_log_loss(x, y, w, b))
    return w, b, cost_history


# regularization
def compute_cost_l2(x, y, w, b, lmb):
    m = len(y)
    error = (x @ w + b) - y
    mse = (1 / (2 * m)) * np.sum(error**2)
    penalty = (lmb / (2 * m)) * np.sum(w**2) 
    return mse + penalty

def compute_gradients_l2(x, y, w, b, lmb):
    m = len(y)
    error = (x @ w + b) - y
    dw = (1 / m) * (x.T @ error) + (lmb / m) * w
    db = (1 / m) * np.sum(error) # Bias has NO regularization penalty
    return dw, db

def grad_desc_l2(x, y, w_init, b_init, lr, iterations, lmb):
    w, b = w_init, b_init
    cost_history = []
    for i in range(iterations):
        dw, db = compute_gradients_l2(x, y, w, b, lmb)
        w = w - lr * dw
        b = b - lr * db
        cost_history.append(compute_cost_l2(x, y, w, b, lmb))
    return w, b, cost_history

#scaling
x_raw = np.array([50.0, 75.0, 100.0, 125.0, 150.0], dtype=float)
x_mean = np.mean(x_raw)
x_std = np.std(x_raw)
x_scaled = (x_raw - x_mean) / x_std  # Correct numerator grouping

# CRITICAL RULE FOR TEST/NEW DATA:
# x_new_scaled = (x_new - x_mean) / x_std  <-- Always use training mean and std!


#confusion 
y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0])

# Manual Confusion Matrix Breakdown
TP = np.sum((y_true == 1) & (y_pred == 1))
TN = np.sum((y_true == 0) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))

# Manual Metrics Calculations
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1_val = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n--- Manual Evaluation Metrics ---")
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1_val:.2f}")

# Scikit-Learn Classification Reports
print("\n--- Scikit-Learn Classification Report ---")
print(classification_report(y_true, y_pred, target_names=["Class 0", "Class 1"], zero_division=0))


#scikit
print("\n--- Scikit-Learn Built-in Models ---")
X_synth = np.array([[0.5, 40], [1.0, 45], [1.5, 50], [2.0, 55], [2.5, 60],
                    [6.0, 70], [6.5, 90], [7.0, 75], [7.5, 92], [8.0, 88]])
y_synth = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Train-Test Split (stratify keeps class proportions balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X_synth, y_synth, test_size=0.3, random_state=42, stratify=y_synth
)

# Standardize features using standalone scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # NEVER fit_transform on test!

# Fit a built-in Logistic Regression classifier
sk_log_reg = LogisticRegression()
sk_log_reg.fit(X_train_scaled, y_train)
sk_preds = sk_log_reg.predict(X_test_scaled)
print("Sklearn Logistic Regression Accuracy:", accuracy_score(y_test, sk_preds))


# poly features
def create_polynomial_features(x_col, degree):
    features = [x_col ** p for p in range(1, degree + 1)]
    return np.column_stack(features)

x_sample = np.array([2, 3, 4])
poly_features_matrix = create_polynomial_features(x_sample, degree=3)
print("\n--- Polynomial Features Matrix (Degree 3) ---")
print(poly_features_matrix)


# matty plotty flatty 
X_house = np.array([50, 75, 100, 125, 150], dtype=float)
y_house = np.array([100, 150, 200, 250, 300], dtype=float)
w_cand, b_cand = 1.5, 10.0

plt.figure(figsize=(6, 4))
plt.scatter(X_house, y_house, s=80, label='Actual Data')

x_line = np.linspace(X_house.min(), X_house.max(), 100)
y_line = predict_lin(w_cand, x_line, b_cand)
plt.plot(x_line, y_line, color='red', label=f"Model Line")

plt.xlabel("House Size (m²)")
plt.ylabel("Price (Thousands)")
plt.title("Matplotlib Visualization Example")
plt.legend()
plt.grid(True)
plt.show()