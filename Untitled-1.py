from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Load the dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels: 0 = malignant, 1 = benign

# 2. Scale the features to mean=0 and std=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split the data into train and test sets (80/20)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 4. Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

# 5. Define the model
model = nn.Sequential(
    nn.Linear(X.shape[1], 16),  # Input layer → Hidden layer
    nn.ReLU(),
    nn.Linear(16, 1),           # Hidden layer → Output layer
    nn.Sigmoid()                # Sigmoid for binary classification
)

# 6. Define the loss function and optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training loop
epochs = 100
train_losses = []  # To store loss values
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())  # Save loss
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.show()


# Evaluate the model on the test set
model.eval()  # Set model to evaluation mode

with torch.no_grad():  # No need to track gradients during inference
    y_pred_test = model(X_test)  # Predict on test features
    predicted_labels = (y_pred_test >= 0.5).float()  # Convert probabilities to 0 or 1

    correct = (predicted_labels == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
