from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels: 0 = malignant, 1 = benign

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#80/20 train & test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1)

model = nn.Sequential(
    nn.Linear(X.shape[1], 16),  # Input layer → Hidden layer
    nn.ReLU(),
    nn.Linear(16, 1),           # Hidden layer → Output layer
    nn.Sigmoid()                # Sigmoid for binary classification
)

#loss function & optimizer
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop
epochs = 100
train_losses = [] 
for epoch in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())  
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

plt.plot(train_losses)
plt.xlabel("epoch")
plt.ylabel("training loss")
plt.title("training loss over time")
plt.grid(True)
plt.show()


#evaluate model on test set
model.eval()  # Set model to evaluation mode

with torch.no_grad():  
    y_pred_test = model(X_test) 
    predicted_labels = (y_pred_test >= 0.5).float()  

    correct = (predicted_labels == y_test).sum().item()
    total = y_test.size(0)
    accuracy = correct / total

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
