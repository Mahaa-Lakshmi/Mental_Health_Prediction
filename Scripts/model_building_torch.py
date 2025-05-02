# depression_model_torch.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import os
import logging

# Configure the logging settings
logging.basicConfig(
    filename='E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/modelLogs.log',  # Specify the name of your log file
    level=logging.INFO,         # Set the minimum logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODEL_PATH = "E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/models/prediction_model.pth"

# ----------------- Load dataset -----------------
df = pd.read_csv("E:/AI engineer/Guvi/Capstone Projects/Project5/Mental_Health_Prediction/processedData/train_scaled.csv")

X = df.drop('Depression', axis=1)
y = df['Depression']

# ----------------- Feature Selection -----------------
rf = RandomForestClassifier(n_estimators=500, max_depth=3, random_state=42)
rf.fit(X, y)
importances = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
selected_cols = importances.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()
print("Selected columns:", selected_cols)

logging.info(f"Selected columns: {selected_cols}")

#Best model selected columns were Selected columns: ['Age', 'Working Professional or Student', 'Have you ever had suicidal thoughts ?', 'Job Satisfaction', 'Financial Stress', 'Work Pressure', 'Work/Study Hours', 'Dietary Habits', 'Degree', 'Sleep Duration']

X = df[selected_cols]

# ----------------- Apply SMOTE -----------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# ----------------- Train-Test Split -----------------
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# ----------------- Focal Loss -----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_term = self.alpha * (1 - pt) ** self.gamma
        loss = focal_term * bce
        return loss.mean()

# ----------------- Model -----------------
class DepressionNet(nn.Module):
    def __init__(self, input_dim):
        super(DepressionNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 40),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(40),

            nn.Linear(40, 50),
            nn.LeakyReLU(),

            nn.Linear(50, 30),
            nn.LeakyReLU(),

            nn.Linear(30, 25),
            nn.LeakyReLU(),

            nn.Linear(25, 20),
            nn.LeakyReLU(0.2),

            nn.Linear(20, 10),
            nn.LeakyReLU(0.3),

            nn.Linear(10, 5),
            nn.LeakyReLU(0.3),

            nn.Dropout(0.3),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = DepressionNet(input_dim=X_train.shape[1])
print(model)

# ----------------- Optimizer and Loss -----------------
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----------------- Training -----------------
def calculate_f1(preds, labels):
    preds = (preds > 0.5).float()
    TP = ((preds == 1) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    FN = ((preds == 0) & (labels == 1)).sum()
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1.item()

best_f1 = 0
patience = 10
patience_counter = 0

for epoch in range(50):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_train_tensor)
        val_preds = val_outputs
        f1 = calculate_f1(val_preds, y_train_tensor)

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Train F1: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(best_model_state)

# ----------------- Evaluation -----------------
model.eval()
with torch.no_grad():
    test_probs = model(X_test_tensor).numpy()
    test_preds = (test_probs > 0.5).astype(int)

report = classification_report(y_test, test_preds)
print("\nClassification Report:\n", report)
logging.info(f"\nClassification Report:\n{report}")

roc_auc = roc_auc_score(y_test, test_probs)
print("ROC AUC Score:", roc_auc)
logging.info(f"ROC AUC Score: {roc_auc}")

# Save model state_dict
torch.save(model.state_dict(), MODEL_PATH)
logging.info(f"Model saved to {MODEL_PATH}")
print(f"Model saved to {MODEL_PATH}")