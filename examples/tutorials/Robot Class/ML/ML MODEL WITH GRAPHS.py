import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the data
data = pd.read_csv('OmegaII_6_6.csv')

# Select the columns that the player has access to
accessible_columns = ['num_of_player', 'num_of_decks', 'dealer_card', 'init_hand', 'hit', 'outcome']
data = data[accessible_columns]

# One-hot encode 'dealer_card'
data = pd.get_dummies(data, columns=['dealer_card'])

# Normalize the data (excluding one-hot and target columns)
feature_columns = ['num_of_player', 'num_of_decks', 'init_hand', 'hit']  # Specify the columns to normalize
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])

data = data.astype('float32')

# Split the dataset into features and target
target_column = 'outcome'  # Specify the target column
X = data.drop(columns=target_column)
y = data[target_column]

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

# Custom PyTorch Dataset
class BlackjackDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features.iloc[idx].values
        label = self.labels.iloc[idx]
        return torch.tensor(feature), torch.tensor(label)

# DataLoader instances
batch_size = 32
train_dataset = BlackjackDataset(train_features, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = BlackjackDataset(test_features, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Design of the neural network model
class BlackjackModel(nn.Module):
    def __init__(self, input_size):
        super(BlackjackModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

        # Weight Initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.xavier_normal_(self.fc4.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the model, loss function, and optimizer
input_size = train_features.shape[1]
model = BlackjackModel(input_size=input_size).to(device)
criterion = nn.BCEWithLogitsLoss().to(device)
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Training loop with loss tracking
num_epochs = 200
print_every = 5
losses = []  # To track the loss at each epoch
roc_aucs = []  # To track ROC AUC at each epoch
accuracies = []  # To track accuracy at each epoch

for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()  # Set model to training mode
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
        targets = targets.view(-1, 1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate the average loss for the epoch
    average_loss = total_loss / len(train_loader)
    losses.append(average_loss)  # Append to the tracking list
    
    # Evaluation for each epoch (to track ROC AUC and Accuracy)
    model.eval()  # Set the model to evaluation mode
    true_epoch_labels = []
    predicted_epoch_labels = []
    predicted_epoch_probs = []  # To store prediction probabilities for ROC AUC
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float())
            preds = torch.sigmoid(outputs)
            predicted = (preds >= 0.5).float().view(-1)
            true_epoch_labels.extend(targets.cpu().numpy())
            predicted_epoch_labels.extend(predicted.cpu().numpy())
            predicted_epoch_probs.extend(preds.cpu().numpy())

    # Calculate ROC AUC for the current epoch predictions
    epoch_roc_auc = roc_auc_score(true_epoch_labels, predicted_epoch_probs)
    roc_aucs.append(epoch_roc_auc)
    
    # Calculate accuracy for the current epoch predictions
    epoch_accuracy = accuracy_score(true_epoch_labels, predicted_epoch_labels)
    accuracies.append(epoch_accuracy)
    
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, ROC AUC: {epoch_roc_auc:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Evaluation on the test set
true_labels = []
predicted_labels = []
predicted_probs = []
model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs.float())
        preds = torch.sigmoid(outputs)
        predicted = (preds >= 0.5).float().view(-1)
        true_labels.extend(targets.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        predicted_probs.extend(preds.cpu().numpy())

# Calculate final evaluation metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
roc_auc = roc_auc_score(true_labels, predicted_probs)

print(f"Final ROC AUC: {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Plotting Loss over epochs
plt.plot(losses)
plt.title("Loss over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Plotting ROC AUC over epochs
plt.plot(roc_aucs)
plt.title("ROC AUC over Epochs")
plt.xlabel("Epochs")
plt.ylabel("ROC AUC Score")
plt.show()

# Plotting Accuracy over epochs
plt.plot(accuracies)
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

# Confusion Matrix Visualization
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Bar plot for evaluation metrics
metrics = [accuracy, precision, recall, f1]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

plt.bar(metric_names, metrics)
plt.title('Evaluation Metrics')
plt.ylim(0, 1)  # since these metrics range from 0 to 1
plt.show()
