"""
Create Hyperparameter Exploration Table as Image
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load data
print("Loading data...")
train_data = pd.read_csv('AmesHousing_train.csv')
test_data = pd.read_csv('AmesHousing_test.csv')

# Remove non-numeric columns
object_cols = train_data.select_dtypes(include=['object']).columns.tolist()
if 'SalePrice' in object_cols:
    object_cols.remove('SalePrice')

if len(object_cols) > 0:
    train_data = train_data.drop(columns=object_cols)
    test_data = test_data.drop(columns=object_cols)

# Prepare data
X_train = train_data.drop('SalePrice', axis=1).values.astype(np.float32)
y_train = train_data['SalePrice'].values.astype(np.float32)
X_test = test_data.drop('SalePrice', axis=1).values.astype(np.float32)
y_test = test_data['SalePrice'].values.astype(np.float32)

X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

# Normalize target
y_mean = y_train.mean()
y_std = y_train.std()
y_train_norm = (y_train - y_mean) / y_std
y_test_norm = (y_test - y_mean) / y_std

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_norm, dtype=torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_norm, dtype=torch.float32).reshape(-1, 1)

# Define RegularizedNN class
class RegularizedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate=0.0, activation='relu'):
        super(RegularizedNN, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def compute_metrics(y_true, y_pred, denormalize=True):
    if denormalize:
        y_true = y_true * y_std + y_mean
        y_pred = y_pred * y_std + y_mean
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {'MSE': mse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

# Define hyperparameter configurations
hyperparameter_configs = [
    {'name': 'Config1', 'layers': [64], 'epochs': 100, 'lr': 0.0001, 'momentum': 0.0, 'activation': 'relu'},
    {'name': 'Config2', 'layers': [64], 'epochs': 100, 'lr': 0.01, 'momentum': 0.0, 'activation': 'relu'},
    {'name': 'Config3', 'layers': [128, 64], 'epochs': 150, 'lr': 0.001, 'momentum': 0.0, 'activation': 'relu'},
    {'name': 'Config4', 'layers': [128, 64], 'epochs': 150, 'lr': 0.001, 'momentum': 0.0, 'activation': 'tanh'},
    {'name': 'Config5', 'layers': [128, 64, 32], 'epochs': 200, 'lr': 0.001, 'momentum': 0.0, 'activation': 'relu'},
    {'name': 'Config6', 'layers': [128, 64, 32], 'epochs': 200, 'lr': 0.001, 'momentum': 0.9, 'activation': 'relu'},
    {'name': 'Config7', 'layers': [256, 128, 64, 32], 'epochs': 250, 'lr': 0.001, 'momentum': 0.0, 'activation': 'relu'},
    {'name': 'Config8', 'layers': [256, 256], 'epochs': 150, 'lr': 0.001, 'momentum': 0.0, 'activation': 'relu'},
    {'name': 'Config9', 'layers': [256, 256], 'epochs': 150, 'lr': 0.001, 'momentum': 0.85, 'activation': 'tanh'},
    {'name': 'Config10', 'layers': [64, 64, 64, 64], 'epochs': 200, 'lr': 0.0005, 'momentum': 0.0, 'activation': 'relu'},
    {'name': 'Config11', 'layers': [256, 128, 64], 'epochs': 200, 'lr': 0.001, 'momentum': 0.9, 'activation': 'relu'},
    {'name': 'Config12', 'layers': [64, 128, 256], 'epochs': 200, 'lr': 0.001, 'momentum': 0.0, 'activation': 'tanh'},
]

print(f"\nTesting {len(hyperparameter_configs)} configurations...")

hyperparameter_results = []
criterion = nn.MSELoss()

for i, config in enumerate(hyperparameter_configs, 1):
    print(f"[{i}/{len(hyperparameter_configs)}] Testing {config['name']}...")

    model = RegularizedNN(X_train.shape[1], config['layers'],
                         dropout_rate=0.0, activation=config['activation'])

    if config['momentum'] > 0:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy().flatten()
        y_true_eval = y_test_tensor.numpy().flatten()

    # Clean predictions from NaN/Inf
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e10, neginf=-1e10)

    metrics = compute_metrics(y_true_eval, y_pred, denormalize=True)

    hyperparameter_results.append({
        'Config': config['name'],
        'Num_Layers': len(config['layers']),
        'Layer_Structure': str(config['layers']),
        'Epochs': config['epochs'],
        'LR': config['lr'],
        'Momentum': config['momentum'],
        'Activation': config['activation'],
        'MSE': metrics['MSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE']
    })

    print(f"  MSE: {metrics['MSE']/1e6:.2f}M, MAE: {metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

# Create DataFrame
df = pd.DataFrame(hyperparameter_results)

# Save CSV
df.to_csv('hyperparameter_exploration_table.csv', index=False)
print("\nCSV saved: hyperparameter_exploration_table.csv")

# Create table image
fig, ax = plt.subplots(figsize=(18, 10))
ax.axis('tight')
ax.axis('off')

# Prepare table data with better formatting
table_data = []
headers = ['Config', 'Layers', 'Structure', 'Epochs', 'LR', 'Mom', 'Act', 'MSE (M)', 'MAE', 'MAPE (%)']

for _, row in df.iterrows():
    table_data.append([
        row['Config'],
        row['Num_Layers'],
        row['Layer_Structure'],
        row['Epochs'],
        f"{row['LR']:.4f}",
        f"{row['Momentum']:.2f}",
        row['Activation'],
        f"{row['MSE']/1e6:.1f}",
        f"{row['MAE']:.1f}",
        f"{row['MAPE']:.2f}"
    ])

# Create table
table = ax.table(cellText=table_data, colLabels=headers,
                cellLoc='center', loc='center',
                colWidths=[0.08, 0.06, 0.15, 0.06, 0.08, 0.06, 0.06, 0.10, 0.08, 0.10])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data) + 1):
    if i % 2 == 0:
        for j in range(len(headers)):
            table[(i, j)].set_facecolor('#E7E6E6')

# Highlight best MSE
best_mse_idx = df['MSE'].idxmin() + 1
for j in range(len(headers)):
    table[(best_mse_idx, j)].set_facecolor('#C6E0B4')

plt.title('Hyperparameter Exploration Results - Neural Network Configurations',
          fontsize=16, weight='bold', pad=20)

plt.savefig('hyperparameter_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Table image saved: hyperparameter_table.png")

# Also create a summary visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: MSE comparison
ax1 = axes[0, 0]
colors = ['#C6E0B4' if i == df['MSE'].idxmin() else '#4472C4' for i in range(len(df))]
ax1.bar(df['Config'], df['MSE'] / 1e6, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Configuration', fontweight='bold')
ax1.set_ylabel('MSE (Millions)', fontweight='bold')
ax1.set_title('MSE by Configuration', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: MAPE comparison
ax2 = axes[0, 1]
colors2 = ['#C6E0B4' if i == df['MAPE'].idxmin() else '#E97451' for i in range(len(df))]
ax2.bar(df['Config'], df['MAPE'], color=colors2, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Configuration', fontweight='bold')
ax2.set_ylabel('MAPE (%)', fontweight='bold')
ax2.set_title('MAPE by Configuration', fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Learning Rate impact
ax3 = axes[1, 0]
lr_mse = df.groupby('LR')['MSE'].mean() / 1e6
ax3.bar(range(len(lr_mse)), lr_mse.values, color='#70AD47', alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(lr_mse)))
ax3.set_xticklabels([f"{lr:.4f}" for lr in lr_mse.index])
ax3.set_xlabel('Learning Rate', fontweight='bold')
ax3.set_ylabel('Avg MSE (Millions)', fontweight='bold')
ax3.set_title('Learning Rate Impact', fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Network depth impact
ax4 = axes[1, 1]
depth_mse = df.groupby('Num_Layers')['MSE'].mean() / 1e6
ax4.plot(depth_mse.index, depth_mse.values, marker='o', linewidth=2.5,
         markersize=10, color='#5B9BD5', markeredgecolor='black', markeredgewidth=1.5)
ax4.set_xlabel('Number of Layers', fontweight='bold')
ax4.set_ylabel('Avg MSE (Millions)', fontweight='bold')
ax4.set_title('Network Depth Impact', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(depth_mse.index)

plt.tight_layout()
plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Analysis image saved: hyperparameter_analysis.png")

print("\nDone!")
print(f"Best configuration: {df.loc[df['MSE'].idxmin(), 'Config']} with MSE = {df['MSE'].min()/1e6:.2f}M")
