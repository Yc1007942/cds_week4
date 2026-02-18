"""
Student Success Prediction Module
================================
This module provides functions to predict student final course scores.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)


def prepare_data(df, target_col='final_course_score'):
    """
    Prepare the data for training
    """
    df_processed = df.copy()
    
    # Handle categorical variables
    # 1. Encode 'prereq_ct_grade' (ordinal encoding - grades have natural order)
    grade_mapping = {
        'Exempted/Di': 3.75, 'C+ or lower': 2.5, 'B-': 2.7, 'B': 3.5,
        'B+': 4.0, 'A-': 4.5, 'A/A+': 5
    }
    df_processed['prereq_ct_grade_encoded'] = df_processed['prereq_ct_grade'].map(grade_mapping)
    
    # 2. Encode binary categorical variables
    binary_mapping = {'Yes': 1, 'No': 0}
    df_processed['used_pytorch_tensorflow_enc'] = df_processed['used_pytorch_tensorflow'].map(binary_mapping)
    df_processed['laptop_or_cloud_ready_enc'] = df_processed['laptop_or_cloud_ready'].map(binary_mapping)
    
    # 3. Encode 'pillar year' (combination of pillar and year)
    # df_processed['pillar'] = df_processed['pillar_year'].apply(lambda x: x.split()[0])
    # df_processed['year'] = df_processed['pillar_year'].apply(lambda x: 'final' if 'final' in x else '3rd')
    
    # One-hot encode pillar
    pillar_dummies = pd.get_dummies(df_processed['pillar_year'], prefix='pillar_year')

    
    # Combine all features
    feature_cols = ['cgpa', 'prereq_ct_grade_encoded', 'used_pytorch_tensorflow_enc',
                    'laptop_or_cloud_ready_enc', 'total_grit_score', 'hidden_knowledge_score',
                    'study_friction_index', 'python_confidence_gap']
    
    # Add dummy variables
    X = pd.concat([df_processed[feature_cols], pillar_dummies], axis=1)
    y = df_processed[target_col]
    
    return X, y


def train_model(X, y, test_size=0.2, random_state=42, batch_size=2, learning_rate=0.01, epochs=100):
    """
    Train the linear regression model
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train_scaled.shape[1]
    model = LinearRegressionModel(input_dim)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
    
    return model, scaler, train_losses, X_test_tensor, y_test_tensor, X.columns.tolist()


def evaluate_model(model, test_loader):
    """
    Evaluate the model on test data
    """
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predictions.extend(outputs.numpy().flatten())
            actuals.extend(batch_y.numpy().flatten())
    
    return np.array(predictions), np.array(actuals)


def predict_new_data(model, scaler, new_data_df, feature_names):
    """
    Make predictions on new data
    """
    # Preprocess new data (same steps as training data)
    X_new, _ = prepare_data(new_data_df)
    
    # Ensure columns match training data
    X_new = X_new[feature_names]
    
    # Scale
    X_new_scaled = scaler.transform(X_new)
    X_new_tensor = torch.FloatTensor(X_new_scaled)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model(X_new_tensor)
    
    return predictions.numpy().flatten()