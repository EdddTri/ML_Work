import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hashlib
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc1(x)

# Load and preprocess the dataset
def load_real_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=["Address"])  # Drop non-numeric columns
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Target

    # Normalize features and target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = (y - y.mean()) / y.std()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=10, shuffle=True)

# Train the local model
def train_local_model(model, train_loader):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    total_loss = 0.0
    for epoch in range(5):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
    avg_loss = total_loss / len(train_loader)
    return model, avg_loss

# Fetch the global model
def fetch_global_model(server_url, input_size):
    logging.info("Fetching the global model from the server...")
    response = requests.get(f"{server_url}/global_model")
    weights = response.json().get("global_model_weights")
    if not weights:
        raise ValueError("Global model weights not found on the server.")
    model = SimpleNN(input_size)
    for param, weight in zip(model.parameters(), weights):
        param.data = torch.tensor(weight)
    logging.info("Global model fetched and initialized.")
    return model

# Send weights to the server
def send_weights_to_server(model, server_url):
    logging.info("Preparing to send weights to the server...")
    weights = [
        [float(value) for value in param.data.flatten()]
        for param in model.parameters()
    ]
    weights_hash = hashlib.sha256(json.dumps(weights).encode()).hexdigest()
    payload = {"weights_hash": weights_hash, "weights": weights}
    response = requests.post(f"{server_url}/add", json=payload)
    logging.info("Weights sent to server. Server Response: %s", response.json())

if __name__ == "__main__":
    server_url = "http://127.0.0.1:5003"  # Update with actual server IP
    data_path = "housing.csv"  # Update with actual dataset path
    input_size = pd.read_csv(data_path).drop(columns=["Address"]).shape[1] - 1

    response = requests.get(f"{server_url}/status").json()
    status = response.get("status")
    logging.info("Server status: %s", status)

    if status == "global weights not initialized":
        init_response = requests.post(f"{server_url}/init_global_model", params={"input_size": input_size})
        logging.info("Server Init Response: %s", init_response.json())
    elif status.startswith("waiting for more submissions"):
        logging.info(f"Server waiting: {status}")
    elif status != "ready":
        raise RuntimeError(f"Unexpected server status: {status}")

    local_model = fetch_global_model(server_url, input_size)
    train_loader = load_real_data(data_path)

    for round_num in range(1, 2):
        logging.info(f"--- Round {round_num} ---")
        trained_model, avg_loss = train_local_model(local_model, train_loader)
        logging.info(f"Training completed for Round {round_num}. Average MSE: {avg_loss}")
        send_weights_to_server(trained_model, server_url)
        logging.info(f"Round {round_num} completed. Waiting for server aggregation...")
