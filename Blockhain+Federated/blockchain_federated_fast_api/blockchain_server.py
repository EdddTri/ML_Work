# Updated Blockchain Server Code with Detailed Logging
from fastapi import FastAPI, Query 
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
app = FastAPI()

# File to store global weights
WEIGHTS_FILE = "global_weights.pth"

# Blockchain to store client weights
blockchain = [{"index": 0, "weights_hash": "genesis", "previous_hash": "0"}]
received_weights = []
EXPECTED_CLIENTS = 2  # Number of clients expected
weights_received_count = 0

# Global Model Definition
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc1(x)

global_model = None  # Placeholder for global model

# Load global weights at server startup
@app.on_event("startup")
def load_global_weights():
    global global_model
    if os.path.exists(WEIGHTS_FILE):
        global_model = torch.load(WEIGHTS_FILE)
        logging.info("Global weights loaded from file.")
    else:
        logging.info("No pre-existing global weights found. Initialization required.")

@app.post("/init_global_model")
def init_global_model(input_size: int = Query(..., description="Input size for the model")):
    global global_model
    if global_model is not None:
        return {"message": "Global model already initialized"}
    global_model = SimpleNN(input_size)
    torch.nn.init.constant_(global_model.fc1.weight, 0.1)  # Initialize weights to avoid vanishing
    torch.nn.init.constant_(global_model.fc1.bias, 0.1)
    logging.info(f"Global weights initialized with input size: {input_size}")
    return {"message": "Global weights initialized", "input_size": input_size}

@app.get("/global_model")
def get_global_model():
    if global_model is None:
        logging.info("Client requested global model, but it has not been initialized.")
        return {"message": "Global model not initialized"}
    weights = [param.data.tolist() for param in global_model.parameters()]
    logging.info("Global model weights sent to a client.")
    return {"global_model_weights": weights}

@app.get("/status")
def server_status():
    if global_model is None:
        return {"status": "global weights not initialized"}
    if weights_received_count < EXPECTED_CLIENTS:
        return {"status": f"waiting for more submissions: {weights_received_count}/{EXPECTED_CLIENTS}"}
    return {"status": "ready"}

class Block(BaseModel):
    weights_hash: str
    weights: List[List[float]]

@app.post("/add")
def add_block(block: Block):
    global weights_received_count
    received_weights.append(block.weights)
    weights_received_count += 1

    # Blockchain functionality
    previous_hash = blockchain[-1]["weights_hash"]
    new_block = {
        "index": len(blockchain),
        "weights_hash": block.weights_hash,
        "previous_hash": previous_hash,
    }
    blockchain.append(new_block)

    logging.info(f"Received weights from a client. Total submissions: {weights_received_count}/{EXPECTED_CLIENTS}")

    if weights_received_count >= EXPECTED_CLIENTS:
        aggregate_weights()

    return {"message": "Block added", "block": new_block}

@app.post("/aggregate")
def aggregate_weights():
    global weights_received_count

    if not received_weights:
        logging.info("Aggregation requested, but no weights to aggregate.")
        return {"message": "No weights to aggregate."}

    aggregated_weights = [
        torch.mean(torch.stack([torch.tensor(client_weights[layer]) for client_weights in received_weights]), dim=0)
        for layer in range(len(received_weights[0]))
    ]

    # Reshape the aggregated weights and apply them to the model
    global_model.fc1.weight.data = aggregated_weights[0].view(global_model.fc1.weight.shape)
    global_model.fc1.bias.data = aggregated_weights[1].view(global_model.fc1.bias.shape)

    # Save aggregated weights to file
    torch.save(global_model, WEIGHTS_FILE)
    logging.info("Global model aggregated and saved to file.")

    received_weights.clear()
    weights_received_count = 0
    logging.info("Aggregation complete. Received weights list cleared, and count reset.")

    return {"message": "Global model updated", "aggregated_weights": [w.tolist() for w in aggregated_weights]}


@app.get("/chain")
def get_chain():
    return blockchain
