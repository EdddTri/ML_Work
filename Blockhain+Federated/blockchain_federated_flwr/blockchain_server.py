import flwr as fl
import json
import time
from Crypto.Cipher import AES
import base64
import numpy as np
import hashlib

# Blockchain block class
class Block:
    def __init__(self, index, data, previous_hash, nonce=0):
        self.index = index
        self.timestamp = time.time()
        self.data = data  # Client weights, nonce, etc.
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def proof_of_work(self, difficulty=0):
        while not self.hash.startswith("0" * difficulty):
            self.nonce += 1
            self.hash = self.compute_hash()


# Blockchain class
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, "Genesis Block", "0")
        genesis_block.proof_of_work()
        self.chain.append(genesis_block)

    def add_block(self, data):
        previous_hash = self.chain[-1].hash
        new_block = Block(len(self.chain), data, previous_hash)
        new_block.proof_of_work()
        self.chain.append(new_block)

    def display_chain(self):
        for block in self.chain:
            print(f"Block {block.index}: {block.__dict__}")


# Encryption utilities
AES_KEY = b'Sixteen byte key'

def encrypt_weights(weights):
    cipher = AES.new(AES_KEY, AES.MODE_EAX)
    ciphertext, nonce = cipher.encrypt_and_digest(np.array(weights).tobytes())
    return base64.b64encode(ciphertext).decode(), base64.b64encode(nonce).decode()

def decrypt_weights(encrypted_weights, nonce):
    encrypted_bytes = base64.b64decode(encrypted_weights)
    nonce_bytes = base64.b64decode(nonce)
    cipher = AES.new(AES_KEY, AES.MODE_EAX, nonce=nonce_bytes)
    decrypted = cipher.decrypt(encrypted_bytes)
    return np.frombuffer(decrypted, dtype=float)


# Custom FedAvg strategy with blockchain integration
class BlockchainFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, blockchain, **kwargs):
        super().__init__(**kwargs)
        self.blockchain = blockchain

    def aggregate_fit(self, server_round, results, failures):
        # Add client weights to the blockchain
        for client_id, (weights, _) in enumerate(results):
            encrypted_weights, nonce = encrypt_weights(weights)
            self.blockchain.add_block({
                "client_id": client_id,
                "weights": encrypted_weights,
                "nonce": nonce,
            })
        print("Blockchain after round", server_round)
        self.blockchain.display_chain()

        # Perform standard FedAvg aggregation
        return super().aggregate_fit(server_round, results, failures)


# Instantiate blockchain
blockchain = Blockchain()

# Define a custom strategy
strategy = BlockchainFedAvg(
    blockchain=blockchain,
    fraction_fit=0.5,
    fraction_evaluate=0.5,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

# Start the server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="192.168.7.97:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
