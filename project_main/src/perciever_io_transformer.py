import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PerceiverConfig, PerceiverModel
import json
import numpy as np

# Configuration
BATCH_SIZE = 16
SEQUENCE_LENGTH = 50  # Number of IMU/WiFi time steps
FEATURE_DIM = 12  # IMU (9) + WiFi (3)
LATENT_DIM = 256  # Latent space dimension
NUM_BOXES = 10  # Max number of bounding boxes per image
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Datat needs to be processed first to sync vision and sensor inputs

# Paths to Data
IMU_WIFI_DATA_PATH = "path/to/imu_wifi_data.json"  # Replace with your IMU/WiFi data path
VISION_LABELS_PATH = "path/to/vision_labels.json"  # Replace with your vision labels path
GROUND_TRUTH_PATH = "path/to/ground_truth_boxes.json"  # Replace with your ground truth path

# Perceiver IO Configuration
config = PerceiverConfig(
    input_dim=LATENT_DIM,  # Dimension of input embeddings
    num_latents=512,  # Number of latent vectors
    d_latents=LATENT_DIM,  # Latent dimension
    output_dim=LATENT_DIM  # Dimension of output embeddings
)
model = PerceiverModel(config).to(DEVICE)

# Embedding Layers
imu_wifi_embedding = nn.Linear(FEATURE_DIM, LATENT_DIM).to(DEVICE)
vision_embedding = nn.Linear(4, LATENT_DIM).to(DEVICE)  # For bounding boxes
regression_head = nn.Linear(LATENT_DIM, 4).to(DEVICE)  # Predict [xmin, ymin, xmax, ymax]

# Loss and Optimizer
loss_fn = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Load Data
def load_data():
    """
    Loads IMU/WiFi data, vision labels, and ground truth bounding boxes.

    Returns:
        tuple: IMU/WiFi data, vision labels, ground truth boxes.
    """
    with open(IMU_WIFI_DATA_PATH, 'r') as f:
        imu_wifi_data = json.load(f)  # Shape: [batch_size, sequence_length, feature_dim]
    
    with open(VISION_LABELS_PATH, 'r') as f:
        vision_labels = json.load(f)  # Shape: [batch_size, num_boxes, 4]

    with open(GROUND_TRUTH_PATH, 'r') as f:
        ground_truth_boxes = json.load(f)  # Shape: [batch_size, num_boxes, 4]

    # Convert to PyTorch tensors
    imu_wifi_data = torch.tensor(imu_wifi_data, dtype=torch.float32)
    vision_labels = torch.tensor(vision_labels, dtype=torch.float32)
    ground_truth_boxes = torch.tensor(ground_truth_boxes, dtype=torch.float32)

    return imu_wifi_data, vision_labels, ground_truth_boxes

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for _ in range(100):  # Number of batches (example)
        # Load actual data
        imu_wifi_data, vision_labels, ground_truth_boxes = load_data()
        
        # Ensure data is of proper batch size
        imu_wifi_data, vision_labels, ground_truth_boxes = (
            imu_wifi_data[:BATCH_SIZE].to(DEVICE),
            vision_labels[:BATCH_SIZE].to(DEVICE),
            ground_truth_boxes[:BATCH_SIZE].to(DEVICE),
        )

        # Embed inputs
        imu_wifi_embedded = imu_wifi_embedding(imu_wifi_data)  # [BATCH_SIZE, SEQUENCE_LENGTH, LATENT_DIM]
        vision_embedded = vision_embedding(vision_labels)  # [BATCH_SIZE, NUM_BOXES, LATENT_DIM]

        # Concatenate embeddings
        multimodal_input = torch.cat([imu_wifi_embedded, vision_embedded], dim=1)  # [BATCH_SIZE, TOTAL_LENGTH, LATENT_DIM]

        # Forward pass
        outputs = model(inputs=multimodal_input)
        predicted_boxes = regression_head(outputs.last_hidden_state[:, :NUM_BOXES, :])  # Extract NUM_BOXES outputs

        # Compute loss
        loss = loss_fn(predicted_boxes, ground_truth_boxes)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / 100:.4f}")

# Save the Model
torch.save(model.state_dict(), "perceiver_io_multimodal.pth")
print("Model saved to perceiver_io_multimodal.pth")
