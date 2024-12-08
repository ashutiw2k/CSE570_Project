import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PerceiverConfig, PerceiverModel
import json
from tqdm import tqdm

# Configuration
BATCH_SIZE = 16
LATENT_DIM = 256
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data files
TRANSFORMER_ALL_SUBJECTS= 'project_main/data/Transformer Input/Subject0'

# Load sensor inputs and bounding box labels
INPUT_DATA_PATH = TRANSFORMER_ALL_SUBJECTS + '/' + "transformer_input.json"  # Output

# Define Feature Types
FEATURE_TYPES = [
    "Accel_x", "Accel_y", "Accel_z",  # IMU Accelerometer
    "Gyro_x", "Gyro_y", "Gyro_z",  # IMU Gyroscope
    "Mag_x", "Mag_y", "Mag_z",  # IMU Magnetometer
    "WiFi_ftm1", "WiFi_ftm2", "WiFi_rssi"  # WiFi features
]

# Number of feature types
NUM_FEATURE_TYPES = len(FEATURE_TYPES)

# Perceiver IO Configuration
config = PerceiverConfig(
    input_dim=LATENT_DIM,  # Dimension of input embeddings
    num_latents=512,  # Number of latent vectors
    d_latents=LATENT_DIM,  # Latent dimension
    output_dim=LATENT_DIM  # Dimension of output embeddings
)
model = PerceiverModel(config).to(DEVICE)

# Embedding Layers
sensor_embedding = nn.Linear(1, LATENT_DIM).to(DEVICE)  # Corrected: Map each feature scalar to latent dimension
feature_type_embedding = nn.Embedding(NUM_FEATURE_TYPES, LATENT_DIM).to(DEVICE)  # Learnable embeddings for feature types
bbox_embedding = nn.Linear(4, LATENT_DIM).to(DEVICE)  # Bounding box [xmin, ymin, xmax, ymax]
regression_head = nn.Linear(LATENT_DIM, 4).to(DEVICE)  # Predict bounding boxes

# Loss and Optimizer
loss_fn = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Add Feature-Type Embeddings to Sensor Features
def embed_features(sensor_features):
    """
    Adds feature-type embeddings to the sensor features.

    Args:
        sensor_features (torch.Tensor): Input sensor data [batch_size, num_features].

    Returns:
        torch.Tensor: Feature embeddings [batch_size, num_features, LATENT_DIM].
    """
    batch_size, num_features = sensor_features.shape

    # Create indices for feature types
    feature_type_indices = torch.arange(num_features).expand(batch_size, -1).to(DEVICE)  # [batch_size, num_features]

    # Get feature-type embeddings
    type_embeddings = feature_type_embedding(feature_type_indices)  # [batch_size, num_features, LATENT_DIM]

    # Reshape sensor_features to [batch_size, num_features, 1] for linear layer
    sensor_features = sensor_features.unsqueeze(-1)  # [batch_size, num_features, 1]

    # Pass each feature through the linear layer to get raw embeddings
    raw_embeddings = sensor_embedding(sensor_features)  # [batch_size, num_features, LATENT_DIM]

    # Add feature-type embeddings to raw embeddings
    return raw_embeddings + type_embeddings  # [batch_size, num_features, LATENT_DIM]

# Load and Process Data
def load_data(input_path):
    """
    Loads the combined dataset of sensor inputs and vision labels.

    Args:
        input_path (str): Path to the JSON file containing combined data.

    Returns:
        tuple: Training and testing datasets for `_left` and `_right`.
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Separate `_left` and `_right`
    left_data = [entry for entry in data if entry["side"] == "left"]
    right_data = [entry for entry in data if entry["side"] == "right"]

    # Extract features and labels
    left_sensors = torch.tensor([entry["features"] for entry in left_data], dtype=torch.float32)
    left_bboxes = torch.tensor([entry["bounding_boxes"] for entry in left_data], dtype=torch.float32)
    right_sensors = torch.tensor([entry["features"] for entry in right_data], dtype=torch.float32)
    right_bboxes = torch.tensor([entry["bounding_boxes"] for entry in right_data], dtype=torch.float32)

    return left_sensors, left_bboxes, right_sensors, right_bboxes

# Training and Testing Loop with Alternation
def train_and_test_model(left_sensors, left_bboxes, right_sensors, right_bboxes):
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        # Alternate training and testing sides
        if epoch % 2 == 0:
            train_sensors, train_bboxes = left_sensors, left_bboxes
            test_sensors, test_bboxes = right_sensors, right_bboxes
        else:
            train_sensors, train_bboxes = right_sensors, right_bboxes
            test_sensors, test_bboxes = left_sensors, left_bboxes

        # Batch training with tqdm progress bar
        for i in tqdm(range(0, len(train_sensors), BATCH_SIZE), desc=f"Training Epoch {epoch + 1}"):
            batch_sensors = train_sensors[i:i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE, 12]
            batch_bboxes = train_bboxes[i:i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE, 4]

            # Embed features with feature-type embeddings
            sensor_embedded = embed_features(batch_sensors)  # [BATCH_SIZE, 12, 256]
            bbox_embedded = bbox_embedding(batch_bboxes)  # [BATCH_SIZE, 256]

            # Reshape sensor_embedded to match the vision sequence size
            # Example: Expand sensor embeddings to match the sequence length of bbox embeddings
            # Assuming bbox_embedded should have a sequence dimension, adjust if necessary
            # Here, we'll treat bbox as part of the sequence
            bbox_embedded = torch.flatten(bbox_embedded.unsqueeze(1), 2)  # [BATCH_SIZE, 1, 256]
            # print(bbox_embedded.shape)
            # print(sensor_embedded.shape)
            multimodal_input = torch.cat([sensor_embedded, bbox_embedded], dim=1)  # [BATCH_SIZE, 13, 256]
            print(f'multimodal input shape: ', multimodal_input.shape)
            # Forward pass
            outputs = model(inputs=multimodal_input)
            predicted_boxes = regression_head(outputs.last_hidden_state[:, :1, :])  # Adjust as needed

            # Compute loss
            loss = loss_fn(predicted_boxes, batch_bboxes)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{EPOCHS}, Training Loss: {total_loss:.4f}")

        # Testing
        test_loss = test_model(test_sensors, test_bboxes)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Testing Loss: {test_loss:.4f}")

# Testing Loop
def test_model(test_sensors, test_bboxes):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_sensors), BATCH_SIZE), desc="Testing"):
            batch_sensors = test_sensors[i:i + BATCH_SIZE].to(DEVICE)
            batch_bboxes = test_bboxes[i:i + BATCH_SIZE].to(DEVICE)

            # Embed features with feature-type embeddings
            sensor_embedded = embed_features(batch_sensors)  # [BATCH_SIZE, 12, 256]

            # Forward pass
            outputs = model(inputs=sensor_embedded)  # Adjust if model expects a different input shape
            predicted_boxes = regression_head(outputs.last_hidden_state)  # [BATCH_SIZE, sequence_length, 4]

            # Compute loss
            loss = loss_fn(predicted_boxes, batch_bboxes)
            total_loss += loss.item()

    return total_loss / len(test_sensors)

# Main Execution
if __name__ == "__main__":
    # Load data
    left_sensors, left_bboxes, right_sensors, right_bboxes = load_data(INPUT_DATA_PATH)

    # Train and test model
    train_and_test_model(left_sensors, left_bboxes, right_sensors, right_bboxes)

    # Save the model
    torch.save(model.state_dict(), "perceiver_io_multimodal.pth")
    print("Model saved to perceiver_io_multimodal.pth")
