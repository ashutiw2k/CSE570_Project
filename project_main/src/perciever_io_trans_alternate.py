import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PerceiverConfig, PerceiverModel
import json
import os
from tqdm import tqdm

# Configuration
BATCH_SIZE = 16
LATENT_DIM = 256  # Ensure this matches d_model
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to data files
TRANSFORMER_ALL_SUBJECTS = 'project_main/data/Transformer Input/Subject0'

# Load sensor inputs and bounding box labels
INPUT_DATA_PATH = os.path.join(TRANSFORMER_ALL_SUBJECTS, "transformer_input.json")  # Output

# Define Feature Types
FEATURE_TYPES = [
    "Accel_x", "Accel_y", "Accel_z",  # IMU Accelerometer
    "Gyro_x", "Gyro_y", "Gyro_z",    # IMU Gyroscope
    "Mag_x", "Mag_y", "Mag_z",        # IMU Magnetometer
    "WiFi_ftm1", "WiFi_ftm2", "WiFi_rssi"  # WiFi features
]

# Number of feature types
NUM_FEATURE_TYPES = len(FEATURE_TYPES)

# Function to initialize the model and its components
def initialize_model(latent_dim, num_feature_types):
    """
    Initializes the Perceiver model along with embedding layers and regression head.

    Args:
        latent_dim (int): Dimension of the latent embeddings.
        num_feature_types (int): Number of distinct feature types.

    Returns:
        tuple: Contains the model, embedding layers, regression head, and configuration.
    """
    # Perceiver IO Configuration
    config = PerceiverConfig(
        input_dim=latent_dim,       # Dimension of input embeddings
        num_latents=512,            # Number of latent vectors
        d_latents=latent_dim,       # Latent dimension
        output_dim=latent_dim,      # Dimension of output embeddings
        d_model=latent_dim          # Ensure d_model matches latent_dim
    )
    model = PerceiverModel(config).to(DEVICE)

    # Embedding Layers
    sensor_embedding = nn.Linear(1, latent_dim).to(DEVICE)  # Map each feature scalar to latent dimension
    feature_type_embedding = nn.Embedding(num_feature_types, latent_dim).to(DEVICE)  # Learnable embeddings for feature types
    bbox_embedding = nn.Linear(4, latent_dim).to(DEVICE)  # Bounding box [xmin, ymin, xmax, ymax]
    regression_head = nn.Linear(latent_dim, 4).to(DEVICE)  # Predict bounding boxes

    return model, sensor_embedding, feature_type_embedding, bbox_embedding, regression_head, config

# Define the training function
def train_model(train_sensors, train_bboxes, model, sensor_embedding, feature_type_embedding, bbox_embedding, regression_head, loss_fn, optimizer, epoch_start=1, epoch_end=EPOCHS):
    """
    Train the model on the provided sensor and bbox data.

    Args:
        train_sensors (torch.Tensor): Sensor data [num_samples, num_features].
        train_bboxes (torch.Tensor): Bounding box data [num_samples, 4].
        model (nn.Module): The Perceiver model.
        sensor_embedding (nn.Module): Embedding layer for sensor data.
        feature_type_embedding (nn.Module): Embedding layer for feature types.
        bbox_embedding (nn.Module): Embedding layer for bounding boxes.
        regression_head (nn.Module): Regression head to predict bounding boxes.
        loss_fn (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch_start (int): Starting epoch number.
        epoch_end (int): Ending epoch number.
    """
    model.train()

    for epoch in range(epoch_start, epoch_end + 1):
        total_loss = 0.0

        # Iterate through batches
        for i in tqdm(range(0, len(train_sensors), BATCH_SIZE), desc=f"Training Epoch {epoch}"):
            batch_sensors = train_sensors[i:i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE, num_features]
            batch_bboxes = train_bboxes[i:i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE, 4]

            # Embed features with feature-type embeddings
            sensor_embedded = embed_features(batch_sensors, sensor_embedding, feature_type_embedding)  # [BATCH_SIZE, num_features, LATENT_DIM]
            bbox_embedded = bbox_embedding(batch_bboxes)      # [BATCH_SIZE, LATENT_DIM]

            # Reshape bbox_embedded to [BATCH_SIZE, 1, LATENT_DIM]
            bbox_embedded = bbox_embedded.unsqueeze(1)        # [BATCH_SIZE, 1, LATENT_DIM]

            # Concatenate sensor and bbox embeddings along the sequence dimension
            multimodal_input = torch.cat([sensor_embedded, bbox_embedded], dim=1)  # [BATCH_SIZE, num_features + 1, LATENT_DIM]

            # Forward pass
            outputs = model(inputs=multimodal_input)
            predicted_boxes = regression_head(outputs.last_hidden_state[:, :1, :])  # [BATCH_SIZE, 1, 4]

            # Compute loss
            loss = loss_fn(predicted_boxes.squeeze(1), batch_bboxes)  # [BATCH_SIZE, 4] vs [BATCH_SIZE, 4]
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / (len(train_sensors) / BATCH_SIZE)
        print(f"Epoch {epoch}/{epoch_end}, Training Loss: {average_loss:.4f}")

# Function to embed features
def embed_features(sensor_features, sensor_embedding, feature_type_embedding):
    """
    Adds feature-type embeddings to the sensor features.

    Args:
        sensor_features (torch.Tensor): Input sensor data [batch_size, num_features].
        sensor_embedding (nn.Module): Embedding layer for sensor data.
        feature_type_embedding (nn.Module): Embedding layer for feature types.

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
        tuple: Training datasets for left and right.
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

# Main Execution
if __name__ == "__main__":
    # Load data
    left_sensors, left_bboxes, right_sensors, right_bboxes = load_data(INPUT_DATA_PATH)

    # Verify data sizes
    num_left = left_sensors.shape[0]
    num_right = right_sensors.shape[0]

    if num_left != num_right:
        print(f"Warning: Number of left samples ({num_left}) does not match number of right samples ({num_right}).")
        # Handle mismatch, e.g., truncate to min(num_left, num_right)
        min_samples = min(num_left, num_right)
        left_sensors = left_sensors[:min_samples]
        left_bboxes = left_bboxes[:min_samples]
        right_sensors = right_sensors[:min_samples]
        right_bboxes = right_bboxes[:min_samples]
        print(f"Truncated to {min_samples} samples.")

    # First Training Phase: Train on left_sensors to predict right_bboxes
    print("=== First Training Phase: Train on left sensors to predict right bounding boxes ===")
    model1, sensor_embedding1, feature_type_embedding1, bbox_embedding1, regression_head1, config1 = initialize_model(LATENT_DIM, NUM_FEATURE_TYPES)
    loss_fn1 = nn.SmoothL1Loss()  # Regression loss
    optimizer1 = optim.Adam(
        list(model1.parameters()) +
        list(sensor_embedding1.parameters()) +
        list(feature_type_embedding1.parameters()) +
        list(bbox_embedding1.parameters()) +
        list(regression_head1.parameters()),
        lr=LEARNING_RATE
    )

    train_model(
        train_sensors=left_sensors,
        train_bboxes=right_bboxes,
        model=model1,
        sensor_embedding=sensor_embedding1,
        feature_type_embedding=feature_type_embedding1,
        bbox_embedding=bbox_embedding1,
        regression_head=regression_head1,
        loss_fn=loss_fn1,
        optimizer=optimizer1,
        epoch_start=1,
        epoch_end=EPOCHS
    )

    # Save the model after first training phase
    model1_save_path = "perceiver_io_multimodal_run1.pth"
    torch.save({
        'model_state_dict': model1.state_dict(),
        'sensor_embedding_state_dict': sensor_embedding1.state_dict(),
        'feature_type_embedding_state_dict': feature_type_embedding1.state_dict(),
        'bbox_embedding_state_dict': bbox_embedding1.state_dict(),
        'regression_head_state_dict': regression_head1.state_dict(),
        'config': config1
    }, model1_save_path)
    print(f"Model from first training phase saved to {model1_save_path}")

    # Second Training Phase: Train on right_sensors to predict left_bboxes
    print("\n=== Second Training Phase: Train on right sensors to predict left bounding boxes ===")
    model2, sensor_embedding2, feature_type_embedding2, bbox_embedding2, regression_head2, config2 = initialize_model(LATENT_DIM, NUM_FEATURE_TYPES)
    loss_fn2 = nn.SmoothL1Loss()  # Regression loss
    optimizer2 = optim.Adam(
        list(model2.parameters()) +
        list(sensor_embedding2.parameters()) +
        list(feature_type_embedding2.parameters()) +
        list(bbox_embedding2.parameters()) +
        list(regression_head2.parameters()),
        lr=LEARNING_RATE
    )

    train_model(
        train_sensors=right_sensors,
        train_bboxes=left_bboxes,
        model=model2,
        sensor_embedding=sensor_embedding2,
        feature_type_embedding=feature_type_embedding2,
        bbox_embedding=bbox_embedding2,
        regression_head=regression_head2,
        loss_fn=loss_fn2,
        optimizer=optimizer2,
        epoch_start=1,
        epoch_end=EPOCHS
    )

    # Save the model after second training phase
    model2_save_path = "perceiver_io_multimodal_alternating.pth"
    torch.save({
        'model_state_dict': model2.state_dict(),
        'sensor_embedding_state_dict': sensor_embedding2.state_dict(),
        'feature_type_embedding_state_dict': feature_type_embedding2.state_dict(),
        'bbox_embedding_state_dict': bbox_embedding2.state_dict(),
        'regression_head_state_dict': regression_head2.state_dict(),
        'config': config2
    }, model2_save_path)
    print(f"Model from second training phase saved to {model2_save_path}")
