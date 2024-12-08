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

# Function to embed features
def embed_features(sensor_features, sensor_embedding, feature_type_embedding, side_flags):
    """
    Adds feature-type embeddings to the sensor features and includes side flags.

    Args:
        sensor_features (torch.Tensor): Input sensor data [batch_size, num_features].
        sensor_embedding (nn.Module): Embedding layer for sensor data.
        feature_type_embedding (nn.Module): Embedding layer for feature types.
        side_flags (torch.Tensor): Side flags indicating the visible side [batch_size].

    Returns:
        torch.Tensor: Feature embeddings [batch_size, num_features + 1, LATENT_DIM].
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
    feature_embeddings = raw_embeddings + type_embeddings  # [batch_size, num_features, LATENT_DIM]

    # Embed the side flag (binary: 0 for left, 1 for right)
    side_flags = side_flags.unsqueeze(1).unsqueeze(2).float().to(DEVICE)  # [batch_size, 1, 1]
    side_embedding = nn.Linear(1, LATENT_DIM).to(DEVICE)(side_flags)     # [batch_size, 1, LATENT_DIM]

    # Concatenate side embedding to feature embeddings
    combined_embeddings = torch.cat([feature_embeddings, side_embedding], dim=1)  # [batch_size, num_features + 1, LATENT_DIM]

    return combined_embeddings  # [batch_size, num_features + 1, LATENT_DIM]

# Function to calculate Intersection over Union (IoU)
def calculate_iou(pred_boxes, true_boxes):
    """
    Calculates the Intersection over Union (IoU) between predicted and true bounding boxes.
    
    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes [batch_size, 4] in [xmin, ymin, xmax, ymax] format.
        true_boxes (torch.Tensor): Ground truth bounding boxes [batch_size, 4] in [xmin, ymin, xmax, ymax] format.
    
    Returns:
        torch.Tensor: IoU scores [batch_size].
    """
    # Calculate coordinates of intersection rectangles
    # print(pred_boxes.shape)
    # print(true_boxes.shape)
    inter_xmin = torch.max(pred_boxes[:, 0], true_boxes[:, 0])
    inter_ymin = torch.max(pred_boxes[:, 1], true_boxes[:, 1])
    inter_xmax = torch.min(pred_boxes[:, 2], true_boxes[:, 2])
    inter_ymax = torch.min(pred_boxes[:, 3], true_boxes[:, 3])
    
    # Compute areas of intersection rectangles
    inter_width = (inter_xmax - inter_xmin).clamp(min=0)
    inter_height = (inter_ymax - inter_ymin).clamp(min=0)
    inter_area = inter_width * inter_height
    
    # Compute areas of individual bounding boxes
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    true_area = (true_boxes[:, 2] - true_boxes[:, 0]).clamp(min=0) * (true_boxes[:, 3] - true_boxes[:, 1]).clamp(min=0)
    
    # Compute union area
    union_area = pred_area + true_area - inter_area + 1e-6  # Add small epsilon to avoid division by zero
    
    # Compute IoU
    iou = inter_area / union_area
    
    return iou

# Define the testing function
def test_model(test_sensors, test_bboxes, test_side_flags, model, sensor_embedding, feature_type_embedding, bbox_embedding, regression_head, loss_fn):
    """
    Evaluates the model on the testing subset.

    Args:
        test_sensors (torch.Tensor): Testing sensor data [num_test_samples, num_features].
        test_bboxes (torch.Tensor): Testing bounding box data [num_test_samples, 4].
        test_side_flags (torch.Tensor): Testing side flags [num_test_samples].
        model (nn.Module): The Perceiver model.
        sensor_embedding (nn.Module): Embedding layer for sensor data.
        feature_type_embedding (nn.Module): Embedding layer for feature types.
        bbox_embedding (nn.Module): Embedding layer for bounding boxes.
        regression_head (nn.Module): Regression head to predict bounding boxes.
        loss_fn (nn.Module): Loss function.

    Returns:
        tuple: (average_testing_loss, average_iou)
    """
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(test_sensors), BATCH_SIZE), desc="Testing"):
            batch_sensors = test_sensors[i:i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE, num_features]
            batch_bboxes = test_bboxes[i:i + BATCH_SIZE].to(DEVICE)    # [BATCH_SIZE, 4]
            batch_side_flags = test_side_flags[i:i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE]

            # Embed features with feature-type embeddings and side flags
            sensor_embedded = embed_features(batch_sensors, sensor_embedding, feature_type_embedding, batch_side_flags)  # [BATCH_SIZE, num_features + 1, LATENT_DIM]
            bbox_embedded = bbox_embedding(batch_bboxes)      # [BATCH_SIZE, LATENT_DIM]

            # Reshape bbox_embedded to [BATCH_SIZE, 1, LATENT_DIM]
            bbox_embedded = torch.flatten(bbox_embedded.unsqueeze(1), 2)        # [BATCH_SIZE, 1, LATENT_DIM]

            # Concatenate sensor and bbox embeddings along the sequence dimension
            multimodal_input = torch.cat([sensor_embedded, bbox_embedded], dim=1)  # [BATCH_SIZE, num_features + 2, LATENT_DIM]

            # Forward pass
            outputs = model(inputs=multimodal_input)
            predicted_boxes = regression_head(outputs.last_hidden_state[:, :1, :])  # [BATCH_SIZE, 1, 4]
            predicted_boxes = predicted_boxes.squeeze(1)  # [BATCH_SIZE, 4]

            # Compute loss
            loss = loss_fn(predicted_boxes, batch_bboxes)  # [BATCH_SIZE, 4] vs [BATCH_SIZE, 4]
            total_loss += loss.item() * batch_sensors.size(0)  # Accumulate loss weighted by batch size

            # Compute IoU
            iou = calculate_iou(predicted_boxes, torch.flatten(batch_bboxes, 1))  # [BATCH_SIZE]
            total_iou += iou.sum().item()  # Accumulate IoU
            total_samples += batch_sensors.size(0)

    average_loss = total_loss / total_samples
    average_iou = total_iou / total_samples
    return average_loss, average_iou

# Define the training function
def train_model(train_sensors, train_bboxes, train_side_flags, test_sensors, test_bboxes, test_side_flags,
               model, sensor_embedding, feature_type_embedding, bbox_embedding, regression_head, loss_fn, optimizer,
               epoch_start=1, epoch_end=EPOCHS):
    """
    Trains the model and evaluates it on the testing subset within each epoch.

    Args:
        train_sensors (torch.Tensor): Training sensor data [num_train_samples, num_features].
        train_bboxes (torch.Tensor): Training bounding box data [num_train_samples, 4].
        train_side_flags (torch.Tensor): Training side flags [num_train_samples].
        test_sensors (torch.Tensor): Testing sensor data [num_test_samples, num_features].
        test_bboxes (torch.Tensor): Testing bounding box data [num_test_samples, 4].
        test_side_flags (torch.Tensor): Testing side flags [num_test_samples].
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
    for epoch in range(epoch_start, epoch_end + 1):
        model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_samples = 0

        # Shuffle the training data at the start of each epoch
        permutation = torch.randperm(train_sensors.size()[0])
        train_sensors_shuffled = train_sensors[permutation]
        train_bboxes_shuffled = train_bboxes[permutation]
        train_side_flags_shuffled = train_side_flags[permutation]

        # Iterate through training batches
        for i in tqdm(range(0, len(train_sensors_shuffled), BATCH_SIZE), desc=f"Training Epoch {epoch}"):
            batch_sensors = train_sensors_shuffled[i:i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE, num_features]
            batch_bboxes = train_bboxes_shuffled[i:i + BATCH_SIZE].to(DEVICE)    # [BATCH_SIZE, 4]
            batch_side_flags = train_side_flags_shuffled[i:i + BATCH_SIZE].to(DEVICE)  # [BATCH_SIZE]

            # Embed features with feature-type embeddings and side flags
            sensor_embedded = embed_features(batch_sensors, sensor_embedding, feature_type_embedding, batch_side_flags)  # [BATCH_SIZE, num_features + 1, LATENT_DIM]
            bbox_embedded = bbox_embedding(batch_bboxes)      # [BATCH_SIZE, LATENT_DIM]

            # Reshape bbox_embedded to [BATCH_SIZE, 1, LATENT_DIM]
            bbox_embedded = torch.flatten(bbox_embedded.unsqueeze(1),2)        # [BATCH_SIZE, 1, LATENT_DIM]

            # Concatenate sensor and bbox embeddings along the sequence dimension
            multimodal_input = torch.cat([sensor_embedded, bbox_embedded], dim=1)  # [BATCH_SIZE, num_features + 2, LATENT_DIM]

            # Forward pass
            outputs = model(inputs=multimodal_input)
            predicted_boxes = regression_head(outputs.last_hidden_state[:, :1, :])  # [BATCH_SIZE, 1, 4]
            predicted_boxes = predicted_boxes.squeeze(1)  # [BATCH_SIZE, 4]

            # Compute loss
            loss = loss_fn(predicted_boxes, batch_bboxes)  # [BATCH_SIZE, 4] vs [BATCH_SIZE, 4]
            total_loss += loss.item() * batch_sensors.size(0)  # Accumulate loss weighted by batch size

            # Compute IoU
            iou = calculate_iou(predicted_boxes, torch.flatten(batch_bboxes, 1))  # [BATCH_SIZE]
            total_iou += iou.sum().item()  # Accumulate IoU
            total_samples += batch_sensors.size(0)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        average_train_loss = total_loss / total_samples
        average_train_iou = total_iou / total_samples

        # After each epoch, compute test loss and IoU
        test_loss, test_iou = test_model(test_sensors, test_bboxes, test_side_flags, model, sensor_embedding, feature_type_embedding, bbox_embedding, regression_head, loss_fn)

        print(f"Epoch {epoch}/{epoch_end}, "
              f"Training Loss: {average_train_loss:.4f}, Training IoU: {average_train_iou:.4f}, "
              f"Testing Loss: {test_loss:.4f}, Testing IoU: {test_iou:.4f}")

# Load and Process Data
def load_and_combine_data(input_path):
    """
    Loads the dataset, pairs left and right data, and combines them into a single dataset.

    Args:
        input_path (str): Path to the JSON file containing combined data.

    Returns:
        tuple: Combined sensors, bounding boxes, and side flags.
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

    # Create side flags: 0 for left, 1 for right
    left_side_flags = torch.zeros(left_sensors.size(0), dtype=torch.long)  # [num_left]
    right_side_flags = torch.ones(right_sensors.size(0), dtype=torch.long)  # [num_right]

    # Combine sensors, bboxes, and side flags
    combined_sensors = torch.cat([left_sensors, right_sensors], dim=0)      # [num_left + num_right, num_features]
    combined_bboxes = torch.cat([left_bboxes, right_bboxes], dim=0)        # [num_left + num_right, 4]
    combined_side_flags = torch.cat([left_side_flags, right_side_flags], dim=0)  # [num_left + num_right]

    return combined_sensors, combined_bboxes, combined_side_flags

# Main Execution
if __name__ == "__main__":
    # Load and combine data
    combined_sensors, combined_bboxes, combined_side_flags = load_and_combine_data(INPUT_DATA_PATH)

    # Verify combined data sizes
    num_combined = combined_sensors.shape[0]
    print(f"Total combined samples: {num_combined}")

    # Split data into training and testing
    TEST_RATIO = 0.2  # 20% for testing

    # Function to split data into training and testing subsets
    def split_data(sensors, bboxes, side_flags, test_ratio=0.2):
        """
        Splits the data into training and testing subsets.

        Args:
            sensors (torch.Tensor): Sensor data [num_samples, num_features].
            bboxes (torch.Tensor): Bounding box data [num_samples, 4].
            side_flags (torch.Tensor): Side flags [num_samples].
            test_ratio (float): Proportion of data to use for testing.

        Returns:
            tuple: (train_sensors, train_bboxes, train_side_flags, test_sensors, test_bboxes, test_side_flags)
        """
        num_samples = sensors.shape[0]
        indices = torch.randperm(num_samples)

        test_size = int(num_samples * test_ratio)
        train_size = num_samples - test_size

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_sensors = sensors[train_indices]
        train_bboxes = bboxes[train_indices]
        train_side_flags = side_flags[train_indices]
        test_sensors = sensors[test_indices]
        test_bboxes = bboxes[test_indices]
        test_side_flags = side_flags[test_indices]

        return train_sensors, train_bboxes, train_side_flags, test_sensors, test_bboxes, test_side_flags

    # Split the data
    train_sensors, train_bboxes, train_side_flags, test_sensors, test_bboxes, test_side_flags = split_data(
        combined_sensors, combined_bboxes, combined_side_flags, test_ratio=TEST_RATIO
    )
    print(f"Training samples: {train_sensors.shape[0]}, Testing samples: {test_sensors.shape[0]}")

    # Initialize single model and its components
    model, sensor_embedding, feature_type_embedding, bbox_embedding, regression_head, config = initialize_model(LATENT_DIM, NUM_FEATURE_TYPES)

    # Define loss function and optimizer
    loss_fn = nn.SmoothL1Loss()  # Regression loss
    optimizer = optim.Adam(
        list(model.parameters()) +
        list(sensor_embedding.parameters()) +
        list(feature_type_embedding.parameters()) +
        list(bbox_embedding.parameters()) +
        list(regression_head.parameters()),
        lr=LEARNING_RATE
    )

    # Train and Test the model
    train_model(
        train_sensors=train_sensors,
        train_bboxes=train_bboxes,
        train_side_flags=train_side_flags,
        test_sensors=test_sensors,
        test_bboxes=test_bboxes,
        test_side_flags=test_side_flags,
        model=model,
        sensor_embedding=sensor_embedding,
        feature_type_embedding=feature_type_embedding,
        bbox_embedding=bbox_embedding,
        regression_head=regression_head,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epoch_start=1,
        epoch_end=EPOCHS
    )

    # Save the trained model
    model_save_path = "perceiver_io_multimodal_combined.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'sensor_embedding_state_dict': sensor_embedding.state_dict(),
        'feature_type_embedding_state_dict': feature_type_embedding.state_dict(),
        'bbox_embedding_state_dict': bbox_embedding.state_dict(),
        'regression_head_state_dict': regression_head.state_dict(),
        'config': config
    }, model_save_path)
    print(f"Combined model saved to {model_save_path}")
