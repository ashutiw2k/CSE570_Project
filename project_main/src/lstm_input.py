import json
import os

# Paths to data files
CENTROID_LABELS_ALL_SUBJECTS= 'project_main/data/Testing/Transformer Input/'
SENSOR_ALL_SUBJECTS= 'project_main/data/Testing/Transformer Input/'
TRANSFORMER_ALL_SUBJECTS= 'project_main/data/Testing/Transformer Input/'


def extract_unique_timestamps(centroids_data):
    """
    Extract unique timestamps from centroids.json by ignoring the left/right suffix.

    Args:
        centroids_data (dict): Parsed JSON data from centroids.json.

    Returns:
        set: A set of unique timestamps.
    """
    unique_timestamps = set()
    for image_key in centroids_data.keys():
        # Example image_key: "2020-12-23 14_09_58.686064_left.png"
        # Extract timestamp by removing the suffix and file extension
        # Split by '_', join the first three parts to get the timestamp
        parts = image_key.split('_')
        if len(parts) < 4:
            # Handle unexpected format gracefully
            continue
        timestamp = '_'.join(parts[:3])  # "2020-12-23 14_09_58.686064"
        unique_timestamps.add(timestamp)
    return unique_timestamps

def filter_sensor_entries(sensor_data, unique_timestamps):
    """
    Filter sensor entries to retain only those with timestamps in unique_timestamps.

    Args:
        sensor_data (list): List of sensor entries from transformer_sensor_input.json.
        unique_timestamps (set): Set of unique timestamps extracted from centroids.json.

    Returns:
        list: Filtered list of sensor entries.
    """
    filtered_data = [entry for entry in sensor_data if entry.get('timestamp') in unique_timestamps]
    return filtered_data

def process_subject(subject, SENSOR_ALL_SUBJECTS, BB_LABELS_ALL_SUBJECTS, TRANSFORMER_ALL_SUBJECTS):
    """
    Process a single subject by filtering sensor data based on centroids.

    Args:
        subject (str): Subject identifier.
        SENSOR_ALL_SUBJECTS (str): Base directory for sensor data.
        BB_LABELS_ALL_SUBJECTS (str): Base directory for centroid data.
        TRANSFORMER_ALL_SUBJECTS (str): Base directory to save filtered data.
    """
    # Define file paths
    sensor_file = os.path.join(SENSOR_ALL_SUBJECTS, subject, "transformer_sensor_input.json")  # Path to sensor inputs
    centroid_file = os.path.join(BB_LABELS_ALL_SUBJECTS, subject, "centroids.json")          # Path to centroid labels
    output_file = os.path.join(TRANSFORMER_ALL_SUBJECTS, subject, "lstm_sensor_input.json") # Output file

    # Check if sensor_file exists
    if not os.path.isfile(sensor_file):
        print(f"Sensor file not found for subject: {subject}. Skipping.")
        return

    # Check if centroid_file exists
    if not os.path.isfile(centroid_file):
        print(f"Centroid file not found for subject: {subject}. Skipping.")
        return

    # Load sensor data
    try:
        with open(sensor_file, 'r') as f:
            sensor_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Invalid JSON in sensor file for subject: {subject}. Skipping.")
        return

    # Load centroid data
    try:
        with open(centroid_file, 'r') as f:
            centroid_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Invalid JSON in centroid file for subject: {subject}. Skipping.")
        return

    # Extract unique timestamps from centroids.json
    unique_timestamps = extract_unique_timestamps(centroid_data)
    print(f"Subject: {subject} - {len(unique_timestamps)} unique timestamps extracted from centroids.json.")

    # Filter sensor data based on unique timestamps
    filtered_sensor_data = filter_sensor_entries(sensor_data, unique_timestamps)
    print(f"Subject: {subject} - {len(filtered_sensor_data)} sensor entries retained after filtering.")

    # Ensure the subject's output directory exists
    subject_output_dir = os.path.dirname(output_file)
    os.makedirs(subject_output_dir, exist_ok=True)

    # Save the filtered sensor data to the output file
    try:
        with open(output_file, 'w') as f:
            json.dump(filtered_sensor_data, f, indent=4)
        print(f"Filtered sensor data saved to {output_file}")
    except IOError as e:
        print(f"Failed to write to {output_file}: {e}")

def main():
    """
    Main function to process all subjects.
    """
    # List all subjects (assuming each subject has a separate folder)
    subjects = os.listdir(SENSOR_ALL_SUBJECTS)

    for subject in subjects:
        process_subject(subject, SENSOR_ALL_SUBJECTS, CENTROID_LABELS_ALL_SUBJECTS, TRANSFORMER_ALL_SUBJECTS)

    print("All subjects processed successfully.")

if __name__ == "__main__":
    main()