"""

Script for generating ground truth feature tracks from event data and synchronized images.

"""

 

import os

import cv2

import h5py

import numpy as np

from pathlib import Path

from tqdm import tqdm

import fire

from collections import defaultdict

 

# Constants

IMG_H = 384  # Image height

IMG_W = 512  # Image width

 

# Shi-Tomasi Corner Detection Parameters

MAX_CORNERS = 100  # Increased to detect more features

MIN_DISTANCE = 30

QUALITY_LEVEL = 0.01  # Lowered to detect more corners

BLOCK_SIZE = 25

K = 0.04

USE_HARRIS_DETECTOR = False

TRACK_NAME = "ground_truth_tracks.gt.txt"

 

# Filtering Parameters

MIN_TRACK_DISPLACEMENT = 5  # Minimum displacement to retain a track (pixels)

 

# Optical Flow Parameters

LK_PARAMS = dict(winSize=(21, 21),

                 maxLevel=3,

                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

 

 

def load_images_timestamps(timestamps_file):

    """

    Load image filenames and their corresponding timestamps from a text file.

 

    :param timestamps_file: Path to images_timestamps.txt

    :return: List of tuples [(image_filename, timestamp), ...] sorted by timestamp

    """

    image_timestamps = []

    with open(timestamps_file, 'r') as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 2:

                continue  # Skip malformed lines

            image_filename, timestamp = parts

            image_timestamps.append((image_filename, float(timestamp)))

    # Sort by timestamp

    image_timestamps.sort(key=lambda x: x[1])

    return image_timestamps

 

 

def load_events(h5_file, sort_events=True):

    """

    Load events from an H5 file.

 

    :param h5_file: Path to the H5 file.

    :param sort_events: Whether to sort events by timestamp.

    :return: Tuple of numpy arrays (p, t, x, y)

    """

    with h5py.File(h5_file, 'r') as f:

        p = np.array(f['p'])

        t = np.array(f['t'])

        x = np.array(f['x'])

        y = np.array(f['y'])

 

    if sort_events:

        sort_idx = np.argsort(t)

        p = p[sort_idx]

        t = t[sort_idx]

        x = x[sort_idx]

        y = y[sort_idx]

 

    return p, t, x, y

 

 

def reconstruct_frame(events_indices, p, x, y):

    """

    Reconstruct a grayscale image from a set of events.

 

    :param events_indices: Indices of events to include in this frame.

    :param p: Polarity array.

    :param x: X-coordinate array.

    :param y: Y-coordinate array.

    :return: Reconstructed grayscale image.

    """

    frame = np.zeros((IMG_H, IMG_W), dtype=np.float32)

 

    # Accumulate events

    # Positive polarity: increase brightness

    # Negative polarity: decrease brightness

    for i in events_indices:

        xi = int(round(x[i]))

        yi = int(round(y[i]))

        if 0 <= xi < IMG_W and 0 <= yi < IMG_H:

            frame[yi, xi] += 1 if p[i] == 1 else -1

 

    # Normalize the frame to 0-255

    frame = np.clip(frame, 0, 255)

    frame = frame.astype(np.uint8)

 

    return frame

 

 

def detect_features(frame):

    """

    Detect Shi-Tomasi corners in a frame.

 

    :param frame: Grayscale image.

    :return: Array of corners or None if no corners are found.

    """

    corners = cv2.goodFeaturesToTrack(

        frame,

        maxCorners=MAX_CORNERS,

        qualityLevel=QUALITY_LEVEL,

        minDistance=MIN_DISTANCE,

        blockSize=BLOCK_SIZE,

        useHarrisDetector=USE_HARRIS_DETECTOR,

        k=K

    )

    if corners is not None:

        corners = np.float32(corners).reshape(-1, 1, 2)

    return corners

 

 

def track_features(prev_frame, current_frame, prev_corners):

    """

    Track features from the previous frame to the current frame using Optical Flow.

 

    :param prev_frame: Previous grayscale image.

    :param current_frame: Current grayscale image.

    :param prev_corners: Array of previous corners.

    :return: Tuple of (next_corners, status)

    """

    next_corners, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, current_frame, prev_corners, None, **LK_PARAMS)

    return next_corners, status

 

 

def generate_tracks(images_corrected_dir, image_timestamps, p, t, x, y, output_dir, dt=0.01):

    """

    Generate feature tracks using synchronized images and event data.

 

    :param images_corrected_dir: Path to the images_corrected directory.

    :param image_timestamps: List of tuples [(image_filename, timestamp), ...]

    :param p: Polarity array.

    :param t: Timestamp array.

    :param x: X-coordinate array.

    :param y: Y-coordinate array.

    :param output_dir: Path to the output directory.

    :param dt: Time delta between frames (seconds).

    """

    tracks = []

    track_id_counter = 0

    active_tracks = dict()  # track_id: [(time, x, y), ...]

 

    # Initialize previous frame and corners

    prev_frame = None

    prev_corners = None

 

    # Iterate over each image based on timestamps

    for idx, (image_filename, image_timestamp) in enumerate(tqdm(image_timestamps, desc="Processing Images")):

        image_path = images_corrected_dir / image_filename

        if not image_path.exists():

            print(f"[WARNING] Image {image_path} does not exist. Skipping.")

            continue

 

        # Determine event indices for this frame

        if idx == 0:

            # First frame: include all events up to its timestamp

            event_indices = np.where(t <= image_timestamp)[0]

        else:

            # Subsequent frames: include events between previous and current timestamp

            prev_timestamp = image_timestamps[idx - 1][1]

            event_indices = np.where((t > prev_timestamp) & (t <= image_timestamp))[0]

 

        # Reconstruct frame from events

        frame_events = event_indices

        frame = reconstruct_frame(frame_events, p, x, y)

 

        # Detect features in the first frame

        if prev_frame is None:

            prev_frame = frame

            corners = detect_features(prev_frame)

            if corners is not None:

                for corner in corners:

                    cx, cy = corner.ravel()

                    active_tracks[track_id_counter] = [(0.0, cx, cy)]

                    track_id_counter += 1

            continue

 

        # Track features using Optical Flow

        if prev_corners is not None and len(prev_corners) > 0:

            next_corners, status = track_features(prev_frame, frame, prev_corners)

 

            # Update active tracks

            new_active_tracks = dict()

            for i, (new_pt, st) in enumerate(zip(next_corners, status)):

                if st:

                    new_x, new_y = new_pt.ravel()

                    track_id = list(active_tracks.keys())[i]

                    time = image_timestamp  # Current frame timestamp

                    active_tracks[track_id].append((time, new_x, new_y))

                    new_active_tracks[track_id] = active_tracks[track_id]

            active_tracks = new_active_tracks

 

        # Detect new features if number of active tracks is low

        if len(active_tracks) < MAX_CORNERS * 0.8:

            mask = np.ones_like(frame)

            for (cx, cy) in [trk[-1][1:] for trk in active_tracks.values()]:

                cv2.circle(mask, (int(cx), int(cy)), MIN_DISTANCE, 0, -1)

            new_corners = detect_features(frame)

            if new_corners is not None:

                for corner in new_corners:

                    cx, cy = corner.ravel()

                    active_tracks[track_id_counter] = [(image_timestamp, cx, cy)]

                    track_id_counter += 1

 

        # Update previous frame and corners

        prev_frame = frame.copy()

        if len(active_tracks) > 0:

            prev_corners = np.float32([[[trk[-1][1], trk[-1][2]]]] for trk in active_tracks.values())

            prev_corners = np.array([trk[-1][1:] for trk in active_tracks.values()], dtype=np.float32).reshape(-1, 1, 2)

        else:

            prev_corners = None

 

    # Compile and filter tracks

    print("[INFO] Compiling and filtering tracks...")

    filtered_tracks = []

    for track_id, track_points in active_tracks.items():

        if len(track_points) < 2:

            continue  # Ignore tracks with less than 2 points

        start_pt = np.array(track_points[0][1:])

        end_pt = np.array(track_points[-1][1:])

        displacement = np.linalg.norm(end_pt - start_pt)

        if displacement < MIN_TRACK_DISPLACEMENT:

            continue  # Filter out short tracks

 

        # Check if all points are within image bounds

        in_bounds = all((0 <= pt[0] < IMG_W and 0 <= pt[1] < IMG_H) for _, pt in track_points)

        if not in_bounds:

            continue  # Filter out out-of-bounds tracks

 

        # Assign track ID and compile track entries

        track_entries = []

        for time, x_pt, y_pt in track_points:

            track_entries.append(f"{time:.2f},{x_pt:.2f},{y_pt:.2f}")

        filtered_tracks.append(track_entries)

 

    print(f"[DEBUG] Number of tracks after filtering: {len(filtered_tracks)}")

 

    if not filtered_tracks:

        print("[INFO] No tracks met the filtering criteria.")

        return

 

    # Write tracks to .gt.txt

    tracks_dir = output_dir / "tracks"

    tracks_dir.mkdir(parents=True, exist_ok=True)

    output_path = tracks_dir / TRACK_NAME

 

    with open(output_path, 'w') as f:

        for track in filtered_tracks:

            track_line = " ".join(track)

            f.write(f"{track_line}\n")

 

    print(f"[SUCCESS] Ground truth tracks saved to {output_path}")

    print(f"[INFO] Total tracks generated: {len(filtered_tracks)}")

 

 

def main(dataset_dir, images_timestamps_file, events_h5_file, output_dir, dt=0.01):

    """

    Main function to generate ground truth feature tracks.

 

    :param dataset_dir: Directory path to the dataset containing images_corrected and pose data.

    :param images_timestamps_file: Path to images_timestamps.txt file.

    :param events_h5_file: Path to the single events_corrected.h5 file.

    :param output_dir: Output directory to save the generated tracks.

    :param dt: Time delta between frames (seconds). Default is 0.01.

    """

    # Define paths

    dataset_dir = Path(dataset_dir)

    images_corrected_dir = dataset_dir / "images_corrected"

    images_timestamps_path = Path(images_timestamps_file)

    events_h5_path = Path(events_h5_file)

    output_dir = Path(output_dir)

 

    # Validate directories and files

    if not images_corrected_dir.exists():

        print(f"[ERROR] Images directory {images_corrected_dir} does not exist.")

        return

    if not images_timestamps_path.exists():

        print(f"[ERROR] Images timestamps file {images_timestamps_path} does not exist.")

        return

    if not events_h5_path.exists():

        print(f"[ERROR] Events H5 file {events_h5_path} does not exist.")

        return

    output_dir.mkdir(parents=True, exist_ok=True)

 

    # Load image timestamps

    print("[INFO] Loading image timestamps...")

    image_timestamps = load_images_timestamps(images_timestamps_path)

    print(f"[INFO] Loaded {len(image_timestamps)} image timestamps.")

 

    # Load events

    print("[INFO] Loading events from H5 file...")

    p, t, x, y = load_events(events_h5_path, sort_events=True)

    print(f"[INFO] Total events loaded: {len(t)}")

 

    # Generate tracks

    generate_tracks(images_corrected_dir, image_timestamps, p, t, x, y, output_dir, dt=dt)

 

 

if __name__ == "__main__":

    fire.Fire(main)