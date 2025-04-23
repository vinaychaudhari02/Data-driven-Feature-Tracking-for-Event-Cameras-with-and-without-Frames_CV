import os

import cv2

import h5py

import numpy as np

from pathlib import Path

from tqdm import tqdm

import fire

 

# Constants

IMG_H = 384  # Image height (Not used in this script but can be used for resizing if needed)

IMG_W = 512  # Image width (Not used in this script but can be used for resizing if needed)

MAX_CORNERS = 30

MIN_DISTANCE = 30

QUALITY_LEVEL = 0.3

BLOCK_SIZE = 25

K = 0.15

USE_HARRIS_DETECTOR = False

TRACK_NAME = "pose3_h5_tracks.gt.txt"

MIN_TRACK_DISPLACEMENT = 5  # Minimum displacement to retain a track

 

def load_pose_from_h5(h5_file):

    """

    Load pose data from an h5 file.

    :param h5_file: Path to the h5 file.

    :return: Pose matrix (4x4).

    """

    with h5py.File(h5_file, "r") as f:

        if "pose" in f:

            pose = np.array(f["pose"])

            if pose.shape != (4, 4):

                raise ValueError(f"Pose matrix in {h5_file} is not 4x4.")

        else:

            raise ValueError(f"No 'pose' dataset found in {h5_file}.")

    return pose

 

def load_intrinsics_from_h5(h5_file):

    """

    Load camera intrinsic matrix from an h5 file.

    :param h5_file: Path to the h5 file.

    :return: Intrinsic matrix (3x3).

    """

    with h5py.File(h5_file, "r") as f:

        if "intrinsics" in f:

            intrinsics = np.array(f["intrinsics"])

            if intrinsics.shape != (3, 3):

                raise ValueError(f"Intrinsics matrix in {h5_file} is not 3x3.")

        else:

            raise ValueError(f"No 'intrinsics' dataset found in {h5_file}.")

    return intrinsics

 

def project_points(points, pose, intrinsics):

    """

    Project 2D points to 2D using pose and camera intrinsics.

    :param points: Array of 2D points (Nx2) in the image plane.

    :param pose: 4x4 transformation matrix representing the relative pose.

    :param intrinsics: 3x3 camera intrinsic matrix.

    :return: Transformed 2D points (Nx2).

    """

    # Convert 2D points to homogeneous coordinates (Nx3)

    points_homog = np.hstack([points, np.ones((points.shape[0], 1))])

 

    # Transform to normalized camera coordinates (inverse intrinsics)

    points_3d = np.dot(np.linalg.inv(intrinsics), points_homog.T).T  # Shape: (N, 3)

 

    # Convert to homogeneous 3D coordinates (Nx4)

    points_3d_homog = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

 

    # Apply relative pose transformation

    transformed_points_3d = np.dot(pose, points_3d_homog.T).T  # Shape: (N, 4)

 

    # Convert back to 3D coordinates (remove homogeneous component)

    transformed_points_3d = transformed_points_3d[:, :3]

 

    # Project back to 2D image plane using intrinsics

    points_2d_homog = np.dot(intrinsics, transformed_points_3d.T).T  # Shape: (N, 3)

 

    # Normalize by depth to get 2D points

    points_2d = points_2d_homog[:, :2] / points_2d_homog[:, 2, np.newaxis]  # Shape: (N, 2)

 

    return points_2d

 

def generate_single_track(images_corrected_dir, poses, intrinsics, output_dir):

    """

    Generate feature tracks for a single sequence using Pose3 data.

    :param images_corrected_dir: Path to the images_corrected directory.

    :param poses: List of 4x4 pose matrices.

    :param intrinsics: 3x3 camera intrinsic matrix.

    :param output_dir: Path to the output directory.

    """

    tracks = []

    img_t0_p = images_corrected_dir / "0400000.png"

    if not img_t0_p.exists():

        print(f"[ERROR] Reference image {img_t0_p} does not exist.")

        return

 

    img_t0 = cv2.imread(str(img_t0_p), cv2.IMREAD_GRAYSCALE)

    if img_t0 is None:

        print(f"[ERROR] Failed to read image {img_t0_p}.")

        return

 

    # Detect corners in the reference image

    corners = cv2.goodFeaturesToTrack(

        img_t0,

        maxCorners=MAX_CORNERS,

        qualityLevel=QUALITY_LEVEL,

        minDistance=MIN_DISTANCE,

        blockSize=BLOCK_SIZE,

        useHarrisDetector=USE_HARRIS_DETECTOR,

        k=K

    )

 

    if corners is None:

        print(f"[WARNING] No corners detected in {img_t0_p}.")

        return

 

    corners = corners.reshape(-1, 2)

 

    # Initialize tracks with the first frame

    for i_track in range(corners.shape[0]):

        track = np.array([0.0, corners[i_track, 0], corners[i_track, 1]])  # [time, x, y]

        tracks.append(track.reshape((1, 3)))

 

    # Track features across poses

    for i_pose in range(1, len(poses)):

        pose_prev = poses[i_pose - 1]

        pose_next = poses[i_pose]

        rel_pose = np.dot(np.linalg.inv(pose_prev), pose_next)  # Relative transformation

 

        # Project current points to the next frame using relative pose

        transformed_points = project_points(corners, rel_pose, intrinsics)  # Shape: (N, 2)

 

        # Update tracks with new points

        for i_track, transformed_point in enumerate(transformed_points):

            new_time = 0.1 * i_pose  # Increment time; adjust as needed

            new_track_entry = np.array([new_time, transformed_point[0], transformed_point[1]])

            tracks[i_track] = np.vstack([tracks[i_track], new_track_entry])

 

    # Filter tracks by minimum displacement

    filtered_tracks = []

    for track in tracks:

        start_pt = track[0, 1:]

        end_pt = track[-1, 1:]

        displacement = np.linalg.norm(end_pt - start_pt)

        if displacement >= MIN_TRACK_DISPLACEMENT:

            filtered_tracks.append(track)

 

    if not filtered_tracks:

        print("[INFO] No tracks met the minimum displacement criteria.")

        return

 

    # Prepare output directory

    # Since all images are in 'images_corrected', use a generic name or timestamp

    tracks_dir = output_dir / "tracks"

    tracks_dir.mkdir(parents=True, exist_ok=True)

    output_path = tracks_dir / TRACK_NAME

 

    # Save tracks to .gt.txt

    with open(output_path, 'w') as f:

        for track in filtered_tracks:

            # Each track is a (N, 3) array: [time, x, y]

            track_entries = " ".join([f"{entry[0]:.2f},{entry[1]:.2f},{entry[2]:.2f}" for entry in track])

            f.write(f"{track_entries}\n")

 

    print(f"[SUCCESS] Ground truth tracks saved to {output_path}")

 

def generate_tracks(dataset_dir, pose_subdir, output_dir):

    """

    Generate feature tracks using Pose3 data.

    :param dataset_dir: Directory path to the dataset containing image sequences and pose data.

    :param pose_subdir: Subdirectory within pose_3 containing pose .h5 files (e.g., 'time_surface_v2_5').

    :param output_dir: Output directory for generated tracks.

    """

    # Define paths

    dataset_dir = Path(dataset_dir)

    images_corrected_dir = dataset_dir / "images_corrected"

    pose_dir = dataset_dir / "pose_3" / pose_subdir

    output_dir = Path(output_dir)

 

    # Validate directories

    if not images_corrected_dir.exists():

        print(f"[ERROR] Images directory {images_corrected_dir} does not exist.")

        return

    if not pose_dir.exists():

        print(f"[ERROR] Pose directory {pose_dir} does not exist.")

        return

    output_dir.mkdir(parents=True, exist_ok=True)

 

    # Load pose files

    pose_files = sorted(pose_dir.glob("*.h5"))

    if not pose_files:

        print(f"[ERROR] No .h5 pose files found in {pose_dir}.")

        return

 

    print(f"[INFO] Found {len(pose_files)} pose files.")

 

    # Load all poses

    poses = []

    for pose_file in tqdm(pose_files, desc="Loading Poses"):

        try:

            pose = load_pose_from_h5(pose_file)

            poses.append(pose)

        except Exception as e:

            print(f"[ERROR] Failed to load pose from {pose_file}: {e}")

            return

 

    # Load intrinsics (assuming all intrinsics are the same, take from the first pose file)

    try:

        intrinsics = load_intrinsics_from_h5(pose_files[0])

    except Exception as e:

        print(f"[ERROR] Failed to load intrinsics from {pose_files[0]}: {e}")

        return

 

    # Generate tracks

    print("[INFO] Generating feature tracks...")

    generate_single_track(images_corrected_dir, poses, intrinsics, output_dir)

 

    print("[DONE] Feature track generation completed.")

 

if __name__ == "__main__":

    fire.Fire(generate_tracks)