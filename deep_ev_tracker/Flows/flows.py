import cv2
import h5py
import numpy as np
import os
import time

# Load images
image_folder = "/home/aircraft-lab/Documents/Deep_Learning_Project/DL_Final_Project_Team6/DL_Final_Project_Team6/DL_Dataset_Fall_2024/Town 01/rgb/rgb_aligned_town01_night/images_corrected"
image_files = sorted(os.listdir(image_folder))  # Ensure sequential order

# Validate the image directory
if not image_files:
    raise FileNotFoundError(f"No images found in {image_folder}")

images = []
for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Unable to read image {img_path}")
        continue
    images.append(img)

if len(images) < 2:
    raise ValueError("Insufficient valid images for optical flow computation.")

# Load event data from HDF5 file
event_file = "/home/aircraft-lab/Documents/Deep_Learning_Project/DL_Final_Project_Team6/DL_Final_Project_Team6/DL_Dataset_Fall_2024/Town 01/rgb/rgb_aligned_town01_night/events_corrected.h5"

if not os.path.exists(event_file):
    raise FileNotFoundError(f"Event file not found at {event_file}")

with h5py.File(event_file, 'r') as f:
    x = f['x'][:]
    y = f['y'][:]
    t = f['t'][:]
    p = f['p'][:]

# Function to visualize flow
def visualize_flow(flow):
    """
    Visualize optical flow as an RGB image.
    """
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

# Function to generate event frame
def generate_event_frame(x, y, p, frame_shape, num_bins):
    """
    Generate an event frame by accumulating events over bins.
    """
    frame = np.zeros(frame_shape, dtype=np.int32)
    for i in range(len(x)):
        frame[int(y[i]), int(x[i])] += p[i]
    frame = np.clip(frame, -num_bins, num_bins)
    return frame

# Define frame dimensions and bins
frame_shape = (images[0].shape[0], images[0].shape[1])
event_frame = generate_event_frame(x, y, p, frame_shape, num_bins=1)

flows = []

# Farneback parameters
farneback_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

# Directory to save flows
flow_save_dir = "/home/aircraft-lab/Documents/Deep_Learning_Project/Flows/generated_flow/"
os.makedirs(flow_save_dir, exist_ok=True)

# Compute optical flow and save
for i in range(len(images) - 1):
    start_time = time.time()
    flow = cv2.calcOpticalFlowFarneback(images[i], images[i + 1], None, **farneback_params)
    end_time = time.time()
    flows.append(flow)
    print(f"Flow {i} computation time: {end_time - start_time:.4f} seconds")
    
    # Save the flow
    flow_file = os.path.join(flow_save_dir, f"flow_{i}.npy")
    np.save(flow_file, flow)

    # Visualize and save flow visualization
    flow_rgb = visualize_flow(flow)
    vis_file = os.path.join(flow_save_dir, f"flow_{i}_visualization.png")
    cv2.imwrite(vis_file, flow_rgb)
    print(f"Flow {i} saved and visualized.")

# Optional: Compute flow between event frame and the first image
start_time = time.time()
event_to_image_flow = cv2.calcOpticalFlowFarneback(event_frame, images[0], None, **farneback_params)
end_time = time.time()
print(f"Event to image flow computation time: {end_time - start_time:.4f} seconds")

# Display visualizations
for i, flow in enumerate(flows):
    flow_rgb = visualize_flow(flow)
    cv2.imshow(f"Optical Flow {i}", flow_rgb)
    cv2.waitKey(0)
cv2.destroyAllWindows()
