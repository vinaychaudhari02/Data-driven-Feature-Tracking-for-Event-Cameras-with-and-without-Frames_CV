import cv2
import h5py
import numpy as np
import os
import time
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Load images
image_folder = "/home/aircraft-lab/Documents/Deep_Learning_Project/eds_subseq/00_peanuts_dark/images_corrected"  # Update with your path
image_files = sorted(os.listdir(image_folder))  # Ensure sequential order

images = [
    cv2.imread(os.path.join(image_folder, img), cv2.IMREAD_GRAYSCALE)
    for img in image_files
]

# Ensure all images have the same size (resize them if necessary)
ref_shape = images[0].shape
images = [
    cv2.resize(img, (ref_shape[1], ref_shape[0])) if img.shape != ref_shape else img
    for img in images
]
images = [img for img in images if img is not None]

if len(images) < 2:
    raise ValueError("Not enough valid images to compute optical flow.")

# Load event data from HDF5 file
event_file = "/home/aircraft-lab/Documents/Deep_Learning_Project/eds_subseq/00_peanuts_dark/events_corrected.h5"  # Update with your path
with h5py.File(event_file, 'r') as f:
    x = f['x'][:]  # x-coordinates of events
    y = f['y'][:]  # y-coordinates of events
    t = f['t'][:]  # timestamps of events
    p = f['p'][:]  # polarities of events

# Function to generate an event frame
def generate_event_frame(x, y, p, frame_shape, num_bins):
    frame = np.zeros(frame_shape, dtype=np.int32)
    for i in range(len(x)):
        frame[int(y[i]), int(x[i])] += p[i]
    frame = np.clip(frame, -num_bins, num_bins)
    return frame

# Define frame dimensions and bins
frame_shape = (images[0].shape[0], images[0].shape[1])
event_frame = generate_event_frame(x, y, p, frame_shape, num_bins=1)

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

# Check if flows are already computed and saved
flow_file = "/home/aircraft-lab/Documents/Deep_Learning_Project/Flows/generated_flow/"  # Path to save flows
if os.path.exists(flow_file):
    print("Flows already computed, loading from file...")
    flows = np.load(flow_file)
else:
    print("Computing optical flows...")
    flows = []
    for i in range(len(images) - 1):
        start_time = time.time()
        flow = cv2.calcOpticalFlowFarneback(images[i], images[i + 1], None, **farneback_params)
        flows.append(flow)
        end_time = time.time()
        print(f"Flow {i} computation time: {end_time - start_time:.4f} seconds")

    # Save computed flows to file for future use
    np.save(flow_file, np.array(flows))

# Validate flows
def warp_image(image, flow):
    h, w = flow.shape[:2]
    flow_map = np.zeros_like(flow, dtype=np.float32)
    flow_map[..., 0] = np.arange(w)
    flow_map[..., 1] = np.arange(h)[:, np.newaxis]
    flow_map += flow
    return cv2.remap(image, flow_map[..., 0], flow_map[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def endpoint_error(predicted_flow, ground_truth_flow):
    diff = predicted_flow - ground_truth_flow
    epe = np.sqrt(np.sum(diff**2, axis=-1))
    return np.mean(epe)

def compute_smoothness(flow):
    dx = np.gradient(flow[..., 0], axis=1)
    dy = np.gradient(flow[..., 1], axis=0)
    smoothness = np.mean(np.sqrt(dx**2 + dy**2))
    return smoothness

def compute_flow_magnitude(flow):
    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(mag)

def warp_events(x, y, flow):
    x_warped = x + flow[y.astype(int), x.astype(int), 0]
    y_warped = y + flow[y.astype(int), x.astype(int), 1]
    return x_warped, y_warped

# Validate and print metrics for the first flow
warped_image = warp_image(images[0], flows[0])
mse = np.mean((images[1] - warped_image)**2)
ssim_value = ssim(images[1], warped_image, data_range=warped_image.max() - warped_image.min())
smoothness = compute_smoothness(flows[0])
magnitude = compute_flow_magnitude(flows[0])
x_warped, y_warped = warp_events(x, y, flows[0])

print(f"MSE between warped and original image: {mse:.4f}")
print(f"SSIM between warped and original image: {ssim_value:.4f}")
print(f"Smoothness metric for Flow[0]: {smoothness:.4f}")
print(f"Average flow magnitude for Flow[0]: {magnitude:.4f}")

# Visualize warped events
plt.scatter(x_warped, y_warped, s=1, c='blue', label='Warped Events')
plt.scatter(x, y, s=1, c='red', label='Original Events')
plt.legend()
plt.title("Warped vs Original Events")
plt.show()

# Visualize a flow map
def visualize_flow(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

flow_rgb = visualize_flow(flows[0])
cv2.imshow("Optical Flow", flow_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
