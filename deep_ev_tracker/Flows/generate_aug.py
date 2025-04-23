import os
from PIL import Image
import numpy as np

# Placeholder corruption functions (you should replace these with actual implementations)
def gaussian_noise(image, severity):
    """Apply Gaussian noise to the image"""
    row, col, ch = image.shape
    mean = 0
    sigma = severity * 25  # severity controls the noise level
    gaussian = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = np.clip(image + gaussian, 0, 255)
    return noisy_image

def shot_noise(image, severity):
    """Apply shot noise to the image"""
    row, col, ch = image.shape
    prob = severity * 0.01  # severity controls the noise density
    noise = np.random.rand(row, col, ch)
    noisy_image = np.copy(image)
    noisy_image[noise < prob] = 0
    return noisy_image

def impulse_noise(image, severity):
    """Apply impulse noise to the image"""
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = severity * 0.01
    noisy_image = np.copy(image)
    num_salt = int(amount * row * col * s_vs_p)
    num_pepper = int(amount * row * col * (1.0 - s_vs_p))
    
    # Salt noise
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1], :] = 255
    
    # Pepper noise
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1], :] = 0
    
    return noisy_image

def speckle_noise(image, severity):
    """Apply speckle noise to the image"""
    row, col, ch = image.shape
    gaussian = np.random.randn(row, col, ch)
    noisy_image = np.clip(image + image * severity * gaussian, 0, 255)
    return noisy_image

def gaussian_blur(image, severity):
    """Apply Gaussian blur to the image"""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, sigma=severity)

def glass_blur(image, severity):
    """Apply glass blur to the image (a blur effect with distortion)"""
    from scipy.ndimage import zoom
    return zoom(image, (1 + severity * 0.1, 1 + severity * 0.1, 1), order=1)

def defocus_blur(image, severity):
    """Apply defocus blur to the image"""
    from scipy.ndimage import uniform_filter
    return uniform_filter(image, size=(severity, severity, 1))

def motion_blur(image, severity):
    """Apply motion blur to the image"""
    from scipy.ndimage import convolve
    kernel_size = severity  # kernel size should depend on severity

    # Ensure kernel size is odd for proper centering
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Create a kernel that simulates motion blur
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size  # Normalize the kernel

    # If the image is RGB, apply the kernel to each channel
    if image.ndim == 3:
        blurred_image = np.zeros_like(image)
        for i in range(3):  # Iterate over RGB channels
            blurred_image[..., i] = convolve(image[..., i], kernel, mode='reflect')
    else:
        blurred_image = convolve(image, kernel, mode='reflect')

    return blurred_image

def zoom_blur(image, severity):
    """Apply zoom blur to the image"""
    from scipy.ndimage import zoom
    zoom_factor = 1 + severity * 0.1
    return zoom(image, (zoom_factor, zoom_factor, 1), order=1)

def fog(image, severity):
    """Simulate fog effect"""
    fog_effect = np.random.normal(0, severity * 10, image.shape)
    foggy_image = np.clip(image + fog_effect, 0, 255)
    return foggy_image

def frost(image, severity):
    """Simulate frost effect"""
    frost_effect = np.random.normal(0, severity * 20, image.shape)
    frosty_image = np.clip(image + frost_effect, 0, 255)
    return frosty_image

def snow(image, severity):
    """Simulate snow effect"""
    snow_effect = np.random.normal(0, severity * 15, image.shape)
    snowy_image = np.clip(image + snow_effect, 0, 255)
    return snowy_image

def spatter(image, severity):
    """Apply spatter effect to the image"""
    spatter_effect = np.random.normal(0, severity * 30, image.shape)
    spattered_image = np.clip(image + spatter_effect, 0, 255)
    return spattered_image

# List of corruption functions and names
corruptions = [
    ("gaussian_noise", gaussian_noise),
    ("shot_noise", shot_noise),
    ("impulse_noise", impulse_noise),
    ("speckle_noise", speckle_noise),
    ("gaussian_blur", gaussian_blur),
    ("glass_blur", glass_blur),
    ("defocus_blur", defocus_blur),
    ("motion_blur", motion_blur),
    ("zoom_blur", zoom_blur),
    ("fog", fog),
    ("frost", frost),
    ("snow", snow),
    ("spatter", spatter),
]

# Function to apply corruptions and save the results
def generate_corruptions(input_folder, output_folder, severity=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")

            # Open the image
            image = Image.open(image_path).convert("RGB")
            image = np.array(image)  # Convert to numpy array for processing

            # Apply all corruptions
            for corruption_name, corruption_fn in corruptions:
                # Create a separate folder for each corruption type
                corruption_folder = os.path.join(output_folder, corruption_name)
                if not os.path.exists(corruption_folder):
                    os.makedirs(corruption_folder)

                # Apply only a single severity level (default is 5)
                corrupted_image = corruption_fn(image, severity)

                # Convert back to image and save
                corrupted_image = np.clip(corrupted_image, 0, 255).astype(np.uint8)
                corrupted_image = Image.fromarray(corrupted_image)

                # Save the corrupted image in the appropriate folder
                save_path = os.path.join(
                    corruption_folder, f"{os.path.splitext(filename)[0]}.png"
                )
                corrupted_image.save(save_path)
                print(f"Saved {save_path}")

if __name__ == "__main__":
    # Specify input and output directories
    input_folder = "/home/aircraft-lab/Documents/Deep_Learning_Project/DL_Final_Project_Team6/DL_Final_Project_Team6/DL_Dataset_Fall_2024/Town01/rgb/rgb_aligned_town01_night/images_corrected"  # Folder containing your images
    output_folder = "/home/aircraft-lab/Documents/Deep_Learning_Project/Flows/"  # Folder to save corrupted images

    # Generate and save corrupted images
    generate_corruptions(input_folder, output_folder, severity=3)

