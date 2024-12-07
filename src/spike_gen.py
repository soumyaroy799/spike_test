import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from astropy.io import fits

# Functions for generating synthetic image components
def add_bright_pixels(image, num_bright_pixels,):

    intensity = np.random.randint(6e3, 2e4, dtype=int)
    x_coords = np.random.randint(0, image.shape[0], num_bright_pixels)
    y_coords = np.random.randint(0, image.shape[1], num_bright_pixels)
    image[x_coords, y_coords] = intensity

def add_bright_clusters(image, num_clusters, max_radius=4):
    
    """Add clusters of bright pixels with a gradient decreasing outward."""
    

    for _ in range(num_clusters):

        core_intensity = np.random.randint(6e3, 2e4, dtype=int)
        cx, cy = np.random.randint(max_radius, image.shape[0] - max_radius, 2)
        
        # Create a grid for the cluster
        y, x = np.ogrid[-max_radius:max_radius+1, -max_radius:max_radius+1]
        distance = np.sqrt(x**2 + y**2)
        
        # Generate the gradient mask
        gradient_mask = core_intensity * np.clip(1 - distance / max_radius, 0, 1)
        
        # Apply the mask to the image
        x_start, x_end = cx - max_radius, cx + max_radius + 1
        y_start, y_end = cy - max_radius, cy + max_radius + 1
        
        # Ensure the bounds are within the image
        x_start, x_end = max(x_start, 0), min(x_end, image.shape[0])
        y_start, y_end = max(y_start, 0), min(y_end, image.shape[1])
        
        # Add the gradient cluster to the image
        image_slice = image[x_start:x_end, y_start:y_end]
        mask_slice = gradient_mask[max(0, -x_start):max_radius*2+1-max(0, x_end-image.shape[0]),
                                   max(0, -y_start):max_radius*2+1-max(0, y_end-image.shape[1])]
        
        np.maximum(image_slice, mask_slice, out=image_slice)

def add_constant_streaks(image, num_streaks, ):
    
    for _ in range(num_streaks):

        intensity = np.random.randint(6e3, 2e4, dtype=int)
        x_start, y_start = np.random.randint(0, image.shape[0], 2)
        if np.random.rand() > 0.5:  # Horizontal streak
            length = np.random.randint(100, image.shape[1] // 4)
            image[x_start, y_start:y_start+length] = intensity
        else:  # Vertical streak
            length = np.random.randint(100, image.shape[0] // 4)
            image[x_start:x_start+length, y_start] = intensity

def add_gradient_streaks(image, num_streaks):
    for _ in range(num_streaks):
        x_start, y_start = np.random.randint(0, image.shape[0], 2)
        length = np.random.randint(3, image.shape[0] // 4)
        val = sorted(np.random.randint(6e3, 2e4, size=(2), dtype=int))
        
        gradient = np.linspace(val[0], val[1], length)
        if np.random.rand() > 0.5:  # Horizontal streak
            image[x_start, y_start:y_start+length] = gradient[:image.shape[1] - y_start]
        else:  # Vertical streak
            image[x_start:x_start+length, y_start] = gradient[:image.shape[0] - x_start]

#def add_dark_and_bright_rows_columns(image, spacing):
#    for i in range(0, image.shape[0], spacing):
#        image[i, :] = 255 if i % (2 * spacing) == 0 else 0  # Bright and dark rows
#        image[:, i] = 255 if i % (2 * spacing) == 0 else 0  # Bright and dark columns

# Image generation function
def generate_synthetic_image(
        size, num_bright_pixels, num_clusters, num_constant_streaks, num_gradient_streaks, bias,): #row_col_spacing, bias):
    
    image = np.zeros((size, size), dtype=np.float32) + bias

    add_bright_pixels(image, num_bright_pixels)
    add_bright_clusters(image, num_clusters)
    add_constant_streaks(image, num_constant_streaks)
    add_gradient_streaks(image, num_gradient_streaks)
    #add_dark_and_bright_rows_columns(image, row_col_spacing)

    return image

# File handling and visualization
def save_image_to_fits(image, output_path):
    hdu = fits.PrimaryHDU(image)
    hdu.writeto(output_path, overwrite=True)

def plot_image(image, title="Synthetic Observation"):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray', origin='lower')
    plt.title(title)
    plt.colorbar(label="Intensity")
    plt.show()

# Main function for pipeline
def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the synthetic image
    synthetic_image = generate_synthetic_image(
        size=args.size,
        num_bright_pixels=args.num_bright_pixels,
        num_clusters=args.num_clusters,
        num_constant_streaks=args.num_constant_streaks,
        num_gradient_streaks=args.num_gradient_streaks,
        #row_col_spacing=args.row_col_spacing,
        bias = args.bias,
    )

    # Save the image as a FITS file
    fits_path = output_dir / "synthetic_image.fits"
    save_image_to_fits(synthetic_image, fits_path)

    # Optionally show the image plot
    if args.show_plot:
        plot_image(synthetic_image)

# Argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic observation images.")
    parser.add_argument("--size", type=int, default=4096, help="Size of the square image (default: 4096)")
    parser.add_argument("--num_bright_pixels", type=int, default=500, help="Number of very bright pixels")
    parser.add_argument("--num_clusters", type=int, default=50, help="Number of bright clusters")
    parser.add_argument("--num_constant_streaks", type=int, default=10, help="Number of constant streaks")
    parser.add_argument("--num_gradient_streaks", type=int, default=10, help="Number of gradient streaks")
    parser.add_argument("--row_col_spacing", type=int, default=512, help="Spacing between dark and bright rows/columns")
    parser.add_argument("--output_dir", type=str, default="./../output", help="Directory to save outputs")
    parser.add_argument("--show_plot", action="store_true", help="Display the generated image plot")
    parser.add_argument("--bias", type=float, default=450., help='Default Bias level')

    args = parser.parse_args()
    main(args)

