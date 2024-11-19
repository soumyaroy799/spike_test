import argparse
import time
from astropy.io import fits
import matplotlib.pyplot as plt

try:
    import cupy
    xp = cupy
except ModuleNotFoundError:
    import numpy
    xp = numpy

def to_numpy(array):
    """
    Convert an array to a NumPy array, regardless of whether it's a NumPy or CuPy array.
    """
    if xp.__name__ == "cupy":
        return xp.asnumpy(array)
    return array  # Already a NumPy array

def pad_data(data, padding=10):
    """
    Pad the data array with zeros.
    """
    padded_data = xp.zeros((data.shape[0] + 2 * padding, data.shape[1] + 2 * padding), dtype=data.dtype)
    padded_data[padding:-padding, padding:-padding] = data
    return padded_data

def detect_spikes(data, th1=400, th2=1.2, padding=10):
    """
    Detect spikes in the image based on intensity thresholds using vectorized operations.
    Returns a spike mask and the corrected data with spikes replaced by the median of neighbors.
    """
    x_offsets = xp.array([-1, 0, 1, -1, 1, -1, 0, 1])
    y_offsets = xp.array([1, 1, 1, 0, 0, -1, -1, -1])

    # Get the neighborhood values for each pixel
    neighbors = xp.array([
        data[padding + x_offsets[i]: -padding + x_offsets[i] or None,
             padding + y_offsets[i]: -padding + y_offsets[i] or None]
        for i in range(8)
    ])

    # Compute the average and median of valid neighbors
    valid_neighbors = neighbors > 0
    valid_counts = xp.sum(valid_neighbors, axis=0)
    avg_neighbors = xp.sum(neighbors * valid_neighbors, axis=0) / valid_counts
    median_neighbors = xp.median(xp.where(valid_neighbors, neighbors, xp.nan), axis=0)

    # Extract the central region
    central_data = data[padding:-padding, padding:-padding]

    # Apply spike detection thresholds
    spike_mask = (central_data > avg_neighbors + th1) & (central_data > avg_neighbors * th2)

    return spike_mask, median_neighbors

def replace_spikes(data, spike_mask, median_neighbors, padding=10):
    """
    Replace spikes in the original data with the median of valid neighbors.
    """
    corrected_data = data.copy()
    corrected_data[spike_mask] = median_neighbors[spike_mask]
    return corrected_data

def process_image(file_path, th1=400, th2=1.2, padding=10, save_output=False, output_spike_mask=None, output_corrected_data=None, verbose=False):
    """
    Process an image to detect and correct spikes.
    Optionally saves the results to disk.
    """
    if verbose:
        print(f"Processing file: {file_path}")

    # Load FITS file
    with fits.open(file_path) as hdul:
        data = xp.array(hdul[0].data, dtype=float)  # Load as float for processing

    if verbose:
        print("Data loaded successfully.")

    # Pad the data
    padded_data = pad_data(data, padding=padding)

    if verbose:
        print("Data padded.")

    # Detect spikes
    spike_mask, median_neighbors = detect_spikes(padded_data, th1=th1, th2=th2, padding=padding)

    if verbose:
        print(f"Spikes detected. Number of spikes: {xp.sum(spike_mask)}")

    # Replace spikes in the original data
    final_corrected_data = replace_spikes(data, spike_mask, median_neighbors, padding=padding)

    if verbose:
        print(f"Spikes replaced. Max value in corrected data: {xp.max(final_corrected_data)}")

    if save_output:
        if output_spike_mask is None or output_corrected_data is None:
            raise ValueError("Output paths for spike mask and corrected data must be specified when --save_output is used.")

        # Save spike mask and corrected data
        xp.save(output_spike_mask, to_numpy(spike_mask))
        xp.save(output_corrected_data, to_numpy(final_corrected_data))

        if verbose:
            print(f"Spike mask saved to {output_spike_mask}, corrected data saved to {output_corrected_data}.")
    else:
        if verbose:
            print("Results not saved to disk.")

    # Return results
    return to_numpy(spike_mask), to_numpy(final_corrected_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect and correct spikes in a FITS image.")
    parser.add_argument("file_path", type=str, help="Path to the input FITS file.")
    parser.add_argument("--th1", type=float, default=400, help="Minimum intensity threshold for spike detection.")
    parser.add_argument("--th2", type=float, default=1.2, help="Median threshold multiplier for spike detection.")
    parser.add_argument("--padding", type=int, default=10, help="Padding size for processing edges.")
    parser.add_argument("--save_output", action="store_true", help="Flag to save the output files.")
    parser.add_argument("--output_spike_mask", type=str, help="Path to save the output spike mask (npy format).")
    parser.add_argument("--output_corrected_data", type=str, help="Path to save the corrected data (npy format).")
    parser.add_argument("--verbose", action="store_true", help="Print detailed log during processing.")

    args = parser.parse_args()

    spike_mask, corrected_data = process_image(
        file_path=args.file_path,
        th1=args.th1,
        th2=args.th2,
        padding=args.padding,
        save_output=args.save_output,
        output_spike_mask=args.output_spike_mask,
        output_corrected_data=args.output_corrected_data,
        verbose=args.verbose
    )

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(spike_mask, origin='lower', cmap='jet')
    ax1.set_title("Spike Mask")

    ax2 = fig.add_subplot(122)
    ax2.imshow(corrected_data, origin='lower', cmap='Greys')
    ax2.set_title("Corrected Data")

    plt.tight_layout()
    plt.show()
