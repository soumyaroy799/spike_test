try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    import numpy as np
    cp = np  # If CuPy is not available, fall back to NumPy
    USE_CUPY = False

import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import time

def detect_spikes(data, th1=400, th2=1.2, plot=False):
    """
    Detect spikes in the image and optionally visualize the results.
    """
    x_offsets = cp.array([-1, 0, 1, -1, 1, -1, 0, 1])
    y_offsets = cp.array([1, 1, 1, 0, 0, -1, -1, -1])

    spike_map = cp.zeros(data.shape, dtype=bool)
    corrected_data = data.copy()

    # Create shifted arrays for neighbors
    neighbors = cp.stack(
        [cp.roll(cp.roll(data, y, axis=0), x, axis=1) for x, y in zip(x_offsets, y_offsets)]
    )

    # Compute mean and median for valid neighbors
    valid_mask = neighbors > 0
    neighbor_sum = cp.sum(neighbors * valid_mask, axis=0)
    neighbor_count = cp.sum(valid_mask, axis=0)
    avg_neighbor = cp.divide(neighbor_sum, neighbor_count, out=cp.zeros_like(neighbor_sum), where=neighbor_count > 0)

    neighbor_sorted = cp.sort(neighbors, axis=0)
    median_neighbor = cp.median(neighbor_sorted, axis=0)

    # Conditions for spike detection
    is_spike = (data > 0) & (data > avg_neighbor + th1) & (data > avg_neighbor * th2)

    # Update spike map and replace spikes
    spike_map[is_spike] = True
    corrected_data[is_spike] = median_neighbor[is_spike]

    # Plot if enabled
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

        # Before: Original data with spikes marked
        axs[0].imshow(data[10:-10, 10:-10], cmap='gray', origin='lower', interpolation='none')
        y, x = cp.where(spike_map[10:-10, 10:-10])
        axs[0].scatter(x, y, color='red', s=5, label='Detected Spikes')
        axs[0].set_title("Original Map with Spikes")
        axs[0].legend()
        axs[0].axis('off')

        # After: Corrected data
        axs[1].imshow(corrected_data[10:-10, 10:-10], cmap='gray', origin='lower', interpolation='none')
        axs[1].set_title("Corrected Map")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return spike_map[10:-10, 10:-10], corrected_data


def process_file(icput_file, th1, th2, save_fits, plot, time_execution):
    """
    Process a single FITS file to detect and remove spikes.
    """
    start_time = time.time() if time_execution else None

    # Load the FITS file
    with fits.open(icput_file) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    # Pad the data
    padded_data = cp.pad(data, pad_width=10, mode='constant', constant_values=0)

    # Detect spikes
    spike_map, corrected_data = detect_spikes(padded_data, th1, th2, plot)

    # Save corrected FITS file if enabled
    if save_fits:
        corrected_file = icput_file.replace(".fits", "_corrected.fits")
        fits.writeto(corrected_file, corrected_data, header, overwrite=True)
        print(f"Corrected file saved to {corrected_file}")

    # Print execution time before plotting
    if time_execution:
        elapsed_time = time.time() - start_time
        print(f"Processing time for {icput_file}: {elapsed_time:.2f} seconds")
        return elapsed_time
    
    else:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Spike detection and correction in FITS files.")
    parser.add_argument("icput_file", help="Path to the icput FITS file")
    parser.add_argument("--th1", type=float, default=400, help="Minimum intensity threshold")
    parser.add_argument("--th2", type=float, default=1.2, help="Median-based intensity threshold")
    parser.add_argument("--save-fits", action="store_true", help="Save corrected FITS file")
    parser.add_argument("--plot", action="store_true", help="Plot spike map and corrected data")
    parser.add_argument("--time", action="store_true", help="Time the execution for this file")
    args = parser.parse_args()

    # Process the icput file and get the execution time before plotting
    execution_time = process_file(
        icput_file=args.icput_file,
        th1=args.th1,
        th2=args.th2,
        save_fits=args.save_fits,
        plot=args.plot,
        time_execution=args.time
    )

    # Print execution time before plotting
    if args.time:
        print(f"Total execution time (excluding plotting): {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()