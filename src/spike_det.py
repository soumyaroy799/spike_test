import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import time
from itertools import  product

def detect_spikes(data, th1=400, th2=1.2, kernel=5, plot=False):
    """
    Detect spikes in the image and optionally visualize the results.
    """
    #x_offsets = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    #y_offsets = np.array([1, 1, 1, 0, 0, -1, -1, -1])


    x_offsets = np.arange(-1*(kernel//2),(kernel//2)+1,1); y_offsets = x_offsets

    spike_map = np.zeros(data.shape, dtype=bool)
    corrected_data = data.copy()

    # Create shifted arrays for neighbors
    neighbors = np.stack(
        [np.roll(np.roll(data, y, axis=0), x, axis=1) for x, y in product(x_offsets, y_offsets)]
    )

    # Compute mean and median for valid neighbors
    valid_mask = neighbors > 0
    neighbor_sum = np.sum(neighbors * valid_mask, axis=0)
    neighbor_count = np.sum(valid_mask, axis=0)
    avg_neighbor = np.divide(neighbor_sum, neighbor_count, out=np.zeros_like(neighbor_sum), where=neighbor_count > 0)

    neighbor_sorted = np.sort(neighbors, axis=0)
    median_neighbor = np.median(neighbor_sorted, axis=0)

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
        y, x = np.where(spike_map[10:-10, 10:-10])
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

    return spike_map[10:-10, 10:-10], corrected_data[10:-10, 10:-10]


def process_file(input_file, th1, th2, kernel, save_fits, plot, time_execution):
    """
    Process a single FITS file to detect and remove spikes.
    """
    start_time = time.time() if time_execution else None

    # Load the FITS file
    with fits.open(input_file) as hdul:
        data = hdul[0].data.astype('float')
        header = hdul[0].header

    # Pad the data
    padded_data = np.pad(data, pad_width=10, mode='constant', constant_values=0)

    # Detect spikes
    spike_map, corrected_data = detect_spikes(padded_data, th1, th2, kernel,  plot)

    # Save corrected FITS file if enabled
    if save_fits:
        corrected_file = input_file.replace(".fits", "_corrected.fits")
        fits.writeto(corrected_file, corrected_data, header, overwrite=True)
        print(f"Corrected file saved to {corrected_file}")

    # Print execution time before plotting
    if time_execution:
        elapsed_time = time.time() - start_time
        print(f"Processing time for {input_file}: {elapsed_time:.2f} seconds")
        return elapsed_time

    else:
        return 0


def main():
    parser = argparse.ArgumentParser(description="Spike detection and correction in FITS files.")
    parser.add_argument("input_file", help="Path to the input FITS file")
    parser.add_argument("--th1", type=float, default=400, help="Minimum intensity threshold")
    parser.add_argument("--th2", type=float, default=1.2, help="Median-based intensity threshold")
    parser.add_argument("--kernel", type=float, default=5, help="Full kernel size")
    parser.add_argument("--save-fits", action="store_true", help="Save corrected FITS file")
    parser.add_argument("--plot", action="store_true", help="Plot spike map and corrected data")
    parser.add_argument("--time", action="store_true", help="Time the execution for this file")
    args = parser.parse_args()

    # Process the input file and get the execution time before plotting
    execution_time = process_file(
        input_file=args.input_file,
        th1=args.th1,
        th2=args.th2,
        kernel=args.kernel,
        save_fits=args.save_fits,
        plot=args.plot,
        time_execution=args.time
    )

    # Print execution time before plotting
    if args.time:
        print(f"Total execution time (excluding plotting): {execution_time:.2f} seconds")


if __name__ == "__main__":
    main()
