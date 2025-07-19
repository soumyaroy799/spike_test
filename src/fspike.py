import numpy as np
from astropy.io import fits
from scipy.ndimage import uniform_filter, median_filter
from tqdm import tqdm
import time
import os

def despike_optimized(file, iterations=2, sigma_threshold=4, dry_run=False):
    start = time.time()
    with fits.open(file) as hdul:
        data0 = hdul[0].data.astype(np.float32)
        img_header = hdul[0].header

    flnm = os.path.splitext(os.path.basename(file))[0]
    print(f'\nDespiking: {flnm}')

    #Sun center and radius
    Rad = img_header['R_SUN'] * 1.1
    cx = img_header['CRPIX1']
    cy = img_header['CRPIX2']

    sz = data0.shape
    padded_shape = (sz[0] + 20, sz[1] + 20)

    #Padding
    data1 = np.zeros(padded_shape, dtype=np.float32)
    circle_mask_l = np.zeros(padded_shape, dtype=bool)

    data1[10:10 + sz[0], 10:10 + sz[1]] = data0

    y, x = np.ogrid[:sz[0], :sz[1]]
    distance_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    circle_mask = distance_from_center >= Rad
    circle_mask_l[10:10 + sz[0], 10:10 + sz[1]] = circle_mask

    # Replacing negative values with 0
    data1[data1 < 0] = 0

    # Iterations
    for itr in range(iterations):
        print(f"Running iteration {itr} with sigma={sigma_threshold}...")

        # Compute local stats using a 21x21 window
        local_mean = uniform_filter(data1, size=21, mode='constant')
        local_sq_mean = uniform_filter(data1**2, size=21, mode='constant')
        local_std = np.sqrt(local_sq_mean - local_mean**2)
        local_median = median_filter(data1, size=21, mode='constant')

        # Checking for Spikes
        spike_mask = (data1 > local_mean + sigma_threshold * local_std) & (data1 > 300) & (~circle_mask_l)

        spike_count = np.sum(spike_mask)
        print(f"Iteration {itr} - Spikes removed: {spike_count}")

        if dry_run:
            # Dry run: Just return spike count
            if itr == iterations - 1:
                return spike_count
            else:
                continue

        # Replace spikes with median of local pixels
        data1 = np.where(spike_mask, local_median, data1)

        # Extract the corrected core data
        data0 = data1[10:10 + sz[0], 10:10 + sz[1]]
        spk_img = spike_mask[10:10 + sz[0], 10:10 + sz[1]]

        # Save results for this iteration
        np.save(flnm + f'_spike_location_{itr}.npy', spk_img.astype(np.uint8))
        np.save(flnm + f'_spike_rm_data_{itr}.npy', data0)

    print(f"Total runtime: {(time.time() - start)/60:.2f} minutes")


if __name__ == "__main__":
    folder_path = '/run/media/weapon-0/PortableSSD/SUIT_NB07/'
    sigma_threshold = 4   # Change to test different thresholds
    # dry_run = False        True for testing without saving

    fits_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.fits')]

    for fits_file in fits_files:
        try:
            despike_optimized(fits_file, iterations=2, sigma_threshold=sigma_threshold, dry_run=dry_run)
        except Exception as e:
            print(f"Error processing {fits_file}: {e}")

