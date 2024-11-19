import numpy as np
from astropy.io import fits
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import mpld3
import sunpy.map
import glob
from matplotlib.backends.backend_pdf import PdfPages
import pathlib

hdul = fits.open('/Analysis/Research_Projects/SUIT_work/Spike_test/NB3_img.fits') # original image
data0 = hdul[0].data
flnm = (hdul[0].header)['F_NAME'][:-5]
hdul.close()
sz=data0.shape
data1 = np.zeros((sz[0] + 20, sz[1] + 20))
#data1[10:10 + sz[0], 10:10 + sz[1]] = data0
spk_img0=np.load('/Analysis/Research_Projects/SUIT_work/Spike_test/SUT_C24_0081_000099_Lev0.5_2024-02-01T05.08.11.357_4081NB03_spike_location.npy') #spike location
spk_img=np.load('/Analysis/Research_Projects/SUIT_work/Spike_test/SUT_C24_0081_000099_Lev0.5_2024-02-01T05.08.11.357_4081NB03_spike_rm_data.npy') #Spike removed data
data1 = spk_img
#print
nn = 0
sz_spk_img0=spk_img0.shape
# Spike replacement with mean of 8 perimeter intensities

with PdfPages(f'{flnm}_spike_loc.pdf') as pdf:
    fig, axs = plt.subplots(10, 10, figsize=(30, 30))
    for i in range(sz_spk_img0[0]):
        for j in range(sz_spk_img0[1]):
            if spk_img0[i, j] == 1:
                ax = axs[nn // 10, nn % 10]
                ax.imshow(data0[i - 10:i + 11, j - 10:j + 11], cmap='gray')
                nn += 1
                if nn >= 100:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, axs = plt.subplots(10, 10, figsize=(30, 30))
                    nn = 0
    if nn > 0:
        pdf.savefig(fig)
        plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 10))

for i in range(sz[0]):
    for j in range(sz[1]):
        if spk_img0[i, j] == 1:
            rect = Rectangle((j - 5, i - 5), 10, 10, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
ax.set_title('Detected Spikes on Source Image')
ax.imshow(data0, cmap='gray', origin='lower')


with PdfPages(f'{flnm}_spike_loc_Verfify.pdf') as pdf:
    nn=0
    fig, axs = plt.subplots(10, 10, figsize=(30, 30))
    for i in range(sz_spk_img0[0]):
        for j in range(sz_spk_img0[1]):
            if spk_img0[i, j] == 1:
                ax = axs[nn // 10, nn % 10]
                ax.imshow(data1[i - 10:i + 11, j - 10:j + 11], cmap='gray')
                nn += 1
                if nn >= 100:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, axs = plt.subplots(10, 10, figsize=(30, 30))
                    nn = 0
    if nn > 0:
        pdf.savefig(fig)
        plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(data1, cmap='gray', origin='lower')

for i in range(sz[0]):
    for j in range(sz[1]):
        if spk_img0[i, j] == 1:
            rect = Rectangle((j - 5, i - 5), 10, 10, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

ax.set_title('Despiked image')
html_str = mpld3.fig_to_html(fig)

with open(f'Plots/{flnm}_spike_boxes.html', 'w') as f:
    f.write(html_str)

plt.show()