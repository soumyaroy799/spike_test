import numpy as np
from astropy.io import fits
from tqdm import tqdm
import matplotlib.pyplot as plt
import timeit

def despike(file,iterations):
    hdul = fits.open(file)
    data0 = hdul[0].data
    img_header=hdul[0].header
    hdul.close()
    flnm=(hdul[0].header)['F_NAME'][:-5]
    print('Despiking :',flnm)

    Rad=img_header['R_SUN']*1.1
    cx=img_header['CRPIX1']
    cy=img_header['CRPIX2']

    # Create a circular mask
    y, x = np.ogrid[:data0.shape[0], :data0.shape[1]]
    distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    circle_mask = distance_from_center >= Rad
    #solar_disc_data=data0[circle_mask] # get me the disc data
    #mean_disc = np.mean(solar_disc_data) #mean and standard deviation of only disc elemnts not the full image
    #std_disc = np.std(solar_disc_data)
    

    # Create a larger array and embed the original data in the center
    sz = data0.shape
    data1 = np.zeros((sz[0] + 20, sz[1] + 20))
    circle_mask_l = np.zeros((sz[0] + 20, sz[1] + 20))
    data1[10:10 + sz[0], 10:10 + sz[1]] = data0
    circle_mask_l[10:10 + sz[0], 10:10 + sz[1]] = circle_mask

    # Replace negative values with 0
    data1[data1 < 0] = 0

    # Define neighbor coordinates
    #x3 = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
    #y3 = np.array([1, 1, 1, 0, 0, -1, -1, -1])    
    #peri = np.zeros(8)
    # Thresholds
    #th1 = 400  # Min intensity threshold
    #th2 = 1.2  # Median threshold as in AIA image
    


    # Spike detection
    def fix_spike(data1):
        spk_img = np.zeros(sz)
        spk_img0 = np.zeros((sz[0] + 20, sz[1] + 20))
        count=0
        for i in tqdm(range(10, 10 + sz[0]), desc='spk detection (%)'):
            for j in range(10, 10 + sz[1]):
                int_val = data1[i, j]
                msk_dsk=circle_mask_l[i,j]

                if int_val > 0 and msk_dsk<1 :
                    im = data1[i-10:i+10, j-10:j+10]
                    valid = im[im > 0]
                    
                    if valid.size > 0:
                        av = valid.mean()
                        sigma=np.std(valid)
                        #print(av+3*sigma,av * th2)
                        if int_val > av + 3*sigma and int_val>300 :
                            count+=1
                            #print(count,int_val,':',(av+300),(av*th2))
                            spk_img[i - 10, j - 10] = 1
                            spk_img0[i, j] = 1
                            data1[i, j]=np.median(valid) #replacing spike pixel with median
                            
                            '''
                            #md=np.median(valid)
                            #sigma=valid.std()
                            bg=valid[valid< (av+400 )]
                            bg=bg[bg<(av*th2)]
                            mn = np.median(bg)
                            Data_[i - 10, j - 10] = mn
                            
                            fig1 = plt.figure()
                            ax1, ax2,ax3= fig1.subplots(1, 3)
                            ax1.imshow(data1[i-10:i+10,j-10:j+10])
                            data1[i,j]=10000
                            ax2.imshow(data1[i-10:i+10,j-10:j+10])
                            data0[i - 10, j - 10] = mn
                            ax3.imshow(data0[i-20:i,j-20:j])
                            plt.show()'''
        
        #data2 = np.zeros((sz[0] + 20, sz[1] + 20))
        #data2[10:10 + sz[0], 10:10 + sz[1]] = data0

        '''
        c2 = np.where(spk_img0 == 1)
        data1[c2] = -200
        
        for i in range(sz[0] + 20):
            for j in range(sz[1] + 20):
                if spk_img0[i, j] == 1:
                    im = data1[i-10:i+10, j-10:j+10] #box around the spike point
                    im = im[im != -200]  # Ignore -200 values in the average
                    if len(im) > 0:
                        
                        int_val = np.median(im)
                        #print(data1[i, j],int_val)
                        data1[i, j] = np.median(im)
        '''
                        

        data0 = data1[10:10 + sz[0], 10:10 + sz[1]]
        spk_rm_img = data0
        return spk_img,spk_img0,data0
                        
    spk_img,spk_img0,data0 = fix_spike(data1)
    num_spikes = np.sum(spk_img == 1)
    print('Number of detected spike =', num_spikes)
    np.save(flnm+f'_spike_location.npy', spk_img)
    np.save(flnm+f'_spike_rm_data.npy', data0)

    print('---> ',iterations)

    # Additional iterations

    for itr in range (iterations):
        data1[10:10 + sz[0], 10:10 + sz[1]] = data0
        spk_img,spk_img0,data0 = fix_spike(data1)
        num_spikes = np.sum(spk_img == 1)
        print('Number of detected spike =', num_spikes)

        np.save(flnm+f'_spike_location_{itr}.npy', spk_img)
    

        # Spike replacement with mean of 8 perimeter intensities
        data2 = np.zeros((sz[0] + 20, sz[1] + 20))
        data2[10:10 + sz[0], 10:10 + sz[1]] = data0
        '''
        c2 = np.where(spk_img0 == 1)
        data2[c2] = -200

        for i in range(sz[0] + 20):
            for j in range(sz[1] + 20):
                if spk_img0[i, j] == 1:
                    im = data2[x3 + i, y3 + j]
                    valid = im[im != -200]
                    if valid.size > 0:
                        int_val = np.mean(valid)
                        data2[i, j] = int_val
        '''

        data3 = data2[10:10 + sz[0], 10:10 + sz[1]]
        spk_rm_img = data3
        np.save(flnm+f'_spike_rm_data_{itr}.npy', spk_rm_img)

        # Output plots
        from matplotlib.backends.backend_pdf import PdfPages

        sz_spk_img0 = spk_img0.shape
        nn = 0

        with PdfPages(flnm+f'_spike_loc_{itr}.pdf') as pdf:
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
        
        Dsp_data = np.zeros((sz[0] + 20, sz[1] + 20))
        Dsp_data[10:10 + sz[0], 10:10 + sz[1]] = data3

        with PdfPages(flnm+f'_post_spike_loc_{itr}.pdf') as pdf:
            fig, axs = plt.subplots(10, 10, figsize=(30, 30))
            i=0
            j=0
            nn=0
            for i in range(sz_spk_img0[0]):
                for j in range(sz_spk_img0[1]):
                    if spk_img0[i, j] == 1:
                        ax = axs[nn // 10, nn % 10]
                        ax.imshow(Dsp_data[i - 10:i + 11, j - 10:j + 11], cmap='gray')
                        nn += 1
                        if nn >= 100:
                            pdf.savefig(fig)
                            plt.close(fig)
                            fig, axs = plt.subplots(10, 10, figsize=(30, 30))
                            nn = 0
            if nn > 0:
                pdf.savefig(fig)
                plt.close(fig)


if __name__ == "__main__":
    startTime = timeit.default_timer()
    despike('NB3_img.fits',2) #image, iterations (n+1 is actual iterations)
    stopTime = timeit.default_timer()
    runtime = (stopTime - startTime)/60
    print('Total time consumed: ',runtime)
    


    


