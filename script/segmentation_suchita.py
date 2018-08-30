def flat_field(image):
    img_average = np.mean(image, axis = 0).astype('uint16')
    img_ave_median = median(img_average, disk(200))
    M = np.mean(image)
    flat_img = np.divide(image, img_ave_median[np.newaxis, :,:])*M
    return flat_img.astype(image.dtype)


def denoise(image, denoise_type = 'nlm'):

    denoise_types = ['nlm', 'bilateral', 'tvd']

    if denoise_type not in denoise_types:
        raise ValueError("Invalid denoise type. Expected one of: %s" % denoise_types)

    # Anscombe tranform considering only Poisson noise
    im_Ansc = 2*np.sqrt(image+3/8)
    # Image, int to float [0,1] for denoising
    im_Ansc_float = (im_Ansc-im_Ansc.min())/(im_Ansc.max()-im_Ansc.min())
    sigma = estimate_sigma(im_Ansc_float)

    #denoise using non local mean (very slow but very good when S/N very low)
    if denoise_type == 'nlm':
        im_denoised = denoise_nl_means(im_Ansc_float,patch_size=7, patch_distance=21, h=sigma, multichannel=False)

    #denoise using denoise image using bilateral filter (very fast)
    elif denoise_type == 'bilateral':
        im_denoised = restoration.denoise_bilateral(im_Ansc_float, multichannel=False)

    #denoise using total-variation denoising
    else:
        im_denoised = restoration.denoise_tv_chambolle(im_Ansc_float, weight=0.01)

    #Scale back for inverse Anscombe
    im_denoised = im_denoised*(image.max()-image.min())+image.min()
    # Close approximation os inverse anscomb transform
    im_denoised = (1/4*np.square(im_denoised) + 1/4*np.sqrt(3/2*np.power(im_denoised, -1))
                  - 11/8*np.power(im_denoised, -2)     + 5/8*np.sqrt(3/2*np.power(im_denoised, -3))
                  - 1/8)

    # Image, int to float [0,1] for denoising
    im_denoised = (im_denoised-im_denoised.min())/(im_denoised.max()-im_denoised.min())

    return im_denoised


def to_array(name):
    n_array = np.empty((1,2))
    n_array[0,:] = np.array([int(name[0]), int(name[1])])
    return n_array

def markers_rw(image):
    # Making the seeds for random walker
    local_maxima = peak_local_max(image, threshold_abs=threshold_otsu(image),
                                  footprint=np.ones((50, 50)),
                                  threshold_rel=0.0, exclude_border=True)
    backmiddle = np.where(image == image[int(x/3):int(-y/3),int(x/3):int(-y/3)].min())

    if len(backmiddle[0]) >= 1:
        backmiddle = (backmiddle[0][0:1], backmiddle[1][0:1])


    seeds = np.concatenate((local_maxima, to_array(backmiddle)), axis= 0)

    markers = np.zeros(image.shape, dtype=np.int)
    markers[seeds[:,0].astype(np.int), seeds[:,1].astype(np.int)] = np.arange(len(seeds[:,0])) + 1
    markers = dilation(markers, disk(10))

    return markers


def rand_walk(image):
    markers = markers_rw(image)
    return random_walker(image, markers, beta=25000, mode='cg_mg')

def water(image):

    elevation_map = sobel(image)

    average = np.mean(image)
    average_above_av = np.mean(image[image>average])
    markers = np.zeros_like(image)
    markers[image < average+(average*10/100)] = 1
    markers[image > average_above_av] = 2

    segmentation = watershed(elevation_map, markers)
    segmentation = (segmentation > 1)
    return segmentation

def join_seg(segmentation_rw, segmentation_ws):
    rw_ws_join = join_segmentations(segmentation_rw, segmentation_ws)
    rw_ws_join[~segmentation_ws] = 0

    labeled_segmentation = label(rw_ws_join)
    labeled_segmentation = remove_small_objects(labeled_segmentation, 800)

    return labeled_segmentation
