import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import median
from skimage import data, img_as_float
from joblib import Parallel, delayed, cpu_count
from skimage.morphology import disk

def flat_field(image):
    img_average = np.mean(image, axis = 0).astype('uint16')
    img_ave_median = median(img_average, disk(200))
    M = np.mean(image)
    flat_img = np.divide(image, img_ave_median[np.newaxis, :,:])*M
    return flat_img.astype(image.dtype)

def _denoised(flat_img, tp=5):

    im_float = img_as_float(flat_img)

    # Evaluate noise in the image assuming noise is gaussian
    sigma = np.mean(estimate_sigma(im_float[tp,:,:], multichannel=False))
    # Using non local mean to remove the noise
    img_denoise = denoise_nl_means(im_float[tp,:,:],patch_size=7, patch_distance=21,
                                   h=sigma, multichannel=False, fast_mode=True)

    return img_denoise

def parallel_denoise(img_flatten, path='', save=False, nt = 4):
    #nt, ny, nx = image_cell.shape
    cores = cpu_count()
    result = Parallel(n_jobs=cores)(delayed(_denoised)(img_flatten, tp=t) for t in range(nt))
    stack = np.array(result)

    if save == True:
        counter = time.strftime("_%Y%m%d_%H%M")
        path, file_name = os.path.split(path)
        filename, extension = os.path.splitext(file_name)
        filename = re.match(r'.+\d+', filename)
        filename = filename.group()
        if not os.path.exists(filename):
            os.makedirs(filename)
        file = os.path.join(filename + str(counter) + "denoised")
        np.save("{}/{}".format(filename, file), stack)

    return(stack)
