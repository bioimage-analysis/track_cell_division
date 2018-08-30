import numpy as np
from skimage.filters import threshold_adaptive, sobel, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (reconstruction, disk, watershed, opening, selem, remove_small_objects,
                                binary_dilation, binary_erosion, dilation)
from skimage.segmentation import random_walker, join_segmentations
from skimage.feature import peak_local_max
from skimage.color import label2rgb
from joblib import Parallel, delayed, cpu_count


def _to_array(name):
    n_array = np.empty((1,2))
    n_array[0,:] = np.array([int(name[0]), int(name[1])])
    return n_array

def _markers_rw(image):
    # Making the seeds for random walker
    y, x = image.shape
    local_maxima = peak_local_max(image, threshold_abs=threshold_otsu(image),
                                  footprint=np.ones((50, 50)),
                                  threshold_rel=0.0, exclude_border=True)
    backmiddle = np.where(image == image[int(x/3):int(-y/3),int(x/3):int(-y/3)].min())

    if len(backmiddle[0]) >= 1:
        backmiddle = (backmiddle[0][0:1], backmiddle[1][0:1])


    seeds = np.concatenate((local_maxima, _to_array(backmiddle)), axis= 0)

    markers = np.zeros(image.shape, dtype=np.int)
    markers[seeds[:,0].astype(np.int), seeds[:,1].astype(np.int)] = np.arange(len(seeds[:,0])) + 1
    markers = dilation(markers, disk(10))

    return markers

def _rand_walk(image):
    markers = _markers_rw(image)
    return random_walker(image, markers, beta=25000, mode='cg_mg')

def _water(image):

    elevation_map = sobel(image)

    average = np.mean(image)
    average_above_av = np.mean(image[image>average])
    markers = np.zeros_like(image)
    markers[image < average+(average*10/100)] = 1
    markers[image > average_above_av] = 2

    segmentation = watershed(elevation_map, markers)
    segmentation = (segmentation > 1)
    return segmentation

def _join_seg(segmentation_rw, segmentation_ws):
    rw_ws_join = join_segmentations(segmentation_rw, segmentation_ws)
    rw_ws_join[~segmentation_ws] = 0

    labeled_segmentation = label(rw_ws_join)
    labeled_segmentation = remove_small_objects(labeled_segmentation, 800)

    return labeled_segmentation

def _segmentation(img_row, img_Red_row, result_denoise, tp = 5):

    # segmentation of img_row:
    #----------------------------------------------------------------------

    img_denoise = result_denoise[tp]
    #img_denoise = result_denoise
    #Random walker
    segmentation_rw = _rand_walk(img_denoise)

    # Watershed
    segmentation_ws = _water(img_denoise)

    labeled_image = _join_seg(segmentation_rw, segmentation_ws)

    # Constructing the h-dome for analysis under the pic:
    # calculate hdome to have stronger difference in S/N
    #----------------------------------------------------------------------

    #seed = np.copy(img_denoise)
    #seed[1:-1, 1:-1] = img_denoise.min()
    #mask = img_denoise
    #dilated = reconstruction(seed, mask, method='dilation')
    #hdome = img_denoise - dilated

    props = regionprops(labeled_image, intensity_image = img_denoise)
    #props_hdome = regionprops(labeled_image, intensity_image = hdome)

    # Denoising / processing / segmentation of img_row red:
    #----------------------------------------------------------------------

    Red_binary = threshold_adaptive(img_Red_row[tp,:,:], block_size=37)
    d = selem.diamond(radius=4)
    Red_binary_open = opening(Red_binary, d)
    Red_binary_open = binary_erosion(Red_binary_open, selem = disk(2))
    Red_binary_open = binary_dilation(Red_binary_open, selem = disk(2))

    Red_binary_open_labeled = label(Red_binary_open)
    Red_binary_open_labeled_overlay = label2rgb(Red_binary_open_labeled, image=img_Red_row[tp,:,:])

    props_Red = regionprops(Red_binary_open_labeled, intensity_image = img_denoise)

    # Getting property out
    #----------------------------------------------------------------------

    #Cells property
    #------------------
    cell_coord=[]
    mean_intensity = []
    numb_para = []
    prop_green = []

    #Red property
    #------------------
    prop_red = []

    for cell, prop in enumerate(props):
        prop_green.append(props[cell])

    for cell, prop in enumerate(props):

        # Cell part:
        #------------------

        ycent,xcent = props[cell]['centroid']
        mean_intensity.append(props[cell]['mean_intensity'])
        cell_coord.append((ycent,xcent))


        # Parasite part:
        #------------------
        Para_masked = np.copy(Red_binary_open_labeled)
        Mask = (labeled_image == cell+1)
        Para_masked[~Mask] = 0
        para_ID = list(np.unique(Para_masked[Para_masked != 0]))
        numb_para.append(len(para_ID))

        ### Get the all parasite not just the piece that overlap i
        lst_prop_para_ID = []

        for para, prop in enumerate(props_Red):
            ID = props_Red[para]['label']
            if ID in para_ID:
                lst_prop_para_ID.append(prop)
        prop_red.append((lst_prop_para_ID))


    result1 = np.array(cell_coord)
    result2 = np.array(mean_intensity)
    result3 = np.array(prop_red)
    result4 = np.array(prop_green)

    result = np.concatenate((result1, result2[:, np.newaxis], result3[:, np.newaxis], result4[:, np.newaxis]) , axis=1)

    return result, labeled_image, prop_red, Red_binary_open

def parallel_segmentation(image_cell, image_parasite, image_denoised, nt = 4):
    #nt, ny, nx = image_cell.shape
    cores = cpu_count()
    result = Parallel(n_jobs=cores)(delayed(_segmentation)(image_cell, image_parasite, image_denoised, tp=t) for t in range(nt))
    result, labeled_image, prop_red, Red_binary_open = zip(*result)
    return(result, labeled_image, prop_red, Red_binary_open)
