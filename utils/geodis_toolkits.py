import cv2
import random
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import binary_erosion, binary_dilation
import torch
import torchio as tio
import GeodisTK

def focusregion_index(pred_array):

    """find index for each axis which has the biggest summation value"""

    # pred_array (H,W,D)

    h, w, d = None, None, None

    thres = 0
    for i in range(pred_array.shape[0]):
        if np.sum(pred_array[i]) > thres:
            h = i
            thres = np.sum(pred_array[i])

    thres = 0
    for i in range(pred_array.shape[1]):
        if np.sum(pred_array[:, i]) > thres:
            w = i
            thres = np.sum(pred_array[:, i])

    thres = 0
    for i in range(pred_array.shape[2]):
        if np.sum(pred_array[:, :, i]) > thres:
            d = i
            thres = np.sum(pred_array[:, :, i])

    return h, w, d


def randompoint(seg):

    # random point selection via component analysis

    """Then the user interactions on each mis-segmented
    region are simulated by randomly sampling n pixels in that
    region. Suppose the size of one connected under-segmented
    or over-segmented region is Nm, we set n for that region to
    0 if Nm < 30 and dNm/100 e otherwise based on experience."""

    seg_shape = seg.shape
    seg_array = np.array(seg, dtype=np.uint8)
    focus_h, focus_w, focus_d = focusregion_index(seg_array)
    output = np.zeros(shape=seg_shape)

    if None not in [focus_h, focus_w, focus_d]:
        # h
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_array[focus_h, :, :]
        )
        for i in range(1, len(stats)):
            region_size = stats[i][4]
            if region_size >= 30:
                number_n = int(np.ceil(region_size / 100))
                index_list = np.random.choice(region_size, number_n, replace=False)
                output[
                    focus_h,
                    np.where(labels == i)[0][index_list],
                    np.where(labels == i)[1][index_list],
                ] = 1

        # w
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_array[:, focus_w, :]
        )
        for i in range(1, len(stats)):
            region_size = stats[i][4]
            if region_size >= 30:
                number_n = int(np.ceil(region_size / 100))
                index_list = np.random.choice(region_size, number_n, replace=False)
                output[
                    np.where(labels == i)[0][index_list],
                    focus_w,
                    np.where(labels == i)[1][index_list],
                ] = 1

        # d
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
            seg_array[:, :, focus_d]
        )
        for i in range(1, len(stats)):
            region_size = stats[i][4]
            if region_size >= 30:
                number_n = int(np.ceil(region_size / 100))
                index_list = np.random.choice(region_size, number_n, replace=False)
                output[
                    np.where(labels == i)[0][index_list],
                    np.where(labels == i)[1][index_list],
                    focus_d,
                ] = 1

    return output


def randominteraction(label_array):           
    kernel = np.ones((5, 5, 5), np.uint8)  
    erode = binary_erosion(label_array, structure=kernel, iterations=1)
    dilate = binary_dilation(label_array, structure=kernel, iterations=1)
    overseg = np.where(dilate - label_array == 1, 1, 0)
    underseg = np.where(label_array - erode == 1, 1, 0)
    sb = randompoint(overseg)
    sf = randompoint(underseg)
    return sb, sf

def geodismap(sf, sb, input_np):

    # shape needs to be aligned.
    # original image shape : h, w, d

    # sf: foreground interaction (under segmented)
    # sb: background interaction (over segmented)

    """
    Get 3D geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    """

    I = np.squeeze(input_np, axis=0).transpose(2, 0, 1)
    sf = np.array(sf, dtype=np.uint8).transpose(2, 0, 1)
    sb = np.array(sb, dtype=np.uint8).transpose(2, 0, 1)
    spacing = tio.ScalarImage(tensor=np.expand_dims(I, axis=0)).spacing
    # print(spacing)
    with Pool(2) as p:
        fore_dist_map, back_dist_map = p.starmap(GeodisTK.geodesic3d_raster_scan, 
                                                 [(I, sf, spacing, 1, 2), (I, sb, spacing, 1, 2)])

    if fore_dist_map.all():
        fore_dist_map = I

    if back_dist_map.all():
        back_dist_map = I
    fore_exp_dis = np.exp(-(fore_dist_map ** 2))  #fore_dist_map.transpose(1, 2, 0), back_dist_map.transpose(1, 2, 0)
    back_exp_dis = np.exp(-(back_dist_map ** 2))
    return fore_exp_dis.transpose(1, 2, 0), back_exp_dis.transpose(1, 2, 0)


def get_geodismaps(inputs_np, true_labels_np):
    fore_dist_map_batch = np.empty(inputs_np.shape, dtype=np.float32)
    back_dist_map_batch = np.empty(inputs_np.shape, dtype=np.float32)
    # print(inputs_np.shape, true_labels_np.shape)
    for i, (input_np,  true_label_np) in enumerate(zip(inputs_np, true_labels_np)):

        sb, sf = randominteraction(true_label_np)
        fore_dist_map, back_dist_map = geodismap(sf, sb, input_np)

        fore_dist_map_batch[i] = np.expand_dims(fore_dist_map, axis=0)
        back_dist_map_batch[i] = np.expand_dims(back_dist_map, axis=0)

    return fore_dist_map_batch, back_dist_map_batch
