##
"""
The script implement three steps for refining the segmentation results based
on the fitted SSM.

Step 1: Close holes in regions dictated by the SSM
step 2: Remove unconnected components outside the span of the SSM
Step 3: Remove voxels that are further than the 95% of contour distances

Processed segmentation masks are saved in the
"./Results/stat_shape/seg_postprocess" folder.

Author: Filip Konstantinos <filip.k@ece.upatras.gr>
Last updated: 2021-02-03

"""

import os
import nibabel as nib
import numpy as np
from scipy.ndimage import morphology
from skimage import measure
import collections
import time

## Help functions

def compute_dsc(A, B):
    """
    Compute the DICE score of volumes A and B.
    """
    numerator = np.sum(np.logical_and(A, np.where(B != 0, 1, 0)))
    denominator = (np.sum(A.astype(bool)) + np.sum(B.astype(bool)))
    return numerator * 200.0 / denominator


def getLargestCC(segmentation):
    """
    Find the largest connected component in volume image.
    """
    labels = measure.label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def surfd(input1, input2, sampling=1, connectivity=1):
    """
    Compute the Hausforff distance of two image volumes.

    """
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn)

    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    ha = np.multiply(dta, np.where(Sprime != 0, 1, 0))
    hb = np.multiply(dtb, np.where(S != 0, 1, 0))
    return sds, ha, hb


## Data preparation

seg_labels_path = './Data/OpenKnee/seg/'
ssm_labels_path = './Results/stat_shape/'
gt_labels_path = './Data/OpenKnee/ground_truth/'

save_path = './Data/OpenKnee/seg/postprocess/'

seg_labels = [f for f in os.listdir(seg_labels_path) if f.endswith('.nii.gz')]
ssm_labels = [f for f in os.listdir(ssm_labels_path) if f.endswith('.nii.gz')]
gt_labels = [f for f in os.listdir(gt_labels_path) if f.endswith('.nii.gz')]

labels = {
    1: 'femur',
    2: 'femoral_cart',
    3: 'tibia',
    4: 'tibial_cart_med',
    5: 'tibial_cart_lat',
}

## Main processing

f = open('Results/results.txt', 'w')
for subject in range(len(seg_labels)):

    print(seg_labels[subject])

    # Load SEGMENTAION mask
    seg_img = nib.load(os.path.join(seg_labels_path + seg_labels[subject]))
    seg_mask = seg_img.get_fdata()

    # Load SSM FIT mask
    ssm_img = nib.load(os.path.join(ssm_labels_path + ssm_labels[subject]))
    ssm_mask = ssm_img.get_fdata()

    # Load SSM Ground Truth (GT) mask
    gt_img = nib.load(os.path.join(gt_labels_path + gt_labels[subject]))
    gt_mask = gt_img.get_fdata()

    hdr = seg_img.header
    spacing = hdr.get_zooms()
    affine = seg_img.affine

    # Apply corrective filters
    Volume = np.zeros(hdr['dim'][1:4])
    for label in range(1, len(labels) + 1):
        seg_volume = np.where(seg_mask == label, label, 0)
        ssm_volume = np.where(ssm_mask == label, label, 0)
        gt_volume = np.where(gt_mask == label, label, 0)

        #STEP 1: Close holes in regions dictated by the SSM
        subvolume = np.logical_and(seg_volume, np.where(ssm_volume != 0, 1, 0))
        fill_holes_vol = morphology.binary_fill_holes(subvolume)
        new_volume = np.logical_or(fill_holes_vol,
                                   seg_volume).astype(int) * label

        # STEP 2: Remove unconnected components outside the span of the SSM
        union_volume = np.logical_or(new_volume,
                                     np.where(ssm_volume != 0, 1, 0))
        conncomp = measure.label(union_volume.astype(int))
        largestCC = getLargestCC(union_volume.astype(int))
        unconncomp = np.logical_xor(conncomp, largestCC)
        new_volume = np.logical_xor(new_volume, unconncomp) * label

        # STEP 3: Remove voxels that are further than the 95% of contour
        # distances
        hd, ha, hb = surfd(new_volume, ssm_volume, spacing, 1)
        threshold = np.percentile(hd, 95)
        indexes = np.where(hb < threshold, 1, 0)
        new_volume = np.multiply(new_volume, indexes)

        Volume = np.add(Volume, new_volume)

        DICE_before = compute_dsc(seg_volume, gt_volume)
        DICE_after = compute_dsc(new_volume, gt_volume)

        print('Dice similarity score for %s BEFORE processing is %f and AFTER \
            processing is %f' % (labels[label], DICE_before, DICE_after))

        hd_before, _, _ = surfd(seg_volume, gt_volume, spacing, 1)
        hd_after, _, _ = surfd(new_volume, gt_volume, spacing, 1)

        print(
            'Hausdorff distance for %s BEFORE processing is %f (mm) and AFTER\
            processing is %f (mm)' %
            (labels[label], np.max(hd_before), np.max(hd_after)))

        f.write('%f %f %f %f ' %
                (DICE_before, DICE_after, np.max(hd_before), np.max(hd_after)))

    new_img = nib.Nifti1Image(Volume, affine, hdr)
    nib.save(
        new_img,
        os.path.join(save_path,
                     ssm_labels[subject].replace('SSM_fit',
                                                 'seg_postprocess')))

    f.write('\n')
f.close()
##
