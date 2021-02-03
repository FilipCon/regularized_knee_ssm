##
"""
This script is used to create .stl
files from Nifti volumetric images (.nii.gz)

Author: Filip Konstantinos <filip.k@ece.upatras.gr>
Last updated: 2021-02-03
"""

import os
import nibabel as nib
import numpy as np
import time

from stl import mesh
from skimage import measure

## Input parameters

# list all labelmaps in folder (single multi-label .nii.gz file)
full_path = './Data/OAI_ZIB/original/labelmaps/'
labelmaps = [f for f in os.listdir(full_path) if f.endswith('.nii.gz')]

# target folder to save .stl files
save_path = './Data/OAI_ZIB/original/geometries/stl_original'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# labels of structures
labels = {
    1: 'femur',
    2: 'femoral_cart',
    3: 'tibia',
    4: 'tibial_cart_med',
    5: 'tibial_cart_lat',
}

## Main functions

# For each .nii.gz file in folder
counter = 0  # verbose counter
for mask in labelmaps:

    #load volume image
    img = nib.load(os.path.join(full_path + mask))
    volume = img.get_fdata()

    # pad with zeros the boundary voxels (needed to create watertight meshes)
    for i in [0, img.shape[1] - 1]:
        volume[:, i, :] = 0
    for j in [0, img.shape[2] - 1]:
        volume[:, :, j] = 0

    hdr = img.header  # nifti header
    M = img.affine  # nifti affine transforamtion
    qoffset_x = hdr.structarr['qoffset_x'].tolist()
    qoffset_y = hdr.structarr['qoffset_y'].tolist()
    qoffset_z = hdr.structarr['qoffset_z'].tolist()
    offeset = np.asarray([qoffset_x, qoffset_y, qoffset_z])

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # create meshes for each label in the multi-label volume image
    for label in range(1, len(labels) + 1):

        labelmap = np.isin(volume, label)
        labelmap = labelmap.astype(int)

        # Marching Cubes
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            labelmap,
            level=None,
            spacing=[1, 1, 1],
            gradient_direction='ascent',
            allow_degenerate=False,
            step_size=1)

        verts2 = np.ones((verts.shape[0], 4))
        verts2[:verts.shape[0], :verts.shape[1]] = verts
        verts = np.dot(verts2, M)[:, 0:3] + offeset[None, :]

        # Create the mesh
        shape = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                shape.vectors[i][j] = verts[f[j], :]

        # Write the mesh to target folder
        stl_save_folder = save_path + '/' + labels[label]
        if not os.path.exists(stl_save_folder):
            os.makedirs(stl_save_folder)

        shape.save(stl_save_folder + '/oai_' + mask.replace('.nii.gz', '') + \
                   '_' + labels[label] + '.stl')

    counter = counter + 1
    print(str(counter) + '/' + str(len(labelmaps)))

print('End')
##
