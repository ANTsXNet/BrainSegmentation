#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et

import os
import sys
import time
import numpy as np
import keras

import ants
import antspynet

args = sys.argv

if len(args) != 5:
    help_message = ("Usage:  python doBrainExtraction.py" +
        " inputFile inputMaskFile outputFilePrefix reorientationTemplate")
    raise AttributeError(help_message)
else:
    input_file_name = args[1]
    input_mask_file_name = args[2]
    output_file_name_prefix = args[3]
    reorient_template_file_name = args[4]

classes = ("Background", "Csf", "GrayMatter", "WhiteMatter",
  "DeepGrayMatter", "BrainStem", "Cerebellum")
number_of_classification_labels = len(classes)
segmentation_labels = list(range(0, number_of_classification_labels))
resampled_image_size = (112, 160, 112)

image_mods = ["T1"]
channel_size = len(image_mods)

print("Reading reorientation template " + reorient_template_file_name)
start_time = time.time()
reorient_template = ants.image_read(reorient_template_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")


unet_model = antspynet.create_unet_model_3d((*resampled_image_size, channel_size),
  number_of_outputs = number_of_classification_labels,
  number_of_layers = 4,
  number_of_filters_at_base_layer = 16,
  dropout_rate = 0.0,
  convolution_kernel_size = (3, 3, 3),
  deconvolution_kernel_size = (2, 2, 2),
  weight_decay = 1e-5 )

print( "Loading weights file" )
start_time = time.time()
weights_file_name = "./brainSegmentationWeights.h5"

if not os.path.exists(weights_file_name):
    weights_file_name = antspynet.get_pretrained_network("brainExtraction", weights_file_name)

unet_model.load_weights(weights_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

start_time_total = time.time()

print( "Reading ", input_file_name )
start_time = time.time()
image = ants.image_read(input_file_name)
mask = ants.image_read( input_mask_file_name)
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print( "Normalizing to template" )
start_time = time.time()
center_of_mass_template = ants.get_center_of_mass(reorient_template)
center_of_mass_image = ants.get_center_of_mass(image)
translation = np.asarray(center_of_mass_image) - np.asarray(center_of_mass_template)
xfrm = ants.create_ants_transform(transform_type="Euler3DTransform",
  center=np.asarray(center_of_mass_template),
  translation=translation)
warped_image = ants.apply_ants_transform_to_image(xfrm, image,
  reorient_template, interpolation='linear')
warped_mask = ants.apply_ants_transform_to_image(xfrm, mask, reorient_template,
  interpolation='linear' )
warped_mask = ants.threshold_image(warped_mask, 0.4999, 1.0001, 1, 0)
warped_mask = ants.iMath(warped_mask, "MD", 3)
warped_cropped_image = ants.crop_image(warped_image, warped_mask, 1)
original_cropped_size = warped_cropped_image.shape
warped_cropped_image = ants.resample_image( warped_cropped_image,
  resampled_image_size, use_voxels = True )
end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

batchX = np.expand_dims(warped_cropped_image.numpy(), axis=0)
batchX = np.expand_dims(batchX, axis=-1)
batchX = (batchX - batchX.mean()) / batchX.std()

print("Prediction and decoding")
start_time = time.time()
predicted_data = unet_model.predict(batchX, verbose=0)

origin_cropped = warped_cropped_image.origin
spacing_cropped = warped_cropped_image.spacing
direction_cropped = warped_cropped_image.direction

probability_images_array = list()
for i in range(number_of_classification_labels):
    probability_images_array.append(
       ants.from_numpy(np.squeeze(predicted_data[0, :, :, :, i]),
         origin=origin_cropped, spacing=spacing_cropped,
         direction=direction_cropped))

end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

print("Renormalize to native space")
start_time = time.time()

zeros = np.zeros(warped_image.shape)
zeros_image = ants.from_numpy( zeros, origin=warped_image.origin,
  spacing=warped_image.spacing, direction=warped_image.direction)

for i in range(number_of_classification_labels):
    probability_image = ants.resample_image(probability_images_array[i],
      original_cropped_size, use_voxels=True, interp_type=0)
    probability_image = ants.decrop_image(probability_image, zeros_image)
    probability_images_array[i] = ants.apply_ants_transform_to_image(
      ants.invert_ants_transform(xfrm), probability_image, image)

end_time = time.time()
elapsed_time = end_time - start_time
print("  (elapsed time: ", elapsed_time, " seconds)")

for i in range(1, number_of_classification_labels):
    print("Writing", classes[i])
    start_time = time.time()
    ants.image_write(probability_images_array[i],
      output_file_name_prefix + classes[i] + "Segmentation.nii.gz")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("  (elapsed time: ", elapsed_time, " seconds)")

probability_images_matrix = ants.image_list_to_matrix(probability_images_array, mask)
segmentation_vector = np.argmax(probability_images_matrix, axis=0)
segmentation_image = ants.make_image(mask, segmentation_vector)
ants.image_write(segmentation_image, output_file_name_prefix + "Segmentation.nii.gz")

end_time_total = time.time()
elapsed_time_total = end_time_total - start_time_total
print("Total elapsed time: ", elapsed_time_total, "seconds")