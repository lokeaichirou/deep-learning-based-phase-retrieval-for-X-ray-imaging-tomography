import numpy as np
import glob
import scipy.ndimage
import os

over_sampling = 2
path = '../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/with_noise/1024*1024_with_oversampling=2/projection/npy_format/projection_stack_in_npy_format/*.npy'
os.makedirs('../projection/projection_images_with_size_for_training/phase_contrast_image_stack_of_4_in_npy_format', exist_ok=True)

input =sorted(glob.glob(path))

for i in range(len(input)):
    input_array = np.load(input[i])
    output_array = scipy.ndimage.zoom(input=input_array, zoom=(1, 1/over_sampling, 1/over_sampling), order=1)
    np.save('../projection/projection_images_with_size_for_training/phase_contrast_image_stack_of_4_in_npy_format/{:05d}.npy'.format(i+1),
            output_array.astype(np.float32))
