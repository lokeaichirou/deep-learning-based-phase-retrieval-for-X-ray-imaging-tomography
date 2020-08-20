# Import code
import msdnet
import glob
import tifffile
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage

# Make folder for output
os.makedirs('../regression_result/result_for_stack_of_input/tif_format/attenuation', exist_ok=True)
os.makedirs('../regression_result/result_for_stack_of_input/tif_format/phase', exist_ok=True)

os.makedirs('../regression_result/result_for_stack_of_input/npy_format/attenuation', exist_ok=True)
os.makedirs('../regression_result/result_for_stack_of_input/npy_format/phase', exist_ok=True)

# Load network from file
n = msdnet.network.MSDNet.from_file('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/Train network/SGD/dilation_rate=12/layer=30/without_elliptical_cylinder/regr_params_for_stack_distance.h5', gpu=False)

# Process all test images
flsin = sorted(glob.glob('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/input/input_as_stack/phase_contrast_test/*.npy'))

for i in range(len(flsin)):
    input_array = np.load(flsin[i])
    # Create datapoint with only input image
    d = msdnet.data.ArrayDataPoint(input_array)
    # Compute network output
    output = n.forward(d.input)
    # Save network output to file
    # attenuation
    tifffile.imsave('../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/regression_result/SGD/layer=30/dilation = 12/without_elliptical_cylinder/result_for_stack_of_input/tif_format/attenuation/attenuation_regr_{:05d}.tif'.format(i + 1), output[0].astype(np.float32))
    np.save('../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/regression_result/SGD/layer=30/dilation = 12/without_elliptical_cylinder/result_for_stack_of_input/npy_format/attenuation/attenuation_regr_{:05d}.npy'.format(i + 1), output[0].astype(np.float32))

    # phase
    tifffile.imsave('../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/regression_result/SGD/layer=30/dilation = 12/without_elliptical_cylinder/result_for_stack_of_input/tif_format/phase/phase_regr_{:05d}.tif'.format(i + 1), output[1].astype(np.float32))
    np.save('../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/regression_result/SGD/layer=30/dilation = 12/without_elliptical_cylinder/result_for_stack_of_input/npy_format/phase/phase_regr_{:05d}.npy'.format(i + 1), output[1].astype(np.float32))
    print(output.shape)
    # plt.figure(i)
    # plt.imshow(output[0],'gray')



ground_truth_path = '../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/desired_output/desired_output_stack_of_2/test'
image_path_attenuation = '../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/regression_result/SGD/layer=30/dilation = 12/without_elliptical_cylinder/result_for_stack_of_input/npy_format/attenuation'
image_path_phase = '../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/regression_result/SGD/layer=30/dilation = 12/without_elliptical_cylinder/result_for_stack_of_input/npy_format/phase'
# SSIM measurement
quality_attenuation = list()
ssim_sum_attenuation = 0
for i in range(100):
    # attenuation
    ground_truth = np.load('{}/{:05d}.npy'.format(ground_truth_path, i+901))
    ground_truth = ground_truth[0]
    image = np.load('{}/attenuation_regr_{:05d}.npy'.format(image_path_attenuation , i+1))
    ssim_attenuation = skimage.measure.compare_ssim(ground_truth, image)
    print("ssim_attenuation is: ",ssim_attenuation)
    ssim_sum_attenuation +=ssim_attenuation
    quality_attenuation.append(ssim_attenuation)

quality_phase = list()
ssim_sum_phase = 0
for i in range(100):
    # phase
    ground_truth = np.load('{}/{:05d}.npy'.format(ground_truth_path, i+901))
    ground_truth = ground_truth[1]
    image = np.load('{}/phase_regr_{:05d}.npy'.format(image_path_phase, i+1))
    ssim_phase = skimage.measure.compare_ssim(ground_truth, image)
    print("ssim_phase is: ",ssim_phase)
    ssim_sum_phase +=ssim_phase
    quality_phase.append(ssim_phase)

print("average ssim_attenuation is: ",ssim_sum_attenuation/100)
print("average ssim_phase is: ",ssim_sum_phase/100)

xx = [x+1 for x in range(100)]
plt.figure()
plt.ylim((0,1))
plt.bar(xx, quality_attenuation)
plt.title('SSIM for each attenuation retrieval picture(compared with original attenuation projection)')
plt.show()

plt.figure()
plt.figure()
plt.ylim((0,1))
plt.bar(xx, quality_phase)
plt.title('SSIM for each phase retrieval picture(compared with original phase projection)')
plt.show()

# PSNR measurement
quality_attenuation = list()
psnr_sum_attenuation = 0
for i in range(100):
    # attenuation
    ground_truth = np.load('{}/{:05d}.npy'.format(ground_truth_path,i+901))
    ground_truth = ground_truth[0]
    image = np.load('{}/attenuation_regr_{:05d}.npy'.format(image_path_attenuation, i+1))
    psnr_attenuation = skimage.measure.compare_psnr(ground_truth,image)
    print("psnr_attenuation is: ",psnr_attenuation)
    psnr_sum_attenuation  +=psnr_attenuation
    quality_attenuation.append(psnr_attenuation)

quality_phase = list()
psnr_sum_phase = 0
for i in range(100):
    # phase
    ground_truth = np.load('{}/{:05d}.npy'.format(ground_truth_path, i+901))
    ground_truth = ground_truth[1]
    image = np.load('{}/phase_regr_{:05d}.npy'.format(image_path_phase, i+1))
    psnr_phase = skimage.measure.compare_psnr(ground_truth,image,255)
    print("psnr_phase is: ",psnr_phase)
    psnr_sum_phase  +=psnr_phase
    quality_phase.append(psnr_phase)

print("average psnr_attenuation is: ",psnr_sum_attenuation/100)
print("average psnr_phase is: ",psnr_sum_phase/100)

xx = [x+1 for x in range(100)]
plt.figure()
plt.ylim((0, 150))
plt.bar(xx, quality_attenuation)
plt.title('PSNR for each attenuation retrieval picture(compared with original attenuation projection)')
plt.show()

plt.figure()
plt.figure()
plt.ylim((0, 150))
plt.bar(xx, quality_phase)
plt.title('PSNR for each phase retrieval picture(compared with original phase projection)')
plt.show()


# MSE measurement
quality_attenuation = list()
mse_sum_attenuation = 0
for i in range(100):
    # attenuation
    ground_truth = np.load('{}/{:05d}.npy'.format(ground_truth_path, i+901))
    ground_truth = ground_truth[0]
    image = np.load('{}/attenuation_regr_{:05d}.npy'.format(image_path_attenuation, i+1))
    mse = skimage.measure.compare_mse(ground_truth, image)
    print("mse_attenuation is: ",mse)
    mse_sum_attenuation  +=mse
    quality_attenuation.append(mse)

quality_phase = list()
mse_sum_phase = 0
for i in range(100):
    # phase
    ground_truth = np.load('{}/{:05d}.npy'.format(ground_truth_path, i+901))
    ground_truth = ground_truth[1]
    image = np.load('{}/phase_regr_{:05d}.npy'.format(image_path_phase, i+1))
    mse = skimage.measure.compare_mse(ground_truth, image)
    print("mse_phase is: ",mse)
    mse_sum_phase  +=mse
    quality_phase.append(mse)

print("average mse_attenuation is: ",mse_sum_attenuation/100)
print("average mse_phase is: ",mse_sum_phase/100)

xx = [x+1 for x in range(100)]
plt.figure()
plt.ylim((0, 5e-7))
plt.bar(xx, quality_attenuation)
plt.title('MSE for each attenuation retrieval picture(compared with original attenuation projection)')
plt.show()

plt.figure()
plt.figure()
plt.ylim((0, 0.5))
plt.bar(xx, quality_phase)
plt.title('MSE for each phase retrieval picture(compared with original phase projection)')
plt.show()

# NMSE measurement

quality_attenuation = list()
nmse_sum_attenuation = 0
for i in range(100):
    # attenuation
    ground_truth = np.load('{}/{:05d}.npy'.format(ground_truth_path, i+901))
    ground_truth = ground_truth[0]
    image = np.load('{}/attenuation_regr_{:05d}.npy'.format(image_path_attenuation, i+1))

    nmse = np.sqrt(np.sum(np.square(np.abs(ground_truth - image)))/np.sum(np.square(ground_truth)))

    print("nmse_attenuation is: ", nmse)
    nmse_sum_attenuation +=nmse
    quality_attenuation.append(nmse)

quality_phase = list()
nmse_sum_phase = 0
for i in range(100):
    # phase
    ground_truth = np.load('{}/{:05d}.npy'.format(ground_truth_path, i+901))
    ground_truth = ground_truth[1]
    image = np.load('{}/phase_regr_{:05d}.npy'.format(image_path_phase, i+1))

    nmse = np.sqrt(np.sum(np.square(np.abs(ground_truth - image))) / np.sum(np.square(ground_truth)))

    print("nmse_phase is: ", nmse)
    nmse_sum_phase += nmse
    quality_phase.append(nmse)

print("average nmse_attenuation is: ", nmse_sum_attenuation/100)
print("average nmse_phase is: ", nmse_sum_phase/100)

xx = [x+1 for x in range(100)]
plt.figure()
plt.ylim((0, 0.5))
plt.bar(xx, quality_attenuation)
plt.title('NMSE for each attenuation retrieval picture(compared with original attenuation projection)')
plt.show()

plt.figure()
plt.figure()
plt.ylim((0, 0.5))
plt.bar(xx, quality_phase)
plt.title('NMSE for each phase retrieval picture(compared with original phase projection)')
plt.show()
