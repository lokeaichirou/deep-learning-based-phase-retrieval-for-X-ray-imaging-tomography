
# Import code
import msdnet
import glob
import numpy as np

dilations = msdnet.dilations.IncrementDilations(12)

num_of_layers = 30
num_of_inputs_channels = 4
num_of_output_channesl = 2

n = msdnet.network.MSDNet(num_of_layers, dilations, num_of_inputs_channels, num_of_output_channesl, gpu= False)

n.initialize()

flsin_train = sorted(glob.glob('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/with_noise/1024*1024_with_oversampling=2/input/Gaussian_noise_PPSNR=24dB/input_as_stack/phase_contrast_train/*.npy'))
flstg_train = sorted(glob.glob('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/with_noise/1024*1024_with_oversampling=2/desired_output/desired_output_stack_of_2/train/*.npy'))

# Create list of datapoints (i.e. input/target pairs)
dats = []
for i in range(len(flsin_train)):
    input_array = np.load(flsin_train[i])
    target_array = np.load(flstg_train[i])
    # Create datapoint with file names
    d = msdnet.data.ArrayDataPoint(input_array,target_array)
    # Augment data by rotating and flipping
    d_augm = msdnet.data.RotateAndFlipDataPoint(d)
    # Add augmented datapoint to list
    dats.append(d_augm)
# Note: The above can also be achieved using a utility function for such 'simple' cases:
# dats = msdnet.utils.load_simple_data('train/noisy/*.tiff', 'train/noiseless/*.tiff', augment=True)

n.normalizeinout(dats)

# Use image batches of a single image when feeding 1, can set other values
bprov = msdnet.data.BatchProvider(dats, 1)

# Define validation data (not using augmentation)
flsin_validate = sorted(glob.glob('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/with_noise/1024*1024_with_oversampling=2/input/Gaussian_noise_PPSNR=24dB/input_as_stack/phase_contrast_validate/*.npy'))
flstg_validate = sorted(glob.glob('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/with_noise/1024*1024_with_oversampling=2/desired_output/desired_output_stack_of_2/validate/*.npy'))

datsv = []
for i in range(len(flsin_validate)):
    input_array = np.load(flsin_validate[i])
    target_array = np.load(flstg_validate[i])
    d = msdnet.data.ArrayDataPoint(input_array,target_array)
    datsv.append(d)
# Note: The above can also be achieved using a utility function for such 'simple' cases:
# datsv = msdnet.utils.load_simple_data('val/noisy/*.tiff', 'val/noiseless/*.tiff', augment=False)

# Validate with Mean-Squared Error
val = msdnet.validate.MSEValidation(datsv)

# Use ADAM training algorithms
t = msdnet.train.AdamAlgorithm(n)

# Log error metrics to console
consolelog = msdnet.loggers.ConsoleLogger()
# Log error metrics to file
filelog = msdnet.loggers.FileLogger('log_regr.txt')
# Log typical, worst, and best images to image files
imagelog = msdnet.loggers.ImageLogger('log_regr', onlyifbetter=True)


msdnet.train.train(n, t, val, bprov, 'regr_params_for_stack_distance.h5',loggers=[consolelog,filelog,imagelog], val_every=len(datsv))

