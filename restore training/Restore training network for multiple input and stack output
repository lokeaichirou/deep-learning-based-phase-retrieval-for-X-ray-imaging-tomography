
# Import code
import msdnet
import glob
import numpy as np

dilations = msdnet.dilations.IncrementDilations(12)

num_of_layers = 30
num_of_inputs_channels = 4
num_of_output_channesl = 2

# Define validation data (not using augmentation)
flsin_validate = sorted(glob.glob('../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/input/input_as_stack/phase_contrast_validate/*.npy'))
flstg_validate = sorted(glob.glob('../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/desired_output/desired_output_stack_of_2/validate/*.npy'))

datsv = []
for i in range(len(flsin_validate)):
    input_array = np.load(flsin_validate[i])
    target_array = np.load(flstg_validate[i])
    d = msdnet.data.ArrayDataPoint(input_array, target_array)
    datsv.append(d)

n, t, val = msdnet.train.restore_training('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/Train network/SGD/dilation_rate=12/layer=100/without_elliptical_cylinder/regr_params_for_stack_distance.checkpoint', msdnet.network.MSDNet, msdnet.train.AdamAlgorithm, msdnet.validate.MSEValidation, datsv, gpu=False)
n.initialize()

flsin_train = sorted(glob.glob('../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/input/input_as_stack/phase_contrast_train/*.npy'))
flstg_train = sorted(glob.glob('../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/desired_output/desired_output_stack_of_2/train/*.npy'))

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

# Use image batches of a single image
bprov = msdnet.data.BatchProvider(dats, 1)

# Log error metrics to console
consolelog = msdnet.loggers.ConsoleLogger()
# Log error metrics to file
filelog = msdnet.loggers.FileLogger('log_regr.txt')
# Log typical, worst, and best images to image files
imagelog = msdnet.loggers.ImageLogger('log_regr', onlyifbetter=True)


msdnet.train.train(n, t, val, bprov, 'regr_params_for_stack_distance.h5',loggers=[consolelog,filelog,imagelog], val_every=len(datsv))

