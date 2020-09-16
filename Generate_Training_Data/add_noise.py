# Adding noise using target SNR
import numpy as np


def add_gaussian_noise_with_given_peak_to_peak_snr_for_longest_distance(input_array, target_snr_db=12, mean_noise=0):
    input_array_max = np.amax(input_array)
    print('input_array_max = ', input_array_max)

    input_array_mean = np.mean(input_array)
    print('input_array_mean = ', input_array_mean)
    # Generate an sample of white noise

    max_after_subtracting_mean = input_array_max - input_array_mean
    std_of_noise_for_longest_distance = 1/4.6*max_after_subtracting_mean/(10**(target_snr_db/20))

    noise_volts = np.random.normal(loc=mean_noise, scale=std_of_noise_for_longest_distance, size=(input_array.shape[0], input_array.shape[1]))
    #noise_volts = 1/2*input_array_max / (10**(target_snr_db/20))*np.random.randn(input_array.shape[0], input_array.shape[1])
    print('max noise = ', np.amax(noise_volts))
    print(np.std(noise_volts))

    # Noise added to the original signal
    output_array = input_array + noise_volts
    snr = 10*np.log10(np.sum(output_array**2)/np.sum(noise_volts**2))
    print('snr =  ', snr)

    ppsnr = 20 * np.log10(max_after_subtracting_mean/np.amax(noise_volts))
    print('ppsnr considering mean of image=  ', ppsnr)

    print('调用完成')

    return output_array, noise_volts, std_of_noise_for_longest_distance

def add_gaussian_noise_with_given_peak_to_peak_snr_for_shorter_distance(input_array, mean_noise,std_of_noise_for_longest_distance):

    input_array_max = np.amax(input_array)
    print('input_array_max = ', input_array_max)

    input_array_mean = np.mean(input_array)
    print('input_array_mean = ', input_array_mean )
    # Generate an sample of white noise

    max_after_subtracting_mean = input_array_max - input_array_mean

    noise_volts = np.random.normal(loc=mean_noise, scale=std_of_noise_for_longest_distance,
                                   size=(input_array.shape[0], input_array.shape[1]))
    # noise_volts = 1/2*input_array_max / (10**(target_snr_db/20))*np.random.randn(input_array.shape[0], input_array.shape[1])
    print('max noise = ', np.amax(noise_volts))
    print(np.std(noise_volts))

    # Noise added to the original signal
    output_array = input_array + noise_volts
    snr = 10*np.log10(np.sum(output_array**2) / np.sum(noise_volts**2))
    print('snr =  ', snr)

    ppsnr = 20 * np.log10(max_after_subtracting_mean/np.amax(noise_volts))
    print('ppsnr considering mean of image =  ', ppsnr)

    print('调用完成')

    return output_array

def add_poisson_noise(input_array, target_snr_db=30):

    # Generate an sample of white noise

    noise_volts = np.random.poisson(lam=10**(target_snr_db/10),
                                   size=(input_array.shape[0], input_array.shape[1]))

    # Noise up the original signal
    output_array = input_array + noise_volts

    return output_array
