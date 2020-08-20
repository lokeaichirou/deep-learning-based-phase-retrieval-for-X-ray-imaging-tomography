import matplotlib.pyplot as plt
import pylab
import numpy as np
import os
import scipy.ndimage
import scipy.fftpack
import tifffile
import add_noise

os.makedirs('../phase_contrast_images/phase_contrast_image_stack_of_4_in_npy_format', exist_ok=True)
os.makedirs('../phase_contrast_images/phase_contrast_image_distance_1', exist_ok=True)
os.makedirs('../phase_contrast_images/phase_contrast_image_distance_2', exist_ok=True)
os.makedirs('../phase_contrast_images/phase_contrast_image_distance_3', exist_ok=True)
os.makedirs('../phase_contrast_images/phase_contrast_image_distance_4', exist_ok=True)

os.makedirs('../phase_contrast_images/phase_contrast_image_distance_1_in_npy_format', exist_ok=True)
os.makedirs('../phase_contrast_images/phase_contrast_image_distance_2_in_npy_format', exist_ok=True)
os.makedirs('../phase_contrast_images/phase_contrast_image_distance_3_in_npy_format', exist_ok=True)
os.makedirs('../phase_contrast_images/phase_contrast_image_distance_4_in_npy_format', exist_ok=True)

for model in range(1, 1001):

    a = np.asarray(tifffile.imread(
        '../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/projection_1_without_elliptical_cylinder/tif_format/attenuation_projection_in_tif_format/{}.tif'.format(str(model).zfill(5))))

    # print(a_00001.shape)
    # fig0 = plt.figure(0)
    # plt.imshow(a_00001, cmap='gray')
    # fig0.suptitle('a_00001', fontsize=16)
    # plt.colorbar()
    # pylab.show()

    # print(a_00001)

    ph = np.asarray(tifffile.imread(
        '../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/projection_1_without_elliptical_cylinder/tif_format/phase_projection_in_tif_format/{}.tif'.format(str(model).zfill(5))))
    # print(ph_00001.shape)
    # fig1 = plt.figure(0)
    # plt.imshow(ph_00001, cmap='gray')
    # fig1.suptitle('ph_00001', fontsize=16)
    # plt.colorbar()
    # pylab.show()

    over_sampling = 2

    # Image part
    length_scale = 1e-6  # downsize it to the real range of distance
    padding = 2  # padding is for the purpose of avoiding circular convolution
    pixel_size = 1  # pixel size
    nx = 1024  # nx = 1448  number of pixels horizontally
    ny = 1024  # ny = 1024  number of pixels vertically

    # Spatial domain
    # ps = (pixel_size / oversampling) * 1e-6  # real range pixel size (resolution;real
    ps = 7.0e-7/over_sampling  # (7.5)*1e-6
    # range distance between two adjunct points)
    x = np.linspace((-ps * nx / 2), ps * (nx / 2 - 1), nx)  # spatial domain
    y = np.linspace((-ps * ny / 2), ps * (ny / 2 - 1), ny)
    xx, yy = np.meshgrid(x, y)

    # Optics part
    energy = 22.5  # Energy of photon, unit is keV
    Lambda = 12.4e-10 / energy  # X-ray wavelength
    # Lambda = 0.5166e-10
    distance = [0.002, 0.01, 0.02, 0.045]  # distance from exit plane of object to diffraction plane. Should be a set of 4 different
    # distances for CTF method according to Dr Max's thesis?

    # Wave part
    wave = np.exp(-a / 2 - 1.0j * ph)  # Refer to Formula (2.40), it is the transmittance function, because we assume
    # the incident wave as uniform flux ; wave at the exit plane of object = Multiplication of original incident wave
    # with transmittance function (decided by complex refractive index including attenuation coefficients and phase shift)
    wave = np.pad(array=wave, pad_width=((ny // 2, ny // 2), (nx // 2, nx // 2)), mode='edge')  # edge padding is done to extend the boundary of 2D
    # wave function representation.Since we want to keep this
    wave = np.fft.fft2(wave)  # Fourier domain representation of wave at exit plane of object, same as u_0,(u node) as in
    # Dr Max's report
    # In numpy,by default, Fourier transform will do with last two axes automatically, don't worry about it
    # wave = np.fft.fftshift(wave)

    # Sampling part and frequency domain for PROPAGATOR
    # Here refer to some knowledge of digital signal processing

    fs = 1 / ps  # corresponds to shannon-nyquist sampling frequency
    f = np.linspace(-fs / 2, fs / 2 - fs / (nx * padding),
                    nx * padding)  # because fs is shannon-nyquist sampling frequency,
    # the maximum detected frequency shoule be half of fs;
    # horizontal representation of frequency domain; 4096 at this moment
    g = np.linspace(-fs / 2, fs / 2 - fs / (ny * padding), ny * padding)  # vertical representation of frequency domain;
    # 4096 at this moment
    ff, gg = np.meshgrid(f, g)  # 4096*4096 at this moment

    # PROPAGATOR part
    # P=np.fft.ifftshift(np.exp(-1j*np.pi*Lambda*(distance)*(ff**2+gg**2)))
    P = [0 for x in range(len(distance))]
    for x in range(len(distance)):
        P[x] = np.fft.ifftshift(np.exp(
            -1j * np.pi * Lambda * distance[x] * (ff ** 2 + gg ** 2)))  # P(f) PROPAGATOR in Fourier domain expression

    # We can consider the free space propagation as a linear space invariant system, knowledge of signal and model could be
    # implemented here for computation purpose

    # Fresnel diffraction intensity part
    Id = [0 for x in range(len(distance))]
    for x in range(len(distance)):
        Id[x] = np.fft.ifft2(wave * P[x])  # complex form wave at certain distance (D) from the object (sample) or
    # let's say wave on diffraction plane can be computed by computed as multiplication of wave and PROPAGATOR in Fourier
    # domain, we recover it to be back to spatial domain, i.e. spatial representation using inverse Fourier transform
    # Id size: 4096*4096 at this moment
    # Because numpy has broadcasting, we don't have to mind the fact that wave is a 3 dimensional tensor while P is a 2
    # dimensional matrix, numpy will copy P for 4 times for the multiplication processing

    Id = np.abs(Id) ** 2  # intensity of wave on diffraction plane is squared modulus of its representation
    # Id = np.abs(Id)  # intensity of wave on diffraction plane is squared modulus of its representation
    Id = Id[::, ny // 2:-ny // 2, nx // 2:-nx // 2]  # We keep only the core part after the convolution (in Fourier domain,it is
    # multiplication done above, at the same moment in spatial domain, it is convolution). This is in line with the wave
    # before the convolution with PROPAGATOR
    # Id size: 2048*2048 at this moment
    Id = scipy.ndimage.zoom(input=Id, zoom=(1, 1/over_sampling, 1/over_sampling), order=1)  # Since the data obtained is oversampling data with oversampling rate = 4,

    # Adding Gaussian noise
    for ii in range(3, -1, -1):
        print('model = ', model, '   distance = ', ii)

        if(ii==3):
            Id[ii], noise_volts_for_longest_distance, std_of_noise_for_longest_distance = add_noise.add_gaussian_noise_with_given_peak_to_peak_snr_for_longest_distance(input_array=Id[ii], target_snr_db=6, mean_noise=0)
        else:
            print('noise_volts_for_longest_distance shape: ', noise_volts_for_longest_distance.shape)
            print('noise_volts_for_longest_distance maximum value: ', np.amax(noise_volts_for_longest_distance))
            Id[ii] = add_noise.add_gaussian_noise_with_given_peak_to_peak_snr_for_shorter_distance(input_array=Id[ii], mean_noise=0, std_of_noise_for_longest_distance=std_of_noise_for_longest_distance)

    # we recover it to be back to 512 as very original
    # Id size: 512*512 at this moment
    print("Id shape", Id.shape)

    # FId = np.fft.fft2(np.pad(Id, ((0, 0), ((Id.shape[1]) // 2, (Id.shape[1]) // 2), ((Id.shape[2]) // 2, (Id.shape[2]) // 2)), 'edge'))
    # compute the 2D Fourier transform of intensity of diffraction pattern, this would be the involved in ctf

    np.save('../phase_contrast_images/phase_contrast_image_stack_of_4_in_npy_format/{:05d}.npy'.format(model), Id.astype(np.float32))
    for x in range(len(distance)):
        tifffile.imsave('../phase_contrast_images/phase_contrast_image_distance_{:0<1d}/{:05d}.tif'.format(x + 1, model), Id[x].astype(np.float32))
        # np.save('../phase_contrast_images/phase_contrast_image_distance_{:0<1d}_in_npy_format/{:05d}.npy'.format(x + 1, model), Id[x].astype(np.float32))
        # show the image of intensity of diffraction pattern
    print("Propagator Congratulation! finished")
