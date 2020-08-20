import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.ndimage
import glob
import pylab
import tifffile
import skimage.measure

length_scale = 1e-6

distance = [0.002, 0.01, 0.02, 0.045]

energy = 22.5
Lambda = 12.4e-10 / energy  # X-ray wavelength

flsin = sorted(glob.glob('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/input/input_as_stack/phase_contrast_test/*.npy'))
input_array = np.load(flsin[0])

# Create folder for tif format
os.makedirs('../ctf_result/result_for_stack_of_input/tif_format/attenuation', exist_ok=True)
os.makedirs('../ctf_result/result_for_stack_of_input/tif_format/phase', exist_ok=True)

# Create folder for npy format
os.makedirs('../ctf_result/result_for_stack_of_input/npy_format/attenuation', exist_ok=True)
os.makedirs('../ctf_result/result_for_stack_of_input/npy_format/phase', exist_ok=True)

class PhaseRetrievalAlgorithm2D:
    """Reconstruction algorithms should be given as an object I guess. Should
    have a function reconstruct that eats dataSet?"""

    def __init__(self):
        print("we have entered the class PhaseRetrievalAlgorithm2D ")

        self.lengthscale = 1e-6   # downsize it to the real range of distance

        self.nx = 512  # length of Fourier transform of diffraction pattern intensity image
        self.ny = 512  # width of Fourier transform of diffraction pattern intensity image
        self.padding = 2  # padding for 2 times
        self.nfx = self.nx * self.padding  # TODO: Double with DS/flex w pad size
        # width of Fourier transform of diffraction pattern intensity image after padding
        self.nfy = self.ny * self.padding
        # length of Fourier transform of diffraction pattern intensity image after padding

        self.pixel_size = 7*1e-7
        # self.sample_frequency = 1 / (self.pixel_size * self.lengthscale)
        self.sample_frequency = self.lengthscale/self.pixel_size
        # Corresponds to Shannon-nyquist sampling frequency
        self.fx, self.fy = self.FrequencyVariable(self.nfx, self.nfy, self.sample_frequency)
        # 1024*1024 at this moment

        self.ND = len(distance)  # number of distances in experiment set-up

        self.coschirp = [0 for xxx in range(self.ND)] # corresponds to cos "chirp" term for different distances in formula 4.14
        self.sinchirp = [0 for xxx in range(self.ND)] # corresponds to sin "chirp" term for different distances in formula 4.14

        for distances in range(self.ND):
            self.coschirp[distances] = np.cos((np.pi * Fresnel_number[distances]) * (self.fx ** 2) + (np.pi * Fresnel_number[distances]) * (self.fy ** 2))
            # corresponds to cos "chirp" term in formula 4.14
            self.sinchirp[distances] = np.sin((np.pi * Fresnel_number[distances]) * (self.fx ** 2) + (np.pi * Fresnel_number[distances]) * (self.fy ** 2))
            # corresponds to sin "chirp" term in formula 4.14

        self.alpha = 1e-5  # should be given as the exponential only?
        self.alpha_cutoff = 0.5
        self.alpha_cutoff_frequency = self.alpha_cutoff * self.sample_frequency
        # TODO: should be a property (dynamically calculated from alpha_cutoff)
        self.alpha_slope = .5e3

    def FrequencyVariable(self, nfx, nfy, sample_frequency):
        # creation of frequency variable
        # According to numpy fft convention;
        # a[0] should contain the zero frequency term,
        # a[1:n//2] should contain the positive-frequency terms,
        # a[n//2 + 1:] should contain the negative-frequency terms, in increasing
        # order starting from the most negative frequency.
        #f = np.linspace(-sample_frequency / 2, sample_frequency / 2 - sample_frequency / (nx * self.padding), nx * self.padding)  # horizontal representation of frequency domain
        #g = np.linspace(-sample_frequency / 2, sample_frequency / 2 - sample_frequency / (ny * self.padding), ny * self.padding)  # vertical representation of frequency domain

        #f = scipy.fftpack.fftfreq(nfx, d=1/sample_frequency)
        #f_shift = scipy.fftpack.fftshift(f)
        #g = scipy.fftpack.fftfreq(nfy, d=1/sample_frequency)
        #g_shift = scipy.fftpack.fftshift(g)

        x = 0
        x = np.append(x, np.linspace(sample_frequency / nfx, sample_frequency / 2, nfx // 2))
        x = np.append(x, np.linspace(-sample_frequency / 2 + sample_frequency / nfx, -sample_frequency / nfx, nfx//2 -1))

        #                            int(nfx//2 - 1 + (np.ceil(nfx/2) - nfx//2))))  # + (np.ceil(nfx / 2) - nfx // 2)

        y = 0
        y = np.append(y, np.linspace(sample_frequency / nfy, sample_frequency / 2, nfy // 2))
        y = np.append(y, np.linspace(-sample_frequency / 2 + sample_frequency / nfy, -sample_frequency / nfy, nfy//2 - 1))

        #                             int(nfy//2 - 1 + (np.ceil(nfy/2) - nfy//2))))  # + (np.ceil(nfy / 2) - nfy // 2)

        #return np.meshgrid(f_shift, g_shift)

        return np.meshgrid(x, y)


class CTF(PhaseRetrievalAlgorithm2D):
    def __init__(self):
        super().__init__()
        print("we have entered the class CTF")
        self.A = np.zeros((self.nfy, self.nfx))
        # corresponds to term A in formula 4.17, A has same shape as fx and fx
        self.B = self.A.copy()
        # corresponds to term B in formula 4.17, A has same shape as fx and fx
        self.C = self.A.copy()
        # corresponds to term C in formula 4.17, A has same shape as fx and fx

        for distances in range(self.ND):
            self.A += self.sinchirp[distances] * self.coschirp[distances]
            # A is summation of product of sin "chirp" and cos "chirp" for different distances according to (4.14)
            self.B += self.sinchirp[distances] * self.sinchirp[distances]
            # B is summation of product of sin "chirp" and sin "chirp" for different distances according to (4.14)
            self.C += self.coschirp[distances] * self.coschirp[distances]
            # C is summation of product of cos "chirp" and cos "chirp" for different distances according to (4.14)

        self.Delta = self.B * self.C - self.A ** 2
        # corresponds to delta in (4.14)

        # FID = [0 for x in range(ND)] #TODO: test variable

        # TODO: normalisation of CTF factors is not explicit in Zabler
        # but should probably be done?
        # firecp = firecp/length(planes);
        # fireca = fireca/length(planes);
        # fig = pyplot.imshow(self.sinchirp[3])
        # pyplot.colorbar()

    def reconstruct_projection(self, *argv):
        # TODO The interface is not nice. Should probably be with all kwargs?
        fID = [0 for x in range(self.ND)]  # different distances have different Fresenel diffraction pattern, and definitely
        # different corresponding Fourier transform result
        for distances in range(0, self.ND):
            fID[distances] = FId[distances]
        # Generate CTF factors
        # TODO: should possibly be done in constructor
        sin_ctf_factor = np.zeros((self.nfy, self.nfx))
        cos_ctf_factor = sin_ctf_factor.copy()

        for distances in range(self.ND):
            sin_ctf_factor = sin_ctf_factor + self.sinchirp[distances] * fID[distances] # first summation term in (4.17)
            cos_ctf_factor = cos_ctf_factor + self.coschirp[distances] * fID[distances] # second summation term in (4.17)
            # TODO: The removal of the delta is not explicit in the paper
            # but should probably be done
            # s{k}(1,1) -= nf*mf; # remove 1 in real space
            # TODO: verify correct padding

        phase = (self.C * sin_ctf_factor - self.A * cos_ctf_factor) / (2 * self.Delta + self.alpha)
        # formula (4.17)
        attenuation = (self.A * sin_ctf_factor - self.B * cos_ctf_factor) / (2 * self.Delta + self.alpha)

        phase = np.real(np.fft.ifft2(phase))
        # take its real part of complex form phase
        attenuation = np.real(np.fft.ifft2(attenuation))

        # truncate to nx-ny
        phase = phase[self.ny // 2:-self.ny // 2, self.nx // 2:-self.nx // 2]
        # We keep only the core part after the convolution (in Fourier domain, it is
        # multiplication done above, at the same moment in spatial domain, it is convolution).
        attenuation = attenuation[self.ny // 2:-self.ny // 2, self.nx // 2:-self.nx // 2]

        return phase, attenuation


for i in range(100):
    Id = np.load(flsin[i])

    FId = np.fft.fft2(
        np.pad(Id, ((0, 0), ((Id.shape[1]) // 2, (Id.shape[1]) // 2), ((Id.shape[2]) // 2, (Id.shape[2]) // 2)),
               mode='edge'))

    Fresnel_number = [0.0 * x for x in range(len(distance))]
    for x in range(len(distance)):
        Fresnel_number[x] = Lambda * distance[x] / (length_scale ** 2)

    ctf = CTF()  # define a ctf object
    phase_retrieval, attenuation_retrieval = ctf.reconstruct_projection()  # apply reconstruct_projection method, obtain the phase retrieval result
    #fig11 = plt.figure(11)
    #plt.imshow(phase_retrieval, cmap='gray')  # plot result image
    #fig11.suptitle("phase_retrieval for acqui_0 based on 4 distances", fontsize=16)
    #plt.colorbar()
    #pylab.show()

    #fig12 = plt.figure(12)
    #plt.imshow(attenuation_retrieval, cmap='gray')  # plot result image
    #fig12.suptitle("attenuation_retrieval for acqui_0 based on 4 distances", fontsize=16)
    #plt.colorbar()
    #pylab.show()

    # attenuation
    tifffile.imsave(
        '/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/ctf_result/result_for_stack_of_input/tif_format/attenuation/attenuation_regr_{:05d}.tif'.format(
            i + 1), attenuation_retrieval.astype(np.float32))
    np.save(
        '/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/ctf_result/result_for_stack_of_input/npy_format/attenuation/attenuation_regr_{:05d}.npy'.format(
            i + 1), attenuation_retrieval.astype(np.float32))

    # phase
    tifffile.imsave(
        '/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/ctf_result/result_for_stack_of_input/tif_format/phase/phase_regr_{:05d}.tif'.format(
            i + 1), phase_retrieval.astype(np.float32))
    np.save(
        '/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/ctf_result/result_for_stack_of_input/npy_format/phase/phase_regr_{:05d}.npy'.format(
            i + 1), phase_retrieval.astype(np.float32))

    print("finished!")

print("all finished!")
print("see you next time")


ground_truth_path = '/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/desired_output/desired_output_stack_of_2/test'
image_path_attenuation = '/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/ctf_result/result_for_stack_of_input/npy_format/attenuation'
image_path_phase = '/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/ctf_result/result_for_stack_of_input/npy_format/phase'
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
plt.ylim((0, 0.0005))
plt.bar(xx, quality_attenuation)
plt.title('MSE for each attenuation retrieval picture(compared with original attenuation projection)')
plt.show()

plt.figure()
plt.figure()
plt.ylim((0, 50))
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
plt.ylim((0, 50))
plt.bar(xx, quality_attenuation)
plt.title('NMSE for each attenuation retrieval picture(compared with original attenuation projection)')
plt.show()

plt.figure()
plt.figure()
plt.ylim((0, 5))
plt.bar(xx, quality_phase)
plt.title('NMSE for each phase retrieval picture(compared with original phase projection)')
plt.show()
