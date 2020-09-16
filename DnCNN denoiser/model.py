import time
from glob import glob
import numpy as np
import tensorflow as tf
import random
import os
import cv2
import tifffile


def dncnn(input, is_training=True, output_channels=1):  # dncnn is a single residual learning block
    with tf.variable_scope('block1'):
        output = tf.layers.conv2d(input, 64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, dilation_rate=1)
        # First convolutional layer of 64 feature maps generated, kernel size = 3, same padding used, activation
        # function used is relu, relu is normally used for image as input

    for layers in range(2, 19 + 1):
        with tf.variable_scope('block%d' % layers):
            output = tf.layers.conv2d(output, 64, kernel_size=(3, 3), padding='same', name='conv%d' % layers, activation=tf.nn.relu)
            # Hidden sequential convolutional layers of 64 feature maps generated, kernel size = 3, same padding used,
            # No activation function being used is relu

            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
            # Batch normalization is being applied in training
    with tf.variable_scope('block20'):
        output = tf.layers.conv2d(output, output_channels, 3, padding='same', use_bias=False)
        # The final layer is convolutional layer without activation function being applied; no bias added
    return input - output  # inout - output to achieve residual learning


filepaths = glob('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/without_noise/1024*1024_with_oversampling=2/phase_contrast_images_1_without_elliptical_cylinder/nd_zoom_order=1/npy_format/phase_contrast_image_distance_4_in_npy_format/train/*.npy')  # takes all the paths of the png files in the train folder
filepaths = sorted(filepaths)  # Order the list of files
filepaths_noisy = glob('/home/mli/tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/with_noise/1024*1024_with_oversampling=2/phase_contrast_images/Gaussian_noise_PPSNR=12dB/npy_format/phase_contrast_image_distance_4_in_npy_format/train/*.npy')
filepaths_noisy = sorted(filepaths_noisy)
ind = list(range(len(filepaths)))


class denoiser(object):
    def __init__(self, sess, input_c_dim=1, batch_size=128):
        self.sess = sess
        self.input_c_dim = input_c_dim  # input image channel
        # build model
        self.Y_ = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')  # Expected image: clean image
        self.is_training = tf.placeholder(tf.bool, name='is_training')  # set training mode
        self.X = tf.placeholder(tf.float32, [None, None, None, self.input_c_dim])  # Input image: dirty image
        self.Y = dncnn(self.X, is_training=self.is_training)  # Perform dncnn residual learning block
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.Y_ - self.Y)  # compute mean square error loss, i.e. l2
        # norm based loss
        self.lr = tf.placeholder(tf.float32, name='learning_rate')  # learning rate
        self.dataset = dataset(sess)
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.999, name='AdamOptimizer')  # apply ADAM optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   # 通过tf.get_collection来获取moving_mean 和 moving_var的op
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)  # train operation
        init = tf.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def evaluate(self, iter_num, eval_files, noisy_files, summary_writer):
        print("[*] Evaluating...")
        psnr_sum = 0

        for i in range(20):
            #clean_image = cv2.imread(eval_files[i])
            clean_image = np.load(eval_files[i])
            clean_image = clean_image.astype('float32') / 255.0
            clean_image = clean_image[np.newaxis, ...]
            clean_image = np.expand_dims(clean_image, axis=3)
            #noisy = cv2.imread(noisy_files[i])
            noisy = np.load(noisy_files[i])
            noisy = noisy.astype('float32') / 255.0
            noisy = noisy[np.newaxis, ...]
            noisy = np.expand_dims(noisy, axis=3)

            output_clean_image = self.sess.run(
                [self.Y], feed_dict={self.Y_: clean_image,
                                     self.X: noisy,
                                     self.is_training: False})
            psnr = psnr_scaled(clean_image, output_clean_image)
            print("img%d PSNR: %.2f" % (i + 1, psnr))
            psnr_sum += psnr

        avg_psnr = psnr_sum / 20

        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)

    def train(self, eval_files, noisy_files, batch_size, ckpt_dir, epoch, lr, eval_every_epoch=1):

        numBatch = int(len(filepaths)/10)
        # load pretrained model
        load_model_status, global_step = self.load(ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0  # iteration number; iteration is each time we update the parameter
            start_epoch = 0  # epoch number; epoch is each tme we have have go through all the samples for parameter update
            start_step = 0  # step number
            print("[*] Not find pretrained model!")
        # make summary
        tf.summary.scalar('loss', self.loss)  # here the loss refers to the l2 norm based loss applied before
        tf.summary.scalar('lr', self.lr)  # learning rate
        writer = tf.summary.FileWriter('./logs', self.sess.graph)  # generate tf event to record
        merged = tf.summary.merge_all()
        clip_all_weights = tf.get_collection("max_norm")  # get

        print("[*] Start training, with start epoch %d start iter %d : " % (start_epoch, iter_num))
        start_time = time.time()
        self.evaluate(iter_num, eval_files, noisy_files, summary_writer=writer)  # eval_data value range is 0-255
        for epoch in range(start_epoch, epoch):
            batch_noisy = np.zeros((batch_size, 64, 64, 1), dtype='float32')
            batch_images = np.zeros((batch_size, 64, 64, 1), dtype='float32')
            for batch_id in range(start_step, numBatch):
                try:
                    res = self.dataset.get_batch()  # If we get an error retrieving a batch of patches we have to reinitialize the dataset
                except KeyboardInterrupt:
                    raise
                except:
                    self.dataset = dataset(self.sess)  # Dataset re init
                    res = self.dataset.get_batch()
                if batch_id == 0:
                    batch_noisy = np.zeros((batch_size, 64, 64, 1), dtype='float32')
                    batch_images = np.zeros((batch_size, 64, 64, 1), dtype='float32')
                ind1 = range(res.shape[0] // 2)
                ind1 = np.multiply(ind1, 2)
                for i in range(batch_size):
                    random.shuffle(ind1)
                    ind2 = random.randint(0, 8 - 1)
                    batch_noisy[i] = res[ind1[0], ind2]
                    batch_images[i] = res[ind1[0] + 1, ind2]
                #              for i in range(64):
                #                cv2.imshow('raw',batch_images[i])
                #                cv2.imshow('noisy',batch_noisy[i])
                _, loss, summary = self.sess.run([self.train_op, self.loss, merged],
                                                 feed_dict={self.Y_: batch_images, self.X: batch_noisy,
                                                            self.lr: lr[epoch],
                                                            self.is_training: True})
                self.sess.run(clip_all_weights)

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1
                writer.add_summary(summary, iter_num)

            if np.mod(epoch + 1, eval_every_epoch) == 0:  ##Evaluate and save model
                self.evaluate(iter_num, eval_files, noisy_files, summary_writer=writer)
                self.save(iter_num, ckpt_dir)
        print("[*] Training finished.")

    def save(self, iter_num, ckpt_dir, model_name='DnCNN-tensorflow'):
        saver = tf.train.Saver()
        checkpoint_dir = ckpt_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print("[*] Saving model...")
        saver.save(self.sess,
                   os.path.join(checkpoint_dir, model_name),
                   global_step=iter_num)

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            global_step = int(full_path.split('/')[-1].split('-')[-1])
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            return False, 0

    def test(self, eval_files, noisy_files, ckpt_dir, save_dir):  # , temporal
        """Test DnCNN"""
        # init variables
        tf.global_variables_initializer().run()
        assert len(eval_files) != 0, 'No testing data!'
        load_model_status, global_step = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        psnr_sum = 0

        for i in range(len(eval_files)):
            #clean_image = cv2.imread(eval_files[i])  # use opencv to read evaluation image
            clean_image = np.load(eval_files[i])
            clean_image = clean_image.astype('float32')
            clean_image = clean_image[np.newaxis, ...]
            clean_image = np.expand_dims(clean_image, axis=3)

            #noisy = cv2.imread(noisy_files[i])   # use opencv to read noisy image
            noisy = np.load(noisy_files[i])
            noisy = noisy.astype('float32')
            noisy = noisy[np.newaxis, ...]
            noisy = np.expand_dims(noisy, axis=3)

            output_clean_image = self.sess.run(
                [self.Y], feed_dict={self.Y_: clean_image, self.X: noisy,
                                     self.is_training: False})

            out1 = np.asarray(output_clean_image)

            psnr = psnr_scaled(clean_image, out1[0, 0])
            psnr1 = psnr_scaled(clean_image, noisy)

            print("img%d PSNR: %.2f , noisy PSNR: %.2f" % (i + 1, psnr, psnr1))
            psnr_sum += psnr

            #cv2.imwrite('./data/denoised/%04d.png' % (i), out1[0, 0] * 255.0)
            tifffile.imsave('./data/denoised/%04d.tif' % (i), out1[0, 0])

        avg_psnr = psnr_sum / len(eval_files)
        print("--- Test ---- Average PSNR %.2f ---" % avg_psnr)


class dataset(object):
    def __init__(self, sess):
        self.sess = sess
        seed = time.time()
        random.seed(seed)

        random.shuffle(ind)

        filenames = list()   # one noisy and one clean image consecutively
        for i in range(len(filepaths)):
            filenames.append(filepaths_noisy[ind[i]])
            filenames.append(filepaths[ind[i]])

        height = np.load(filenames[1]).shape[0]
        width = np.load(filenames[1]).shape[1]
        self.data = np.ones(shape=[len(filenames), height, width])
        for i in range(len(filenames)):
            self.data[i] = np.load(filenames[i])

        self.data = np.expand_dims(self.data, axis=3)

        # Parameters
        num_patches = 8  # number of patches to extract from each image
        patch_size = 64  # size of the patches
        num_parallel_calls = 1  # number of threads
        batch_size = 32  # size of the batch
        get_patches_fn = lambda image: get_patches(image, num_patches=num_patches, patch_size=patch_size)


        dataset = (
            tf.data.Dataset.from_tensor_slices(tensors=self.data).map(get_patches_fn, num_parallel_calls=num_parallel_calls).batch(batch_size).prefetch(batch_size)
        )
        #.map(im_read, num_parallel_calls=num_parallel_calls)
        iterator = dataset.make_one_shot_iterator()
        self.iter = iterator.get_next()

    def get_batch(self):
        res = self.sess.run(self.iter)
        return res


def im_read(filename):
    """Decode the png image from the filename and convert to [0, 1]."""
    image_string = tf.read_file(filename)
    #image_decoded = tf.image.decode_png(image_string, channels=1)
    #npy_image = cv2.imread(filename, 1)
    #npy_image = tf.convert_to_tensor(filename, dtype=tf.float32)
    #load_npy=np.load(filename)
    #image_decoded=tf.convert_to_tensor(load_npy, dtype=tf.float32)
    image_decoded=tf.expand_dims(image_string, 0)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image


def get_patches(image, num_patches=128, patch_size=64):
    """Get `num_patches` from the image"""
    patches = []
    for i in range(num_patches):
        point1 = random.randint(0, 116)  # 116 comes from the image source size (180) - the patch dimension (64)
        point2 = random.randint(0, 116)
        patch = tf.image.crop_to_bounding_box(image, point1, point2, patch_size, patch_size)
        patches.append(patch)
    patches = tf.stack(patches)
    assert patches.get_shape().dims == [num_patches, patch_size, patch_size, 1]
    return patches


def cal_psnr(im1, im2):  # PSNR function for 0-255 values
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr


def psnr_scaled(im1, im2):  # PSNR function for 0-1 values
    mse = ((im1 - im2) ** 2).mean()
    mse = mse * (255 ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse)
    return psnr

