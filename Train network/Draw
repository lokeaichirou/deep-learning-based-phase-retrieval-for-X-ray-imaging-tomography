import re
import numpy as np
import matplotlib.pyplot as plt


def list_generation(input_file):
    reader = open(input_file, 'r')
    output_list = []
    while True:
        line = reader.readline()
        if len(line) == 0:
            break
        strlist = line.split('Current error: ')

        for item in strlist[1:]:
            try:
                list_link = item.split(',')[0]
                list_link = float(list_link)
                output_list.append(list_link)
            except:
                pass

    return output_list


input_file_1 = '../tomograms/pycharm_demos/single_density/without_Gaussian_shapes/normal_test/with_noise/1024*1024_with_oversampling=2/Train network/SGD/layer=30/dilation_width=12/without_elliptical_cylinder/Gaussian=24dB_ppsnr/log_regr.txt'

d1 = np.asarray(list_generation(input_file_1))

print(d1)

x = np.linspace(1, len(d1), num=len(d1))
plt.plot(x, d1, label='distance = 0.002m')

plt.xlabel('training epochs')
plt.ylabel('MSE error')
plt.title('Loss variation during training for number of layer = 30')
# plt.legend()
plt.show()
