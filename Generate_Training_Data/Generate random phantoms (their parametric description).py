import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import random
from copy import deepcopy
import numpy as np

from tomophantom import TomoP3D

# Define the path and name of the `.dat` file that will contain the phantoms model description. If the file exists, the new models will be appended.python

path_1 = '../tomograms/pycharm_demos/single_density/Generate_Training_Data/Phantom3DLibrary_For_att_without_Gaussian.dat'
path_2 = '../tomograms/pycharm_demos/single_density/Generate_Training_Data/Phantom3DLibrary_For_phase_without_Gaussian.dat'

# Generate the models:

number_of_phantoms = 1000

# components = ['gaussian','paraboloid', 'ellipsoid', 'cone', 'cuboid', 'elliptical_cylinder']
# the unused components are not uniform in density.

components = ['paraboloid', 'ellipsoid']
mu = [0.328]
delta = [481]

for n in range(1, number_of_phantoms + 1):
    num_of_components = random.randint(1, 5)

    objects_1 = []
    objects_2 = []
    component_subset = random.choices(components, k=random.randint(1, 3))
    for i in range(num_of_components):
        obj_1 = random.choice(component_subset)
        obj_2 = deepcopy(obj_1)
        upper_lim_half_width = random.choice([0.5])

        c0 = 0
        m0 = mu[c0]
        d0 = delta[c0]

        x0 = round(random.uniform(-0.5, 0.5), 2)
        y0 = round(random.uniform(-0.5, 0.5), 2)
        z0 = round(random.uniform(-0.5, 0.5), 2)
        a = round(random.uniform(0.01, upper_lim_half_width), 3)
        b = round(random.uniform(0.01, upper_lim_half_width), 3)
        c = round(random.uniform(0.01, upper_lim_half_width), 3)
        alpha = round(random.uniform(-180, 180), 2)
        beta = round(random.uniform(-180, 180), 2)
        gamma = round(random.uniform(-180, 180), 2)

        obj_1 += ' {c0} {x0} {y0} {x0} {a} {b} {c} {alpha} {beta} {gamma}'.format(c0=m0,
                                                                                  # round(random.uniform(0.1,1),2),
                                                                                  x0=x0,
                                                                                  y0=y0,
                                                                                  z0=z0,
                                                                                  a=a,
                                                                                  b=b,
                                                                                  c=c,
                                                                                  alpha=alpha,
                                                                                  beta=beta,
                                                                                  gamma=gamma)

        obj_2 += ' {c0} {x0} {y0} {x0} {a} {b} {c} {alpha} {beta} {gamma}'.format(c0=d0,
                                                                                  # round(random.uniform(0.1,1),2),
                                                                                  x0=x0,
                                                                                  y0=y0,
                                                                                  z0=z0,
                                                                                  a=a,
                                                                                  b=b,
                                                                                  c=c,
                                                                                  alpha=alpha,
                                                                                  beta=beta,
                                                                                  gamma=gamma)

        objects_1.append(obj_1)
        objects_2.append(obj_2)

    phantom_string_1 = '''#----------------------------------------------------
# random phantom number {num}
Model : {num};
Components : {comp};
TimeSteps : 1;
'''.format(num=n, comp=num_of_components)

    phantom_string_2 = '''#----------------------------------------------------
# random phantom number {num}
Model : {num};
Components : {comp};
TimeSteps : 1;
'''.format(num=n, comp=num_of_components)

    for obj in objects_1:
        phantom_string_1 += 'Object : ' + obj + '\n'

    with open(path_1, 'a') as file:
        file.write(phantom_string_1)

    for obj in objects_2:
        phantom_string_2 += 'Object : ' + obj + '\n'

    with open(path_2, 'a') as file:
        file.write(phantom_string_2)


