import numpy as np

""" Select DNN-2D or DNN-3D """
import dnn2dr as dnn
# import dnn3dr as dnn


def ani_inplane(tex_info, ave_num=50):
    r_val, std = dnn.estimate_rvalue(tex_info, ave_num)
    r_ave = (r_val[0] + 2. * r_val[1] + r_val[2]) / 4.
    standard_deviation = (r_val[0] - r_ave)**2. + 2. * (r_val[1] - r_ave)**2. + (r_val[2] - r_ave)**2.
    return np.sqrt(standard_deviation / 4.)


def inplane_and_normal_anistropy(tex_info, weight=0.01, ave_num=50):
    r_val, _ = dnn.estimate_rvalue(tex_info, ave_num)
    r_ave = (r_val[0] + 2. * r_val[1] + r_val[2]) / 4.
    standard_deviation = (r_val[0] - r_ave)**2. + 2. * (r_val[1] - r_ave)**2. + (r_val[2] - r_ave)**2.
    return np.sqrt(standard_deviation / 4.) - weight * r_ave
