# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import common.rawdata as rdat
from tex_util.tex import Texture

""" from tf_models.($dnn_model_you_developed$).model import Network """
from tf_models.dnn3d.model import NetWork

tf.config.set_visible_devices([], 'GPU')

mvec = rdat.max_r

model = NetWork().loadModel().model


def estimate_rvalue(tex_info, num):
    ret = np.zeros([num, 3])  # "3" means r_0, r_45 and r_90
    x = np.empty((num, 32, 16, 32, 1))
    for cnt in range(num):
        tex = Texture(volume=1000, tex_info=tex_info)
        x[cnt] = tex.voxel()
    y = model.predict(x)
    r_value_hat = y * mvec
    ret[:, 0] = r_value_hat[:, 0]**(1. / rdat.order_n['0'])
    ret[:, 1] = r_value_hat[:, 1]**(1. / rdat.order_n['45'])
    ret[:, 2] = r_value_hat[:, 2]**(1. / rdat.order_n['90'])
    std = np.std(ret, axis=0)
    mean = np.average(ret, axis=0)
    return mean, std
