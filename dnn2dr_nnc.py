# -*- coding: utf-8 -*-
import nnabla as nn
import numpy as np
import common.rawdata as rdat
from tex_util import Texture

""" from nnc_proj.($$project_name_of_nnc) import network """
from nnc_proj.model import network

nn.clear_parameters()
nn.parameter.load_parameters('./nnc_proj/model.nnp')

mvec = rdat.max_r


def estimate_rvalue(tex_info, num):
    x = nn.Variable((num, 1, 128, 128))
    y = network(x, test=True)
    ret = np.zeros([num, 3])
    for cnt in range(num):
        tex = Texture(volume=1000, tex_info=tex_info)
        x.d[cnt, 0] = tex.pole_figure() / 255.
    y.forward()
    r_value_hat = y.d[:, :, 0] * mvec
    ret[:, 0] = r_value_hat[:, 0]**(1. / rdat.order_n['0'])
    ret[:, 1] = r_value_hat[:, 1]**(1. / rdat.order_n['45'])
    ret[:, 2] = r_value_hat[:, 2]**(1. / rdat.order_n['90'])
    std = np.std(ret, axis=0)
    mean = np.average(ret, axis=0)
    return mean, std
