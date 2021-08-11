import numpy as np
from matplotlib import pyplot as plt
import common.graph_setup as graph 
import common.rawdata as rdat
import dnn2dr
# import dnn3dr


if __name__ == '__main__':
    fig, ax = graph.rvalue_graph_setup()

    # texture = '0_00514_04109_00406_02209_01611'
    texture = '0_00611_02907_00507_01507_01311'
    # texture = '0_01813_00808_01012_00213_00405'

    import time
    start = time.time()
    texture_sample_num = 10
    rvalue, std = dnn2dr.estimate_rvalue(texture, texture_sample_num)
    # rvalue, std = dnn3dr.estimate_rvalue(texture, texture_sample_num)
    elapsed = time.time() - start
    print('elapsed time: {} [sec]'.format(elapsed))
    graph.draw_rvalue(ax, [0., 45., 90.], rvalue, std)
    plt.show()
