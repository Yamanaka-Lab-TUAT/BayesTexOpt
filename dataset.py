# -*- coding: utf-8 -*-
import os
import csv
import random
import sys
from os import makedirs
from os.path import exists
from PIL import Image
import numpy as np
import common.rawdata as rdat
from common.rawdata import Datatype
from tex_util import Texture

save_dir = './label/'  # Path where to save the label
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

cache_dir = rdat.rvalue_dir + 'cache/'
if not exists(cache_dir):
    makedirs(cache_dir)


def std_print(out):
    sys.stdout.write(out)
    sys.stdout.flush()


def get_texture(tex_info, data_type):
    dataPath = rdat.traindata_dir
    if data_type == Datatype.valid:
        dataPath = rdat.evaldata_dir
    elif data_type == Datatype.test:
        dataPath = rdat.testdata_dir
    data = np.genfromtxt(
        dataPath + 'texture/' + tex_info + '.txt', delimiter='')
    phi1 = data[:, 2]
    phi = data[:, 3]
    phi2 = data[:, 4]
    return np.array([phi1, phi, phi2]).T


def create_voxel_input():
    if not os.path.exists(rdat.voxel_dir):
        os.makedirs(rdat.voxel_dir)
    for lt in rdat.train_listdir:
        tex_info = lt.rstrip('.txt')
        raw_data = get_texture(tex_info, Datatype.train)
        tex_data = Texture(texture=raw_data)
        try:
            vox = tex_data.voxel()
        except IndexError as e:
            print(e)
            print(tex_info + '\n')
            return
        np.save(rdat.voxel_dir + 'data_' + tex_info, vox)
    for lt in rdat.eval_listdir:
        tex_info = lt.rstrip('.txt')
        raw_data = get_texture(tex_info, Datatype.valid)
        tex_data = Texture(texture=raw_data)
        vox = tex_data.voxel()
        np.save(rdat.voxel_dir + 'data_' + tex_info, vox)


def create_image_input():
    if not os.path.exists(rdat.image_dir):
        os.makedirs(rdat.image_dir)
    for lt in rdat.train_listdir:
        tex_info = lt.rstrip('.txt')
        raw_data = get_texture(tex_info, Datatype.train)
        tex_data = Texture(texture=raw_data)
        tex_data.savePoleFigure(rdat.image_dir + 'data_' + tex_info + '.png')
    for lt in rdat.eval_listdir:
        tex_info = lt.rstrip('.txt')
        raw_data = get_texture(tex_info, Datatype.valid)
        tex_data = Texture(texture=raw_data)
        tex_data.savePoleFigure(rdat.image_dir + 'data_' + tex_info + '.png')


def save_rvalue_dataset(file_name, data):
    with open(file_name, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['x:input', 'r:r-value'])
        writer.writerows(data)


def create_teacher_data():
    save_rvalue_dir = rdat.rvalue_dir
    if not os.path.exists(save_rvalue_dir):
        os.makedirs(save_rvalue_dir)
    angles = list(rdat.angles.keys())

    # --- create training data of r-value
    r_data = np.empty([len(rdat.train_listdir), len(angles)])
    for cnt, lt in enumerate(rdat.train_listdir):
        tex_info = lt.rstrip('.txt')
        r_value = np.array([rdat.get_rvalue(angle, tex_info, Datatype.train)
                            for angle in angles])
        if np.any(r_value < 0.):
            print(tex_info, r_value)
            r_value[np.where(r_value < 0.)] = 0.
        r_data[cnt, :] = r_value

    # --- normalize using a mapping f = r_value**n
    for a, angle in enumerate(angles):
        r_data[:, a] = r_data[:, a]**rdat.order_n[angle]
    r_max = np.max(r_data, axis=0)
    np.savetxt(rdat.rvalue_dir + 'max_r.csv', r_max, delimiter=',')
    r_data /= r_max

    # --- saving training data
    for cnt, lt in enumerate(rdat.train_listdir):
        tex_info = lt.rstrip('.txt')
        np.savetxt(save_rvalue_dir + 'data_' + tex_info + '.csv',
                   r_data[cnt], delimiter=',')
    print('finish r-value train')

    # --- saving validation data
    for lt in rdat.eval_listdir:
        tex_info = lt.rstrip('.txt')
        r_value = np.array([rdat.get_rvalue(angle, tex_info, Datatype.valid)
                            for angle in angles])
        if np.any(r_value < 0.):
            r_value[np.where(r_value < 0.)] = 0.
            print(tex_info)
        r_value = np.array([r_value[a]**rdat.order_n[angle] for a, angle in enumerate(angles)])
        r_value /= r_max
        np.savetxt(save_rvalue_dir + 'data_' + tex_info + '.csv',
                   r_value, delimiter=',')
    print('finish r-value validation')


def createdataset():
    dnn2d_train_rvalue_label = []
    dnn2d_valid_rvalue_label = []
    dnn3d_train_rvalue_label = []
    dnn3d_valid_rvalue_label = []
    for lt in rdat.train_listdir:
        tex_info = lt.rstrip('.txt')
        dnn2d_train_rvalue_label.append([rdat.image_dir + 'data_' + tex_info + '.png',
                                         rdat.rvalue_dir + 'data_' + tex_info + '.csv'])
        dnn3d_train_rvalue_label.append([rdat.voxel_dir + 'data_' + tex_info + '.npy',
                                         rdat.rvalue_dir + 'data_' + tex_info + '.csv'])
    for lt in rdat.eval_listdir:
        tex_info = lt.rstrip('.txt')
        dnn2d_valid_rvalue_label.append([rdat.image_dir + 'data_' + tex_info + '.png',
                                         rdat.rvalue_dir + 'data_' + tex_info + '.csv'])
        dnn3d_valid_rvalue_label.append([rdat.voxel_dir + 'data_' + tex_info + '.npy',
                                         rdat.rvalue_dir + 'data_' + tex_info + '.csv'])
    save_rvalue_dataset(save_dir + 'rvalue_train_dnn2d.csv', dnn2d_train_rvalue_label)
    save_rvalue_dataset(save_dir + 'rvalue_eval_dnn2d.csv', dnn2d_valid_rvalue_label)

    save_rvalue_dataset(save_dir + 'rvalue_train_dnn3d.csv', dnn3d_train_rvalue_label)
    save_rvalue_dataset(save_dir + 'rvalue_eval_dnn3d.csv', dnn3d_valid_rvalue_label)

    print('r-value Train    : ' + str(len(dnn3d_train_rvalue_label)) + ' data')
    print('r-value Evaluate : ' + str(len(dnn3d_valid_rvalue_label)) + ' data')


def load_data(input_data_type=1, shuffle=True, use_cache=True):
    nn_type = 'dnn2d'
    if input_data_type != 0:
        nn_type = 'dnn3d'
    if use_cache:
        try:
            x_train = np.load(cache_dir + 'x_train.npy')
            y_train = np.load(cache_dir + 'y_train.npy')
            x_test = np.load(cache_dir + 'x_test.npy')
            y_test = np.load(cache_dir + 'y_test.npy')
            print('load from cache!')
            return (x_train, y_train), (x_test, y_test)
        except IOError as e:
            print(e)

    # -- load training dataset info files
    train_data = []
    valid_data = []
    with open(save_dir + 'rvalue_train_{}.csv'.format(nn_type)) as f:
        reader = csv.reader(f)
        train_data = [row for row in reader]
        train_data = train_data[1:]
    with open(save_dir + 'rvalue_eval_{}.csv'.format(nn_type)) as f:
        reader = csv.reader(f)
        valid_data = [row for row in reader]
        valid_data = valid_data[1:]
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(valid_data)

    # --- training dataset
    x_train = []
    y_train = []
    percentage = 0
    data_num = len(train_data)
    for i, tr_dat in enumerate(train_data):
        x_train.append(np.load(tr_dat[0]).astype(np.float16) if input_data_type != 0
                       else np.array(Image.open(tr_dat[0])).reshape((128, 128, 1)) / 255.)
        y_train.append(np.loadtxt(tr_dat[1], delimiter=',', dtype='float32'))
        tmp = int((i / data_num) * 100)
        if percentage != tmp:
            std_print('\rtraining data : {} %'.format(percentage + 1))
        percentage = tmp
    x_train = np.array(x_train)
    np.save(cache_dir + 'x_train', x_train)
    y_train = np.array(y_train)
    np.save(cache_dir + 'y_train', y_train)
    del train_data
    std_print('\rtraining data : 100 %\n')
    print('train data {} loaded\n'.format(data_num))

    # --- validation dataset
    x_test = []
    y_test = []
    percentage = 0
    data_num = len(valid_data)
    for i, ev_dat in enumerate(valid_data):
        x_test.append(np.load(ev_dat[0]).astype(np.float16) if input_data_type != 0
                      else np.array(Image.open(ev_dat[0])).reshape((128, 128, 1)) / 255.)
        y_test.append(np.loadtxt(ev_dat[1], delimiter=',', dtype='float32'))
        tmp = int((i / data_num) * 100)
        if percentage != tmp:
            std_print('\rvalidation data : {} %'.format(percentage + 1))
        percentage = tmp
    x_test = np.array(x_test)
    np.save(cache_dir + 'x_test', x_test)
    y_test = np.array(y_test)
    np.save(cache_dir + 'y_test', y_test)
    del valid_data
    std_print('\rvalidation data : 100 %\n')
    print('eval data {} loaded\n'.format(data_num))
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    create_teacher_data()  # Output data of the both DNNs

    create_voxel_input()  # Input data of DNN-3D
    create_image_input()  # Input data of DNN-2D
    print('finish creating training data')

    createdataset()
