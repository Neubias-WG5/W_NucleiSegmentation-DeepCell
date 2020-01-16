import sys
import os
import shutil
import skimage.io
import skimage.morphology
import numpy as np
from cnn_functions import run_models_on_directory
from model_zoo import sparse_bn_feature_net_61x61 as nuclear_fn

"""
The code in this script is mostly copied and restructured
from jccaicedo/deepcell
"""

def prepare_data(in_path, tmp_path):
    image_list = os.listdir(in_path)
    for im_name in image_list:
        img = skimage.io.imread(os.path.join(in_path, im_name))
        outpath = os.path.join(tmp_path, im_name.split('.')[0])
        os.mkdir(outpath)
        skimage.io.imsave(os.path.join(outpath, "nuclear.png"), img)

def predict(in_path, out_path):
    win_nuclear = 30
    nuclear_channel_names = ['nuclear']
    trained_network_dir = "/app/DeepCell/trained_networks/Nuclear"
    prefix = "2016-07-12_nuclei_all_61x61_bn_feature_net_61x61_"
    
    image_list = os.listdir(in_path)
    models_to_use = 5
    list_of_weights = [os.path.join(trained_network_dir, prefix + str(i) + ".h5") for i in range(models_to_use)]

    for i in range(len(image_list)):
        data_location = os.path.join(in_path, image_list[i])
        nuclear_location = os.path.join(in_path, image_list[i])
        im = skimage.io.imread(os.path.join(in_path, image_list[i], "nuclear.png"))
        nuclear_predictions = run_models_on_directory(data_location,
                                                      nuclear_channel_names,
                                                      nuclear_location,
                                                      model_fn = nuclear_fn,
                                                      list_of_weights = list_of_weights,
                                                      image_size_x = im.shape[1],
                                                      image_size_y = im.shape[0],
                                                      win_x = win_nuclear,
                                                      win_y = win_nuclear,
                                                      split = False)

def postprocess(tmp_path, out_path, min_size, boundary_weight):
    predictions = "feature_2_frame_0.tif feature_1_frame_0.tif feature_0_frame_0.tif".split()
    image_list = os.listdir(tmp_path)
    for iname in image_list:
        nuclear_location = os.path.join(tmp_path, iname)
        probmap = to_rgb(predictions, nuclear_location)
        pred = probmap_to_pred(probmap, boundary_weight)
        labels = pred_to_label(pred, min_size).astype(np.uint16)
        skimage.io.imsave(os.path.join(out_path,iname+'.tif'), labels)

def to_rgb(names, nuclear_location):
    pred = []
    for im in names:
        img = skimage.io.imread(nuclear_location + "/" + im)
        pred.append(img.reshape(img.shape + (1,)))
    pred = np.concatenate(pred, -1)
    return pred

def probmap_to_pred(probmap, boundary_boost_factor=1):
    # we need to boost the boundary class to make it more visible
    # this shrinks the cells a little bit but avoids undersegmentation
    pred = np.argmax(probmap * [1, 1, boundary_boost_factor], -1)
    return pred

def pred_to_label(pred, cell_min_size, cell_label=1):
    cell = (pred == cell_label)
    # fix cells
    cell = skimage.morphology.remove_small_holes(cell, min_size=cell_min_size)
    cell = skimage.morphology.remove_small_objects(cell, min_size=cell_min_size)
    # label cells only
    [label, num] = skimage.morphology.label(cell, return_num=True)
    return label

def main():
    in_path = sys.argv[1]
    tmp_path = sys.argv[2]
    out_path = sys.argv[3]
    min_size = int(sys.argv[4])
    boundary_weight = float(sys.argv[5])
    prepare_data(in_path, tmp_path)
    predict(tmp_path, out_path)
    postprocess(tmp_path, out_path, min_size, boundary_weight)

if __name__ == "__main__":
    main()
