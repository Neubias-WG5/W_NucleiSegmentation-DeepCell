import sys
import os
import shutil
import skimage.io
from cnn_functions import run_models_on_directory
from model_zoo import sparse_bn_feature_net_61x61 as nuclear_fn


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

def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    tmp_path = sys.argv[3]
    min_size = int(sys.argv[4])
    boundary_weight = float(sys.argv[5])
    prepare_data(in_path, tmp_path)
    predict(tmp_path, out_path)

if __name__ == "__main__":
    main()
