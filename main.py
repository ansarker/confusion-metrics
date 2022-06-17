import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import argparse

from plotting import plot_images
from total_conf_mat import total_conf_mat


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./data/input', help='path to input images')
    parser.add_argument('--target', type=str, default='./data/target', help='path to target images')
    parser.add_argument('--predict', type=str, default='./data/predict', help='path to predicted images')
    parser.add_argument('--num_classes', type=int, default=6, help='number of classes for confusion matrix')
    parser.add_argument('--results', type=str, default='./results', help='path to saving results directory')
    parser.add_argument('--norm_conf', action='store_true', help='plot normalize confusion matrix')
    parser.add_argument('--figure', action='store_true', help='plot the result with scores')
    parser.add_argument('--write_csv', action='store_true', help='whether save csv or not')
    opt = parser.parse_args()
    print('============ Options ============')
    print(opt)
    print()

    if os.path.exists(opt.results):
        print('Directory already exists. Skipping')
    else:
        print('Directory created')
        os.makedirs(opt.results)

    # Write time in file
    f = open(os.path.join(opt.results, 'time.txt'), 'a')

    # Starting Time
    start_ms = int(round(time.time() * 1000))
    f.write('Start time: ' + str(start_ms) + '\n')

    ind=0
    for img_name in os.listdir(opt.input):
        if ind < 4:
            print(f'Current Image: {img_name.rstrip(".png")}')

            # Read images as PIL Image and convert it to numpy array
            original = np.asarray(Image.open(os.path.join(opt.input, img_name)))
            target = np.asarray(Image.open(os.path.join(opt.target, img_name)))
            predict = np.asarray(Image.open(os.path.join(opt.predict, img_name)))

            plot_images(opt=opt, fname=img_name, original=original, target=target, predict=predict, pos=ind)
        else:
            pass
        ind += 1
    
    # Ending Time
    end_ms = int(round(time.time() * 1000))
    time_in_min = (end_ms - start_ms)/3600
    print(f'Time: {round(time_in_min, 2)} minutes')
    print(f'Avg time: {round(time_in_min/60, 2)} minutes')

    f.write('End time: ' + str(round(end_ms, 2)) + '\n')
    f.write('Time: ' + str(round(time_in_min, 2)) + 'minutes\n')
    f.write('Avg time: ' + str(round(time_in_min/60, 2)) + 'minutes\n')
    f.close()

    # conf_mat_dataframe = pd.read_csv('result/csvs/confusion_matrix.csv')
    # total_conf_mat(conf_mat_dataframe)