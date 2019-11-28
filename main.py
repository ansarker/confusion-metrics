from PIL import Image
import pandas as pd
import numpy as np
import time
import os

from plotting import plot_images
from total_conf_mat import total_conf_mat


if __name__ == "__main__":

    # org_dir ='Original/'
    # tar_dir ='Target/'
    # pred_dir ='Prediction/'

    # # Write time in file
    # f = open(os.path.join('result', 'time.txt'), 'a')

    # # Starting Time
    # start_ms = int(round(time.time() * 1000))
    # f.write('Start time: ' + str(start_ms) + '\n')

    # ind=0
    # for fname in os.listdir('Original'):
    #     if ind < 4:
    #         print(f'Current Image: {fname.rstrip(".png")}')

    #         # Read images as PIL Image and convert it to numpy array
    #         original = np.asarray(Image.open(os.path.join(org_dir, fname)))
    #         target = np.asarray(Image.open(os.path.join(tar_dir, fname)))
    #         predict = np.asarray(Image.open(os.path.join(pred_dir, fname)))

    #         plot_images(fname=fname, original=original, target=target, predict=predict, pos=ind)
    #     else:
    #         pass
    #     ind += 1
    
    # # Ending Time
    # end_ms = int(round(time.time() * 1000))
    # time_in_min = (end_ms - start_ms)/3600
    # print(f'Time: {time_in_min} minutes')
    # print(f'Avg time: {time_in_min/60} minutes')

    # f.write('End time: ' + str(end_ms) + '\n')
    # f.write('Time: ' + str(time_in_min) + 'minutes\n')
    # f.write('Avg time: ' + str(time_in_min/60) + 'minutes\n')
    # f.close()

    conf_mat_dataframe = pd.read_csv('result/csvs/confusion_matrix.csv')
    total_conf_mat(conf_mat_dataframe)