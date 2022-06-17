import numpy as np 
import pandas as pd


def conf_mat_csv(opt, dataframe, percent, img_title, pos):
    add_to_first_col = []
    csv_header = []
    csv_rows = []
    data_frame_np_ = np.zeros((7,7), dtype=np.object)
    
    dataframe.reset_index(level=0, inplace=True)
    [add_to_first_col.append(round(p,2)) for p in percent]
    data_frame_np = dataframe.to_numpy()
    data_frame_np_[0,:] = dataframe.columns.values
    data_frame_np_[1:] = data_frame_np

    """
        Create header
    """
    # Image title header
    csv_header.append('Title')

    # Area header
    for i in range(data_frame_np_.shape[0]):
        if data_frame_np_[0,i] == 'index':
            pass
        else:
            csv_header.append('Area-' + data_frame_np_[0, i])

    # Confusion matrix header
    for i in range(data_frame_np_.shape[0]):
        for j in range(data_frame_np_.shape[1]):
            if data_frame_np_[0,j] == 'index' or data_frame_np_[i, 0] == 'index':
                pass
            else:
                str_class = data_frame_np_[i, 0] + '-' + data_frame_np_[j, 0]
                csv_header.append(str_class)

    """
        Title rows, percent and confusion matrix rows
    """
    csv_rows.append(img_title)
    [csv_rows.append(round(i,2)) for i in percent]
    for i in range(1, data_frame_np_.shape[0]):
        for j in range(1, data_frame_np_.shape[1]):
            csv_rows.append(data_frame_np_[i, j])
    
    data_frame_csv = pd.DataFrame(columns=csv_header)
    with open(f'{opt.results}/confusion_matrix.csv', 'a') as f:
        data_frame_csv.loc[pos] = np.array(csv_rows).reshape(-1, len(csv_rows))[0]
        data_frame_csv.to_csv(f, header=True)
    data_frame_csv