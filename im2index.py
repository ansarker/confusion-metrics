import numpy as np

def im2index(im):
    assert len(im.shape) == 3
    r = im[:, :, 0].ravel()
    g = im[:, :, 1].ravel()
    b = im[:, :, 2].ravel()
    label = np.zeros((r.shape[0], 1), dtype=np.uint8)
    for i in range(r.shape[0]):
        # Unknown
        if r[i] == 0 and g[i] == 0 and b[i] == 0:
            label[i] = 0
        # Forest
        elif r[i] == 0 and g[i] == 255 and b[i] == 255:
            label[i] = 1
        # Built-up
        elif r[i] == 255 and g[i] == 0 and b[i] == 0:
            label[i] = 2
        # Water
        elif r[i] == 0 and g[i] == 0 and b[i] == 255:
            label[i] = 3
        # Farmland
        elif r[i] == 0 and g[i] == 255 and b[i] == 0:
            label[i] = 4
        # Meadow
        elif r[i] == 255 and g[i] == 255 and b[i] == 0:
            label[i] = 5
        else:
            label[i] = 0
    indx_img = np.reshape(label, (-1, im.shape[1]))
    return indx_img