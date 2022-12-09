from PIL import Image
import numpy as np
import math
from skimage import color

def my_imread(path: str):
    return np.asarray(Image.open(path), dtype=np.uint8)

class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return "<class 'Point'>: x={}, y={}".format(self.x, self.y)
    __repr__ = __str__

def gen_seeds(numRow: int, numCol: int, numSeed: int) -> tuple:
    '''
    :param numRow: the image row number
    :param numCol: the image column number
    :param numSeed: the seed number should be generated
    :return: 
        seeds_list: a tuple contains all the seeds
    '''
    seeds_list = []
    num_seeds_col = int(math.sqrt(numSeed * numCol / numRow))
    num_seeds_row = int(numSeed / num_seeds_col)
    step_x = int(numRow / num_seeds_row)
    step_y = int(numCol / num_seeds_col)
    row_remain = numRow - num_seeds_row * step_x
    col_remain = numCol - num_seeds_col * step_y
    current_row_remain = 1
    current_col_remain = 1
    current_seeds_count = 0
    for i in range(num_seeds_row):
        for j in range(num_seeds_col):
            center_x = int(i * step_x + 0.5 * step_x + current_row_remain)
            center_y = int(j * step_y + 0.5 * step_y + current_col_remain)
            if center_x > numRow - 1:
                center_x = numRow - 1
            if center_y > numCol - 1:
                center_y = numCol - 1
            if current_col_remain < col_remain:
                current_col_remain += 1
            seeds_list.append(Point(x=center_x, y=center_y))
            current_seeds_count += 1
        if current_row_remain < row_remain:
            current_row_remain += 1
    return tuple(seeds_list)

def rgbtolab(I:np.ndarray, num_cols:int, num_rows:int):
    """
    https://graphicdesign.stackexchange.com/questions/76824/what-are-the-pros-and-cons-of-using-lab-color
    change rgb to lab format
    :param I: rgb format image
    :return:
        L: L channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
        a: a channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
        b: b channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
    """
    # 2: channel (3), 0,1: HxW
    lab_img = color.rgb2lab(I).transpose([2, 1, 0])
    # R -> L
    L = lab_img[0].copy().reshape([num_rows * num_cols])
    a = lab_img[1].copy().reshape([num_rows * num_cols])
    b = lab_img[2].copy().reshape([num_rows * num_cols])
    L /= (100 / 255)  # L is [0, 100], change it to [0, 255]
    L += 0.5
    a += 128 + 0.5  # A is [-128, 127], change it to [0, 255]
    b += 128 + 0.5  # B is [-128, 127], change it to [0, 255]
    return L.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)

def gstolab(I:np.ndarray, num_cols:int, num_rows:int):
    """
    https://graphicdesign.stackexchange.com/questions/76824/what-are-the-pros-and-cons-of-using-lab-color
    change rgb to lab format
    :param I: rgb format image
    :return:
        L: L channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
        a: a channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
        b: b channel, range from 0 to 255, dtype: uint8, shape: (row_num * col_num,)
    """
    L = I.reshape([num_rows * num_cols])
    a = np.zeros([num_rows * num_cols])
    b = np.zeros([num_rows * num_cols])
    return L.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)

def display_superpixel(label_2D: np.ndarray, img: np.ndarray, para, mode):
    img = img.copy()
    nRows, nCols = label_2D.shape
    for i in range(nRows):
        for j in range(nCols):
            minX = 0 if i - 1 < 0 else i - 1
            minY = 0 if j - 1 < 0 else j - 1
            maxX = nRows - 1 if i + 1 >= nRows else i + 1
            maxY = nCols - 1 if j + 1 >= nCols else j + 1
            count = (label_2D[minX:maxX + 1, minY:maxY + 1] != label_2D[i][j]).sum()
            if count >= 2:
                if mode == "RGB":
                    img[i][j] = [0, 0, 0]
                else:
                    img[i][j] = 0
    if mode == "RGB":
        PIL_image = Image.fromarray(img, 'RGB')
    else:
        PIL_image = Image.fromarray(img, 'L')
    #PIL_image.show()
    PIL_image.save("data/sp/" + str(para)+"_sp"+".png")

def display_superpixel_seed(label_2D: np.ndarray, img: np.ndarray, para, mode, seed):
    img = img.copy()
    nRows, nCols = label_2D.shape
    for i in range(nRows):
        for j in range(nCols):
            minX = 0 if i - 1 < 0 else i - 1
            minY = 0 if j - 1 < 0 else j - 1
            maxX = nRows - 1 if i + 1 >= nRows else i + 1
            maxY = nCols - 1 if j + 1 >= nCols else j + 1
            count = (label_2D[minX:maxX + 1, minY:maxY + 1] != label_2D[i][j]).sum()
            #if count >= 2:
            #    if mode == "RGB":
            #        img[i][j] = [0, 0, 0]
            #    else:
            #        img[i][j] = 0
            for s in range(seed.shape[1]):
                if i == seed[0][s] and j== seed[1][s]:
                    if mode == "RGB":
                        img[i][j] = [0, 0, 0]
                    else:
                        img[i][j] = 0
    if mode == "RGB":
        PIL_image = Image.fromarray(img, 'RGB')
    else:
        PIL_image = Image.fromarray(img, 'L')
    #PIL_image.show()
    PIL_image.save("data/sp_seed/" + str(para)+"_sp"+".png")