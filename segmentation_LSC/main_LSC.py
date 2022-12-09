from PIL import Image, ImageOps
from superpixel_LSC import LSCProcessor
from utils import my_imread
import cv2

## params
img_dir = 'data/examples/img_rgb.jpg'
img_arr = my_imread(img_dir)
img = Image.fromarray(img_arr,'RGB')
#img = cv2.imread(img_dir)

## superpixel segmentation
# def __init__(self, filename, colorWeight, distWeight, threshold, numSeed, numIter, imgFormat):
lsc_sps = LSCProcessor(img_dir,20,5,4,400,10,'RGB')
lsc_sps.sp_segmentation()

## draw lines
seed_1 = (int(lsc_sps.seedArray[0][1]), int(lsc_sps.seedArray[1][1]))
seed_2 = (int(lsc_sps.seedArray[0][31]), int(lsc_sps.seedArray[1][31]))
img_sp_arr = lsc_sps.array_sp_rgb
cv2.line(img_sp_arr, seed_1, seed_2, (0,0,0), 1)
cv2.imwrite('data/sp_rout/img_test_2.jpg', img_sp_arr)