from skimage import io, color
import cmath
from utils import *
from gen_superpixel_LSC import gen_superpixel_LSC, initialize_LSC
from PIL import Image, ImageOps

class LSCProcessor(object):
    def __init__(self, filename, colorWeight, distWeight, threshold, numSeed, numIter, imgFormat):
        self.filename = filename
        self.data = my_imread(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.colorWeight = colorWeight
        self.distWeight = distWeight
        self.threshold = threshold
        self.numIter = numIter
        self.numSeed = numSeed
        self.imgFormat = imgFormat
        self.numCol = int(cmath.sqrt(numSeed * self.image_width / self.image_height).real)
        self.numRow = int(numSeed / self.numCol)
        self.stepX = int(self.image_width/self.numCol)
        self.stepY = int(self.image_height/self.numRow)
        # The grid interval (to produce roughly equally sized superpixels)
        self.dis = np.full((self.image_height, self.image_width), np.inf)
        self.label_mask = np.ones((self.image_height, self.image_width), dtype=int)
        self.name = (filename.split("/")[2]).split(".")[0]
        self.implement()

    def init_clusters(self):
        # Sampling K seeds or K clusters
        self.seedArray = gen_seeds(self.image_height, self.image_width, self.numSeed)
        self.numSeed = len(self.seedArray)

    def init_featurespace(self):
        if self.imgFormat == "RGB":
            L, a, b = rgbtolab(self.data, self.image_height, self.image_width)
        else:
            L, a, b = gstolab(self.data, self.image_height, self.image_width)
        return initialize_LSC(L, a, b, self.image_height, self.image_width, self.stepX, self.stepY, self.colorWeight, self.distWeight)

    def generate_sp(self):
        L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, W = self.init_featurespace()
        new_label = np.empty([self.image_height, self.image_width], dtype=np.uint16)
        init_label = np.empty([self.image_height*self.image_width], dtype=np.uint16)
        self.label, self.label_nonmerged, self.seedArray = gen_superpixel_LSC(L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, W, init_label, self.seedArray,
                                                                        self.numSeed, self.image_height, self.image_width, self.stepX,
                                                                        self.stepY, self.numIter, self.threshold, new_label)

    def implement(self):
        self.init_clusters()
        self.init_featurespace()
        self.generate_sp()
        self.label_2D =  self.label.reshape([self.image_width, self.image_height]).transpose([1, 0])
        self.label_nonmerged_2D = self.label_nonmerged.reshape([self.image_width, self.image_height]).transpose([1, 0])
        if self.imgFormat == "RGB":
            display_superpixel(self.label_2D, self.data, self.name, "RGB")
            display_superpixel_seed(self.label_2D, self.data, self.name, "RGB", self.seedArray)
        else:
            display_superpixel(self.label_2D, self.data, self.name, "L", self.seedArray)
            display_superpixel_seed
    def sp_segmentation(self):

        if self.imgFormat == "RGB":
            image_sp = np.zeros([self.label.max() + 1, 3])
        else:
            image_sp = np.zeros([self.label.max() + 1])
        cluster_size = np.zeros([self.label.max()+1])
        for i in range(self.image_height):
            for j in range(self.image_width):
                label_sp = self.label_2D[i][j]
                if self.imgFormat == "RGB":
                    image_sp[label_sp][0] += self.data[i][j][0]
                    image_sp[label_sp][1] += self.data[i][j][1]
                    image_sp[label_sp][2] += self.data[i][j][2]
                else:
                    image_sp[label_sp] += self.data[i][j]
                cluster_size[label_sp] += 1
        for i in range(self.label.max()+1):
            cluster_size[i] = (1 if cluster_size[i] == 0 else cluster_size[i])
            if self.imgFormat == "RGB":
                image_sp[i][0] /= cluster_size[i]
                image_sp[i][1] /= cluster_size[i]
                image_sp[i][2] /= cluster_size[i]
            else:
                image_sp[i] /= cluster_size[i]
        self.cluster_size = cluster_size
        self.array_sp = image_sp[cluster_size != 1]
        if self.imgFormat == "RGB":
            image_scale = np.zeros([self.image_height, self.image_width,3],dtype = np.uint8)
        else:
            image_scale = np.zeros([self.image_height, self.image_width], dtype=np.uint8)
        for i in range(self.image_height):
            for j in range(self.image_width):
                label_sp2 = self.label_2D[i][j]
                if self.imgFormat == "RGB":
                    image_scale[i][j][0] = np.round(image_sp[label_sp2][0])
                    image_scale[i][j][1] = np.round(image_sp[label_sp2][1])
                    image_scale[i][j][2] = np.round(image_sp[label_sp2][2])
                else:
                    image_scale[i][j] = np.round(image_sp[label_sp2])
        if self.imgFormat == "RGB":
            self.array_sp_rgb = image_scale
            image_sp_rgb = Image.fromarray(self.array_sp_rgb, 'RGB')
            #image_sp_rgb.show()
            image_sp_rgb.save("data/sp2/" + str(self.name)+"_sp2"+".png")
            self.array_sp_grayscale = np.asarray(ImageOps.grayscale(Image.fromarray(np.uint8(self.array_sp.reshape(-1, 1, 3)), 'RGB')))
            self.image_sp_rgb = image_sp_rgb
            return self.image_sp_rgb, self.array_sp_rgb, self.array_sp_grayscale
        else:
            self.array_sp_gs = image_scale
            image_sp_gs = Image.fromarray(self.array_sp_gs, 'L')
            image_sp_gs.save("data/sp2/" + str(self.name) + "_sp2" + ".png")