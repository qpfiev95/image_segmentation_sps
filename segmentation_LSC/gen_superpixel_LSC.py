import numpy as np
import sys
import time
import math

DBL_MAX = sys.float_info[0]  # max float value
TEST_INITIALIZATION = False
TEST_KMEANS_LABEL = False
FAKE_KMEANS_LABEL = False
TEST_PEC_LABEL = False
FAKE_EC_LABEL = False
PI = 3.1415926

def initialize_LSC(L: np.ndarray, a: np.ndarray, b: np.ndarray, nRows: int, nCols: int, StepX: int, StepY: int,
               Color: float, Distance: float):
    '''
    :param L: Lab
    :param a: lAb
    :param b: laB
    :param nRows: the number of rows of the image
    :param nCols: the number of columns of the image
    :param StepX:
    :param StepY:
    :param Color: color weight
    :param Distance: distance weight
    :return:
    '''
    # test:
    #L, a, b = rgbtolab(I, nRows, nCols)
    print("\t[{}] [Initialize.py]".format(time.ctime()[11:19]))
    vcos = np.vectorize(math.cos)
    vsin = np.vectorize(math.sin)
    thetaL = (np.resize(L.copy(), [nRows, nCols]) / 255.0 * PI / 2.0).astype(np.float64)
    thetaa = (np.resize(a.copy(), [nRows, nCols]) / 255.0 * PI / 2.0).astype(np.float64)
    thetab = (np.resize(b.copy(), [nRows, nCols]) / 255.0 * PI / 2.0).astype(np.float64)
    thetax = np.empty([nRows, nCols], dtype=np.float64)
    thetay = np.empty([nRows, nCols], dtype=np.float64)
    for i in range(thetax.shape[0]):
        thetax[i, :] = i
    for j in range(thetay.shape[1]):
        thetay[:, j] = j
    thetax = (thetax / StepX) * PI / 2
    thetay = (thetay / StepY) * PI / 2
    # 10-dimension feature space phi(p)
    # phi(p) = [L1, L2, a1, a2, b1, b2, x1, x2, y1, y2]
    L1 = Color * vcos(thetaL)
    L2 = Color * vsin(thetaL)
    a1 = Color * vcos(thetaa) * 2.55
    a2 = Color * vsin(thetaa) * 2.55
    b1 = Color * vcos(thetab) * 2.55
    b2 = Color * vsin(thetab) * 2.55
    x1 = Distance * vcos(thetax)
    x2 = Distance * vsin(thetax)
    y1 = Distance * vcos(thetay)
    y2 = Distance * vsin(thetay)

    size = nRows * nCols
    sigmaL1 = L1.sum() / size
    sigmaL2 = L2.sum() / size
    sigmaa1 = a1.sum() / size
    sigmaa2 = a2.sum() / size
    sigmab1 = b1.sum() / size
    sigmab2 = b2.sum() / size
    sigmax1 = x1.sum() / size
    sigmax2 = x2.sum() / size
    sigmay1 = y1.sum() / size
    sigmay2 = y2.sum() / size

    # In the weighted K-means clustering, each data_1 point p is assigned with a weight w(p)
    W = L1 * sigmaL1 + L2 * sigmaL2 + a1 * sigmaa1 + a2 * sigmaa2 + b1 * sigmab1 + \
        b2 * sigmab2 + x1 * sigmax1 + x2 * sigmax2 + y1 * sigmay1 + y2 * sigmay2
    L1 /= W
    L2 /= W
    a1 /= W
    a2 /= W
    b1 /= W
    b2 /= W
    x1 /= W
    x2 /= W
    y1 /= W
    y2 /= W
    return L1.astype(np.float32), L2.astype(np.float32), a1.astype(np.float32), \
           a2.astype(np.float32), b1.astype(np.float32), b2.astype(np.float32), \
           x1.astype(np.float32), x2.astype(np.float32), y1.astype(np.float32), \
           y2.astype(np.float32), W.astype(np.float64)

# Perform weighted kmeans iteratively in the ten dimensional feature space.
def gen_superpixel_LSC(L1: np.ndarray, L2: np.ndarray, a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray,
                 x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray,
                 W: np.ndarray, label: np.ndarray, seedArray: list, seedNum: int, nRows: int, nCols: int, StepX: int,
                 StepY: int, iterationNum: int, thresholdCoef: int, new_label: np.ndarray):
    print("\t[{}] [gen_superpixel_LSC.py]: Pre-treatment".format(time.ctime()[11:19]))
    dist = np.empty([nRows, nCols], dtype=np.float64)
    centerL1 = np.empty([seedNum], dtype=np.float64)
    centerL2 = np.empty([seedNum], dtype=np.float64)
    centera1 = np.empty([seedNum], dtype=np.float64)
    centera2 = np.empty([seedNum], dtype=np.float64)
    centerb1 = np.empty([seedNum], dtype=np.float64)
    centerb2 = np.empty([seedNum], dtype=np.float64)
    centerx1 = np.empty([seedNum], dtype=np.float64)
    centerx2 = np.empty([seedNum], dtype=np.float64)
    centery1 = np.empty([seedNum], dtype=np.float64)
    centery2 = np.empty([seedNum], dtype=np.float64)
    WSum = np.empty([seedNum], dtype=np.float64)
    clusterSize = np.empty([seedNum], dtype=np.int32)

    print("\t[{}] [gen_superpixel_LSC.py]: Initialization".format(time.ctime()[11:19]))
    # Ex: seedNum = 200
    for i in range(seedNum):
        centerL1[i] = 0
        centerL2[i] = 0
        centera1[i] = 0
        centera2[i] = 0
        centerb1[i] = 0
        centerb2[i] = 0
        centerx1[i] = 0
        centerx2[i] = 0
        centery1[i] = 0
        centery2[i] = 0
        # The coordinates of seeds
        x = seedArray[i].x
        y = seedArray[i].y
        # Ex: StepX = 54; StepY = 192; x,y = seedArray[1] = 50,161
        # X: [37,63]; Y: [113, 209]
        minX = int(0 if x - StepX // 4 <= 0 else x - StepX // 4)
        minY = int(0 if y - StepY // 4 <= 0 else y - StepY // 4)
        maxX = int(nRows - 1 if x + StepX // 4 >= nRows - 1 else x + StepX // 4)
        maxY = int(nCols - 1 if y + StepY // 4 >= nCols - 1 else y + StepY // 4)
        Count = 0
        # For each superpixel:
        # the Center parameter are the mean of each parameter L,a,b,x,y
        for j in range(minX, maxX + 1):
            for k in range(minY, maxY + 1):
                Count += 1
                centerL1[i] += L1[j][k]
                centerL2[i] += L2[j][k]
                centera1[i] += a1[j][k]
                centera2[i] += a2[j][k]
                centerb1[i] += b1[j][k]
                centerb2[i] += b2[j][k]
                centerx1[i] += x1[j][k]
                centerx2[i] += x2[j][k]
                centery1[i] += y1[j][k]
                centery2[i] += y2[j][k]
        # weighted means m_k
        centerL1[i] /= Count
        centerL2[i] /= Count
        centera1[i] /= Count
        centera2[i] /= Count
        centerb1[i] /= Count
        centerb2[i] /= Count
        centerx1[i] /= Count
        centerx2[i] /= Count
        centery1[i] /= Count
        centery2[i] /= Count

    print("\t[{}] [gen_superpixel_LSC.py]: K-means".format(time.ctime()[11:19]))
    for iteration in range(iterationNum + 1):
        print("\t\t[{}] [DoSuperpixel.py]: K-means_iter_{}".format(time.ctime()[11:19], iteration))
        # set distance = infinity
        for i in range(nRows):
            for j in range(nCols):
                dist[i][j] = DBL_MAX
        # For each weighted means and search center of each cluster
        for i in range(seedNum):
            # x,y: search center c_k
            # t*stepX x t*stepY region (t = 2)
            # for each point in the 2*stepX x 2*stepY neighborhood of c_k
            ################################
            x = seedArray[i].x
            y = seedArray[i].y
            ################################
            minX = int(0 if x - StepX <= 0 else x - StepX)
            minY = int(0 if y - StepY <= 0 else y - StepY)
            maxX = int(nRows - 1 if x + StepX >= nRows - 1 else x + StepX)
            maxY = int(nCols - 1 if y + StepY >= nCols - 1 else y + StepY)
            #print("minX: {}; maxX: {}; minY: {}; maxY: {}".format(minX, maxX, minY, maxY))
            step1_min_x = minX
            step1_max_x = maxX + 1
            step1_min_y = minY
            step1_max_y = maxY + 1
            #print("step1_minX: {}; step1_maxX: {}; step1_minY: {};"
            #     " step1_maxY: {}".format(step1_min_x, step1_max_x, step1_min_y,  step1_max_y))

            # phi(p): denote the function that maps data_1 points to a 10-dimension feature space.
            # The Euclidean distance between phi(p) and m_k in the feature space
            step1_vpow = np.vectorize(lambda _: _ * _)
            step1_L1_pow = step1_vpow(L1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerL1[i])
            step1_L2_pow = step1_vpow(L2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerL2[i])
            step1_a1_pow = step1_vpow(a1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centera1[i])
            step1_a2_pow = step1_vpow(a2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centera2[i])
            step1_b1_pow = step1_vpow(b1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerb1[i])
            step1_b2_pow = step1_vpow(b2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerb2[i])
            step1_x1_pow = step1_vpow(x1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerx1[i])
            step1_x2_pow = step1_vpow(x2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centerx2[i])
            step1_y1_pow = step1_vpow(y1[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centery1[i])
            step1_y2_pow = step1_vpow(y2[step1_min_x:step1_max_x, step1_min_y: step1_max_y] - centery2[i])
            # D (Euclidean distance)
            step1_D = step1_L1_pow + step1_L2_pow + step1_a1_pow + step1_a2_pow + step1_b1_pow + step1_b2_pow + \
                      step1_x1_pow + step1_x2_pow + step1_y1_pow + step1_y2_pow

            # if D - d(p) < 0 then d(p) = D and L(p) = k
            # Init = 1
            step1_if = (step1_D - dist[step1_min_x: step1_max_x, step1_min_y: step1_max_y] < 0).astype(np.uint16)
            step1_neg_if = 1 - step1_if
            # new_label:
            # new_label(p) = 0 if D - d(p) < 0
            new_label[step1_min_x: step1_max_x, step1_min_y: step1_max_y] *= step1_neg_if
            # new_label(p) = 0 + 1*i (label i_th)
            new_label[step1_min_x: step1_max_x, step1_min_y: step1_max_y] += (step1_if * i)
            # d(p) = 0 if D - d(p) < 0
            dist[step1_min_x: step1_max_x, step1_min_y: step1_max_y] *= step1_neg_if
            # d(p) = 0 + D
            step1_D_to_plus = step1_D * step1_if
            dist[step1_min_x: step1_max_x, step1_min_y: step1_max_y] += step1_D_to_plus

        # Reset centers or move cluster
        for i in range(seedNum):
            centerL1[i] = 0
            centerL2[i] = 0
            centera1[i] = 0
            centera2[i] = 0
            centerb1[i] = 0
            centerb2[i] = 0
            centerx1[i] = 0
            centerx2[i] = 0
            centery1[i] = 0
            centery2[i] = 0
            WSum[i] = 0
            clusterSize[i] = 0
            seedArray[i].x = 0
            seedArray[i].y = 0

        # Update weighted means m_k and search centers c_k for all clusters
        label = new_label.copy().reshape([nRows * nCols])
        for i in range(nRows):
            for j in range(nCols):
                L = label[i * nCols + j]  # int
                Weight = W[i][j]  # double
                centerL1[L] += Weight * L1[i][j]
                centerL2[L] += Weight * L2[i][j]
                centera1[L] += Weight * a1[i][j]
                centera2[L] += Weight * a2[i][j]
                centerb1[L] += Weight * b1[i][j]
                centerb2[L] += Weight * b2[i][j]
                centerx1[L] += Weight * x1[i][j]
                centerx2[L] += Weight * x2[i][j]
                centery1[L] += Weight * y1[i][j]
                centery2[L] += Weight * y2[i][j]
                clusterSize[L] += 1
                WSum[L] += Weight
                seedArray[L].x += i
                seedArray[L].y += j

        for i in range(seedNum):
            WSum[i] = 1 if WSum[i] == 0 else WSum[i]
            clusterSize[i] = 1 if clusterSize[i] == 0 else clusterSize[i]

        cluster_center = np.zeros([2,seedNum])
        for i in range(seedNum):
            centerL1[i] /= WSum[i]
            centerL2[i] /= WSum[i]
            centera1[i] /= WSum[i]
            centera2[i] /= WSum[i]
            centerb1[i] /= WSum[i]
            centerb2[i] /= WSum[i]
            centerx1[i] /= WSum[i]
            centerx2[i] /= WSum[i]
            centery1[i] /= WSum[i]
            centery2[i] /= WSum[i]
            seedArray[i].x /= clusterSize[i]
            seedArray[i].y /= clusterSize[i]
            cluster_center[0][i] = int(seedArray[i].x)
            cluster_center[1][i] = int(seedArray[i].y)

    label_nonmerged = label.reshape([nCols, nRows]).transpose([0, 1])
    tmp = label
    threshold = int((nRows * nCols) / (seedNum * thresholdCoef))
    #preEnforceConnectivity(tmp, nRows, nCols)
    #label_merged = EnforceConnectivity(L1, L2, a1, a2, b1, b2, x1, x2, y1, y2, W, tmp, threshold, nRows, nCols)
    #label_merged = label_merged.reshape([nCols, nRows]).transpose([0, 1])
    label_merged = label.reshape([nCols, nRows]).transpose([0, 1])
    return label_merged, label_nonmerged, cluster_center