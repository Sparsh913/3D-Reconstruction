from submission import epipolarCorrespondence
from submission import eightpoint
import numpy as np
from helper import epipolarMatchGUI
import cv2

I1 = cv2.imread('data/im1.png')
I2 = cv2.imread('data/im2.png')
r,c = I1.shape[:2]

correspondences = np.load('data/some_corresp.npz')
pts1, pts2 = correspondences['pts1'], correspondences['pts2']
M = np.max((r,c))
F = eightpoint(pts1, pts2, M)

x2, y2 = epipolarCorrespondence(I1, I2, F, 100,100)
print("x2, y2", x2, y2)
epipolarMatchGUI(I1, I2, F)