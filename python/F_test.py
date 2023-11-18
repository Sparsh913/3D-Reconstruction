from submission import eightpoint, ransacF
import numpy as np
from helper import displayEpipolarF
import cv2

I1 = cv2.imread('data/im1.png')
I2 = cv2.imread('data/im2.png')
r,c = I1.shape[:2]

correspondences = np.load('data/some_corresp.npz')
pts1, pts2 = correspondences['pts1'], correspondences['pts2']
M = np.max((r,c))
F = eightpoint(pts1, pts2, M)
print("Fundamental Matrix", F)

# noisy_corresp = np.load('data/some_corresp_noisy.npz')
# pts1, pts2 = noisy_corresp['pts1'], noisy_corresp['pts2']
# print("Shape of pts1", pts1.shape)
# best_F, inliers = ransacF(pts1, pts2, M)
# print("Best Fundamental Matrix", best_F)
# print("Number of inliers", len(inliers))
# displayEpipolarF(I1, I2, best_F)

# F = np.array([[-6.87848171e-08, -1.43797186e-06,  3.22495080e-04],
#        [ 1.39849376e-06,  8.57583525e-07, -6.82992808e-04],
#        [-2.45691981e-04,  2.89104175e-04,  2.89670250e-02]])
# displayEpipolarF(I1, I2, F)

# F = eightpoint(pts1, pts2, M)
# print("Fundamental Matrix", F)
displayEpipolarF(I1, I2, F)
