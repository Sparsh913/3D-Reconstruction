'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

from submission import *
import numpy as np
from helper import camera2
# import cv2
# import scipy.ndimage as ndimage
import imageio.v2 as imageio

I1 = imageio.imread('data/im1.png')
I2 = imageio.imread('data/im2.png')
r,c = I1.shape[:2]

correspondences = np.load('data/some_corresp.npz')
pts1, pts2 = correspondences['pts1'], correspondences['pts2']

M = np.max((r,c))
F = eightpoint(pts1, pts2, M)
print("Fundamental Matrix", F)

K = np.load('data/intrinsics.npz')
# print("K1", K['K1'])
K1, K2 = K['K1'], K['K2']

E = essentialMatrix(F, K1, K2)

M2s = camera2(E) # M = [R|t]
# print("M2s", M2s)

M1 = np.concatenate((np.eye(3), np.zeros((3,1))), axis = 1)
C1 = K1 @ M1 # C = K[R|t]; C is the camera projection matrix -> 3x4

for i in range(4):
    M2 = M2s[:,:,i]
    print("M2 shape", M2.shape)
    print(M2)
    C2 = K2 @ M2
    P, err = triangulate(C1, pts1, C2, pts2)
    if (np.all(P[:,2] > 0)):
        print("Reprojection Error", err) # 3D point has to be in front of the camera
        break
# print("P", P)
# print("Reprojection Error", err)
np.savez('output/q3_3.npz', M2 = M2, C2 = C2, P = P)

##====================================== Test ======================================##
# for i in range(4):
#     M2 = M2s[:,:,i]
#     C2 = K2 @ M2
#     P, err = triangulate(C1, pts1, C2, pts2)
#     print(f"Test Reprojection error {i}", err)