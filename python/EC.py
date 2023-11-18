import numpy as np
import matplotlib.pyplot as plt
import cv2
from submission import *
from helper import *

I1 = cv2.imread('data/im1.png')
I2 = cv2.imread('data/im2.png')
r,c = I1.shape[:2]
M = np.max((r,c))

# F = np.load('output/q2_1.npz')['F']
data = np.load('data/some_corresp_noisy.npz')
pts1 = data['pts1']
pts2 = data['pts2']
F, inlier = ransacF(pts1, pts2, M)
print(" Number of inliers", len(inlier))

K = np.load('data/intrinsics.npz')
# print("K1", K['K1'])
K1, K2 = K['K1'], K['K2']

E = essentialMatrix(F, K1, K2)

M1 = np.concatenate((np.eye(3), np.zeros((3,1))), axis = 1)
# print("Shape M1", M1.shape) # 3 x 4
C1 = K1 @ M1 # C = K[R|t]; C is the camera projection matrix -> 3x4

# M2_init = np.load('output/q3_3.npz')['M2'] # 3 x 4
# C2_init = K2 @ M2_init
M2s = camera2(E)

for i in range(4):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2, M2)
        P, err = triangulate(C1, pts1[inlier], C2, pts2[inlier])
        if (np.all(P[:, 2] > 0)):
            print("Initial reprojection Error", err)
            break

# P_init = triangulate(C1, pts1, C2, pts2)[0]
# Visualize the 3D points before optimization
fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection='3d')
ax0.scatter(P[:,0], P[:,1], P[:,2], c='b', marker='o')
ax0.set_title("Before Optimization")
plt.show()

# Optimization
M2, P2 = bundleAdjustment(K1, M1, pts1[inlier], K2, M2, pts2[inlier], P)
print("M2", M2)
n = len(P2)
P = np.zeros((n,3))
P_hom = np.zeros((n,4))
C2 = K2 @ M2
for j in range(n):
      P = P2[j]
      P_hom[j,:] = np.append(P, 1)
proj_left_hom = C1 @ P_hom.T
proj_left = proj_left_hom/proj_left_hom[-1,:] # Shape should be 3 x n
# proj_left = np.concatenate((proj_left_hom[0,:]/proj_left_hom[2, :], proj_left_hom[1,:]/proj_left_hom[2, :]), axis = 0).reshape(n,2) # Shape should be n x 2
proj_right_hom = C2 @ P_hom.T
proj_right = proj_right_hom/proj_right_hom[-1,:] # Shape should be 3 x n
# proj_right = np.concatenate((proj_right_hom[0,:]/proj_right_hom[2, :], proj_right_hom[1,:]/proj_right_hom[2, :]), axis = 0).reshape(n,2) # Shape should be n x 2    
# err = (np.linalg.norm((pts1, proj_left)))**2 + (np.linalg.norm((pts2, proj_right)))**2
# err = np.sum((pts1 - proj_left)**2) + np.sum((pts2 - proj_right)**2)
opt_rpr_err = np.sum((proj_left[:2,:].T-pts1[:n,:])**2)+ np.sum((proj_right[:2,:].T-pts2[:n,:])**2)
print("Optimized reprojection error", opt_rpr_err)
# Visualize the 3D points through scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(P2[:,0], P2[:,1], P2[:,2], c='b', marker='o')
ax.set_title("After Optimization")
plt.show()