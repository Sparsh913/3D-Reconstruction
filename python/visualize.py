'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

from submission import *
import numpy as np
from matplotlib import pyplot as plt
import imageio.v2 as imageio
from helper import *

def q42():
    print("Running q4.2")
    # load images
    I1 = imageio.imread('data/im1.png')
    I2 = imageio.imread('data/im2.png')
    r,c = I1.shape[:2]
    # print("Shape of I1", I1.shape)

    # load necessary points and matrices
    pts1 = np.concatenate((np.load('data/templeCoords.npz')['x1'], np.load('data/templeCoords.npz')['y1']), axis = 1)
    # print("Shape of pts1", pts1.shape)
    F = np.load('output/q2_1.npz')['F']
    # print("Fundamental Matrix", F)
    M2 = np.load('output/q3_3.npz')['M2'] # 3 x 4
    # print("Shape M2", M2.shape)
    C2 = np.load('output/q3_3.npz')['C2']
    # print("Shape C2", C2.shape) # 3 x 4
    K = np.load('data/intrinsics.npz')
    # print("K1", K['K1'])
    K1, K2 = K['K1'], K['K2']
    # print("Shape K1", K1.shape) # 3 x 3
    M1 = np.concatenate((np.eye(3), np.zeros((3,1))), axis = 1)
    # print("Shape M1", M1.shape) # 3 x 4
    C1 = K1 @ M1 # C = K[R|t]; C is the camera projection matrix -> 3x4
    # print("Shape C1", C1.shape) # 3 x 4

    # Finding the corresponding points in the second image
    pts2 = []
    for k in range(pts1.shape[0]):
        x1, y1 = pts1[k,0], pts1[k,1]
        print("x1, y1", x1, y1)
        x2, y2 = epipolarCorrespondence(I1, I2, F, x1, y1)
        pts2.append([x2, y2])
    pts2 = np.array(pts2)
    # print("Shape of pts2", pts2.shape)

    # Find the 3D points
    P = triangulate(C1, pts1, C2, pts2)[0]
    # print(P)
    # print("Shape of P", P.shape)

    # Visualize the 3D points through scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:,0], P[:,1], P[:,2], c='b', marker='o')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    plt.show()

    np.savez('output/q4_2.npz', F = F, M1 = M1, M2 = M2, C1 = C1, C2 = C2)
    return

def q53():
    I1 = imageio.imread('data/im1.png')
    I2 = imageio.imread('data/im2.png')
    r,c = I1.shape[:2]
    M = np.max((r,c))

    # F = np.load('output/q2_1.npz')['F']
    data = np.load('data/some_corresp_noisy.npz')
    pts1 = data['pts1']
    pts2 = data['pts2']
    F, inlier = ransacF(pts1, pts2, M)

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
    opt_rpr_err = np.sum((proj_left[:2,:].T-pts1[:n,:])**2)+ np.sum((proj_right[:2,:].T-pts2[:n,:])**2)
    print("Optimized reprojection error", opt_rpr_err)
    # Visualize the 3D points through scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P2[:,0], P2[:,1], P2[:,2], c='b', marker='o')
    plt.show()

if __name__ == "__main__":
    q42()
    # q53()