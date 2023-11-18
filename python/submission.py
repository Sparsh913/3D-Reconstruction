"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
import util
import scipy.ndimage as ndimage
import random
import scipy
# corr = np.load('data/some_corresp.npz')
# pts1, pts2 = corr['pts1'], corr['pts2']
# print("Shape Correspondences", pts1)

# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    n = pts1.shape[0]

    # Normalization
    # T = 1/M * np.eye(n,n)
    # pts1_n, pts2_n = T @ pts1, T @ pts2
    pts1_n, pts2_n = pts1*(1/M), pts2*(1/M)

    xl, yl = pts1_n[:,0].reshape(n,1), pts1_n[:,1].reshape(n,1)
    xr, yr = pts2_n[:,0].reshape(n,1), pts2_n[:,1].reshape(n,1)
    U = np.concatenate((np.multiply(xr, xl), np.multiply(xr, yl), xr, np.multiply(yr, xl), np.multiply(yr, yl), yr, xl, yl, np.ones((n,1))), axis=1)
    print("Shape U", U.shape) # Shape should be n x 9
    # Solving UF = 0
    u, s, vt = np.linalg.svd(U, full_matrices=False)
    F = vt[-1, :]
    F = F.reshape(3,3)

    # F = util._singularize(F)
    F = util.refineF(F, pts1_n, pts2_n)
    # s[-1] = 0
    # F = u @ np.diag(s) @ vt
    # F = F[-1,:]

    # Unnormalization
    T = np.diag([1./M, 1./M, 1])
    F_unn = T[:3,:3].T @ F @ T[:3, :3]
    np.savez('output/q2_1.npz', F = F_unn, M = M)
    
    return F_unn
    # pass



'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.linalg.inv((np.linalg.inv(K2)).T) @ F @ K1
    np.savez('output/q3_1.npz', E=E)
    
    return E
    # pass


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation
    n = pts1.shape[0]
    ml1_T, ml2_T, ml3_T = C1[0,:].reshape(1,4), C1[1,:].reshape(1,4), C1[2,:].reshape(1,4)
    mr1_T, mr2_T, mr3_T = C2[0,:].reshape(1,4), C2[1,:].reshape(1,4), C2[2,:].reshape(1,4)
    # print("Shape ml1_T", ml1_T.shape) # Shape should be 1 x 4
    xl, yl = pts1[:,0].reshape(n,1), pts1[:,1].reshape(n,1)
    xr, yr = pts2[:,0].reshape(n,1), pts2[:,1].reshape(n,1)
    # print("Shape xl", xl.shape) # Shape should be n x 1

    P = np.zeros((n,3))
    P_hom = np.zeros((n,4))
    for i in range(n):
        A1 = yl[i]*ml3_T - ml2_T
        A2 = ml1_T - xl[i]*ml3_T
        A3 = yr[i]*mr3_T - mr2_T
        A4 = mr1_T - xr[i]*mr3_T

        A = np.concatenate((A1, A2, A3, A4), axis = 0)
    # A1 = np.multiply(xl, ml3_T) - ml1_T
    # A2 = np.multiply(yl, ml3_T) - ml2_T
    # A3 = np.multiply(xr, mr3_T) - mr1_T
    # A4 = np.multiply(yr, mr3_T) - mr2_T
    # A = np.concatenate((A1, A2, A3, A4), axis = 0) # Shape should be 4*n x 4
        # print("Shape A", A.shape)

        # Solving AP = 0 through SVD
        u, s, vt = np.linalg.svd(A, full_matrices = False)
        p = vt[-1,:]
        p = p/p[-1]
        P[i, :] = p[:3] # First row has x,y,z
        P_hom[i, :] = p

    # Compute Reprojection Error
    # P_hom = np.concatenate((P, np.ones((n,1)))).T  # Converting P into a set of 4 x n homogeneous coordinates
    proj_left_hom = C1 @ P_hom.T
    proj_left = proj_left_hom/proj_left_hom[-1,:] # Shape should be 3 x n
    # proj_left = np.concatenate((proj_left_hom[0,:]/proj_left_hom[2, :], proj_left_hom[1,:]/proj_left_hom[2, :]), axis = 0).reshape(n,2) # Shape should be n x 2
    proj_right_hom = C2 @ P_hom.T
    proj_right = proj_right_hom/proj_right_hom[-1,:] # Shape should be 3 x n
    # proj_right = np.concatenate((proj_right_hom[0,:]/proj_right_hom[2, :], proj_right_hom[1,:]/proj_right_hom[2, :]), axis = 0).reshape(n,2) # Shape should be n x 2    
    # err = (np.linalg.norm((pts1, proj_left)))**2 + (np.linalg.norm((pts2, proj_right)))**2
    # err = np.sum((pts1 - proj_left)**2) + np.sum((pts2 - proj_right)**2)
    err=np.sum((proj_left[:2,:].T-pts1)**2)+ np.sum((proj_right[:2,:].T-pts2)**2)

    return P, err
    # pass


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation
    pl = np.array([x1, y1, 1]).reshape(3,1)
    lr = F @ pl # Epipolar line in right image

    # Find the points on the epipolar line in the right image
    sy, sx, _ = im2.shape
    if lr[0] != 0:
        ye = y1 + 30 #sy-1 # y coordinate of the "last" pixel in the right image on the epipolar line near y1
        ys = y1 - 30 # y coordinate of the "first" pixel in the right image on the epipolar line
        xe = -(lr[1] * ye + lr[2])/lr[0] # x coordinate of the "last" pixel in the right image on the epipolar line 
        xs = -(lr[1] * ys + lr[2])/lr[0] # x coordinate of the "first" pixel in the right image on the epipolar line
    else:
        xe = x1 + 30 #sx-1
        xs = x1 - 30 #0
        ye = -(lr[0] * xe + lr[2])/lr[1]
        ys = -(lr[0] * xs + lr[2])/lr[1]
    
    print("ye", ye)
    print("ys", ys)
    print("xe", xe)
    print("xs", xs)
    
    # Smmothen the images
    im1 = ndimage.gaussian_filter(im1, sigma=1, output=np.float64)
    im2 = ndimage.gaussian_filter(im2, sigma=1, output=np.float64)

    # Find the patch in the left image
    patch_left = im1[int(y1-5):int(y1+6), int(x1-5):int(x1+6), :] # Shape should be 11 x 11 x 3
    patch_left = patch_left.reshape(363,1) # Shape should be 363 x 1
    # print("Shape patch_left", patch_left.shape)
    # Gaussian Weighting
    


    # Find the patch in the right image
    min_err = np.inf
    # print("Loop in next")
    for i in range(int(xs)-1, int(xe)+1):
        # print("Entered 1st loop") # Iterate over the x coordinates of the epipolar line in the right image
        for j in range(int(ys), int(ye)): # Iterate over the y coordinates of the epipolar line in the right image
            patch_right = im2[j-5:j+6, i-5:i+6, :]
            # print("Shape patch_right", patch_right.shape)
            patch_right = patch_right.reshape(363,1)
            err = np.linalg.norm(patch_left - patch_right)
            if err < min_err:
                min_err = err
                x2, y2 = i, j
                # print("x2, y2", x2, y2)

    # np.savez('output/q4_1.npz', F = F, pts1 = np.array([x1, y1]), pts2 = np.array([x2, y2]))
    return x2, y2

    # pass

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    # Replace pass by your implementation
    assert(pts1.shape[0] == pts2.shape[0])
    n = pts1.shape[0]
    # # inlier = np.zeros(nIters)
    # s = 8 # At least 8 points are needed to compute F
    # Fs = []
    # num_corresp = 0
    # for i in range(nIters):
    #     rn = np.random.choice(n, s, replace=False)
    #     x1, x2 = np.zeros((s, 2)), np.zeros((s, 2))
    #     # for j in range(s):
    #     #     x1[j, :], x2[j, :] = pts1[rn[j], :], pts2[rn[j], :]
    #     x1, x2 = pts1[rn, :], pts2[rn, :]
    #     print("Shape x1", x1.shape)
    #     F = eightpoint(x1, x2, M)
    #     Fs.append(F)
    #     # inlier[i] = np.sum(np.abs(np.sum(pts2 * (F @ pts1.T).T, axis=1)) < tol)
    #     corresp = []
    #     for k in range(n):
    #         pr = np.append(pts2[k ,:],1).reshape((1,3))
    #         print("Shape pr", pr.shape)
    #         pl = np.append(pts1[k ,:],1).reshape((3,1))
    #         prod = np.abs(pr @ F @ pl)
    #         if prod < tol:
    #             corresp.append(k)
    #             # inlier[i] += 1
    #     # comp = 
    #     if len(corresp) > num_corresp:
    #         num_corresp = len(corresp)
    #         inlier = corresp
        
    # inliers1, inliers2 = pts1[inlier, :], pts2[inlier, :]
    # F = eightpoint(inliers1, inliers2, M)
            
    # return F, inlier

    # pass

    max_inliers = -1
    # F=np.empty([3,3])
    # r1=np.empty([8,2])
    # r2=np.empty([8,2])
    pts1_homo = np.vstack((pts1.T,np.ones([1,pts1.shape[0]])))
    pts2_homo = np.vstack((pts2.T,np.ones([1,pts2.shape[0]])))
    s = 8 # At least 8 points for F
    
    for i in range(nIters):
        print(i)
        inliers_tot=0
        # idx_rand=np.random.choice(pts1.shape[0],8)
        # r1=pts1[idx_rand,:]
        # r2=pts2[idx_rand,:]
        rn = np.random.choice(n, s, replace=False)
        x1, x2 = pts1[rn, :], pts2[rn, :] 
        F = eightpoint(x1, x2, M)
        epipolar_r = np.dot(F, pts1_homo)
        predt_x2 = epipolar_r/np.sqrt(np.sum(epipolar_r[:2,:]**2, axis=0))        
        err = abs(np.sum(pts2_homo*predt_x2, axis=0))
        n_inliers = err < tol
        inliers_tot = n_inliers[n_inliers.T].shape[0]
        print(inliers_tot)
        if inliers_tot > max_inliers:
            bestF = F
            max_inliers = inliers_tot
            inliers = n_inliers
    # c = 0
    # for r in range(len(inliers)):
    #     if inliers[r] == True:
    #         c += 1
    # Total = len(inliers)
    # print(inliers)
    # print('count', (c/Total)*100)
            
            
    return bestF, inliers

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    # r = r.flatten()
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    K = np.array([[0,-r[-1], r[1]], [r[-1], 0, -r[0]], [-r[1], r[0], 0]])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)
    return R
    # pass

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    # Replace pass by your implementation
    theta = np.arccos((np.trace(R)-1)/2)
    r = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(theta))
    return theta*r
    # pass

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    # M2 = np.concatenate((rodrigues(x[3:6]), x[6:9].reshape((3,1))), axis = 1)
    # P = np.hstack((x[:-6].reshape(p1.shape[0], 3), np.ones((p1.shape[0], 1))))
    # C1 = K1 @ M1
    # C2 = K2 @ M2
    # proj_left_hom = (C1 @ P.T)
    # proj_left = (proj_left_hom/proj_left_hom[-1,:]).reshape(-1, 1)
    # proj_right_hom = (C2 @ P.T)
    # proj_right = (proj_right_hom/proj_right_hom[-1,:]).reshape(-1, 1)
    # residuals = np.concatenate(((p1-proj_left[:, :2]).reshape([-1]), (p2-proj_right[:, :2]).reshape([-1])), axis=0)

    M2 = np.hstack((rodrigues(x[-6:-3]), x[-3:].reshape(3, 1)))
    P = np.hstack((x[:-6].reshape(p1.shape[0], 3), np.ones((p1.shape[0], 1))))

    p1_hat = (K1 @ M1 @ P.T).T
    p2_hat = (K2 @ M2 @ P.T).T
    p1_hat = p1_hat/p1_hat[:, -1].reshape(p1.shape[0], 1)
    p2_hat = p2_hat/p2_hat[:, -1].reshape(p2.shape[0], 1)

    residuals = np.concatenate([(p1-p1_hat[:, :2]).reshape([-1]), (p2-p2_hat[:, :2]).reshape([-1])])
    return residuals
    # pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    R2, t2 = M2_init[:, :3], M2_init[:, 3]
    r2 = invRodrigues(R2)
    x0 = np.concatenate((P_init.flatten(), r2.flatten(), t2.flatten()), axis=0)

    # Optimization Least Squares
    func = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)
    out = scipy.optimize.least_squares(func, x0)

    M2 = np.hstack((rodrigues(out.x[-6:-3]), out.x[-3:].reshape(3, 1)))
    P2 = out.x[:-6].reshape(p1.shape[0], 3)

    # scipy.optimize.least_squares(fun, x0, jac='2-point', bounds=None, method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, jac_sparsity=None, max_nfev=None, verbose=0, args=(), kwargs={})
    # pass
    return M2, P2