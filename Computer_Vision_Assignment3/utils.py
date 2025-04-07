import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr


def get_normalization_matrix(x):
    """
    get_normalization_matrix Returns the transformation matrix used to normalize
    the inputs x
    Normalization corresponds to subtracting mean-position and positions
    have a mean distance of sqrt(2) to the center
    """
    # Input: x 3*N
    #
    # Output: T 3x3 transformation matrix of points

    # TO DO TASK:
    # --------------------------------------------------------------
    # Estimate transformation matrix used to normalize
    # the inputs x
    # --------------------------------------------------------------

    # Get centroid and mean-distance to centroid
    centroid = np.mean(x, axis=1).reshape(-1,1)
    mean_distance = np.mean(np.linalg.norm(x - centroid,axis=0))
    T = np.zeros((3, 3))
    T[0, 0] = np.sqrt(2) / mean_distance
    T[1, 1] = np.sqrt(2) / mean_distance
    T[0, 2] = -centroid[0] * np.sqrt(2) / mean_distance
    T[1, 2] = -centroid[1] * np.sqrt(2) / mean_distance
    T[2, 2] = 1
    return T


def eight_points_algorithm(x1, x2, normalize=True):
    """
    Calculates the fundamental matrix between two views using the normalized 8 point algorithm.
    Inputs:
        x1      3xN     homogeneous coordinates of matched points in view 1
        x2      3xN     homogeneous coordinates of matched points in view 2
    Outputs:
        F       3x3     fundamental matrix
    """
    N = x1.shape[1]
    norm_mat_x1 = np.eye(3)
    norm_mat_x2 = np.eye(3)

    if normalize:
        # Normalize inputs
        norm_mat_x1 = get_normalization_matrix(x1)
        norm_mat_x2 = get_normalization_matrix(x2)
        norm_x1 = norm_mat_x1 @ x1
        norm_x2 = np.dot(norm_mat_x2, x2)
    else:
        norm_x1 = x1
        norm_x2 = x2

    # Construct matrix A encoding the constraints on x1 and x2
    A = np.zeros((N, 9))
    for i in range(N):
        x1i = norm_x1[:, i]
        x2i = norm_x2[:, i]
        A[i] = [
            x2i[0]*x1i[0], x2i[0]*x1i[1], x2i[0],
            x2i[1]*x1i[0], x2i[1]*x1i[1], x2i[1],
            x1i[0],        x1i[1],        1
        ]


    # Solve for F using SVD
    _, _, Vt = np.linalg.svd(A)
    F_vec = Vt[-1,:]  # The last row of Vt contains the solution
    F = F_vec.reshape(3, 3)


    # Enforce that rank(F) = 2
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # Set the smallest singular value to 0
    F = U @ np.diag(S) @ Vt

    if normalize:
        # Transform F back to original coordinates
        F = norm_mat_x2.T @ F @ norm_mat_x1
    F = F / F[2, 2]
    return F

def right_epipole(F):
    """
    Computes the (right) epipole from a fundamental matrix F.
    (Use with F.T for left epipole.)
    """

    # The epipole is the null space of F (F * e = 0)
    # TODO
    # Perform Singular Value Decomposition (SVD) of the fundamental matrix F
    _, _, Vt = np.linalg.svd(F)
    
    # The right epipole is the last column of Vt (or the last row of V)
    e = Vt[-1]
    e = e / e[-1]
    return e


def plot_epipolar_line(im, F, x, e):
    """
    Plot the epipole and epipolar line F*x=0 in an image. F is the fundamental matrix
    and x a point in the other image.
    """
    m, n = im.shape[:2]
    # TODO
    # Compute the epipolar line l = Fx
    l = F @ x
    # Compute two points on the line
    p1 = np.array([0, -l[2] / l[1]])
    p2 = np.array([n, -(l[0] * n + l[2]) / l[1]])
    # Plot the epipolar line
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]])


def ransac(x1, x2, threshold, num_steps=10000, random_seed=42):
    if random_seed is not None:
        np.random.seed(random_seed)  # we are using a random seed to make the results reproducible
    # TODO setup variables
    F = np.zeros((3, 3))
    best_inliers = np.zeros(x1.shape[1], dtype=bool)
    for _ in range(num_steps):
        rand = np.random.choice(x1.shape[1], 8, replace=False)
        F_candidate = eight_points_algorithm(x1[:,rand], x2[:, rand], normalize=True)
        epipolar_lines_r = F_candidate @ x1
        epipolar_lines_l = F_candidate.T @ x2
        error_r = np.abs(np.sum(x2*epipolar_lines_r, axis=0))
        error_l = np.abs(np.sum(x1*epipolar_lines_l, axis=0))
        inliers = (error_r <= threshold) * (error_l <= threshold)
        
        if np.sum(inliers) > np.sum(best_inliers):
            F = F_candidate.copy()
            best_inliers = inliers.copy()
    
    return F, inliers  # F is estimated fundamental matrix and inliers is an indicator (boolean) numpy array



def decompose_essential_matrix(E, x1, x2):
    """
    Decomposes E into a rotation and translation matrix using the
    normalized corresponding points x1 and x2.
    """

    # Fix left camera-matrix
    Rl = np.eye(3)
    tl = np.array([[0, 0, 0]]).T
    Pl = np.concatenate((Rl, tl), axis=1)

    # TODO: Compute possible rotations and translations
    # Compute SVD of E
    U, S, Vt = np.linalg.svd(E)
    print(S)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]
    print(R1.shape, t1.shape)
    # Four possibilities
    Pr = [np.concatenate((R1, t1.reshape(-1, 1)), axis=1),
          np.concatenate((R1, t2.reshape(-1, 1)), axis=1),
          np.concatenate((R2, t1.reshape(-1, 1)), axis=1),
          np.concatenate((R2, t2.reshape(-1, 1)), axis=1)]

    # Compute reconstructions for all possible right camera-matrices
    X3Ds = [infer_3d(x1[:, 0:1], x2[:, 0:1], Pl, x) for x in Pr]

    # Compute projections on image-planes and find when both cameras see point
    test = [np.prod(np.hstack((Pl @ np.vstack((X3Ds[i], [[1]])), Pr[i] @ np.vstack((X3Ds[i], [[1]])))) > 0, 1) for i in
            range(4)]
    test = np.array(test)
    idx = np.where(np.hstack((test[0, 2], test[1, 2], test[2, 2], test[3, 2])) > 0.)[0][0]

    # Choose correct matrix
    Pr = Pr[idx]

    return Pl, Pr


def infer_3d(x1, x2, Pl, Pr):
    # INFER3D Infers 3d-positions of the point-correspondences x1 and x2, using
    # the rotation matrices Rl, Rr and translation vectors tl, tr. Using a
    # least-squares approach.

    M = x1.shape[1]
    # Extract rotation and translation
    Rl = Pl[:3, :3]
    tl = Pl[:3, 3]
    Rr = Pr[:3, :3]
    tr = Pr[:3, 3]

    # Construct matrix A with constraints on 3d points
    row_idx = np.tile(np.arange(4 * M), (3, 1)).T.reshape(-1)
    col_idx = np.tile(np.arange(3 * M), (1, 4)).reshape(-1)

    A = np.zeros((4 * M, 3))
    A[:M, :3] = x1[0:1, :].T @ Rl[2:3, :] - np.tile(Rl[0:1, :], (M, 1))
    A[M:2 * M, :3] = x1[1:2, :].T @ Rl[2:3, :] - np.tile(Rl[1:2, :], (M, 1))
    A[2 * M:3 * M, :3] = x2[0:1, :].T @ Rr[2:3, :] - np.tile(Rr[0:1, :], (M, 1))
    A[3 * M:4 * M, :3] = x2[1:2, :].T @ Rr[2:3, :] - np.tile(Rr[1:2, :], (M, 1))

    A = csr_matrix((A.reshape(-1), (row_idx, col_idx)), shape=(4 * M, 3 * M))

    # Construct vector b
    b = np.zeros((4 * M, 1))
    b[:M] = np.tile(tl[0], (M, 1)) - x1[0:1, :].T * tl[2]
    b[M:2 * M] = np.tile(tl[1], (M, 1)) - x1[1:2, :].T * tl[2]
    b[2 * M:3 * M] = np.tile(tr[0], (M, 1)) - x2[0:1, :].T * tr[2]
    b[3 * M:4 * M] = np.tile(tr[1], (M, 1)) - x2[1:2, :].T * tr[2]

    # Solve for 3d-positions in a least-squares way
    w = lsmr(A, b)[0]
    x3d = w.reshape(M, 3).T

    return x3d


