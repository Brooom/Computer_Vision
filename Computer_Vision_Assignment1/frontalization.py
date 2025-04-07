import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

from utils import read_image, show_image


# BEGIN YOUR IMPORTS

# END YOUR IMPORTS


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def find_edges(image):
    """
    Args:
        image (np.array): (grayscale) image of shape [H, W]
    Returns:
        edges (np.array): binary mask of shape [H, W]
    """
    # BEGIN YOUR CODE
    edges = cv2.Canny(image,140,150, L2gradient = True)
    
    return edges
    
    # END YOUR CODE
    
    raise NotImplementedError


def highlight_edges(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        highlighted_edges (np.array): binary mask of shape [H, W]
    """
    # BEGIN YOUR CODE
    kernel = np.ones((3, 3), np.uint8) 
    highlited_edges = cv2.dilate(edges, kernel, iterations=2) 
    
    return highlited_edges
    
    # END YOUR CODE

    raise NotImplementedError


def find_contours(edges):
    """
    Args:
        edges (np.array): binary mask of shape [H, W]
    Returns:
        contours (np.array, np.array, ...): tuple of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    """
    # BEGIN YOUR CODE
    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours[0]
    
    # END YOUR CODE

    raise NotImplementedError


def get_max_contour(contours):
    """
    Args:
        contours (np.array, np.array, ...): tuple of arrays of contours, where each contour is an array of points of shape [N, 1, 2]
    Returns:
        max_contour (np.array): an array of points (vertices) of the contour with the maximum area of shape [N, 1, 2]
    """
    # BEGIN YOUR CODE

    max_contour = max(contours, key = cv2.contourArea)
    
    return max_contour
    
    # END YOUR CODE

    raise NotImplementedError


def order_corners(corners):
    """
    Args:
        corners (np.array): an array of corner points (corners) of shape [4, 2]
    Returns:
        ordered_corners (np.array): an array of corner points in order [top left, top right, bottom right, bottom left]
    """
    #print(corners)
    # BEGIN YOUR CODE
    corners = corners[corners[:,0].argsort()]
    left_corners = corners[:len(corners) // 2]
    right_corners = corners[len(corners) // 2:]
    left_corners = left_corners[left_corners[:,1].argsort()]
    right_corners = right_corners[right_corners[:,1].argsort()]
    top_left = left_corners[0]
    top_right = right_corners[0]
    bottom_right = right_corners[-1]
    bottom_left = left_corners[-1]

    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left])
    return ordered_corners
    
    # END YOUR CODE
    
    raise NotImplementedError


def find_corners(contour, epsilon=0.42):
    """
    Args:
        contour (np.array): an array of points (vertices) of the contour of shape [N, 1, 2]
        epsilon (float): how accurate the contour approximation should be
    Returns:
        ordered_corners (np.array): an array of corner points (corners) of quadrilateral approximation of contour of shape [4, 2]
                                    in order [top left, top right, bottom right, bottom left]
    """
    # BEGIN YOUR CODE

    epsilon = epsilon*cv2.arcLength(contour,True)
    corners = cv2.approxPolyDP(contour, epsilon,True)

    if len(corners) != 4:
        corners += np.array([[0, 0],
                              [0, 1],
                              [1, 0],
                              [1, 1]])
        corners = corners[:4]
    ordered_corners = order_corners(corners[:,0,:])
    
    return ordered_corners
    
    # END YOUR CODE

    raise NotImplementedError

from skimage.transform import rescale
def rescale_image(image, scale=0.42):
    """
    Args:
        image (np.array): input image
        scale (float): scale factor
    Returns:
        rescaled_image (np.array): 8-bit (with range [0, 255]) rescaled image
    """
    # BEGIN YOUR CODE

    #rescaled_image = cv2.resize(image, (0,0), fx=scale, fy=scale)
    rescaled_image = rescale(image, scale, preserve_range=True,anti_aliasing=True).astype(np.uint8)
    return rescaled_image
    
    # END YOUR CODE
    
    raise NotImplementedError


def gaussian_blur(image, sigma):
    """
    Args:
        image (np.array): input image
        sigma (float): standard deviation for Gaussian kernel
    Returns:
        blurred_image (np.array): 8-bit (with range [0, 255]) blurred image
    """
    # BEGIN YOUR CODE

    blurred_image = cv2.GaussianBlur(image, (5, 5), 3*sigma)
    
    return blurred_image
    
    # END YOUR CODE

    raise NotImplementedError


def distance(point1, point2):
    """
    Args:
        point1 (np.array): n-dimensional vector
        point2 (np.array): n-dimensional vector
    Returns:
        distance (float): Euclidean distance between point1 and point2
    """
    # BEGIN YOUR CODE

    distance = np.linalg.norm(point1 - point2)

    return distance
    
    # END YOUR CODE

    raise NotImplementedError


def frontalize_image(image, ordered_corners):
    """
    Args:
        image (np.array): input image
        ordered_corners (np.array): corners in order [top left, top right, bottom right, bottom left]
    Returns:
        warped_image (np.array): warped with a perspective transform image of shape [H, H]
    """
    # 4 source points
    top_left, top_right, bottom_right, bottom_left = ordered_corners

    # BEGIN YOUR CODE

    # # the side length of the Sudoku grid based on distances between corners
    side = int(max(distance(top_left, top_right), distance(top_right, bottom_right), distance(top_left, bottom_left), distance(bottom_left, bottom_right)))

    # # what are the 4 target (destination) points?
    destination_points = np.array([[0, 0], [side, 0], [side, side], [0, side]], dtype=np.int32)

    # # perspective transformation matrix
    transform_matrix = cv2.getPerspectiveTransform(ordered_corners.astype(np.float32), destination_points.astype(np.float32))

    # # image warped using the found perspective transformation matrix
    warped_image = cv2.warpPerspective(image, transform_matrix, (side, side))

    assert warped_image.shape[0] == warped_image.shape[1], "height and width of the warped image must be equal"

    return warped_image

    # END YOUR CODE

    raise NotImplementedError


def show_frontalized_images(image_paths, pipeline, figsize=(16, 12)):
    nrows = len(image_paths) // 4 + 1
    ncols = 4
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if len(axes.shape) == 1:
        axes = axes[np.newaxis, ...]

    for j in range(len(image_paths), nrows * ncols):
        axis = axes[j // ncols][j % ncols]
        show_image(np.ones((1, 1, 3)), axis=axis)
    
    for i, image_path in enumerate(tqdm(image_paths)):
        axis = axes[i // ncols][i % ncols]
        axis.set_title(os.path.split(image_path)[1])
        
        sudoku_image = read_image(image_path=image_path)
        frontalized_image, _ = pipeline(sudoku_image)

        show_image(frontalized_image, axis=axis, as_gray=True)

