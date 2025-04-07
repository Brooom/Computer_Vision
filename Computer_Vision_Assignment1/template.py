from pipeline import Pipeline

# BEGIN YOUR IMPORTS
from const import SUDOKU_SIZE

from recognition import resize_image, get_sudoku_cells

from frontalization import rescale_image, frontalize_image, show_frontalized_images, gaussian_blur, find_edges, highlight_edges, find_contours, get_max_contour, find_corners
# END YOUR IMPORTS

# BEGIN YOUR CODE

"""
create dict of cell coordinates like in this example

CELL_COORDINATES = {"image_0.jpg": {1: (0, 0),
                                    2: (1, 1)},
                    "image_2.jpg": {1: (2, 3),
                                    3: [(2, 1), (0, 4)],
                                    9: (5, 6)}}
"""

CELL_COORDINATES = {"image_4.jpg": {1: (3, 0),
                                    2: (1, 0),
                                    3: (2, 1),
                                    4: (8, 4),
                                    5: (1, 6),
                                    6: (0, 2),
                                    7: (1, 2),
                                    8: (7, 3),
                                    9: (1, 4)},
                    "image_7.jpg": {1: (0, 0),
                                    2: (4, 5),
                                    3: (0, 8),
                                    4: (8, 1),
                                    5: (2, 2),
                                    6: (1, 4),
                                    7: (5, 3),
                                    8: (0, 1),
                                    9: (3, 4)}}

# END YOUR CODE


# BEGIN YOUR FUNCTIONS

# END YOUR FUNCTIONS


def get_template_pipeline():
    # BEGIN YOUR CODE

    pipeline = Pipeline(functions=[rescale_image, find_edges, highlight_edges, find_contours, get_max_contour, find_corners, frontalize_image,
                               resize_image, get_sudoku_cells],
                    parameters={"rescale_image": {"scale": 0.4},
                                "find_corners": {"epsilon": 0.1},
                                "resize_image": {"size": SUDOKU_SIZE},
                                "get_sudoku_cells": {"crop_factor": 0.6, 
                                "binarization_kwargs": {}}
                               })
    
    return pipeline

    # END YOUR CODE

    raise NotImplementedError
