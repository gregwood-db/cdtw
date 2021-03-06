# IMPORT PACKAGES
import numpy as np
import numbers
from cdtw_classes import Node, Curve, Patch

# DESCRIPTION:
#
# A package for measuring the distance between curves using CDTW.
#
# The cdtw_noplot package includes functions to calculate a manifold between two curves, then use CDTW to calculate the
# distance across this manifold. Note that this package contains only methods for computing the distance between two
# curves; if visualization is required, use cdtw_plot instead. This version of the algorithm contains optimizations to
# avoid computing and storing the entire manifold. As a result, this version is ~O(n).
#
# In general, this code requires as inputs text files with two columns, representing the X and Y coordinates of the
# curves to be compared. More formats will be added in the future. Once the files are converted using text_to_curve(),
# they can be fed into the classes and functions detailed below.
#
# FUNCTIONS:
# text_to_curve(filename)
#   Create a curve from a text file, where column 1 is X and column 2 is Y.
#
# df_to_curve(filename)
#   Create a curve from a pandas or pyspark dataframe.
#
# list_to_curve(filename)
#   Create a curve from a list of numbers representing X and Y.
#
# list_to_curve(filename)
#   Create a curve from a numpy array representing X and Y.
#
# cdtw(c1, c2, ns, interp, r)
#   Return the distance between two curves, as computed by the Continuous
#   Dynamic Time Warping (CDTW) algorithm.
#
# cdtw_fast(c1, c2, interp, num_steiner, radius, rounds)
#   Return the distance between two curves, as computed by the fast Continuous
#   Dynamic Time Warping (CDTW) algorithm. This is an optimized version of cdtw.
#
# USAGE:
# In general, a workflow will look like this:
#   c1 = text_to_curve(text1.txt)
#   c2 = text_to_curve(text2.txt)
#   c_dist = cdtw(c1, c2)


# ------------------- Helper Functions ------------------- #
# Helper function to make a patch from two curve points
def __make_patch(c1, c2, c1_ind, c2_ind):
    return Patch([c2(c2_ind + 1) - c1(c1_ind + 1),
                 c2(c2_ind + 1) - c1(c1_ind),
                 c2(c2_ind) - c1(c1_ind + 1),
                 c2(c2_ind) - c1(c1_ind)])


# Simplify a curve object using the Douglas-Peuker algorithm
def __simplify_curve(curve, eps):
    """Simplify a curve using the Douglas-Peuker algorithm.
    Inputs are a Curve object, and eps, the tolerance.

    A higher eps results in higher compression, but lower quality."""
    if eps < 0:
        raise ValueError('eps must be a nonzero number.')

    d_max = 0
    index = 0
    for i in range(1, len(curve)):
        d = __line_dist(curve[i], curve[0], curve[-1])
        if d > d_max:
            index = i
            d_max = d

    if d_max > eps:
        rec1 = __simplify_curve(Curve(curve[:index + 1]), eps)
        rec2 = __simplify_curve(Curve(curve[index:]), eps)

        return Curve(rec1[:-1]) + rec2
    else:
        return Curve([curve[0], curve[-1]])


# Perpendicular distance from node to line
def __line_dist(point, line_st, line_end):
    """Calculate the perpendicular distance between Node 'point' and the line
    defined by the nodes line_st and line_end"""
    x2 = np.array([line_end.x, line_end.y])
    x1 = np.array([line_st.x, line_st.y])
    x0 = np.array([point.x, point.y])
    return np.divide(np.linalg.norm(np.linalg.det([x2 - x1, x1 - x0])),
                     np.linalg.norm(x2 - x1))


# Make a warping path based on the distance map from a CDTW operation. This takes the distance calculated to each
# patch's top-left node, and stores it in a matrix. This can also be visualized to spot-check the warping performance.
def __make_path(dist_map):

    # initialize variables
    h = max([x[0] for x in dist_map.keys()])
    w = max([x[1] for x in dist_map.keys()])
    i = 0
    j = 0
    path = [(0, 0)]

    # loop from (0,0) to (h,w)
    while i < h and j < w:

        # edge cases: at the top or bottom of the map
        if i + 1 >= h:
            next_node = (i, j+1)
        elif j + 1 >= w:
            next_node = (i+1, j)
        else:
            # only possible next nodes are to the right, bottom, and bottom-right
            poss_nodes = [(i+1, j), (i, j+1), (i+1, j+1)]
            node_dists = [dist_map.get((i+1, j), np.inf), dist_map.get((i, j+1), np.inf),
                          dist_map.get((i+1, j+1), np.inf)]

            # find the min distance to the next node and then append
            node_dict = dict(zip(node_dists, poss_nodes))
            next_node = node_dict[min(node_dict)]

        path.append(next_node)

        i = next_node[0]
        j = next_node[1]

    return path


# Project a warping path onto a higher-resolution space, and return the mask based on this path and the radius r. This
# function will return a matrix of size 2*m, 2*n, where m and n are the lengths of the two curves used to
# generate the path that is the input. Using a larger r will results in a more accurate warp but more computation.
def __project_path(path, x_size, y_size, r):

    # form the band to hold the logical map
    mask = np.zeros((2, y_size*2))
    mask[1, :] = np.inf

    # walk through the path and set elements of new_band to 1 according to path
    for (x, y) in path:

        if 2 * x - r < 0:
            x_start = 0
            x_end = 2 * x + r
        elif 2 * x + r > 2 * x_size:
            x_start = 2 * x - r
            x_end = 2 * x_size
        else:
            x_start = 2 * x - r
            x_end = 2 * x + r

        if 2 * y - r < 0:
            y_start = 0
            y_end = 2 * y + r
        elif 2 * y + r > 2 * y_size:
            y_start = 2 * y - r
            y_end = 2 * y_size
        else:
            y_start = 2 * y - r
            y_end = 2 * y + r

        for i in range(y_start, y_end):
            if mask[0, i] < x_end:
                mask[0, i] = x_end

            if mask[1, i] > x_start:
                mask[1, i] = x_start

    return mask


# Internal function that builds the compacted curve objects, then walks through them in order. It starts from the most
# compact curve (last in the list), and uses the path from this to generate the valid warping mask for the next most
# compact curve, and so on, until it reaches the original input.
def __cdtw_fast(c1, c2, radius, rounds, num_steiner):

    curve_rounds = [(c1, c2)]
    dist_map = None
    d = None
    min_size = 25

    for r in range(1, rounds):
        curve_rounds.append((curve_rounds[r-1][0].halve(), curve_rounds[r-1][1].halve()))

    for r in range(rounds-1, -1, -1):

        curr_c1 = curve_rounds[r][0]
        curr_c2 = curve_rounds[r][1]
        h = len(curr_c2)
        w = len(curr_c1)

        if w < min_size or h < min_size:
            mask = np.zeros((2, w))
            mask[0, :] = h-1
            d, dist_map = _cdtw(curr_c1, curr_c2, mask=mask)
        elif dist_map is None:
            mask = np.zeros((2, w))
            mask[0, :] = h - 1
            d, dist_map = _cdtw(curr_c1, curr_c2, mask=mask)
        else:
            path = __make_path(dist_map)
            mask = __project_path(path, len(curve_rounds[r+1][1]), len(curve_rounds[r+1][0]), radius)
            d, dist_map = _cdtw(curr_c1, curr_c2, mask=mask, num_steiner=num_steiner)

    return d, dist_map


def _cdtw(c1, c2, mask, num_steiner=5):
    """
    Perform CDTW on two input curves. This is an internal function only.

    Take two curves as arguments and perform CDTW on them. This will compute the warping distance for any valid node
    (i.e., nodes that are included in the mask matrix).

    Parameters:
    c1 (cdtw.Curve): the first curve to perform CDTW on.
    c2 (cdtw.Curve): the second curve to perform CDTW on.
    mask (np.ndarray): a 2*n array that holds the upper and lower bounds for the valid warping region.
    num_steriners(int): the number of interpolating points per edge in the manifold. Higher is more accurate.
    r(int): the width of the Sakoe-Chiba band. Higher is more accurate.

    Returns:
    dist (float): the distance between the two curves.
    dist_map (dict): a dict containing the warping distance for each node, i.e., {(i,j): dist}
    """

    if (not isinstance(c1, Curve)) | (not isinstance(c2, Curve)):
        raise ValueError('cdtw takes 2 curves as inputs')

    # initialize loop variables
    w = len(c1) - 1  # width of the graph
    h = len(c2) - 1  # height of the graph
    bot_mat = np.zeros((w, num_steiner+2))  # matrix to hold the previous row's distances
    cur = []
    rgt = []

    # hold dist for each valid patch; used for projections to more compact space
    dist_map = {}

    # loop through patches from bottom right to top left along row
    for i in range(h - 1, -1, -1):

        for j in range(w - 1, -1, -1):

            # if we are outside of the provided mask, we can skip this element
            mask_upper = mask[0, j]
            mask_lower = mask[1, j]
            if i < mask_lower or i > mask_upper:
                bot_mat[j][0:num_steiner + 2] = np.inf
                continue

            # make current patch with steiners
            cur = __make_patch(c1, c2, w - 1 - j, h - 1 - i)
            cur.add_steiners('even', num_steiner)

            # handle edge cases on right and bottom edges
            # on first node, initialize bottom right as 0
            if (i == h-1) & (j == w-1):
                cur.br.distance = 0
                cur.initialize_dist()
                rgt = cur  # set right edge for next iteration

            # case where we are on bottom row
            elif i == h-1:
                # current right is former left
                for n in range(0, num_steiner+2):
                    cur.right[n].distance = rgt.left[n].distance
                    cur.right[n].visited = True

                cur.initialize_dist()  # initialize nodes that are still 0
                rgt = cur

            # case where we are on right edge
            elif j == w-1:
                # set current bottom to previous top
                for n in range(0, num_steiner + 2):
                    cur.bottom[n].distance = bot_mat[j][n]
                    cur.bottom[n].visited = True

                cur.initialize_dist()
                rgt = cur

            # case where we are on the edge of the SCB
            elif i > mask[0, j+1] or i < mask[1, j+1]:
                # set current bottom to previous top
                for n in range(0, num_steiner + 2):
                    cur.bottom[n].distance = bot_mat[j][n]
                    cur.bottom[n].visited = True

                cur.br.distance = cur.bottom[-1].distance
                cur.initialize_dist()
                rgt = cur

            # case where we are in the middle of the graph
            else:
                for n in range(0, num_steiner+2):
                    cur.bottom[n].distance = bot_mat[j][n]
                    cur.bottom[n].visited = True

                for n in range(0, num_steiner+2):
                    cur.right[n].distance = rgt.left[n].distance
                    cur.right[n].visited = True

                rgt = cur

            # set the distance for left/top nodes of the current patch
            cur.set_distance()

            for n in range(0, num_steiner + 2):
                bot_mat[j][n] = cur.top[n].distance

            # update the dist_map dict with the current node's distance
            dist_map[(i, j)] = cur.tl.distance

    # final distance is the top left node distance
    return cur.tl.distance, dist_map


# ------------------- Public Functions ------------------- #

# Create a Curve object from a text file containing 2 rows.
# First column is assumed to be X values, second is Y.
def text_to_curve(filename):
    """
    Convert a text file to a cdtw.Curve object.

    Creates a new Curve object from a text file containing (x,y) coordinates. The file should be structured as two
    space-separated columns; the first contains the x-coordinates, and the second contains the y-coordinates. See
    /sample-data for examples.

    Parameters:
    filename (string): the text file containing the (x,y) coordinates.

    Returns:
    c (cdtw.Curve): the curve created from the text file.
    """

    c = Curve([])
    with open(filename, 'r') as f:
        next(f)  # skip header line
        for line in f:
            n = Node(line.split()[0], line.split()[1])
            c.add_node(n)
    return c


def df_to_curve(df):
    """
    Convert a pandas or pyspark dataframe to a cdtw.Curve object.

    Creates a new Curve object from a dataframe containing (x,y) coordinates. The dataframe should be structured as two
    columns; the first contains the x-coordinates, and the second contains the y-coordinates.

    Parameters:
    df (dataframe): the dataframe containing the (x,y) coordinates. If df is a pandas dataframe, the first two columns
    will be used as x and y, respectively. If df is a pyspark dataframe, the x and y columns must be named.

    Returns:
    c (cdtw.Curve): the curve created from the text file.
    """

    df_type = type(df).__module__ + '.' + type(df).__name__

    # check DF type; must be pandas or pyspark
    if df_type == "pandas.core.frame.DataFrame":
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]
    elif df_type == "pyspark.sql.dataframe.DataFrame":
        x = df.select('x').rdd.map(lambda i: i[0]).collect()
        y = df.select('y').rdd.map(lambda i: i[0]).collect()
    else:
        raise ValueError("Dataframe type must be either \
                        'pandas.core.frame.DataFrame' or 'pyspark.sql.dataframe.DataFrame'")

    # enforce all x,y coords are numbers
    if not all(isinstance(i, numbers.Number) for i in x):
        raise ValueError("All x-values must be numbers.")
    elif not all(isinstance(i, numbers.Number) for i in y):
        raise ValueError("All y-values must be numbers.")

    # create the curve object
    c = Curve([])

    for x, y in zip(x, y):
        n = Node(x, y)
        c.add_node(n)

    return c


def list_to_curve(*args):
    """
    Convert a list of numbers to a cdtw.Curve object.

    Creates a new Curve object from a list containing (x,y) coordinates. This function can either take two lists (ie
    separate x and y) or a single array where the first list is x and the second list is y.

    Parameters:
    x (list): the list containing x coordinates (floats)
    y (list): the list containing y coordinates (floats)
     - OR -
    xy_list (list): the array containing the (x,y) coordinates (floats).

    Returns:
    c (cdtw.Curve): the curve created from the text file.
    """

    # enforce all inputs are lists
    if not all(isinstance(i, list) for i in args):
        raise ValueError("All inputs must be lists.")

    len_args = len(args)

    # check DF type; must be pandas or pyspark
    if len_args == 2:
        x = args[0]
        y = args[1]
    elif len_args == 1:
        x = args[0][0]
        y = args[0][1]
    else:
        raise ValueError("Input must either be two lists of x and y, or a single list containing both.")

    # enforce all x,y coords are numbers
    if not all(isinstance(i, numbers.Number) for i in x):
        raise ValueError("All x-values must be numbers.")
    elif not all(isinstance(i, numbers.Number) for i in y):
        raise ValueError("All y-values must be numbers.")

    # create the curve object
    c = Curve([])

    for x, y in zip(x, y):
        n = Node(x, y)
        c.add_node(n)

    return c


def array_to_curve(*args):
    """
    Convert a numpy array to a cdtw.Curve object.

    Creates a new Curve object from a np.ndarray containing (x,y) coordinates. This function can either take two arrays
    (ie separate x and y) or a single array where the first column is x and the second column is y.

    Parameters:
    x (np.ndarray): the array containing x coordinates (floats)
    y (np.ndarray): the array containing y coordinates (floats)
     - OR -
    xy_list (np.ndarray): the array containing the (x,y) coordinates (floats).

    Returns:
    c (cdtw.Curve): the curve created from the text file.
    """

    # enforce all inputs are lists
    if not all(isinstance(i, np.ndarray) for i in args):
        raise ValueError("All inputs must be lists.")

    len_args = len(args)

    # check DF type; must be pandas or pyspark
    if len_args == 2:
        x = args[0]
        y = args[1]
    elif len_args == 1:
        x = args[0][:, 0]
        y = args[0][:, 1]
    else:
        raise ValueError("Input must either be two lists of x and y, or a single list containing both.")

    # enforce all x,y coords are numbers
    if not all(isinstance(i, numbers.Number) for i in x):
        raise ValueError("All x-values must be numbers.")
    elif not all(isinstance(i, numbers.Number) for i in y):
        raise ValueError("All y-values must be numbers.")

    # create the curve object
    c = Curve([])

    for x, y in zip(x, y):
        n = Node(x, y)
        c.add_node(n)

    return c


# Main function to perform standard CDTW. This is a wrapper to _cdtw.
def cdtw(c1, c2, interp=0.3, num_steiner=5, r=100):
    """
    Perform CDTW on two input curves.

    Take two curves as arguments and perform standard CDTW on them. This uses a slightly optimized method that employs
    curve interpolation (using the interp parameter) and warping truncaction (using Sakoe-Chiba bands with width
    controlled by the r parameter).

    Parameters:
    c1 (cdtw.Curve): the first curve to perform CDTW on.
    c2 (cdtw.Curve): the second curve to perform CDTW on.
    interp (float): the interpolation factor for the cuves. Higher is more compressed.
    num_steriners(int): the number of interpolating points per edge in the manifold. Higher is more accurate.
    r(int): the width of the Sakoe-Chiba band. Higher is more accurate.

    Returns:
    dist (float): the distance between the two curves.
    """

    if not isinstance(interp, numbers.Number) | (interp < 0):
        raise ValueError('interp must be a non-negative number')

    if not isinstance(num_steiner, int):
        raise ValueError('num_steiner must be an integer')

    if not isinstance(r, int):
        raise ValueError('r must be an integer')

    if interp > 0:
        c1 = __simplify_curve(c1, interp)
        c2 = __simplify_curve(c2, interp)

    # form the sakoe-chiba band
    h = len(c2)
    w = len(c1)
    if r == 0:
        scb = np.zeros((2, w))
        scb[0, :] = h
    else:
        scb = np.zeros((2, w))
        scale = h / w
        for i in range(0, w):
            h_fill_center = int(np.ceil(i * scale))
            h_fill_upper = min(h, h_fill_center + r)
            h_fill_lower = max(0, h_fill_center - r)
            scb[0, i] = h_fill_upper
            scb[1, i] = h_fill_lower

    dist, dist_map = _cdtw(c1, c2, mask=scb, num_steiner=num_steiner)

    return dist


# Main function to perform fastCDTW. This is a wrapper to __cdtw_fast.
def cdtw_fast(c1, c2, interp=0.3, num_steiner=5, radius=10, rounds=4):
    """
    Perform fast CDTW on two input curves.

    Take two curves as arguments and perform fast CDTW on them. This is a highly optimized version of CDTW that is
    usually about as accurate as standard cdtw, but runs in O(n) instead of O(n^2) time. The accuracy can be controlled
    primarily via the radius parameter; a higher value will tend towards the full CDTW calculation, while smaller values
    reduce the number of cells calculated in the manifold.

    Parameters:
    c1 (cdtw.Curve): the first curve to perform CDTW on.
    c2 (cdtw.Curve): the second curve to perform CDTW on.
    interp (float): the interpolation factor for the curves. Higher is more compressed.
    num_steriners(int): the number of interpolating points per edge in the manifold. Higher is more accurate.
    radius (int): the width of the path projection. Higher is more accurate.
    rounds (int): the number of compaction-projection rounds to be performed. Higher can be more better for large data.

    Returns:
    dist (float): the distance between the two curves.
    """

    if not isinstance(interp, numbers.Number) | (interp < 0):
        raise ValueError('interp must be a non-negative number')

    if not isinstance(num_steiner, int):
        raise ValueError('num_steiner must be an integer')

    if not isinstance(radius, int):
        raise ValueError('radius must be an integer')

    if not isinstance(rounds, int):
        raise ValueError('rounds must be an integer')

    if interp > 0:
        c1 = __simplify_curve(c1, interp)
        c2 = __simplify_curve(c2, interp)

    dist, dist_map = __cdtw_fast(c1, c2, radius, rounds, num_steiner)

    return dist
