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

# FUNCTIONS:
# text_to_curve(filename)
#   Create a curve from a text file, where column 1 is X and column 2 is Y.
#
# make_patch(<Curve>, <Curve>)
#   Return a patch constructed from two curves. Takes two curves as inputs.
#
# cdtw(<Curve>, <Curve>, ns, interp, r)
#   Return the distance between two curves, as computed by the Continuous
#   Dynamic Time Warping (CDTW) algorithm. Inputs:
#       - c1: curve 1
#       - c2: curve 2
#       - interp: optional parameter in the range 0-1 that specifies the
#         epsilon value for the Douglas-Peuker curve simplification algorithm.
#         Higher numbers mean higher compression. Default=0.5
#       - ns: optional parameter specifying number of steiner points per edge.
#         A higher value leads to more accurate distance metrics, but also
#         more computational overhead. Default=5
#       - r: Sakoe-Chiba Band width. This limits the search domain to a band of
#         width +/- r around the diagonal. In general, max(dist) ~ 2*r. Default=100

# USAGE:
# In general, a workflow will look like this:
#   c1 = text_to_curve(text1.txt)
#   c2 = text_to_curve(text2.txt)
#   c_dist = cdtw(c1, c2, 5, 0.5, 100)


# ------------------- Helper Functions ------------------- #
# Helper function to make a patch from two curve points
def make_patch(c1, c2, c1_ind, c2_ind):
    return Patch([c2(c2_ind + 1) - c1(c1_ind + 1),
                 c2(c2_ind + 1) - c1(c1_ind),
                 c2(c2_ind) - c1(c1_ind + 1),
                 c2(c2_ind) - c1(c1_ind)])


# Simplify a curve object using the Douglas-Peuker algorithm
def simplify_curve(curve, eps):
    """Simplify a curve using the Douglas-Peuker algorithm.
    Inputs are a Curve object, and eps, the tolerance.

    A higher eps results in higher compression, but lower quality."""
    if eps < 0:
        raise ValueError('eps must be a nonzero number.')

    d_max = 0
    index = 0
    for i in range(1, len(curve)):
        d = line_dist(curve[i], curve[0], curve[-1])
        if d > d_max:
            index = i
            d_max = d

    if d_max > eps:
        rec1 = simplify_curve(Curve(curve[:index + 1]), eps)
        rec2 = simplify_curve(Curve(curve[index:]), eps)

        return Curve(rec1[:-1]) + rec2
    else:
        return Curve([curve[0], curve[-1]])


# Perpendicular distance from node to line
def line_dist(point, line_st, line_end):
    """Calculate the perpendicular distance between Node 'point' and the line
    defined by the nodes line_st and line_end"""
    x2 = np.array([line_end.x, line_end.y])
    x1 = np.array([line_st.x, line_st.y])
    x0 = np.array([point.x, point.y])
    return np.divide(np.linalg.norm(np.linalg.det([x2 - x1, x1 - x0])),
                     np.linalg.norm(x2 - x1))


# Create a Curve object from a text file containing 2 rows.
# First column is assumed to be X values, second is Y.
def text_to_curve(filename):
    c = Curve([])
    with open(filename, 'r') as f:
        next(f)  # skip header line
        for line in f:
            n = Node(line.split()[0], line.split()[1])
            c.add_node(n)
    return c


# Perform CDTW on c1 and c2
def cdtw(c1, c2, num_steiner=5, interp=0.3, r=100):
    """Perform CDTW on two input curves.
       INPUTS (Defaults):
        * c1: curve 1
        * c2: curve 2
        * num_steiner (5): number of steiners per edge
        * interp (0.3): Douglas-Peuker epsilon value
        * r (100): Sakoe-Chiba Band width"""

    if (not isinstance(c1, Curve)) | (not isinstance(c2, Curve)):
        raise ValueError('cdtw takes 2 curves as inputs')

    if not isinstance(num_steiner, int):
        raise ValueError('num_steiner (input 3) must be an integer')

    if not isinstance(interp, numbers.Number):
        raise ValueError('interp (input 4) input must be a number')

    if not isinstance(num_steiner, int):
        raise ValueError('r (input 5) must be an integer')

    if interp > 0:
        c1 = simplify_curve(c1, interp)
        c2 = simplify_curve(c2, interp)

    # initialize loop variables
    w = len(c1) - 1  # width of the graph
    h = len(c2) - 1  # height of the graph
    bot_mat = np.zeros((w, num_steiner+2))  # matrix to hold the previous row's distances
    cur = []
    rgt = []

    # form the sakoe-chiba band
    scb = np.zeros((h, w))
    for i in range(0, h):
        for j in range(0, w):
            if (j-r <= i) & (j+r >= i):
                scb[i, j] = 1

    # loop through patches from bottom right to top left along row
    for i in range(h - 1, -1, -1):

        for j in range(w - 1, -1, -1):

            # if we are outside of the SCB, we can skip this element
            if scb[i, j] == 0:
                bot_mat[j][0:num_steiner + 2] = np.inf
                continue

            # make current patch with steiners
            cur = make_patch(c1, c2, w - 1 - j, h - 1 - i)
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
            elif scb[i, j + 1] == 0:
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

    # final distance is the top left node distance
    return cur.tl.distance
