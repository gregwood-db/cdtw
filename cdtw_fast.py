# IMPORT PACKAGES
import numpy as np

# DESCRIPTION:
#
# A package for measuring the distance between curves using CDTW.
#
# The cdtw_fast package includes functions to calculate a manifold between two curves, then use CDTW to calculate the
# distance across this manifold. Note that this package contains only methods for computing the distance between two
# curves; if visualization is required, use cdtw_plot instead. This version of the algorithm contains optimizations to
# avoid computing and storing the entire manifold by using methods similar to the fastDTW algorithm. As a result,
# this version is ~O(n).
#
# In general, this code requires as inputs text files with two columns, representing the X and Y coordinates of the
# curves to be compared. More formats will be added in the future. Once the files are converted using text_to_curve(),
# they can be fed into the classes and functions detailed below.

# CLASSES:
#   Curve([Node1, Node2, Node3, ..., NodeN])
#       Represents a 2D, N-length curve made up of Nodes.
#       F--- add_node(Node), show()
#
#   Node(x, y)
#       Represents a point in euclidean space. Makes up patches and curves.
#       F--- show(), dist(Node)
#
#   Patch([NodeTL, NodeTR, NodeBL, NodeBR])
#       Contains 4 native points and potentially more "steiner" points added as interpolative elements.
#       These are the basis of a graph.
#       F--- add_node(Node), reset_nodes(), refresh_nodes(), set_distance(),
#            initialize_dist(), add_steiners(method, num)

# FUNCTIONS:
# text_to_curve(filename)
#   Create a curve from a text file, where column 1 is X and column 2 is Y.
#
# fast_cdtw(<Curve>, <Curve>, radius, interp, rounds)
#   Return the distance between two curves, as computed by the fast Continuous
#   Dynamic Time Warping (CDTW) algorithm. Inputs:
#       - c1: curve 1
#       - c2: curve 2
#       - radius: The neighborhood to use around the path obtained from the previous warping. Must be a positive int.
#                 A higher value will be more accurate but take longer.
#       - interp: The amount of interpolation to perform using Douglas-Peuker. Must be >= 0.
#                 A lower value will be more accurate but more computationally intense.
#       - rounds: The number of rounds to use when performing compaction-projection.
#                 A higher number may be beneficial when computing large timeseries.

# USAGE:
# In general, a workflow will look like this:
#   c1 = text_to_curve(text1.txt)
#   c2 = text_to_curve(text2.txt)
#   c_dist = cdtw(c1, c2, 5, 0.5, 100)


class Node:
    def __init__(self, x, y, node_type='native'):
        # initial distance should be 0
        self.distance = 0

        # node type, steiner or native
        self.type = node_type

        # Location of node (x,y)
        try:
            self.x = float(x)
            self.y = float(y)
        except ValueError:
            raise ValueError('Coordinates must be numbers.')

        # Visit status, T/F
        self.visited = False

        # Unique node ID
        self.id = id(self)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        repr_str = 'node with id %d' % self.id
        return repr_str

    def __add__(self, other):
        # return node object adding x and y
        return Node(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        # return node object subtracting x and y
        return Node(self.x - other.x, self.y - other.y)

    def show(self):
        print('Node with properties:')
        print('Location: (%d,%d)' % (self.x, self.y))
        print('Distance: %d' % self.distance)
        print('Node type: %s' % self.type)

    def dist(self, other):
        """Return euclidean distance FROM self TO other.
           Override this for some more interesting systems."""
        return ((self.x - other.x) ** 2
                + (self.y - other.y) ** 2) ** 0.5


class Curve:
    def __init__(self, *args):
        # Curve objects can be initiated one of two ways:
        #   -pass in a list of pre-initialized nodes, i.e [n1, n2, n3,...]
        #   -pass in two lists of coordinates, x and y, from which nodes will be created
        # In the latter case, make sure x and y have equal length.
        self.nodeList = []

        if len(args) == 2:
            x_list = args[0]
            y_list = args[1]

            if len(x_list) != len(y_list):
                raise ValueError("List of x and y points must be the same length.")

            for x, y in zip(x_list, y_list):
                self.add_node(Node(x, y))
        else:
            for k in args[0]:
                if not isinstance(k, Node):
                    raise ValueError('Tried to add a non-node to nodeList in curve initialization.')
                self.add_node(k)

    def __getitem__(self, item):
        return self.nodeList[item]

    def __call__(self, item):
        if not isinstance(item, int):
            raise ValueError('Non-integer nodeList reference.')
        return self.nodeList[item]

    def __len__(self):
        return int(len(self.nodeList))

    def __repr__(self):
        curve_str = "Curve with %d points. Use curve.show() to list points." % (len(self))
        return curve_str

    def __add__(self, other):
        if not isinstance(other, Curve):
            raise ValueError('Tried to add a non-node to nodeList.')
        return Curve(self.nodeList + other.nodeList)

    def trim(self, start, end):
        if start < 0 or start > len(self) or start > end or end > len(self):
            raise ValueError('Index out of bounds for curve.')
        trimmed_nodes = self.nodeList[start:end]
        return Curve(trimmed_nodes)

    def add_node(self, new_node):
        if not isinstance(new_node, Node):
            raise ValueError('Tried to add a non-node to nodeList.')
        self.nodeList.append(new_node)

    def show(self):
        print("Curve with %d points. Points are:" % (len(self)))
        for k in self.nodeList:
            print(k)

    # cut the size of the curve in half by merging adjacent points. This is used by the fast CDTW method.
    def halve(self):
        return Curve([(self[i].x + self[1 + i].x) / 2 for i in range(0, len(self) - len(self) % 2, 2)],
                     [(self[i].y + self[1 + i].y) / 2 for i in range(0, len(self) - len(self) % 2, 2)])


class Patch:
    def __init__(self, node_list):
        # init should be called with a list of nodes, i.e., patch=([n1,n2,n3,n4])
        # order must be tl, tr, bl, br
        self.nodeList = node_list
        self.tl = self.nodeList[0]
        self.tr = self.nodeList[1]
        self.bl = self.nodeList[2]
        self.br = self.nodeList[3]
        self.left = [self.tl, self.bl]
        self.top = [self.tl, self.tr]
        self.right = [self.tr, self.br]
        self.bottom = [self.bl, self.br]

    def __getitem__(self, item):
        if not isinstance(item, int):
            raise ValueError('Non-integer nodeList reference.')
        return self.nodeList[item]

    def __setitem__(self, index, item):
        if not isinstance(index, int):
            raise ValueError('Tried to set nodeList using non-integer index.')
        elif not isinstance(item, Node):
            raise ValueError('Tried to set non-node item in nodeList')
        self.nodeList[index] = item

    def __repr__(self):
        repr_str = 'patch with id %d' % sum([n.id for n in self.nodeList])
        return repr_str

    def add_node(self, new_node):
        if not isinstance(new_node, Node):
            raise ValueError('Tried to add non-node item to nodeList.')
        self.nodeList.append(new_node)

    # reset nodes to initial state, removing steiners
    def _reset_nodes(self):
        self.tl = self.top[0]
        self.tr = self.top[-1]
        self.bl = self.bottom[0]
        self.br = self.bottom[-1]
        self.left = [self.tl, self.bl]
        self.top = [self.tl, self.tr]
        self.right = [self.tr, self.br]
        self.bottom = [self.bl, self.br]
        self.nodeList = [self.tl, self.tr, self.bl, self.br]

    # refresh nodes to include changes from steiner additions or dedups
    def refresh_nodes(self):
        self.left[0] = self.tl
        self.left[-1] = self.bl
        self.right[0] = self.tr
        self.right[-1] = self.br
        self.top[0] = self.tl
        self.top[-1] = self.tr
        self.bottom[0] = self.bl
        self.bottom[-1] = self.br
        self.nodeList = ([self.tl] + self.top[1:-1] +
                         [self.tr] + self.right[1:-1] +
                         [self.bl] + self.bottom[1:-1] +
                         [self.br] + self.left[1:-1])

    # set the distance for nodes in the top and left of the patch
    def set_distance(self):
        for j in self.left + self.top:
            if j.visited:
                continue

            # node distance is minimum from right/bottom
            dists = [j.dist(n) + n.distance for n in self.right + self.bottom]
            j.distance = min(dists)
            j.visited = True

    # initial distance for bottom/right nodes in patch
    def initialize_dist(self):
        for n in self.bottom + self.right:
            if n.visited:
                continue
            else:
                n.distance = n.dist(self.br) + self.br.distance
                n.visited = True

    # add steiner nodes to a patch
    def add_steiners(self, method, num):
        # validate the inputs
        if not isinstance(num, int):
            raise ValueError('Number of steiners must be an integer.')
        weight_unit = 0.25

        if method not in ['even', 'weighted']:
            raise ValueError('Steiner method must be either even or weighted.')

        # refresh the node list to eliminate existing steiners
        self._reset_nodes()

        # loop through edges to add points
        edges = [self.top, self.right, self.bottom, self.left]
        for idx, edge in enumerate(edges):

            if method == 'weighted':
                length = edge[1].dist(edge[0])
                num = np.ceil(length / weight_unit)

            dx = (edge[1].x - edge[0].x) / (num + 1)
            dy = (edge[1].y - edge[0].y) / (num + 1)
            new_x = np.linspace(edge[0].x + dx, edge[1].x - dx, num)
            new_y = np.linspace(edge[0].y + dy, edge[1].y - dy, num)
            new_nodes = [Node(x, y, node_type='steiner') for x, y in zip(new_x, new_y)]

            if idx == 0:
                self.top = [edges[idx][0]] + new_nodes + [edges[idx][-1]]
            elif idx == 1:
                self.right = [edges[idx][0]] + new_nodes + [edges[idx][-1]]
            elif idx == 2:
                self.bottom = [edges[idx][0]] + new_nodes + [edges[idx][-1]]
            else:
                self.left = [edges[idx][0]] + new_nodes + [edges[idx][-1]]

        self.refresh_nodes()


# ------------------- Helper Functions ------------------- #
# Helper function to make a patch from two curve points
def __make_patch(c1, c2, c1_ind, c2_ind):
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
        d = __line_dist(curve[i], curve[0], curve[-1])
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
def __line_dist(point, line_st, line_end):
    """Calculate the perpendicular distance between Node 'point' and the line
    defined by the nodes line_st and line_end"""
    x2 = np.array([line_end.x, line_end.y])
    x1 = np.array([line_st.x, line_st.y])
    x0 = np.array([point.x, point.y])
    return np.divide(np.linalg.norm(np.linalg.det([x2 - x1, x1 - x0])),
                     np.linalg.norm(x2 - x1))


# Perform CDTW on c1 and c2. Note that in this code, this method should not usually be directly called. Use cdtw_fast
# instead, as this performs the compaction and projection to leverage the fastDTW method.
def cdtw(c1, c2, num_steiner=5, mask=None):
    """Perform CDTW on two input curves.
       INPUTS (Defaults):
        * c1: curve 1
        * c2: curve 2
        * num_steiner (5): number of steiners per edge
        * mask: a mask formed from a more course CDTW run"""

    if (not isinstance(c1, Curve)) | (not isinstance(c2, Curve)):
        raise ValueError('cdtw takes 2 curves as inputs')

    if not isinstance(num_steiner, int):
        raise ValueError('num_steiner (input 3) must be an integer')

    if mask is None:
        mask = np.ones((len(c2) - 1, len(c1) - 1))
    else:
        if not isinstance(mask, np.ndarray):
            raise ValueError('mask must be of type numpy.ndarray')

    # initialize loop variables
    w = len(c1) - 1  # width of the graph
    h = len(c2) - 1  # height of the graph
    bot_mat = np.zeros((w, num_steiner+2))  # matrix to hold the previous row's distances
    cur = []
    rgt = []

    # hold dist for each patch; used for projections to more compact space
    dist_map = np.full((h, w), np.inf)

    # loop through patches from bottom right to top left along row
    for i in range(h - 1, -1, -1):

        for j in range(w - 1, -1, -1):

            if mask[i, j] == 0:
                bot_mat[j][0:num_steiner + 2] = np.inf
                continue

            # make current patch with steiners
            cur = __make_patch(c1, c2, w - 1 - j, h - 1 - i)
            cur.add_steiners('even', num_steiner)

            # handle edge cases on right and bottom edges
            # on first node, initialize bottom right as 0
            if (i == h-1) and (j == w-1):
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

            # case where we are on the edge of the warping boundary
            elif mask[i, j + 1] == 0:
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

            dist_map[i, j] = cur.tl.distance

    # final distance is the top left node distance
    return cur.tl.distance, dist_map


# Make a warping path based on the distance map from a CDTW operation. This takes the distance calculated to each
# patch's top-left node, and stores it in a matrix. This can also be visualized to spot-check the warping performance.
def __make_path(dist_map):

    # initialize variables
    w = np.size(dist_map, 0)
    h = np.size(dist_map, 1)
    i = 0
    j = 0
    path = [(i, j)]

    # loop from (0,0) to (h,w)
    while i <= h and j <= w:

        # edge cases: at the top or bottom of the map
        if i + 1 >= h:
            next_node = (i, j+1)
        elif j + 1 >= w:
            next_node = (i+1, j)
        else:
            # only possible next nodes are to the right, bottom, and bottom-right
            poss_nodes = [(i+1, j), (i, j+1), (i+1, j+1)]
            node_dists = [dist_map[(i+1, j)], dist_map[(i, j+1)],
                          dist_map[(i+1, j+1)]]

            # find the min distance to the next node and then append
            node_dict = dict(zip(node_dists, poss_nodes))
            next_node = node_dict[min(node_dict)]

        path.append(next_node)

        i = next_node[0]
        j = next_node[1]

    return path


# Project a warping path onto a high-resolution space, and return the mask based on this path and the radius r. This
# function will return a matrix double of size 2*m, 2*n, where m and n are the lengths of the two curves used to
# generate the path that is input. Using a larger r will results in a more accurate warp but more computation.
def __project_path(path, x_size, y_size, r):

    # form the band to hold the logical map
    new_band = np.zeros((x_size*2, y_size*2))

    # walk through the path and set elements of new_band to 1 according to path
    for (y, x) in path:

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

        new_band[x_start:x_end, y_start:y_end] = 1

    return new_band


# Internal function that builds the compacted curve objects, then walks through them in order. It starts from the most
# compact curve (last in the list), and uses the path from this to generate the valid warping band for the next most
# compact curve, and so on, until it reaches the original inputs.
def __fast_cdtw(c1, c2, radius, rounds):

    curve_rounds = [(c1, c2)]
    dist_map = None
    d = None
    min_size = 25

    for r in range(1, rounds):
        curve_rounds.append((curve_rounds[r-1][0].halve(), curve_rounds[r-1][1].halve()))

    for r in range(rounds-1, -1, -1):

        curr_c1 = curve_rounds[r][0]
        curr_c2 = curve_rounds[r][1]

        if len(curr_c1) < min_size or len(curr_c2) < min_size:
            d, dist_map = cdtw(curr_c1, curr_c2)
        elif dist_map is None:
            d, dist_map = cdtw(curr_c1, curr_c2)
        else:
            path = __make_path(dist_map)
            band = __project_path(path, len(curve_rounds[r+1][1]), len(curve_rounds[r+1][0]), radius)
            d, dist_map = cdtw(curr_c1, curr_c2, mask=band)

    return d, dist_map


# -------- MAIN PUBLIC FUNCTIONS -------- #

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


# Main function to perform fastCDTW. This is a wrapper to __fast_cdtw to allow interpolation if desired.
def fast_cdtw(c1, c2, radius=10, interp=0.3, rounds=4):

    if interp > 0:
        c1 = simplify_curve(c1, interp)
        c2 = simplify_curve(c2, interp)

    dist, dist_map = __fast_cdtw(c1, c2, radius, rounds)

    return dist
