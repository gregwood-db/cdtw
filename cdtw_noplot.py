# IMPORT PACKAGES
import numpy as np
import numbers

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
    def __init__(self, point_list):
        # init should be called with a list of points, i.e., c1=([p1,p2,p3,...])
        self.nodeList = []
        for k in point_list:
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

    # loop through patches from bottom right to top left
    for i in range(h - 1, -1, -1):

        for j in range(w - 1, -1, -1):

            # if we are outside of the SCB, we can skip this element
            if scb[i, j] == 0:
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

            # set the distance for the current patch
            cur.set_distance()

            for n in range(0, num_steiner + 2):
                bot_mat[j][n] = cur.top[n].distance

    # final distance is the top left node distance
    return cur.tl.distance
