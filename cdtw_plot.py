# IMPORT PACKAGES
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

# DESCRIPTION:
#
# A package for measuring and visualizing the distance between curves using CDTW.
#
# The cdtw_plot package includes functions to calculate a manifold between two curves, visualize this manifold, and use
# CDTW to calculate the distance across this manifold. Note that if only calculation is needed, the cdtw_noplot package
# is more performant (since full manifold calculation is not required), and also forgoes the matplotlib packages req'd
# in this package (which may cause problems in some environments). It is recommended to use that pacakge for all non-
# visualization purposes.
#
# In general, this code requires as inputs text files with two columns, representing the X and Y coordinates of the
# curves to be compared. More formats will be added in the future. Once the files are converted using text_to_curve(),
# they can be fed into the classes and functions detailed below.

# CLASSES:
#   Curve([Node1, Node2, Node3, ..., NodeN])
#       Represents a 2D, N-length curve made up of Nodes.
#       F--- add_node(Node), show(), plot()
#
#   Node(x, y)
#       Represents a point in euclidean space. Makes up patches and curves.
#       F--- show(), dist(Node)
#
#   Patch([NodeTL, NodeTR, NodeBL, NodeBR])
#       Contains 4 native points and potentially more "steiner" points added
#       as interpolative elements. These are the basis of a graph.
#       F--- add_node(Node), reset_nodes(), refresh_nodes(), set_distance(),
#            initialize_dist(), show(), add_steiners(method, num)
#
#   Graph(Curve1, Curve2)
#       Contains the patches created as a result of the Minkowski sum of two
#       Curves, as well as methods to calculate distance and visualize.
#       F--- show(), fill(Curve1, Curve2), dedup_nodes(), place_steiners(method, num),
#            find_node(NodeID), traverse(), trace()

# FUNCTIONS:
# text_to_curve(filename)
#   Create a curve from a text file, where column 1 is X and column 2 is Y.
#
# graph_plot(Graph, <Curve>, <Curve>, <#Steiner>, <Method>)
#   Plot the graph, or compute the graph from curves C1 and C2 and then plot.
#   This function can take arguments in any order, but must have either a graph
#   object, or two curves.
#
# graph_distance()
#   Return the graph distance for the input graph, or that built from the two
#   input curves. This function can take arguments in any order, but must have
#   either a graph object, or two curves.
#
# graph_build(Curve, Curve, <#Steiner>, <Method>)
#   Builds and returns a graph from two curves. This graph can then be plugged into
#   the other functions, and can give more direct access to underlying functions.

# USAGE:
# Most usage should be through the public functions (graph_build, graph_plot). They wrap all necessary variables and
# functions without exposing internal variables. If lower-level functions are required, the graph_build() function
# can be used. In general, a workflow will look like this:
#   c1 = text_to_curve(text1.txt)
#   c2 = text_to_curve(text2.txt)
#   c_dist = graph_distance(c1, c2)
#   graph_plot(c1, c2)

# -------------- CURVE CLASS --------------- #


class Curve:
    def __init__(self, point_list):
        # init should be called with a list of points, i.e., c1=([p1,p2,p3,...])
        self.nodeList = []
        for k in point_list:
            assert isinstance(k, Node)
            self.add_node(k)

    def __getitem__(self, item):
        return self.nodeList[item]

    def __call__(self, item):
        assert isinstance(item, int)
        return self.nodeList[item]

    def __len__(self):
        return int(len(self.nodeList))

    def __repr__(self):
        curve_str = "Curve with %d points. Use curve.show() to list points." % (len(self))
        return curve_str

    def __add__(self, other):
        assert isinstance(other, Curve)
        return Curve(self.nodeList + other.nodeList)

    def trim(self, start, end):
        if start < 0 or start > len(self) or start > end or end > len(self):
            raise ValueError('Index out of bounds for curve.')
        trimmed_nodes = self.nodeList[start:end]
        return Curve(trimmed_nodes)

    def add_node(self, new_node):
        assert isinstance(new_node, Node)
        self.nodeList.append(new_node)

    def show(self):
        print("Curve with %d points. Points are:" % (len(self)))
        for k in self.nodeList:
            print(k)

    def plot(self, show=True):
        """Plot the curve. By default, creates a new figure; use show=False
        to not plot the curve in new figure."""
        
        curve_x = [node.x for node in self.nodeList]
        curve_y = [node.y for node in self.nodeList]
        pl.plot(curve_x, curve_y)
        
        if show:
            pl.show()


# -------------- NODE CLASS --------------- #


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

        # Link to prev/next node in traversal chain
        self.prev_node = []
        self.next_node = []

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


# -------------- PATCH CLASS --------------- #


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
        assert isinstance(item, int)
        return self.nodeList[item]

    def __setitem__(self, index, item):
        assert isinstance(index, int)
        assert isinstance(item, Node)
        self.nodeList[index] = item

    def __repr__(self):
        repr_str = 'patch with id %d' % sum([n.id for n in self.nodeList])
        return repr_str

    def add_node(self, new_node):
        assert isinstance(new_node, Node)
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

    def set_distance(self):
        for j in self.left + self.top:
            if j.visited:
                continue
            node_list = self.right + self.bottom
            node_list = list(filter(lambda p: p != j, node_list))
            dists = [j.dist(n) + n.distance for n in self.right + self.bottom]
            j.distance = min(dists)
            if isinstance(np.argmin(dists), np.ndarray):
                j_ind = np.argmin(dists)[0]
            else:
                j_ind = np.argmin(dists)
            j.prev_node = node_list[j_ind]
            j.prev_node.next_node = j
            j.visited = True

    def initialize_dist(self):
        for n in self.bottom + self.right:
            if n.visited:
                continue
            else:
                n.distance = n.dist(self.br) + self.br.distance
                n.visited = True

    def show(self):
        patches = []
        fig, ax = pl.subplots()
        c_x = [self.tl.x, self.tr.x, self.br.x, self.bl.x]
        c_y = [self.tl.y, self.tr.y, self.br.y, self.bl.y]
        min_y = min(c_y)
        min_x = min(c_x)
        max_y = max(c_y)
        max_x = max(c_x)
        xy = np.c_[c_x, c_y]
        poly = Polygon(xy)
        patches.append(poly)
        p = PatchCollection(patches, alpha=0.4)
        ax.set_ybound(min_y - 1, max_y + 1)
        ax.set_xbound(min_x - 1, max_x + 1)
        ax.add_collection(p)

    def add_steiners(self, method, num):
        # validate the inputs
        assert isinstance(num, int)
        weight_unit = 0.25

        if method not in ['even', 'weighted']:
            raise ValueError('steiner method must be either even or weighted.')

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


# -------------- GRAPH CLASS --------------- #


class Graph:
    def __init__(self, c1, c2):
        self.w = len(c1) - 1
        self.h = len(c2) - 1
        self.min_x = np.inf
        self.min_y = np.inf
        self.max_x = -np.inf
        self.max_y = -np.inf
        self.patch_list = np.empty((self.h, self.w), dtype=object)
        self.fill(c1, c2)
        self.total_dist = np.inf
        self._num_nodes = []

    # length of graph is equal to number of elements
    def __len__(self):
        return self.h * self.w

    # create a graphical patch mesh of the graph
    def show(self, ax=0, pct=False, nodes=False):
        patches = []

        # check if existing axis was passed in
        if ax == 0:
            fig, ax = pl.subplots()

        count = 0
        total = len(self)
        curr_pct = 0
        print('Building graph with %d patches...' % total)

        for i in range(self.h):
            for j in range(self.w):
                c_x = [self.patch_list[(i, j)].tl.x, self.patch_list[(i, j)].tr.x,
                       self.patch_list[(i, j)].br.x, self.patch_list[(i, j)].bl.x]

                if min(c_x) < self.min_x:
                    self.min_x = min(c_x)
                if max(c_x) > self.max_x:
                    self.max_x = max(c_x)

                c_y = [self.patch_list[(i, j)].tl.y, self.patch_list[(i, j)].tr.y,
                       self.patch_list[(i, j)].br.y, self.patch_list[(i, j)].bl.y]

                if min(c_y) < self.min_y:
                    self.min_y = min(c_y)
                if max(c_y) > self.max_y:
                    self.max_y = max(c_y)

                xy = np.c_[c_x, c_y]
                poly = Polygon(xy)
                patches.append(poly)

                if nodes:
                    for n in self.patch_list[(i, j)].nodeList:
                        ax.plot(n.x, n.y, '.k', ms=4)

                count += 1
                if pct:
                    if 10 * count / total >= curr_pct + 1:
                        curr_pct = np.floor(10 * count / total)
                        print('%d percent done...' % (10 * curr_pct))

        p = PatchCollection(patches, alpha=0.4)
        # colors = 100*np.random.rand(len(patches))
        # p.set_array(np.array(colors))
        ax.set_ybound(self.min_y - 1, self.max_y + 1)
        ax.set_xbound(self.min_x - 1, self.max_x + 1)
        ax.add_collection(p)

    # fill graph with patches based on minkowski sum of curve segments
    # needs to be done before calling place_steiners
    def fill(self, c1, c2):

        # filling is done from bottom right to top left
        for i in range(self.h - 1, -1, -1):
            for j in range(self.w - 1, -1, -1):
                # each segment is consecutive points in curve
                c2_ind = self.h - 1 - i
                c1_ind = self.w - 1 - j
                br = c2(c2_ind) - c1(c1_ind)
                bl = c2(c2_ind) - c1(c1_ind + 1)
                tr = c2(c2_ind + 1) - c1(c1_ind)
                tl = c2(c2_ind + 1) - c1(c1_ind + 1)

                self.patch_list[(i, j)] = Patch([tl, tr, bl, br])

        self._dedup_nodes()

    # eliminate redundant nodes
    # for P(i,j), i.e., set left edge = right edge from P(i,j-1)
    def _dedup_nodes(self):

        # loop top left to bottom right along rows
        for i in range(0, self.h):
            for j in range(0, self.w):

                # try to replace the left edge
                try:
                    if j == 0:
                        raise IndexError('Index is negative. We hit a left boundary.')
                    self.patch_list[(i, j)].left = self.patch_list[(i, j - 1)].right
                    self.patch_list[(i, j)].tl = self.patch_list[(i, j)].left[0]
                    self.patch_list[(i, j)].bl = self.patch_list[(i, j)].left[-1]
                    self.patch_list[(i, j)].refresh_nodes()

                except IndexError:
                    pass

                # try to replace the top edge
                try:
                    if i == 0:
                        raise IndexError('Index is negative. We hit a top boundary.')
                    self.patch_list[(i, j)].top = self.patch_list[(i - 1, j)].bottom
                    self.patch_list[(i, j)].tl = self.patch_list[(i, j)].top[0]
                    self.patch_list[(i, j)].tr = self.patch_list[(i, j)].top[-1]
                    self.patch_list[(i, j)].refresh_nodes()

                except IndexError:
                    pass

    # wrapper around the add_steiners method of the patch object
    def place_steiners(self, method='even', num=3):
        for i in range(0, self.h):
            for j in range(0, self.w):
                self.patch_list[(i, j)].add_steiners(method, num)

        self._dedup_nodes()

    # take in a numerical node ID, return the node object with that ID
    def _find_node(self, node_id):
        all_nodes = np.reshape([n.nodeList for n in np.reshape(self.patch_list[:], -1)], -1)
        try:
            return list(filter(lambda p: p.id == node_id, all_nodes))[0]
        except IndexError:
            return []

    # traverse the graph and set distances for all nodes
    def traverse(self):
        _all_nodes = np.empty(0)
        self.patch_list[(self.h - 1, self.w - 1)].initialize_dist()

        for i in range(self.h - 1, -1, -1):
            for j in range(self.w - 1, -1, -1):
                np.append(_all_nodes, self.patch_list[(i, j)].nodeList)
                self.patch_list[(i, j)].initialize_dist()
                self.patch_list[(i, j)].set_distance()

        self._num_nodes = len(np.unique(_all_nodes))
        self.total_dist = self.patch_list[(0, 0)].tl.distance

    # plot a trace of the path across the graph
    def trace(self):
        trace_run = True
        c_x = [self.patch_list[(0, 0)].tl.x]
        c_y = [self.patch_list[(0, 0)].tl.y]
        curr_node = self.patch_list[(0, 0)].tl

        while trace_run:
            try:
                curr_node = curr_node.prev_node
                c_x.append(curr_node.x)
                c_y.append(curr_node.y)
            except AttributeError:
                break

        c_x.append(self.patch_list[(self.h - 1, self.w - 1)].br.x)
        c_y.append(self.patch_list[(self.h - 1, self.w - 1)].br.y)

        pl.plot(c_x, c_y, '-or', linewidth=3.0)
        pl.gca().set_ybound(self.min_y - 1, self.max_y + 1)
        pl.gca().set_xbound(self.min_x - 1, self.max_x + 1)


# Private generic wrapper to the Graph methods.
# Parses and prepares arguments to pass to the Graph class
def _graph_actions(action, **kwargs):
    """Either take in a graph or two curves, create the graph if necessary,
    and then perform some action. Failure to provide either a graph or 2 curves
    will cause an error. Additionally, if action is left blank, an error occurs.

    KWARGS must contain either a graph or two curves:
        -graph=<Graph>: a pre-constructed graph object. Must be properly created, but
        plot_manifold will place steiners and traverse the graph if necessary.
        -c1=<Curve>: the first curve to use to construct the graph. If a graph object
        is provided, this will be ignored.
        -c2=<Curve>: the second curve to use to construct the graph. If a graph object
        is provided, this will be ignored.
        -nsteiners=<int>: the number of steiners to use when constructing the graph.
        -steiner_method=<string>: the method to use when placing the steiners, either
        'even' or 'weighted'.
        -ax=<axis>: the axis on which to plot (if plotting is being called). A new one
        will be created if necessary.

    ACTION can be one of 'build', 'plot', or 'dist'. Other values will cause an error."""
    if action is None:
        raise ValueError("Error: graph_actions called without action specified.")

    if action not in ['plot', 'dist', 'build']:
        raise ValueError("Error: valid actions are 'build, 'plot' and 'dist', received %s" % action)

    if kwargs is None:
        raise ValueError("No arguments provided.")

    try:
        # if the graph has been passed to the function, pick it up
        # and determine if it needs to be traversed or not
        gr = kwargs['graph']

        # if the graph has not been traversed, dist is INF
        # we need to place steiners and traverse the graph
        if gr.total_dist == np.inf:
            try:
                nsteiners, steiner_method = kwargs['nsteiners'], kwargs['steiner_method']
            except KeyError:
                nsteiners = 5
                steiner_method = 'even'

            gr.place_steiners(num=nsteiners, method=steiner_method)
            gr.traverse()

        if action == 'plot':
            # check for axis; if none, create a new one
            try:
                ax = kwargs['ax']
            except KeyError:
                ax = 0

            gr.show(ax)
            gr.trace()
            pl.title('C1-C2 Manifold, D=%f' % gr.total_dist)
        elif action == 'dist':
            return gr.total_dist
        elif action == 'build':
            return gr

    # a KeyError means the graph was not passed, so try curves
    except KeyError:
        try:
            curve1, curve2 = kwargs['c1'], kwargs['c2']
            gr = Graph(curve1, curve2)
            try:
                nsteiners, steiner_method = kwargs['nsteiners'], kwargs['steiner_method']
            except KeyError:
                nsteiners = 5
                steiner_method = 'even'

            gr.place_steiners(num=nsteiners, method=steiner_method)
            gr.traverse()

            if action == 'plot':
                try:
                    ax = kwargs['ax']
                except KeyError:
                    ax = 0

                gr.show(ax)
                gr.trace()
                pl.title('C1-C2 Manifold, D=%f' % gr.total_dist)
            elif action == 'dist':
                return gr.total_dist
            elif action == 'build':
                return gr

        # a KeyError here means we have neither a graph nor two curves
        except KeyError:
            raise ValueError('Wrong arguments. Pass either graph=Graph OR c1=Curve1,c2=Curve2.')


# ----------- PUBLIC-FACING FUNCTIONS ----------- #

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


# Public wrapper to the Graph distance
def graph_distance(*args, **kwargs):
    """Either take in 2 curves or a graph, and return the distance of
    the curves (or the curves represented by the input graph).

    Arguments can be passed in any order, but can include:
        -Graph
        -Curve (must have 2, first will be read as c1)
        -Int (number of steiners)
        -String (steiner method)

    There is one valid argument for KWARGS:
        -interp=<FLOAT>

    This argument provides an the epsilon value, by which curves will be
    simplified using the Douglas-Peuker algorithm. Default is 0, i.e., no
    compression. Must be a positive float/decimal.

    An error is thrown if either a graph or two curves are not provided."""
    if args is None:
        raise ValueError("No arguments provided.")

    gr = []
    c1 = []
    c2 = []
    num_steiner = []
    steiner_method = []

    for a in args:
        if isinstance(a, Graph):
            gr = a
        elif isinstance(a, Curve):
            if not c1:
                c1 = a
            else:
                c2 = a
        elif isinstance(a, str):
            steiner_method = a
        elif isinstance(a, int):
            num_steiner = a

    if kwargs == {}:
        c1_interp = c1
        c2_interp = c2
    else:
        try:
            interp = kwargs['interp']
            if interp > 0:
                c1_interp = simplify_curve(c1, interp)
                c2_interp = simplify_curve(c2, interp)
            else:
                c1_interp = c1
                c2_interp = c2
        except KeyError:
            print('Warning: inappropriate KWARGS provided for graph_plot. Using interp value of 0.')
            c1_interp = c1
            c2_interp = c2

    if not steiner_method:
        steiner_method = 'even'
    if not num_steiner:
        num_steiner = 5

    if gr:
        d = _graph_actions('dist', graph=gr, nsteiners=num_steiner, steiner_method=steiner_method)
    else:
        if not (c1 and c2):
            raise ValueError('Error: must provide 2 curves if graph not provided.')
        d = _graph_actions('dist', c1=c1_interp, c2=c2_interp, nsteiners=num_steiner, steiner_method=steiner_method)

    return d


# Public wrapper to the Graph plot method
def graph_plot(*args, **kwargs):
    """Either take in 2 curves or a graph, and plot the graph.

    Non-explicit arguments can be passed in any order, but can include:
        -Graph
        -Curve (must have 2, first will be read as c1)
        -Int (number of steiners)
        -String (steiner method)

    There are two arguments for KWARGS:
        -interp=<FLOAT>
        -ax=<MATPLOTLIB.AX>

    Interp provides an the epsilon value, by which curves will be
    simplified using the Douglas-Peuker algorithm. Default is 0, i.e., no
    compression. Must be a positive float/decimal.

    Ax provides an axis on which to plot the manifold. This is useful for
    making subplots. If no axis is provided, a new figure will be created.

    An error is thrown if either a graph or two curves are not provided."""
    if args is None:
        raise ValueError("No arguments provided.")

    gr = []
    c1 = []
    c2 = []
    num_steiner = []
    steiner_method = []

    for a in args:
        if isinstance(a, Graph):
            gr = a
        elif isinstance(a, Curve):
            if not c1:
                c1 = a
            else:
                c2 = a
        elif isinstance(a, str):
            steiner_method = a
        elif isinstance(a, int):
            num_steiner = a

    try:
        ax = kwargs['ax']
    except KeyError:
        ax = 0

    if gr:
        _graph_actions('plot', ax=ax, graph=gr, nsteiners=num_steiner, steiner_method=steiner_method)
    else:
        if kwargs == {}:
            c1_interp = c1
            c2_interp = c2
        else:
            try:
                interp = kwargs['interp']
                if interp > 0:
                    c1_interp = simplify_curve(c1, interp)
                    c2_interp = simplify_curve(c2, interp)
                else:
                    c1_interp = c1
                    c2_interp = c2
            except KeyError:
                print('Using interp value of 0.')
                c1_interp = c1
                c2_interp = c2

        if not steiner_method:
            steiner_method = 'even'
        if not num_steiner:
            num_steiner = 5

        if not (c1 and c2):
            raise ValueError('Error: must provide 2 curves if graph not provided.')
        _graph_actions('plot', ax=ax, c1=c1_interp, c2=c2_interp, nsteiners=num_steiner, steiner_method=steiner_method)


# Public wrapper to return a graph object
def graph_build(*args, **kwargs):
    """Take in two curves, and return the graph constructed from them.

    Arguments can be passed in any order, but can include:
        -Curve (must have 2, first will be read as c1)
        -Int (number of steiners)
        -String (steiner method)

    There is one valid argument for KWARGS:
        -interp=<FLOAT>

    This argument provides an the epsilon value, by which curves will be
    simplified using the Douglas-Peuker algorithm. Default is 0, i.e., no
    compression. Must be a positive float/decimal.

    An error is thrown if two curves are not provided."""
    if args is None:
        raise ValueError("No arguments provided.")

    c1 = []
    c2 = []
    num_steiner = []
    steiner_method = []

    for a in args:
        if isinstance(a, Curve):
            if not c1:
                c1 = a
            else:
                c2 = a
        elif isinstance(a, str):
            steiner_method = a
        elif isinstance(a, int):
            num_steiner = a

    if kwargs == {}:
        c1_interp = c1
        c2_interp = c2
    else:
        try:
            interp = kwargs['interp']
            if interp > 0:
                c1_interp = simplify_curve(c1, interp)
                c2_interp = simplify_curve(c2, interp)
            else:
                c1_interp = c1
                c2_interp = c2
        except KeyError:
            print('Warning: inappropriate KWARGS provided for graph_plot. Using interp value of 0.')
            c1_interp = c1
            c2_interp = c2

    if not steiner_method:
        steiner_method = 'even'
    if not num_steiner:
        num_steiner = 5

    if not (c1 and c2):
        raise ValueError('Error: must provide 2 curves if graph not provided.')
    g_out = _graph_actions('build', c1=c1_interp, c2=c2_interp, nsteiners=num_steiner, steiner_method=steiner_method)

    return g_out
