import numpy as np

# DESCRIPTION:
#
# A package for measuring the distance between curves using CDTW.
#
# The cdtw package includes functions to calculate a manifold between two curves, then use CDTW to calculate the
# distance across this manifold. Note that this package contains only methods for computing the distance between two
# curves; if visualization is required, use cdtw_plot instead.
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
