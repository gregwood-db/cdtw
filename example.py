# example usage of cdtw package

import cdtw-plot as cdp
import cdtw-noplot as cd

c1 = cd.text_to_curve('sample-data/s001000')
c2 = cd.text_to_curve('sample-data/s001001')

d = cd.cdtw(c1, c2)

g1 = cdp.graph_plot(c1, c2, interp=0.5)
g1.total_dist
