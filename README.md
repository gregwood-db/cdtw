<a href="databricks.com"><img src="https://i.imgur.com/mPcCDfT.png" title="CDTW Manifolds"></a>

# cdtw
Continuous Dyamic Time Warping with Python.
 
Includes versions that forego plotting libraries, for use on distributed or remote backends.

# Usage
 - Import the appropriate version of the code: if you need to do plotting or visualization, import cdtw-plot. In general, cdtw-noplot is lighter-weight, includes some additional optimization, and is MUCH faster, so should be used for any actual comparisons that don't need graphing.
 - Create a curve from a text file by using `c1 = text_to_curve(text1.txt)`. The format of the file should be two columns with (x,y) coordinates of each point in the curve. Some sample data is included in cdtw/sample-data.
 - Run the comparison of the two curves by using `c_dist = cdtw(c1, c2)`. 
 - If you are using cdtw-plot, you can also use `c_dist = graph_distance(c1, c2)`. This will create a graph from the two curves, and then compute the distance of this graph. Alternatively, you can explicitly create a graph by calling `g1 = graph_build(c1, c2)`. This will create a graph object, which can be manipulated and viewed at a lower level.
 - To directly view a graph, use `graph_plot(c1, c2)` or `g1.show()`.
