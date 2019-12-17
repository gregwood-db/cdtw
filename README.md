<a href="databricks.com"><img src="https://i.imgur.com/X4irUJs.png" title="CDTW Manifolds"></a>

# cdtw
Continuous Dyamic Time Warping with Python.
 
Includes versions that forego plotting libraries, for use on distributed or remote backends.

# Usage
 - Import the appropriate version of the code: if matplotlib is unavailable on the environment you plan to use, import cdtw-noplot. Otherwise, use cdtw-plot.
 - Create a curve from a text file by using `c1 = text_to_curve(text1.txt)`. The format of the file should be two columns with (x,y) coordinates of each point in the curve. Some sample data is included in cdtw/sample-data.
 - Run the comparison of the two curves by using `c_dist = graph_distance(c1, c2)`. This will create a graph from the two curves, and then compute the distance of this graph. Alternatively, you can explicitly create a graph by calling `g1 = graph_build(c1, c2)`. This will create a graph object, which can be manipulated and viewed at a lower level.
 - To directly view a graph, use `graph_plot(c1, c2)` or `g1.show()`.
