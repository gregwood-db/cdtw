<a href="databricks.com"><img src="https://i.imgur.com/mPcCDfT.png" title="CDTW Manifolds"></a>

# cdtw
Continuous Dyamic Time Warping with Python.
 
Includes tools to visualize data, as well as efficient implementation for raw computation.

# Usage
 - Import the appropriate version of the code: if you need to do plotting or visualization, import cdtw-plot. In general, cdtw-noplot and cdtw_fast are lighter-weight, include additional optimization, and are MUCH faster, so should be used for any actual comparisons that don't need graphing. `cdtw` is O(n) in space, but close to O(n^2) in memory. `cdtw_fast` is O(n) in both space and memory, but can be less accurate. Usually, cdtw_fast is acceptable and much faster.
 - Create a curve from a text file by using `c1 = text_to_curve(text1.txt)`. The format of the file should be two columns with (x,y) coordinates of each point in the curve. Some sample data is included in cdtw/sample-data.
 - Run the comparison of the two curves by using `c_dist = cdtw(c1, c2)` or `c_dist = cdtw_fast(c1, c2)`. 
 - If you are using cdtw-plot, you can also use `c_dist = graph_distance(c1, c2)`. This will create a graph from the two curves, and then compute the distance of this graph. Alternatively, you can explicitly create a graph by calling `g1 = graph_build(c1, c2)`. This will create a graph object, which can be manipulated and viewed at a lower level.
 - To directly view a graph, use `graph_plot(c1, c2)` or `g1.show()`.
 
## Usage example
```
import cdtw as cd
c1 = cd.text_to_curve('sample-data/s001000')
c2 = cd.text_to_curve('sample-data/s001001')
c3 = cd.text_to_curve('sample-data/s001f001')

d1 = cd.cdtw_fast(c1, c2)
d2 = cd.cdtw_fast(c1, c3)
print("Distance of real signature: {}\nDistance of forged signature: {}".format(d1, d2))
```

# Important Functions and Parameters (non-graphical)
(Parameters are similar across cdtw-plot, cdtw_fast and cdtw, but not identical, since the implementations vary slightly, and the graphical requirements are more restrictive. This section will cover parameters only for cdtw).

Create curve objects for comparison: `text_to_curve(filename)`
- filename is just a path to a text file that contains (x,y) coordinate pairs that make up the curve. The file should have two columns; the first is X-coordinates and the second is Y-coordinates. Check the included files for examples.

Perform CDTW on two curves using standard CDTW: `cdtw(c1, c2, num_steiner=N, interp=E, r=R)`
- `c1` and `c2` are curves, as created by `text_to_curve()`.
- `interp` is an interpolant factor. This determines the simplication factor of the input curve, according to the Douglas-Peuker algorithm. A higher value for `interp` means the algorithm will run faster, but will be less accurate. Default is 0.3.
- `num_steiner` is another interpolant factor; N determines the number of points placed along each segment of the manifold. Setting this lower results in faster computation, but less accuracy in results. Default is 5.
- `r` controls the width of the Sakoe-Chiba band applied to the manifold during calculation of the optimal warping path. This essentially sets upper and lower bounds on the distance from the main diagonal. Setting this lower will decrease the time required to calculate the optimal path, but may result in local minima, reducing the accuracy of the algorithm. Default is 100.

Perform CDTW on two curves using fast CDTW: `cdtw_fast(c1, c2, interp=E, num_steiner=S, radius=E, rounds=N)`
- `c1` and `c2` are curves, as created by `text_to_curve()`.
- `interp` and `num_steiner` are the same as above.
- `radius` is used to determine the neighborhood when projecting a low-resolution warping path onto a higher-resolution space. See [the original FastDTW implementation](https://github.com/rmaestre/FastDTW) for more detail.
- `rounds` determines how many compaction-projection-refinement rounds are carried out by the FastCDTW algorithm.

# To-Do
- Add more methods for data import
- Add more plotting methods for cdtw-plot
- Explore further efficiency in execution
