from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import numpy as np

s1 = np.array([0, 2, 5, 3, 1, 3, 5, 0, 2, 5, 2])
s2 = np.array([4, 3, 1, 2, 0, 2, 1, 2, 2])
distance = dtw.distance(s1, s2)
print ("distance: %s"%distance)
path = dtw.warping_path(s1, s2)
dtwvis.plot_warping(s1, s2, path, filename="demo.png")

