import numpy as np
import scipy.ndimage as ndimage     
subs = 10  # this is the size of the (square) sub-windows
img = np.random.rand(500, 500)
img_std = ndimage.filters.generic_filter(img, np.std, size=subs)
