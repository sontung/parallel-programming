from os import listdir
from os.path import isfile, join
from skimage import io
import numpy as np

mypath = "/home/sontung/Downloads/blog/media"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for i in onlyfiles:
    im = io.imread("%s/%s" % (mypath, i), as_grey=True)
    info = np.finfo(im.dtype)  # Get the information of the incoming image type
    data = 255 * im  # Now scale by 255
    im = data.astype(np.uint8)
    io.imsave('%s2/%s.png' % (mypath, i), im)
