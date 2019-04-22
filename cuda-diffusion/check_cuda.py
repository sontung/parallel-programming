import numpy as np
import skimage.io

class Block:
    def __init__(self, _id, block_dim):
        self.id = _id
        self.block_dim = block_dim
        self.threads = [Thread(i, self.id, self.block_dim) for i in range(block_dim)]


class Thread:
    def __init__(self, _id, block_id, block_dim):
        self.id = _id
        self.block_id = block_id
        self.block_dim = block_dim

    def determine_which_pixel(self, w, mat):
        x = self.block_dim*self.block_id+self.id
        i = x % w
        j = x / w
        # print(self.block_id, self.id, i, j, i*w+j, x)
        print(i, j)
        mat[i, j] = 5


# width = 368
# height = 640
# n = width*height
# dim = 16
# MAT = np.ones(shape=(width, height))
# all_threads = []
# blocks = [Block(idx, dim) for idx in range(int(n/dim))]
# for b in blocks:
#     all_threads.extend(b.threads)
#
# for t in all_threads:
#     t.determine_which_pixel(width, MAT)
#
# print(MAT.sum()/n)

ref = skimage.io.imread("output.png")
res = skimage.io.imread("output2.png")
for i in range(ref.shape[0]):
    if not np.array_equal(ref[i], res[i]):
        print(i, np.sum(ref[i]-res[i])/res.shape[0])