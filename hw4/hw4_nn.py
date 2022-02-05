import numpy as np
from skimage.util.shape import view_as_windows


#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, filter_width, filter_height, input_size, in_ch_size, num_filters, std=1e0):
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * filter_width * filter_height / 2),
                                  (num_filters, in_ch_size, filter_width, filter_height))
        self.b = 0.01 + np.zeros((1, num_filters, 1, 1))
        self.input_size = input_size

        #######
        ## If necessary, you can define additional class variables here
        #######

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        # x.shape -> (batch_size, input_channel_size, in_width, in_height)
        # self.W.shape -> (num_filters, input_channel_size, filter_width, filter_height)
        # self.b.shape -> (1, num_filters, 1, 1)

        # out.shape -> (batch_size, num_filters, W-F+1, H-F+1)
        out = np.empty((x.shape[0], self.W.shape[0], x.shape[2]-self.W.shape[2]+1, x.shape[3]-self.W.shape[3]+1))

        b = np.squeeze(self.b) # (8,)
        for i in range(x.shape[0]): # batch_size
            y = view_as_windows(x[i], self.W.shape[1:])
            y = y.reshape((y.shape[1], y.shape[2], -1))

            result = np.empty(out.shape[1:])
            for j in range(self.W.shape[0]):  # num_filters
                out_s = y.dot(self.W[j].reshape((-1,1))) # W[j] -> (27, 1) column vector
                out_s = np.squeeze(out_s, axis=2)
                result[j] = out_s + b[j]
            out[i] = result
        return out

    #######
    # Q2. Complete this method
    #######
    def backprop(self, x, dLdy):
        # x.shape -> (batch_size, input_channel_size, in_width, in_height)
        # dLdy.shape -> (batch_size, num_filter, out_width, out_height)
        # self.W.shape -> (num_filters, input_channel_size, filter_width, filter_height)
        # self.b.shape -> (1, num_filters, 1, 1)

        dLdx = np.zeros(x.shape)
        dLdW = np.zeros(self.W.shape)
        dLdb = np.zeros(self.b.shape)

        # W flip
        flip_W = np.zeros(self.W.shape)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                flip_W[i][j] = np.flip(self.W[i][j])

        # add zero padding into dLdy
        pad = self.W.shape[2] - 1
        dLdy_pad = np.zeros((dLdy.shape[0], dLdy.shape[1], dLdy.shape[2]+pad*2, dLdy.shape[3]+pad*2))
        for i in range(dLdy.shape[0]):
            for j in range(dLdy.shape[1]):
                dLdy_pad[i][j] = np.pad(dLdy[i][j], (pad, pad))
        
        # calculate dLdx
        for i in range(x.shape[0]): # batch_size
            for j in range(self.W.shape[0]):  # num_filter
                for k in range(self.W.shape[1]):  # depth
                    y = view_as_windows(dLdy_pad[i][j], flip_W.shape[2:])
                    y = y.reshape((y.shape[0], y.shape[1], -1))

                    out = y.dot(flip_W[j][k].reshape(-1,1))
                    out = np.squeeze(out)
                    dLdx[i][k] += out

        # calculate dLdW
        for i in range(x.shape[0]): # batch_size
            for j in range(self.W.shape[0]):  # num_fliter
                for k in range(self.W.shape[1]):  # depth
                    y = view_as_windows(x[i][k], dLdy.shape[2:])
                    y = y.reshape((y.shape[0], y.shape[1], -1))

                    out = y.dot(dLdy[i][j].reshape((-1,1)))
                    out = np.squeeze(out)
                    dLdW[j][k] += out

        # calculate dLdb
        for i in range(dLdy.shape[0]):  # batch_size
            for j in range(self.b.shape[1]):  # num_filter
                dLdb[0][j][0][0] += np.sum(dLdy[i][j])
                
        return dLdx, dLdW, dLdb

    #######
    ## If necessary, you can define additional class methods here
    #######


class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size
        #######
        ## If necessary, you can define additional class variables here
        #######

    #######
    # Q3. Complete this method
    #######
    def forward(self, x):
        # x.shape -> (batch_size, channel_size, in_width, in_height)

        # out.shape -> (batch_size, channel_size, out_width, out_height)
        out = np.zeros((x.shape[0], x.shape[1], (x.shape[2]-pool_size)//stride + 1, (x.shape[3]-pool_size)//stride + 1))

        for i in range(x.shape[0]): # batch_size
            for j in range(x.shape[1]): # channel_size
                y = view_as_windows(x[i][j], (pool_size, pool_size), step=stride)
                out[i][j] = np.max(y, axis=(2,3))
        return out

    #######
    # Q4. Complete this method
    #######
    def backprop(self, x, dLdy):
        # dLdy.shape -> (batch_size, channel_size, out_width, out_height)
        dLdx = np.zeros(x.shape)

        for i in range(x.shape[0]): # batch_size
            for j in range(x.shape[1]): # channel_size
                y = view_as_windows(x[i][j], (pool_size, pool_size), step=stride)

                for k in range(y.shape[0]):
                    for l in range(y.shape[1]):
                        index = np.argmax(y[k][l])

                        # (k,l) index -> real_index
                        # (0,0) 3 -> (1,1)
                        # (0,1) 3 -> (1, 3)
                        row = (k * pool_size) + (index // pool_size)
                        col = (l * pool_size) + (index % pool_size)
                        dLdx[i,j,row,col] += dLdy[i,j,k,l]
        return dLdx

    #######
    ## If necessary, you can define additional class methods here
    #######


# testing the implementation

# data sizes
batch_size = 8
input_size = 32
filter_width = 3
filter_height = filter_width
in_ch_size = 3
num_filters = 8

std = 1e0
dt = 1e-3

# number of test loops
num_test = 20

# error parameters
err_dLdb = 0
err_dLdx = 0
err_dLdW = 0
err_dLdx_pool = 0

for i in range(num_test):
    # create convolutional layer object
    cnv = nn_convolutional_layer(filter_width, filter_height, input_size, in_ch_size, num_filters, std)

    x = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    # dLdx test
    print('dLdx test')
    y1 = cnv.forward(x)
    y2 = cnv.forward(x + delta)

    bp, _, _ = cnv.backprop(x, np.ones(y1.shape))

    exact_dx = np.sum(y2 - y1) / dt
    apprx_dx = np.sum(delta * bp) / dt
    print('exact change', exact_dx)
    print('apprx change', apprx_dx)

    err_dLdx += abs((apprx_dx - exact_dx) / exact_dx) / num_test * 100

    # dLdW test
    print('dLdW test')
    W, b = cnv.get_weights()
    dW = np.random.normal(0, 1, W.shape) * dt
    db = np.zeros(b.shape)

    z1 = cnv.forward(x)
    _, bpw, _ = cnv.backprop(x, np.ones(z1.shape))
    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_dW = np.sum(z2 - z1) / dt
    apprx_dW = np.sum(dW * bpw) / dt
    print('exact change', exact_dW)
    print('apprx change', apprx_dW)

    err_dLdW += abs((apprx_dW - exact_dW) / exact_dW) / num_test * 100

    # dLdb test
    print('dLdb test')

    W, b = cnv.get_weights()

    dW = np.zeros(W.shape)
    db = np.random.normal(0, 1, b.shape) * dt

    z1 = cnv.forward(x)

    V = np.random.normal(0, 1, z1.shape)

    _, _, bpb = cnv.backprop(x, V)

    cnv.update_weights(dW, db)
    z2 = cnv.forward(x)

    exact_db = np.sum(V * (z2 - z1) / dt)
    apprx_db = np.sum(db * bpb) / dt

    print('exact change', exact_db)
    print('apprx change', apprx_db)
    err_dLdb += abs((apprx_db - exact_db) / exact_db) / num_test * 100

    # max pooling test
    # parameters for max pooling
    stride = 2
    pool_size = 2

    mpl = nn_max_pooling_layer(stride=stride, pool_size=pool_size)

    x = np.arange(batch_size * in_ch_size * input_size * input_size).reshape(
        (batch_size, in_ch_size, input_size, input_size)) + 1
    delta = np.random.normal(0, 1, (batch_size, in_ch_size, input_size, input_size)) * dt

    print('dLdx test for pooling')
    y1 = mpl.forward(x)
    dLdy = np.random.normal(0, 10, y1.shape)
    bpm = mpl.backprop(x, dLdy)

    y2 = mpl.forward(x + delta)

    exact_dx_pool = np.sum(dLdy * (y2 - y1)) / dt
    apprx_dx_pool = np.sum(delta * bpm) / dt
    print('exact change', exact_dx_pool)
    print('apprx change', apprx_dx_pool)

    err_dLdx_pool += abs((apprx_dx_pool - exact_dx_pool) / exact_dx_pool) / num_test * 100

# reporting accuracy results.
print('accuracy results')
print('conv layer dLdx', 100 - err_dLdx, '%')
print('conv layer dLdW', 100 - err_dLdW, '%')
print('conv layer dLdb', 100 - err_dLdb, '%')
print('maxpool layer dLdx', 100 - err_dLdx_pool, '%')