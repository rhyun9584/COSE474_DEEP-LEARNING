import numpy as np
from skimage.util.shape import view_as_windows

##########
#   convolutional layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_convolutional_layer:

    def __init__(self, Wx_size, Wy_size, input_size, in_ch_size, out_ch_size, std=1e0):
    
        # initialization of weights
        self.W = np.random.normal(0, std / np.sqrt(in_ch_size * Wx_size * Wy_size / 2),
                                  (out_ch_size, in_ch_size, Wx_size, Wy_size))
        self.b = 0.01 + np.zeros((1, out_ch_size, 1, 1))
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

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

    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

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

##########
#   max pooling layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_max_pooling_layer:
    def __init__(self, stride, pool_size):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # x.shape -> (batch_size, channel_size, in_width, in_height)

        # out.shape -> (batch_size, channel_size, out_width, out_height)
        pool_size = self.pool_size
        stride = self.stride

        out = np.zeros((x.shape[0], x.shape[1], (x.shape[2]-pool_size)//stride + 1, (x.shape[3]-pool_size)//stride + 1))

        for i in range(x.shape[0]): # batch_size
            for j in range(x.shape[1]): # channel_size
                y = view_as_windows(x[i][j], (pool_size, pool_size), step=stride)
                out[i][j] = np.max(y, axis=(2,3))

        return out

    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # dLdy.shape -> (batch_size, channel_size, out_width, out_height)
        dLdx = np.zeros(x.shape)

        pool_size = self.pool_size
        stride = self.stride

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



##########
#   fully connected layer
##########
# fully connected linear layer.
# parameters: weight matrix matrix W and bias b
# forward computation of y=Wx+b
# for (input_size)-dimensional input vector, outputs (output_size)-dimensional vector
# x can come in batches, so the shape of y is (batch_size, output_size)
# W has shape (output_size, input_size), and b has shape (output_size,)

class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        # Xavier/He init
        self.W = np.random.normal(0, std/np.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+np.zeros((output_size))

    def forward(self,x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        num_data = x.shape[0]
        out = np.zeros((num_data, self.b.shape[0]))
        for i in range(num_data):
            out[i] = self.W@(x[i].reshape(-1)) + self.b

        return out

    def backprop(self,x,dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        dLdW = np.zeros(self.W.shape)
        dLdb = np.zeros(self.b.shape)
        dLdx = np.zeros(x.shape)

        for i in range(x.shape[0]):
            # calculate dLdW
            dLdW += np.outer(dLdy[i], x[i])

            # calculate dLdb
            # dLdb += dLdy[i]
            dLdb += dLdy[i] @ np.eye(self.b.shape[0])

            # calculate dLdx
            dLdx[i] = (dLdy[i] @ self.W).reshape(x.shape[1:])

        dLdW /= x.shape[0]
        dLdb /= x.shape[0]

        return dLdx,dLdW,dLdb

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W, self.b

    def set_weights(self, W, b):
        self.W = W
        self.b = b

##########
#   activation layer
##########
#   This is ReLU activation layer.
##########

class nn_activation_layer:
    
    # performs ReLU activation
    def __init__(self):
        pass
    
    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # elementwise max(x, 0)
        out = np.maximum(x, 0)
        
        return out
    
    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        dLdx = 1. * (x > 0) * dLdy

        return dLdx


##########
#   softmax layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_softmax_layer:

    def __init__(self):
        pass

    def forward(self, x):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # print("softmax Input:", x.shape)

        out = np.exp(x)
        result_sum = out.sum(axis=1).reshape(out.shape[0], 1)
        out /= result_sum

        return out

    def backprop(self, x, dLdy):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        dLdx = np.zeros(x.shape)

        dydx = np.zeros((x.shape[1], x.shape[1]))
        for i in range(x.shape[0]):
            # calculate dydx
            sqrexp = (np.sum(np.exp(x[i]))) ** 2
            for j in range(dydx.shape[0]):
                for k in range(dydx.shape[1]):
                    if j == k:
                        dydx[j][k] = (np.exp(x[i][j])*(1-np.exp(x[i][j]))) / sqrexp
                    else:
                        dydx[j][k] = -np.exp(x[i][j]+x[i][k]) / sqrexp
            
            dLdx[i] = dLdy[i] @ dydx

        return dLdx

##########
#   cross entropy layer
#   you can re-use your previous implementation, or modify it if necessary
##########

class nn_cross_entropy_layer:

    def __init__(self):
        pass

    def forward(self, x, y):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        # print("cross entropy Input:", x.shape)

        num_data = x.shape[0]

        out = 0
        for i in range(num_data):
            # y[i] = 0이면, x[i][0]에 대해서 loss 계산 
            out -= np.log(x[i][y[i]])
        out /= num_data

        return out

    def backprop(self, x, y):
        ##########
        ##########
        #   Complete the method with your implementation
        ##########
        ##########

        dLdx = np.zeros(x.shape)

        for i in range(x.shape[0]):
            true_label = y[i]
            dLdx[i][true_label] = -1/x[i][true_label]

        return dLdx
