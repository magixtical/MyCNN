import numpy  as np
import math

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.heaviside(x, 0)

def sigmoid(x):
    return np.exp(-x)/(1+np.exp(-x))

def sigmoid_derative(x):
    return sigmoid(x)*(1-sigmoid(x))

def softmax(x):
    x_max = np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=1, keepdims=True)

def softmax_derivative(x):
    return softmax(x) * (1 - softmax(x))

def cross_entropy(y_pred,y_true):
    m=y_pred.shape[0]
    logs_probs=-np.log(y_pred[np.arange(m),y_true]+1e-8)
    loss=np.sum(logs_probs)/m
    return loss

def cross_entropy_derivative(y_pred,y_true):
    m=y_pred.shape[0]
    grad=y_pred.copy()
    grad[np.arange(m),y_true]-=1
    return grad/m

def im2col(img,ksize,stride,padding):
    N,C,H,W=img.shape
    out_h=(H+2*padding-ksize)//stride+1
    out_w=(W+2*padding-ksize)//stride+1

    img_padded=np.pad(img, [(0,0), (0,0), (padding,padding), (padding,padding)], 'constant')
    col=np.zeros((N,C,ksize,ksize,out_h,out_w))

    for y in range(ksize):
        y_max=y+stride*out_h
        for x in range(ksize):
            x_max=x+stride*out_w
            col[:, :, y, x, :, :] = img_padded[:, :, y:y_max:stride, x:x_max:stride]

    return col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w, -1)

def col2im(col,img_shape,ksize,stride,padding):
    N,C,H,W=img_shape
    out_h=(H+2*padding-ksize)//stride+1
    out_w=(W+2*padding-ksize)//stride+1

    col=col.reshape(N,out_h,out_w,C,ksize,ksize).transpose(0,3,4,5,1,2)
    img_padded=np.zeros((N,C,H+2*padding,W+2*padding))

    for y in range(ksize):
        y_max=y+stride*out_h
        for x in range(ksize):
            x_max=x+stride*out_w
            img_padded[:,:,y:y_max:stride,x:x_max:stride]+=col[:, :, y, x, :, :]
        
    return img_padded[:, :, padding:H+padding, padding:W+padding]

class FuncLayer:
    def __init__(self,name):
        self.name=name
        self.input=None
    
    def __call__(self,x):
        self.input=x.copy()
        if self.name=='relu':
            return relu(x)
        elif self.name=='sigmoid':
            return sigmoid(x)
        elif self.name=='softmax':
            return softmax(x)
        else:
            raise ValueError('Invalid activation function')
    
    def backward(self,dout):

        if self.name=='relu':
            return dout * relu_derivative(self.input)
        
        elif self.name=='sigmoid':
            return dout * sigmoid_derative(self.input)
        elif self.name=='softmax':
            return dout * softmax_derivative(self.input)
        else:
            raise ValueError('Invalid activation function')
        
class Pooling2d:

    def __init__(self,ksize,stride,method='max'):
        self.ksize=ksize
        self.stride=stride
        self.method=method
        self.mask=None

    def __call__(self,x):
        N,C,H,W=x.shape
        out_h=(H-self.ksize)//self.stride+1
        out_w=(W-self.ksize)//self.stride+1

        self.mask=np.zeros_like(x)
        out=np.zeros((N,C,out_h,out_w))

        for i in range(out_h):
            for j in range(out_w):
                h_start=i*self.stride
                h_end=h_start+self.ksize
                w_start=j*self.stride
                w_end=w_start+self.ksize
                region=x[:, :, h_start:h_end, w_start:w_end]
                if self.method == 'max':
                    max_val = np.max(region, axis=(2,3), keepdims=True)
                    out[:, :, i, j] = max_val.squeeze()
                    self.mask[:, :, h_start:h_end, w_start:w_end] = (region == max_val)
                elif self.method == 'avg':
                    out[:, :, i, j] = np.mean(region, axis=(2,3))

        return out

    def backward(self,dout):
        if self.method == 'max':
            return self.mask*dout.repeat(self.ksize, axis=2).repeat(self.ksize, axis=3)
        elif self.method=='avg':
            return dout.repeat(self.kszie,axis=2).repeat(self.ksize, axis=3) / (self.ksize**2)

class FullyConnected:
    def __init__(self,input_size,output_size,lr=0.01):
        self.input_size=input_size
        self.output_size=output_size
        self.lr=lr
        self.scale=math.sqrt(input_size/2)
        self.weights=np.random.standard_normal((input_size,output_size))/self.scale
        self.bias=np.random.standard_normal((output_size))/self.scale

    def __call__(self,x):
        self.x=x
        return np.dot(x,self.weights)+self.bias
    
    def backward(self,dout):
        
        dw=np.dot(self.x.T,dout)
        db=np.sum(dout,axis=0)
        dx=np.dot(dout,self.weights.T)

        self.weights-=self.lr*dw
        self.bias-=self.lr*db

        return dx


class Conv2d:
    def __init__(self,in_channels,out_channels,ksize,stride,padding,lr=0.01):

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.ksize=ksize
        self.stride=stride
        self.padding=padding
        self.lr=lr

        scale = np.sqrt(2./(self.in_channels*self.ksize**2))
        self.weights = np.random.randn(out_channels, in_channels, ksize, ksize) * scale
        self.bias = np.random.randn(out_channels) * scale

        self.col_img=None
        self.input_shape=None

    def __call__(self,x):
        self.input_shape=x.shape
        N,C,H,W=x.shape
        out_h=(H+2*self.padding-self.ksize)//self.stride+1
        out_w=(W+2*self.padding-self.ksize)//self.stride+1

        self.col_img=im2col(x,self.ksize,self.stride,self.padding)
        col_weights = self.weights.reshape(self.out_channels, -1).T

        out=np.dot(self.col_img,col_weights)+self.bias
        out=np.reshape(out,(N,out_h,out_w,self.out_channels)).transpose(0,3,1,2)

        return out
    
    def backward(self,dout):
        N,OC,OH,OW=dout.shape
        dout=dout.transpose(0,2,3,1).reshape(-1,OC)

        dw=np.dot(self.col_img.T,dout).reshape(self.weights.shape)
        db=np.sum(dout,axis=0)

        rot_weights = np.rot90(self.weights, 2, axes=(2,3))
        col_rot = rot_weights.reshape(self.out_channels, -1).T

        d_col=np.dot(dout,col_rot.T)

        dx=col2im(d_col,self.input_shape,self.ksize,self.stride,self.padding)

        self.weights-=self.lr*dw
        self.bias-=self.lr*db

        return dx
    
class Flatten:
    def __call__(self,x):
        self.x=x
        self.input_shape=x.shape
        return x.reshape(x.shape[0],-1)
    
    def backward(self,dout):
        return dout.reshape(self.input_shape)
    

class CNN():
    def __init__(self, input_shape, num_classes,lr=0.01):
        self.layers = [
            Conv2d(input_shape[0], out_channels=32,ksize=3, stride=1, padding=1, lr=lr),
            FuncLayer('relu'),
            Pooling2d(2, 2, 'max'),
            
            Conv2d(32, 64, 3,stride=1 ,padding=1, lr=lr),
            FuncLayer('relu'),
            Pooling2d(2, 2, 'max'),
            
            Flatten(),
            FullyConnected(64*(input_shape[1]//4)*(input_shape[2]//4), 512, lr),
            FuncLayer('relu'),
            
            FullyConnected(512, num_classes, lr),
            FuncLayer('softmax')
        ]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def backward(self, grad):
        for layer in reversed(self.layers):
            if hasattr(layer, 'backward'):
                grad = layer.backward(grad)
        return grad

if __name__ == "__main__":

    model = CNN(input_shape=(1, 28, 28), num_classes=10)
    for i in enumerate(model.layers):
        print(i[0], i[1].__class__.__name__)

    

