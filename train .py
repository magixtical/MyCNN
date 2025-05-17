import numpy as np
import torchvision
import math
from CNN import *

def onehot(targets, num_classes):
    results=np.zeros((num_classes,10))
    for i in range(num_classes):
        results[i][targets[i]]=1
    return results


def save_model(model,filename):
    params={}
    for i ,layer in enumerate(model.layers):
        if hasattr(layer, 'weights') and hasattr(layer,'bias'):
            params[f'layer_{i}_weights']=layer.weights
            params[f'layer_{i}_bias']=layer.bias
    print("params.keys():", params.keys())
    np.savez(filename,**params)

def train():

    dataset_path = "D:/examples/python/datasets/mnist"
    train_data = torchvision.datasets.MNIST(root=dataset_path, train=True, download=True)
    train_data.data = train_data.data.numpy()  # [60000,28,28]
    train_data.targets = train_data.targets.numpy()  # [60000]
    train_data.data = train_data.data.reshape(60000, 28, 28, 1) / 255.   # 输入向量处理

    print("train_data.data.shape:", train_data.data.shape)
    print("train_data.targets.shape:", train_data.targets.shape)

    train_data.data=train_data.data.transpose((0, 3, 1, 2))  # 转换为[60000,1,28,28]

    print("train_data.data.shape:", train_data.data.shape)


    input_shape=(1,28,28)
    num_classes=10
    lr=0.01
    batch_size=3

    model=CNN(input_shape, num_classes, lr=lr)

    best_loss = math.inf
    best_epoch = 0

    for epoch in range(10):
        loss=0.0

        for i in range(0, 60000, batch_size):
            x_batch = train_data.data[i:i+batch_size]
            y_batch = train_data.targets[i:i+batch_size]

            output = model.forward(x_batch)

            loss=cross_entropy(output,y_batch)

            grad=cross_entropy_derivative(output,y_batch)

            model.backward(grad)

            print("Epoch-{}-{:05d}".format(str(epoch), i), ":", "loss:{:.4f}".format(loss))

        lr *= 0.95**(epoch+1)
        if loss<best_loss:
            best_loss=loss
            best_epoch=epoch
            save_model(model,"mnist_cnn")

def eval():

    r = np.load("mnist_cnn.npz")
    dataset_path = "D:/examples/python/datasets/mnist"
    test_data = torchvision.datasets.MNIST(root=dataset_path, train=False)
    test_data.data = test_data.data.numpy()  # [60000,28,28]
    test_data.targets = test_data.targets.numpy()  # [60000]
    test_data.data = test_data.data.reshape(10000, 28, 28, 1) / 255.   # 输入向量处理


    test_data.data=test_data.data.transpose((0, 3, 1, 2))  # 转换为[60000,1,28,28]

    input_shape=(1,28,28)
    num_classes=10
    lr=0.01
    batch_size=3

    model=CNN(input_shape, num_classes, lr=lr)
    model.layers[0].weights = r['layer_0_weights']
    model.layers[0].bias = r['layer_0_bias']
    model.layers[3].weights = r['layer_3_weights']
    model.layers[3].bias = r['layer_3_bias']
    model.layers[7].weights = r['layer_7_weights']
    model.layers[7].bias = r['layer_7_bias']
    model.layers[9].weights = r['layer_9_weights']
    model.layers[9].bias = r['layer_9_bias']

    correct=0

    for i in range(10000):
        
        X=test_data.data[i]
        X=X[np.newaxis,:]
        Y=test_data.targets[i]

        output=model.forward(X)
        pred=np.argmax(output)
        if pred==Y:
            correct+=1

    print("accuracy:",correct/10000)


if __name__== "__main__":

    eval()