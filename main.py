import os
import struct
import numpy as np
import pickle
from nueralnet import NeuralNetMLP

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2

    return images, labels

if os.path.exists('data/mnist_scaled.npz'):
    mnist = np.load('data/mnist_scaled.npz')
    X_train, y_train, X_test, y_test = [mnist[f] for f in mnist.files]
else:
    X_train, y_train = load_mnist('data/', kind='train')
    X_test, y_test = load_mnist('data/', kind='t10k')

    np.savez_compressed('data/mnist_scaled.npz',
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test
                        )
    
print('행: %d, 열: %d' % (X_train.shape[0], X_train.shape[1]))
print('행: %d, 열: %d' % (X_test.shape[0], X_test.shape[1]))

if os.path.exists('model/nn.pickle'):
    with open('model/nn.pickle', 'rb') as f:
        nn = pickle.load(f)
else:
    nn = NeuralNetMLP(n_hidden=100,
                    l2=0.1,
                    epochs=200,
                    eta=0.0005,
                    minibatch_size=100,
                    shuffle=True,
                    seed=1
                    )
    nn.fit(X_train=X_train[:55000],
        y_train=y_train[:55000],
        X_valid=X_train[55000:],
        y_valid=y_train[55000:]
        )
    with open('model/nn.pickle', 'wb') as f:
        pickle.dump(nn, f)

