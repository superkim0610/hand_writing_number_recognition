from main import X_train, y_train, X_test, y_test, nn
import numpy as np
import matplotlib.pyplot as plt
import time

def show_mnist_data():
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = X_train[y_train == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def show_cost_changes():
    plt.plot(range(nn.epochs), nn.eval_['cost'])
    plt.ylabel('Cost')
    plt.xlabel('Epochs')
    plt.show()

def show_accuracy():
    plt.plot(range(nn.epochs), nn.eval_['train_acc'], label='training')
    plt.plot(range(nn.epochs), nn.eval_['valid_acc'], label='validation', linestyle='--')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def show_test_predict():
    y_test_pred = nn.predict(X_test)
    acc = (np.sum(y_test == y_test_pred).astype(np.float64) / X_test.shape[0])
    print('테스트 정확도: %.2f%%' % (acc * 100))

    miscl_img = X_test[y_test != y_test_pred][:25]
    correct_lab = y_test[y_test != y_test_pred][:25]
    miscl_lab = y_test_pred[y_test != y_test_pred][:25]

    fig, ax = plt.subplots(nrows=5,
                           ncols=5,
                           sharex=True,
                           sharey=True,
                           )
    ax = ax.flatten()
    for i in range(25):
        img = miscl_img[i].reshape(28, 28)
        ax[i].imshow(img,
                     cmap='Greys',
                     interpolation='nearest'
                     )
        ax[i].set_title('%d) t: %d p: %d' % (i+1, correct_lab[i], miscl_lab[i]))

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()

def show_user_img():
    a = [-1. for _ in range(28)]
    l = [1., 1., 1.] + [-1. for _ in range(25)]
    r = [-1. for _ in range(25)] + [1., 1., 1.]
    m = [-1. for _ in range(12)] + [1., 1., 1., 1.] + [-1. for _ in range(12)]

    # x = np.array([l, l, l] + [a for _ in range(22)] + [r, r, r])
    x = np.array([a, a, a] + [m for _ in range(22)] + [a, a, a])
    _x = x.flatten()
    print(x.shape)
    t1 = time.time()
    print(nn.predict([_x])) # nn.predict([data, data, ...]), data.shape = (784,)
    print('predicting time : %d' % (time.time() - t1))
    plt.imshow(x, cmap='Greys')
    plt.show()

# show_accuracy()
# show_test_predict()
show_user_img()