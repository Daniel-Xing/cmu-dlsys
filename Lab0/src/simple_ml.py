import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    # BEGIN YOUR CODE
    return x + y
    # END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    # 打开并解压缩图像文件
    with gzip.open(image_filename, 'rb') as image_file:
        # 读取文件头，跳过前16字节
        image_file.read(16)
        # 读取图像数据
        image_data = image_file.read()

    # 打开并解压缩标签文件
    with gzip.open(label_filename, 'rb') as label_file:
        # 读取文件头，跳过前8字节
        label_file.read(8)
        # 读取标签数据
        label_data = label_file.read()

    # 将图像和标签数据转换为NumPy数组
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    label_array = np.frombuffer(label_data, dtype=np.uint8)

    # 获取图像的维度
    num_images = len(label_array)
    image_dim = len(image_array) // num_images

    # 将图像数据转换为浮点数并归一化
    image_array = image_array.astype(np.float32) / 255.0

    # 返回图像数据和标签数据的元组
    return image_array.reshape(num_images, image_dim), label_array


def softmax_loss(Z, y):
    """
    计算softmax损失函数。
    
    参数:
        Z (np.ndarray[np.float32]): 2D numpy数组，形状为(batch_size, num_classes)，
            包含每个类别的logit预测值。
        y (np.ndarray[np.uint8]): 1D numpy数组，形状为(batch_size, )，
            包含每个样本的真实标签。
    
    返回:
        整个样本的平均softmax损失。
    """

    # 确保输入的logits Z是浮点数，以便进行exp计算
    Z = Z.astype(np.float64)

    # 计算log-sum-exp，即对每个样本的logits应用exp，然后按行求和，最后取对数。
    # 这一步会得到每个样本的log-sum-exp值。
    log_sum_exp = np.log(np.sum(np.exp(Z), axis=1))

    # 从log-sum-exp中减去每个样本真实类别的logit。
    # np.arange(y.shape[0])生成一个索引数组，用于选择每个样本的真实类别的logit。
    correct_logit = Z[np.arange(y.shape[0]), y]

    # 计算每个样本的softmax损失
    softmax_loss_per_sample = log_sum_exp - correct_logit

    # 计算所有样本的平均softmax损失
    average_loss = np.mean(softmax_loss_per_sample)

    return average_loss

def softmax_grad(Z, y):
    """
    计算相对于logits Z的softmax损失的梯度。
    
    参数:
        Z (np.ndarray): logits数组，形状为(batch_size, num_classes)
        y (np.ndarray): 真实标签数组，形状为(batch_size,)
        
    返回:
        np.ndarray: 梯度数组，形状为(batch_size, num_classes)
    """
    # 为了数值稳定性，使用最大值技巧。从每个样本的logits中减去最大值，避免指数运算结果过大。
    probabilities = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    
    # 将这些指数值标准化，得到softmax概率。
    # 这是通过将每个指数值除以每个样本指数值的总和来实现的，确保它们加起来等于1。
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    
    # 从正确类别的概率中减去1。对于正确的类别，梯度将是(p-1)，对于所有错误的类别，梯度将是p。
    # 这一步实现了梯度公式中的(z - e_y)部分。
    probabilities[np.arange(len(y)), y] -= 1
    
    # 计算批次中所有样本的平均梯度。因为损失是在批次上平均的，所以这一步是必要的。
    gradient = probabilities / len(y)
    
    # 返回计算好的梯度。
    return gradient



def softmax_regression_epoch(X, y, theta, lr=0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    num_examples = X.shape[0]
    for start_idx in range(0, num_examples, batch):
        end_idx = min(start_idx + batch, num_examples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        # Compute logits for the current batch
        logits = np.dot(X_batch, theta)
        
        # Compute softmax gradient
        grad = softmax_grad(logits, y_batch)
        
        # Update the Theta parameters
        theta -= lr * np.dot(X_batch.T, grad)


def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # 防止溢出
    return e_Z / e_Z.sum(axis=1, keepdims=True)

def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    
    # Convert y to one-hot encoding
    y_one_hot = np.eye(num_classes)[y]
    
    for start_idx in range(0, num_examples, batch):
        end_idx = min(start_idx + batch, num_examples)
        X_batch = X[start_idx:end_idx]
        y_batch = y_one_hot[start_idx:end_idx]
        
        # Forward pass
        Z1 = ReLU(X_batch @ W1)  # Activation from first layer
        logits = Z1 @ W2  # Logits for the current batch
        probabilities = softmax(logits)
        
        # Compute the gradients for softmax loss
        G2 = probabilities - y_batch  # Gradient for softmax/cross-entropy layer
        G1 = (Z1 > 0).astype(float) * (G2 @ W2.T)  # Gradient for ReLU layer
        
        # Compute the gradients for weights
        grad_W1 = (X_batch.T @ G1) / batch
        grad_W2 = (Z1.T @ G2) / batch
        
        # Update the weights
        W1 -= lr * grad_W1
        W2 -= lr * grad_W2


# CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(
        np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1, 0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1, 0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr=0.2)
