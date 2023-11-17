"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
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
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # 打开并解压缩图像文件
    with gzip.open(image_filesname, 'rb') as image_file:
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


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # 计算 softmax 的每个元素的指数
    exp_Z = ndl.Exp()(Z)

    # 计算 softmax 分母部分：对每行的所有指数求和
    sum_exp_Z = ndl.Summation(axes=(1,))(exp_Z)

    # 计算 softmax 分子部分：获取每个样本的真实类别的 logits 的指数
    # 使用 y_one_hot 作为掩码选择每行的真实类别对应的元素
    correct_logit_exp = ndl.Summation(axes=(1,))(exp_Z * y_one_hot)

    # 计算损失：对于每个样本，计算 -log(分子/分母)
    # 也就是 -log(correct_logit_exp / sum_exp_Z)
    softmax_loss_per_sample = -ndl.Log()(correct_logit_exp / sum_exp_Z)

    # 计算平均损失
    average_loss = ndl.Summation()(softmax_loss_per_sample) / Z.shape[0]

    return average_loss


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """
    num_examples = X.shape[0]
    num_classes = W2.shape[1]
    
    # Convert y to one-hot encoding
    y_one_hot = np.eye(num_classes)[y]
    
    for start_idx in range(0, num_examples, batch):
        end_idx = min(start_idx + batch, num_examples)
        X_batch = ndl.Tensor(X[start_idx:end_idx])
        y_batch = ndl.Tensor(y_one_hot[start_idx:end_idx])
        
        Z1 = ndl.relu(X_batch @ W1)  # Activation from first layer
        logits = Z1 @ W2  # Logits for the current batch
        # Compute softmax loss
        loss = softmax_loss(logits, y_batch)

        # Backward pass
        loss.backward()

        # Update weights
        W1 = ndl.Tensor(W1.data - lr * W1.grad.data)
        W2 = ndl.Tensor(W2.data - lr * W2.grad.data)

        # Clear gradients
        W1.grad = None
        W2.grad = None
    return W1, W2


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
