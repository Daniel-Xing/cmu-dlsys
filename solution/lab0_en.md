# Deep learning system from entry to abandonment - CMU-DL System Lab0

> Open a new pit, open a new pit

[CMU-DL System ](https://dlsyscourse.org/assignments)is a course taught by Chen Tianqi, which aims to reveal the core principles of ML systems and cooperate with some high-quality assignments. If you study a complete course, you will have a more comprehensive understanding of the basic principles of pytorch and tensorflow. Wouldn't it be fulfilling to complete the assignment?

The relevant code is on GitHub: [danielxing-cmu-system](https://github.com/Daniel-Xing/cmu-dlsys)

# Lab0: Self-test before class

Lab0 provides some test questions that help learners evaluate whether their background meets the prerequisites for completing these courses.

First, you need to go to Github to clone Lab0. The address is:

After the cloning is complete, we can see that the main directory structure is:

```Bash
data/
	train-images-idx3-ubyte.gz
	train-labels-idx1-ubyte.gz
	t10k-images-idx3-ubyte.gz
	t10k-labels-idx1-ubyte.gz
src/
	simple_ml.py
	simple_ml_ext.cpp
tests/
	test_simple_ml.py
Makefile
```

## Problem 1: Implementing the add method

First, you need to implement a simple add method in your `simple_ml.py`. The skeleton given is as follows:

```python
def add(x, y):
    """ A trivial 'add' function you should implement to get used to the 
    autograder and submission system.  The solution to this problem is in
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
```

The implementation is very simple, just add it directly.

```Python
def add(x, y):
	...

	### BEGIN YOUR CODE

	return x + y

	### END YOUR CODE
```

## Problem 2: Loading the Minist dataset

Read the dataset according to the prompts

```python
import gzip
import numpy as np

def parse_mnist(image_filename, label_filename):
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

```

## Problem 3: Implementing SoftMax Loss

Implement the softmax loss (aka cross entropy loss) function in the `src/simple_ml.py`file, the `softmax_loss()`function. To recap (hopefully this is a refresher, but we will also cover it in the September 1st class), for multi-class outputs that can take the value $y \in \{1,\ldots,k\}$, the softmax loss function accepts a vector $z \in \mathbb{R}^k$As input, this vector contains the logarithmic probability, and a real class $y \in \{1,\ldots,k\}$, and returns the loss defined as follows:

$$
\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.
$$

Note that, as described in its documentation string, `softmax_loss()`takes a _ two-dimensional array _ logits (i.e., the (k) -dimensional logits of different samples in a batch), and a corresponding one-dimensional array's real label, should output the average _ of the _softmax loss for the entire batch. Note that in order to perform this calculation correctly, you should _ not use any looping _, but use numpy's vectorization operation entirely (to set the expected value, we should note that our reference solution consists of one line of code).

Note that for a "real" softmax loss implementation, you'll want to scale the logits to prevent numeric overflow, but here we don't need to worry about that (even if you don't think about it, the rest of the job will work fine). The following code runs the test case.

```Python
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
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
```

## Problem 4: Random layer descent for Softmax regression

In this problem, you will implement random layer descent (SGD) for (linear) Softmax regression. In other words, as we discussed in our September 1 class, we will consider a hypothesis function that converts $n$-dimensional inputs to $k$-dimensional logits (logits) by the following function:

$$
h(x) = \Theta^T x
$$

Where $x \in \mathbb{R}^n$is the input and $\Theta \in \mathbb{R}^{n \times k}$is the model parameter. Given a dataset $\{(x^{(i)} \in \mathbb{R}^n, y^{(i)} \in \{1,\ldots,k\})\}$, for $i=1,\ldots,m$, the optimization problem associated with Softmax regression is given by:

$$
\DeclareMathOperator*{\minimize}{minimize}
\minimize_{\Theta} \; \frac{1}{m} \sum_{i=1}^m \ell_{\mathrm{softmax}}(\Theta^T x^{(i)}, y^{(i)}).
$$

Recall from the class that the layer of the linear Softmax target is given as follows:

$$
\nabla_\Theta \ell_{\mathrm{softmax}}(\Theta^T x, y) = x (z - e_y)^T
$$

among them

$$
z = \frac{\exp(\Theta^T x)}{1^T \exp(\Theta^T x)} \equiv \text{normalize}(\exp(\Theta^T x))
$$

(That is, $z$is just the normalized Softmax probability), and $e_y$represents the $y$-th unit basis, that is, a vector that is 1 at the $y$-th position and 0 at all other positions.

We can also express this in the more compact notation we discussed in class. That is, if we let $X \in \mathbb{R}^{m \times n}$represent the design matrix of some $m$inputs (whether an entire dataset or a small batch), $y \in \{1,\ldots,k\}^m$corresponding label vector, and overload $\ell_{\mathrm{softmax}}$to refer to the average Softmax loss, then

$$
\nabla_\Theta \ell_{\mathrm{softmax}}(X \Theta, y) = \frac{1}{m} X^T (Z - I_y)
$$

among them

$$
Z = \text{normalize}(\exp(X \Theta)) \quad \text{（归一化应用于逐行）}
$$

Represents a logarithmic probability matrix, and $I_y \in \mathbb{R}^{m \times k}$represents the merging of one-hot encodings of labels in $y$.

Using these layers, implement the `softmax_regression_epoch()`function that runs the SGD for a single cycle (one pass of the dataset) with the specified learning rate/step `lr`and small batch size `batch`. As described in the documentation string, your function should modify the `Theta`array in place. Once implemented, run the test.

```Python
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
```

## Problem 5: Random layer descent (SGD) in two-layer neural networks

Now that you've written random layer descent (SGD) for a linear classifier, let's consider the case of a simple two-layer neural network. Specifically, for the input (x\ in\ mathbb {R} ^ n), we'll consider a two-layer neural network without a bias term of the form:

$$
z = W_2^T \mathrm{ReLU}(W_1^T x)
$$

Here (W_1\ in\ mathbb {R} ^ {n\ times d}) and (W_2\ in\ mathbb {R} ^ {d\ times k}) represent the weights of the network (the network has a hidden unit of (d) dimension), and (z\ in\ mathbb {R} ^ k) represent the logits output by the network. Again, we use softmax/cross entropy loss, which means we want to solve the following optimization problem:

$$
\minimize_{W_1, W_2} \;\; \frac{1}{m} \sum_{i=1}^m \ell_{\mathrm{softmax}}(W_2^T \mathrm{ReLU}(W_1^T x^{(i)}), y^{(i)}).
$$

Or, using the matrix (X\ in\ mathbb {R} ^ {m\ times n}) to describe the case in batch form, it can also be written as:

$$
\minimize_{W_1, W_2} \;\; \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y).
$$

Using the chain rule, we can derive backpropagation updates to this network (which we will briefly cover in class 9/8, but the final form is also provided here for ease of implementation). Specifically, let:

$$
\begin{split}
Z_1 \in \mathbb{R}^{m \times d} & = \mathrm{ReLU}(X W_1) \\
G_2 \in \mathbb{R}^{m \times k} & = \normalize(\exp(Z_1 W_2)) - I_y \\
G_1 \in \mathbb{R}^{m \times d} & = \mathrm{1}\{Z_1 > 0\} \circ (G_2 W_2^T)
\end{split}
$$

Here (\ mathrm {1} {Z_1 > 0}) is a binary matrix with elements equal to zero or one, depending on whether each term in (Z_1) is strictly positive, and (\ circ) represents multiplication by elements. Then the layer of the target is given by:

$$
\begin{split}
\nabla_{W_1} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} X^T G_1  \\
\nabla_{W_2} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} Z_1^T G_2.  \\
\end{split}
$$

Using these layers, now write the `nn_epoch()`function in the `src/simple_ml.py`file. As with the previous problem, your solution should modify the `W1`and `W2`arrays in place. After implementing the function, run the following test. Be sure to implement the function using the matrix operation indicated by the above expression: This will be much faster _ than trying to use a loop _, and more efficient (and requires far less code).

In the above course description, the corresponding formula has been given very fully, and it can be implemented in comparison.

```Python
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
```

## Problem 6: Implementing Softmax Regression with C++

Use C++ to rewrite question 4. Since the native C++ is used, there is a lot of code that needs to be rewritten. Here is the code given in the title.

```cpp
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y, 
								  float *theta, size_t m, size_t n, size_t k, 
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a 
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     * 
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row 
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     * 
     * Returns:
     *     (None)
     */

    /// YOUR CODE HERE
  
    /// END YOUR CODE
}
```

According to the thinking behind question 4, the main steps are divided into three steps:

1. According to the small batch, the logits are calculated.
2. Compute layer
3. Update parameters according to layer

Therefore, we should be able to write the following code framework

```cpp
// 定义softmax回归训练迭代的函数
// 输入参数包括：特征矩阵X，目标值数组y，参数矩阵theta，样本数量m，特征数量n，类别数量k，
// 学习率lr，以及每批处理的样本数量batch
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    // 为logits（逻辑函数的输出）和梯度分配内存空间
    std::vector<float> logits(batch * k);
    std::vector<float> gradients(batch * k);

    // 通过迭代处理每个批次的样本
    for (size_t i = 0; i < m; i += batch) {
        // 确定当前批次的实际大小（处理最后一个批次时可能不满）
        size_t current_batch_size = std::min(batch, m - i);

        // 计算当前批次的logits。'X + i * n'定位到当前批次的第一个样本的特征数据起始点。
        dot_product(X + i * n, theta, logits.data(), current_batch_size, n, k);

        // 将原始logits数组的内容复制到std::vector中，以方便后续处理
        std::vector<float> logits_vector(logits.data(), logits.data() + current_batch_size * k);
  
        // 为当前批次初始化梯度向量
        std::vector<float> gradient_vector(current_batch_size * k);

        // 调用softmax_grad函数计算梯度
        softmax_grad(logits_vector, std::vector<unsigned char>(y + i, y + i + current_batch_size),
                     current_batch_size, k, gradient_vector);

        // 更新theta参数矩阵
        // 对于当前批次中的每个样本（由外循环j控制）
        for (size_t j = 0; j < current_batch_size; ++j) {
            // 对于每个类别（由中循环c控制）
            for (size_t c = 0; c < k; ++c) {
                // 对于每个特征（由内循环d控制）
                for (size_t d = 0; d < n; ++d) {
                    // 'theta[d * k + c]'是theta矩阵中对应于第d个特征对第c个类别的权重
                    // 它是一维数组中的索引，但代表二维矩阵的位置（d行c列）
                    // 'X[(i + j) * n + d]'是当前批次中第j个样本的第d个特征值
                    // 这个值与对应的梯度和学习率相乘后，用于更新theta矩阵的对应权重
                    // 这里进行的操作是梯度下降步骤，用于优化损失函数
                    theta[d * k + c] -= lr * gradient_vector[j * k + c] * X[(i + j) * n + d];
                }
            }
        }
    }
}

```

At this time, only two functions need to be completed dot_product and softmax_grad the implementation of these two functions. The reference implementation is given below:

```cpp
void dot_product(const float* A, const float* B, float* C, size_t A_rows, size_t A_cols, size_t B_cols) {
    // Initialize C with zeros
    std::fill(C, C + A_rows * B_cols, 0.0f);

    // Compute the dot product
    for (size_t i = 0; i < A_rows; ++i) {            // Iterate over the rows of A
        for (size_t j = 0; j < B_cols; ++j) {        // Iterate over the columns of B
            for (size_t k = 0; k < A_cols; ++k) {    // Dot product calculation
                C[i * B_cols + j] += A[i * A_cols + k] * B[k * B_cols + j];
            }
        }
    }
}
```

```cpp
// Helper function to compute the softmax probabilities for a vector
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0;

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum_exp += probabilities[i];
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] /= sum_exp;
    }

    return probabilities;
}

// Function to compute the gradient of the softmax loss
void softmax_grad(const std::vector<float>& Z, const std::vector<unsigned char>& y,
                  size_t batch_size, size_t num_classes, std::vector<float>& gradient) {
    std::vector<float> probabilities;

    // Compute softmax probabilities
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<float> logits(Z.begin() + i * num_classes, Z.begin() + (i + 1) * num_classes);
        std::vector<float> probs = softmax(logits);

        // Subtract 1 from the probability of the correct class
        probs[y[i]] -= 1;

        // Copy the probabilities back into the gradient vector
        std::copy(probs.begin(), probs.end(), gradient.begin() + i * num_classes);
    }

    // Average the gradient over the batch
    for (size_t i = 0; i < gradient.size(); ++i) {
        gradient[i] /= static_cast<float>(batch_size);
    }
}
```

Finally, run the test function

![F6.png](https://p0-bytetech-private.bytetech.info/tos-cn-i-93o7bpds8j/1424e4b589564634b384339c1b77db8d~tplv-93o7bpds8j-compressed.awebp?policy=eyJ2bSI6MiwiY2siOiJieXRldGVjaCJ9&rk3s=5aaa0ea2&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1700029061&x-orig-sign=EVE9FicCIybDR3iLMLWPaLWHZ3o%3D)

## summary

Lab0 is mainly used for self-inspection, and it gives a good introduction from loading data, implementing loss, and implementing layer updates. The relevant tips are all in place.

But the author almost forgot about c ++, problem 6 is still more laborious to do 😅
