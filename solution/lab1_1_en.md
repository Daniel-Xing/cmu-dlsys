# Deep learning systems from entry to abandonment - CMU-DL System Lab1-1 Forward Computing and Layer Propagation

Hello everyone, we meet again. Today we officially start to conquer lab1 (sharpening knives)

Starting with lab1, we are officially entering the development of the deep learning system framework, lab1 will help you start implementing a library called needle. Specifically, lab1 aims to build a basic **automatic differentiation **framework and then use it to re-implement the simple two-layer neural networks used in HW0 for MNIST number classification problems.

In this issue, we mainly focus on the first two problems of lab1: forward computing and layer. At the same time, we will introduce the key tensor and value types in lab in the second part of the problem. Without further ado, we will officially start ✍️.

# What is Neddle?

`needle`is an automatic differentiation library based on the `numpy`CPU backend, which will be gradually expanded over the course to include a linear algebra library with GPU code. In this assignment, you will use the Python language to implement the basics of automatic differentiation.

In the `needle`library, there are two important files: `python/needle/autograd.py`(which defines the basis of the computational graph framework and will become the basis of the automatic differentiation framework) and `python/needle/ops/ops_mathematic.py`(which contains implementations of various operators that you will use in assignments and courses).

While `autograd.py`document has established the basic framework for automatic differentiation, you should be familiar with the basic concepts of the library, especially the classes defined below:

- `Value`: The value computed in a computational graph, which can be the output of operations applied to other `Value`objects, or a constant (leaf) `Value`object. A generic class is used here (then dedicated to e.g. tensors) to make it easier to use other data structures in later versions of needle, but currently you interact with this class mainly through its subclass `Tensor`(see below).
- `Op`: Computes an operator in the graph. Operators need to define their "forward" procedure in the `compute()`method (i.e. how to compute the operator on the `Value`object's underlying data), and via the `gradient()`method to define their "reverse" procedure, i.e. how to multiply the passed output layer. Details of how to write such an operator are given below.
- `Tensor`: A subclass of `Value`that corresponds to the actual tensor output in the computational graph, a multidimensional array. All your code in this job (and most subsequent jobs) will use this subclass of `Value`instead of the generic class above. We provide several convenience functions (e.g. operator overloading) that allow you to manipulate tensors using normal Python conventions, but these functions will not work properly until you implement the corresponding operations.
- `TensorOp`: is a subclass of `Op`, used to return tensor operators. All operations you implement in this job will be of this type.

# Problem 1: Implementing Forward Computing

In question 1, we will implement the `compute`method for multiple classes to perform forward computation. For example, the `EWiseAdd`operator in the `ops/ops_mathematic.py`file, its `compute()`function performs forward propagation computation, that is, the direct computation operation itself, the input is the `NDArray`object. The `gradient()`function is responsible for computing the layer, and its argument is the `Tensor`object, which means that any calls inside the function should be done through the `TensorOp`operation. In addition, to simplify the operation, the auxiliary function `add()`is defined to more concisely implement the addition of two `Tensor`objects. Here is a list of `compute`methods for the operators we need to implement:

- `PowerScalar`: Boosts input to an integer (scalar) power level.
- `EWiseDiv`: True element-by-element division of the input (2 inputs).
- `DivScalar`: Divides the input element by element by a scalar (1 input, `scalar`- number).
- `MatMul`: Matrix multiplication of the inputs (2 inputs).
- `Summation`: Sum array elements on the specified axis (1 input, `axes`- tuple).
- `BroadcastTo`: Broadcast the array to a new shape (1 input, `shape`- tuple).
- `Reshape`: Provides a new shape for the array without changing the data (1 input, `shape`- tuple).
- `Negate`: Calculates the negative value of the input value, element by element (1 input).
- `Transpose`: Reverses the order of the two axes, defaults to the last two axes (1 input, `axes`- tuple).

Note that since in future jobs we will be using a backend other than `numpy`, `numpy`will be imported as a `array_api`, so we need to call functions like `array_api.add()`if we want to use a typical `np.X()`call.

## PowerScalar

`PowerScalar`class's `compute`method takes care of the forward computation. Specifically, it raises the input NDArray `a`to the integer power specified by `self.scalar`. In Python's `numpy`library, this computation can be done directly with the `numpy.power`function. Here is the corresponding formula:

$$
a^{\text{scalar}}
$$

Here, `a`is the input multidimensional array (NDArray), and `scalar`is an integer representing the power to which we want to raise each element of `a`.

In the code, this calculation is implemented as follows:

```python
def compute(self, a: NDArray) -> NDArray:
    return numpy.power(a, self.scalar)
```

## EWiseDiv

This class represents an operation that divides two nodes element by element.

`compute`method performs forward computation, which takes two NDArray objects `a`and `b`as input and returns the result of dividing them element by element. Element-by-element division means that each element in the output array is the quotient of the corresponding element of the input array. In `numpy`, this can be done simply using the `/`operator.

Here is the corresponding formula:

$$
a_i / b_i
$$

Here `a_i`and `b_i`are the corresponding elements in arrays `a`and `b`. In the code, the `compute`method is implemented as:

```python
def compute(self, a, b):
    return a / b
```

## DivScalar

`DivScalar`class implements the operation of dividing each element of a tensor `a`by a `scalar`. This is mathematically expressed as dividing each element `a_i`of a tensor `a`by `scalar`:

$$
\frac{a_i}{\text{scalar}}
$$

In the `compute`method, we use this simple mathematical operation. This is implemented in the code as:

```python
def compute(self, a):
    return a / self.scalar
```

Here, `a`is an NDArray object and `self.scalar`is the scalar value passed in when initializing the `DivScalar`class.

## MatMul

`MatMul``compute`method of class MatMul implements matrix multiplication between two matrices `a`and `b`. Matrix multiplication is a fundamental operation in linear algebra where each element is the dot product of the rows of the first matrix and the columns of the second. In Python, this can be simplified using the `@`operator:

$$
\mathbf{C} = \mathbf{A} @ \mathbf{B}
$$

Here, $\mathbf{A}$and $\mathbf{B}$are the input matrices and $\mathbf{C}$is the result matrix. In the code, the `compute`method is implemented as:

```python
def compute(self, a, b):
    return a @ b
```

## Summation

`Summation`class implements the operation of summing tensors. The method `compute`of this class is responsible for performing the actual summation computation.

If the `axes`parameter is not specified when initializing the `Summation`object, all elements of tensor `a`are globally summed by default. If `axes`are specified, only elements on the specified axis are summed.

`compute`method determines how to sum based on whether `axes`are specified:

- **Global Sum **: If `self.axes`is `None`, all elements of tensor `a`are summed.
- **Specify axis summation **: If `self.axes`is specified, summation is done only on those specific axes.

The code implementation is as follows:

```python
def compute(self, a):
    if self.axes is None:
        return array_api.sum(a)
  
    return array_api.sum(a, axis=self.axes)
```

In this code, `array_api.sum`is the function that performs the sum operation, `a`is the input tensor, and `self.axes`is a tuple of axes indicating the dimensions of the sum operation.

## BroadcastTo

`BroadcastTo`class is designed to extend the shape of a tensor to fit a new shape. This operation is common in deep learning, such as when you need to scale small-scale data to operate with large-scale data.

The `compute` method is responsible for executing the actual broadcasting operation, which uses the `array_api.broadcast_to` function. This function expands the input tensor `a` to the new shape defined by `self.shape`.

- If the shape of `a` can be extended to `self.shape` without copying data, broadcasting is carried out.
- If the shape of `a` cannot be broadcast to `self.shape`, an exception is usually thrown.

The code implementation is as follows:

```python
def compute(self, a):
    return array_api.broadcast_to(a, shape=self.shape)
```

In this code, `a` is the input tensor, and `self.shape` is the target shape to broadcast to. In this way, the `BroadcastTo` operation allows tensors of different shapes to be compatible in mathematical operations.

## Reshape

The `Reshape` class is used to change the shape of the input tensor `a` without altering its data. This is particularly useful in data preprocessing or when passing data between network layers, when you need to change the dimensions of the data to fit a specific operation. The `compute` method performs the actual shape-changing operation. It relies on the `array_api.reshape` function, which takes the original tensor `a` and a target shape `self.shape` as arguments, and returns a tensor with the new shape.

The code implementation is as follows:

```python
def compute(self, a):
    return array_api.reshape(a, self.shape)
```

Here, `a`is the input tensor and `self.shape`is the new shape we want `a`to reshape. The `Reshape`operation ensures that the total number of elements of tensor `a`remains the same, while allowing us to arrange these elements in new dimensions.

## Negate

`Negate`class implements a numeric inversion operation, which reverses the sign of all elements in the input tensor `a`. In mathematics and programming, inversion is a basic operation, often used to change the positive and negative of a numerical value. The `compute`method is responsible for performing the inversion operation. It uses the `array_api.negative`function, which takes the input tensor `a`and returns a new tensor. Each element of the new tensor is the negative value of the corresponding element in `a`.

The code implementation is as follows:

```python
def compute(self, a):
    return array_api.negative(a)
```

In this code, `a`is the input tensor. After executing `array_api.negative(a)`, we get a new tensor that contains the result of the inversion of the value of `a`. This operation is particularly useful in layer calculations and optimizations because it often involves layer direction inversion.

## Transpose

This operation can swap the positions of any two axes in a multidimensional array. Its purpose is to transpose the input multidimensional array `a`. A transpose operation usually means swapping rows and columns in a matrix (a two-dimensional array), but in a multidimensional array, a transpose can be more generalized to swap any two dimensions.

The code first checks if `self.axes` is provided, which is a tuple containing two elements that specify the axes to be swapped. If `self.axes` is not provided, the last two axes of the array are swapped by default. This is common in linear algebra operations with multidimensional arrays, especially when performing matrix multiplication.

If `self.axes` is provided, the code creates a new axis order based on the provided axes. Then, it uses the `array_api.transpose` function to perform the transposition, which takes the array `a` and the axis order `axes_to_use` as parameters.

The transpose operation can be described with the following pseudocode:

```markdown
# Use the last two axes by default if no axes are provided
if axes is None:
    transpose(a, axes=(-2, -1))
# If axes are provided, use the specified axes
else:
    transpose(a, axes=(axes[0], axes[1]))
```

In actual Python code, this logic would be implemented like this:

```python
def compute(self, a):
    if self.axes is None:
        axes_to_use = list(range(a.ndim))
        axes_to_use[-2], axes_to_use[-1] = axes_to_use[-1], axes_to_use[-2]
    else:
        axes_to_use = list(range(a.ndim))
        axes_to_use[self.axes[0]], axes_to_use[self.axes[1]] = axes_to_use[self.axes[1]], axes_to_use[self.axes[0]]
  
    return array_api.transpose(a, axes=axes_to_use)
```

In this code snippet, `a.ndim` indicates the number of dimensions of the array `a`, and `list(range(a.ndim))` creates an integer list from 0 to `a.ndim - 1`, representing the original axis order of the array `a`. Next, by swapping two elements within the list, we obtain a new axis order, which is then passed to the `array_api.transpose` function to perform the actual transposition operation.

## Execution Result

All tests passed ✅

![f1.png](https://p0-bytetech-private.bytetech.info/tos-cn-i-93o7bpds8j/11518998df6f40f78c36ff70dfb88afc~tplv-93o7bpds8j-compressed.awebp?policy=eyJ2bSI6MiwiY2siOiJieXRldGVjaCJ9&rk3s=5aaa0ea2&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1700035117&x-orig-sign=nWnP7%2B0MyRvardDJjvUZgd6UgZw%3D)

# Question 2: Implement Reverse Computation

## Necessary Background Knowledge

Primarily based on the chain rule, but before we delve into that, let's first take a look at a few data structures, mainly `Tensor` and `Value`.

In `python/needle/autograd.py`, a class named `Value` is defined, representing a node in a computational graph. This class is designed for use in automatic differentiation systems or deep learning frameworks to track and compute gradients. Here are some of the main features and methods of the class:

- **Attributes**:

  - `op`: An optional `Op` type representing the operation associated with the value.
  - `inputs`: A list of `Value` types representing the list of input values required to compute the current value.
  - `cached_data`: Cached data used to store the current value of the node to avoid redundant computation.
  - `requires_grad`: A boolean indicating whether the value needs gradient computation.
- **Methods**:

  - `realize_cached_data`: Compute or retrieve cached data. If `cached_data` is not empty, it returns the cached data directly; otherwise, it computes the data using the `compute` method of `op` and the list of input values.
  - `is_leaf`: Determine whether the value is a leaf node, i.e., a value without an `op`.
  - `__del__`: Destructor method used to update the global `TENSOR_COUNTER`, decreasing the counter when a value is deleted.
  - `_init`: Initialization method used to set up `op`, `inputs`, `num_outputs`, `cached_data`, and `requires_grad`. If `requires_grad` is not explicitly provided, it is determined based on the `requires_grad` attribute of the input values.
  - `make_const`: A class method for creating a constant value, i.e., a value without an operation and inputs.
  - `make_from_op`: A class method for creating a `Value` object based on an operation and a list of inputs. If not in lazy mode (`LAZY_MODE` is false), and the value does not require gradients, it will return a detached value. Otherwise, it immediately computes and caches the data.

The design of this class allows for the construction of a computational graph containing all intermediate operations and values, enabling automatic computation of gradients through the backward propagation algorithm. This is foundational to modern deep learning libraries such as PyTorch and TensorFlow.

`Tensor`class, which is a subclass of the `Value`class for implementing tensor operations in the deep learning framework. A tensor is a multidimensional array, which is at the heart of data representation in deep learning. `Tensor`class provides an interface for creating and manipulating these multidimensional arrays, while supporting automatic layer computation for use in backpropagation. Here are some of the main features and methods of the class:

| type                         | name                  | describe                                                                             |
| ---------------------------- | --------------------- | ------------------------------------------------------------------------------------ |
| **attribute**          | `grad`              | Tensor of the storage layer.                                                         |
| **constructor**        | `__init__`          | Initialize tensor to specify array, device, data type, and layer requirements.       |
| **static method**      | `_array_from_numpy` | Convert a NumPy array to a tensor.                                                   |
|                              | `make_from_op`      | Create tensors based on operations and input lists.                                  |
|                              | `make_const`        | Create a constant tensor.                                                            |
| **attribute accessor** | `data`              | Gets or sets tensor data.                                                            |
|                              | `shape`             | Returns the tensor shape.                                                            |
|                              | `dtype`             | Returns the data type.                                                               |
|                              | `device`            | Returns the device where the data resides.                                           |
| **method**             | `detach`            | Create a new tensor that shares data but does not participate in layer computations. |
|                              | `backward`          | Start backpropagation to compute the gradient.                                       |
|                              | `numpy`             | Convert the tensor into a NumPy array.                                               |
| **魔术方法**           | `__add__`等         | Overload arithmetic operators to support direct tensor computation.                  |
|                              | `__matmul__`        | Overload the matrix multiplication operator `@`.                                   |
|                              | `__neg__`           | Overload the negation operation `-`.                                               |
| **Other methods**      | `sum`               | Calculate the sum.                                                                   |
|                              | `broadcast_to`      | Broadcast the tensor to a new shape.                                                 |
|                              | `reshape`           | Change the shape of the tensor.                                                      |
|                              | `transpose`         | Transpose the tensor.                                                                |

## ElementWise Add: A Small Example

### Method of Gradient Computation

For the `EWiseAdd` operation, the given output gradient (`out_grad`) is directly the gradient for each input tensor of the operation. In other words, if we have a scalar function $L$ which is the final loss function, and `a` and `b` are the inputs to the `EWiseAdd` operation, then according to the chain rule:

$$
\frac{\partial L}{\partial a} = \frac{\partial L}{\partial (a+b)} \cdot \frac{\partial (a+b)}{\partial a}
$$

Since the derivative of $a + b$ with respect to $a$ is 1, we can simplify to:

$$
\frac{\partial L}{\partial a} = \frac{\partial L}{\partial (a+b)}
$$

The same applies to $b$:

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial (a+b)}
$$

Therefore, regardless of the specific contents of the input tensors `a` and `b`, the gradient for the `EWiseAdd` operation is the output gradient (`out_grad`) itself.

### Implementation in Code

```python
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        # Directly return the element-wise sum of input tensors a and b
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Return the same gradient as the output gradient to each input node
        return out_grad, out_grad
```

The `gradient` method in this code returns a tuple `(out_grad, out_grad)`, where each `out_grad` is the gradient propagated to each input tensor `a` and `b`. In the context of automatic differentiation, this means that the gradient of the loss function relative to each input of the `EWiseAdd` operation is the same as the gradient of the loss function relative to the output of that operation.

## PowerScalar

The `PowerScalar` operator takes an integer scalar `scalar` and multiplies each element $ a_i $ of the input tensor `a` by itself `scalar` times. The mathematical expression is:

$$
a_i^{\text{scalar}}
$$

### Gradient Computation

When we need to perform backpropagation on the `PowerScalar` operator, we need to compute the gradient with respect to the input `a`. According to the chain rule, if there is a scalar function $ L $ which is the final loss function, then the gradient of `a` is:

$$
\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial (a_i^{\text{scalar}})} \cdot \frac{\partial (a_i^{\text{scalar}})}{\partial a_i}
$$

Given that the derivative of $ a_i^{\text{scalar}} $ with respect to $ a_i $ is $ \text{scalar} \cdot a_i^{\text{scalar} - 1} $, we can further expand the expression:

$$
\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial (a_i^{\text{scalar}})} \cdot \text{scalar} \cdot a_i^{\text{scalar} - 1}
$$

This means that the gradient of each element $ a_i $ in the input tensor `a` is the externally passed gradient (i.e., $ \frac{\partial L}{\partial (a_i^{\text{scalar}})} $) multiplied by `scalar` and then by $ a_i $ to the power of `scalar - 1`.

### Code Implementation

```python
class PowerScalar(TensorOp):
    """Op to raise a tensor to an (integer) power."""
  
    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        # Compute the scalar power of each element in the input tensor a
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Compute the gradient
        return Tensor(out_grad * self.scalar * array_api.power(node, self.scalar - 1))
```

## EWiseDiv

The `EWiseDiv` operator element-wise divides one node (tensor) `a` by another node `b`. The mathematical expression is:

$$
c_i = \frac{a_i}{b_i}
$$

where $ c_i $ is the ith element of the result tensor.

### Gradient Computation

To backpropagate errors in neural networks, we need to compute the gradients of the `EWiseDiv` operation with respect to its two inputs. According to the chain rule, if there is a scalar function $ L $ that represents the final loss function, the gradients with respect to `a` and `b` are respectively:

For $ a $:

$$
\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial c_i} \cdot \frac{\partial c_i}{\partial a_i}
$$

Since $ \frac{\partial c_i}{\partial a_i} = \frac{1}{b_i} $, we have:

$$
\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial c_i} \cdot \frac{1}{b_i}
$$

For $ b $:

$$
\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial c_i} \cdot \frac{\partial c_i}{\partial b_i}
$$

Given that $ \frac{\partial c_i}{\partial b_i} = -\frac{a_i}{b_i^2} $, we get:

$$
\frac{\partial L}{\partial b_i} = -\frac{\partial L}{\partial c_i} \cdot \frac{a_i}{b_i^2}
$$

### Code Implementation

```python
class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""
  
    def compute(self, a, b):
        # Return the element-wise quotient of input tensors a and b
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs 
        # Gradient for a
        grad_a = out_grad / b
        # Gradient for b
        grad_b = -out_grad * a / array_api.power(b, 2)
        return Tensor(grad_a), Tensor(grad_b)
```

## DivScalar

The `DivScalar` operator divides each element $ a_i $ of the input tensor `a` by a scalar value `scalar`. The mathematical expression is:

$$
c_i = \frac{a_i}{\text{scalar}}
$$

where $ c_i $ is the ith element of the result tensor.

### Gradient Computation

For the `DivScalar` operator, we need to compute its gradient relative to the input tensor `a`. According to the chain rule, if there is a scalar function $ L $ representing the final loss function, then the gradient for each element $ a_i $ of the input tensor `a` is:

$$
\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial c_i} \cdot \frac{\partial c_i}{\partial a_i}
$$

Since `scalar` is a constant, the value of $ \frac{\partial c_i}{\partial a_i} $ is $ \frac{1}{\text{scalar}} $. Thus, we get:

$$
\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial c_i} \cdot \frac{1}{\text{scalar}}
$$

### Code Implementation

```python
class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        # Return the quotient of input tensor a and scalar
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        # Return the gradient for input tensor a
        return out_grad * (1 / self.scalar)
```

## MatMul

Matrix multiplication is a fundamental and commonly used operation in neural networks for linear transformations between layers. The `MatMul` operator implements matrix multiplication between two tensors. For gradient computation, especially when involving batch processing and tensors of different dimensions, special handling is required to ensure that the gradient's shape matches the original tensor.

The `MatMul` operator performs matrix multiplication between two tensors `a` and `b`. If the shape of tensor `a` is $ m \times n $ and the shape of tensor `b` is $ n \times p $, then the shape of the result tensor will be $ m \times p $. The mathematical expression is:

$$
C = AB
$$

where $ C $ is the result matrix, $ A $ is the left matrix, and $ B $ is the right matrix.

### Gradient Computation

Assuming there is a scalar function $ L $ representing the loss function, we need to compute the gradients of the `MatMul` operation with respect to its inputs `a` and `b`. Using the chain rule, we have:

The gradient for $ A $:

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T
$$

The gradient for $ B $:

$$
\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}
$$

where $ B^T $ and $ A^T $ denote the transpose of $ B $ and $ A $ respectively.

### Code Implementation

```python
class MatMul(TensorOp):
    def compute(self, a, b):
        # Perform matrix multiplication
        return array_api.matmul(a, b)

    @staticmethod
    def match_shape(grad, original_shape):
        # Adjust the shape of the gradient to match the shape of the original tensor
        # If the gradient has more dimensions than the original shape, sum along the extra axes
        # If the gradient has fewer dimensions than the original shape, prepend dimensions
        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0)
        while grad.ndim < len(original_shape):
            grad = np.expand_dims(grad, axis=0)
        return grad

    def gradient(self, out_grad, node):
        a, b = node.inputs
        # Compute the gradient for a
        grad_a = self.compute(out_grad.cached_data, transpose(b).cached_data)
        grad_a = self.match_shape(grad_a, a.shape)
        # Compute the gradient for b
        grad_b = self.compute(transpose(a).cached_data, out_grad.cached_data)
        grad_b = self.match_shape(grad_b, b.shape)
        return Tensor(grad_a), Tensor(grad_b)
```

In the `gradient` method, we first calculate the gradients for tensors `a` and `b` using the chain rule. Then, we use the `match_shape` static method to adjust the shape of the gradient to ensure that the gradient tensor matches the shape of the original input tensor.

## Summation

The `Summation` operator performs a summation operation on the input tensor `a`, which can be specified along certain axes or for the entire tensor. The mathematical expressions are as follows:

- Summing over all axes (if `axes` is `None`):

$$
S = \sum_i a_i
$$

- Summing over specific axes (if `axes` is not `None`):

$$
S_{j_1, \ldots, j_m} = \sum_{k} a_{j_1, \ldots, j_{m-1}, k, j_{m+1}, \ldots}
$$

where $ j_1, \ldots, j_m $ are indices on the non-summed axes, and $ k $ is the index on the summed axis.

### Gradient Computation

According to the chain rule of differentiation, the gradient computation for the summation operation is as follows:

- If the summation is over all axes, the gradient will be a tensor with the same shape as the original input tensor `a`, where each element is the gradient of the summation result `out_grad`.
- If the summation is over specific axes, the gradient will be a broadcast tensor with the size of 1 on the summed axis and consistent with the input tensor `a` on the other axes.

The mathematical expression is:

$$
\frac{\partial L}{\partial a_i} =
\begin{cases}
out\_grad, & \text{if sum over all axes} \\
broadcast(out\_grad), & \text{if sum over certain axes}
\end{cases}
$$

where $ L $ is the loss function.

### Code Implementation

```python
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # Sum along the specified axes
        return array_api.sum(a, axis=self.axes) if self.axes is not None else array_api.sum(a)

    def gradient(self, out_grad, node):
        input_value = node.inputs[0]  # The input tensor a

        # Adjust the shape of the gradient to match the shape of the input tensor a, based on the summed axes
        if self.axes is not None:
            # Create a new shape with the size of 1 for the summed axes and unchanged for the other axes
            output_shape = [size if axis not in self.axes else 1 for axis, size in enumerate(input_value.shape)]
            # Broadcast the gradient to the new shape
            grad = array_api.broadcast_to(Reshape(output_shape).compute(out_grad.cached_data), input_value.shape)
        else:
            # If summing over all axes, the gradient is a tensor filled with the out_grad value and with the same shape as the input tensor a
            grad = array_api.full_like(input_value, out_grad.cached_data)

        return Tensor(grad)
```

## BroadcastTo

The `BroadcastTo` operator expands the input tensor `a` to a given shape `shape`. Mathematically, for each element $ a_{i_1, \ldots, i_n} $ of tensor `a`, if broadcasting is performed on the given dimension, then all elements along that dimension will have the same value as $ a_{i_1, \ldots, i_n} $.

### Gradient Computation

When performing backpropagation, we must reduce the broadcasted gradient `out_grad` to the original shape of the input tensor `a`. Since in the broadcasting process, a single element of the original tensor is copied to multiple locations, the computation of the gradient needs to sum over the broadcasted dimensions. Specifically, if a dimension was broadcasted (i.e., the original size in that dimension is 1), then the gradients on that dimension will be summed.

The mathematical expression is:

$$
\frac{\partial L}{\partial a_{i_1, \ldots, i_n}} = \sum_{j_1, \ldots, j_m} \frac{\partial L}{\partial c_{j_1, \ldots, j_m}}
$$

where $ \frac{\partial L}{\partial c_{j_1, \ldots, j_m}} $ represents the gradient of the broadcasted tensor, and $ \{j_1, \ldots, j_m\} $ is the set of indices where $ a_{i_1, \ldots, i_n} $ was copied in the broadcasting operation.

### Code Implementation

```python
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # Perform the broadcast operation
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape  # The shape of the input tensor
        output_shape = self.shape  # The shape after broadcasting

        # Initialize the gradient shape to all 1s to determine the summed dimensions
        grad_shape = [1] * len(output_shape)

        # For each dimension, if not broadcasted (size is not 1), make the gradient shape the same as the input
        for i, (input_size, output_size) in enumerate(zip(input_shape, output_shape)):
            if input_size != 1:
                grad_shape[i] = input_size

        # Sum over the broadcasted dimensions to match the input shape
        grad = array_api.sum(out_grad.cached_data, axis=tuple(i for i, size in enumerate(grad_shape) if size == 1))
  
        # Adjust the shape of the gradient
        grad = array_api.reshape(grad, input_shape)

        return Tensor(grad)
```

## Reshape

The `Reshape` operator allows us to change the shape of a tensor without altering the contained data and its order. Suppose we have a tensor of shape $ m \times n $, we can reshape it into a new tensor of shape $ p \times q $, as long as $ m \times n = p \times q $.

### Gradient Computation

The `Reshape` operation itself does not change the data in the tensor, only the arrangement of the data. Therefore, in backpropagation, propagating the gradient to the `Reshape` operation simply requires reshaping the gradient back to the tensor shape before the operation. Specifically, the transformation of the gradient shape must follow the rule:

$$
\text{If} \quad a \in \mathbb{R}^{m \times n} \quad \text{is reshaped to} \quad b \in \mathbb{R}^{p \times q}
$$

$$
\text{Then} \quad \frac{\partial L}{\partial a} \quad \text{can be obtained by reshaping} \quad \frac{\partial L}{\partial b} \quad \text{back to a shape of} \quad m \times n
$$

where $ L $ is the loss function.

### Code Implementation

```python
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        # Perform the reshape operation
        return array_api.reshape(a, self.shape) if self.shape is not None else a

    def gradient(self, out_grad, node):
        # Get the shape of the input tensor
        input_shape = node.inputs[0].cached_data.shape

        # Reshape the output gradient back to the shape of the input tensor
        grad = array_api.reshape(out_grad.cached_data, input_shape)
        return Tensor(grad)
```

## Negate

The `Negate` operator negates each element in the input tensor `a`. If the elements in the input tensor `a` are $ a_i $, then after the `Negate` operator, the elements become $ -a_i $. The mathematical expression is:

$$
c_i = -a_i
$$

where $ c_i $ is the ith element of the result tensor.

### Gradient Computation

For the `Negate` operator, we need to compute its gradient with respect to the input tensor `a`. According to the chain rule, if there is a scalar function $ L $ representing the final loss function, the gradient for the input tensor `a` is:

$$
\frac{\partial L}{\partial a_i} = \frac{\partial L}{\partial c_i} \cdot \frac{\partial c_i}{\partial a_i}
$$

Since the derivative of $ c_i $ with respect to $ a_i $ is -1, we obtain:

$$
\frac{\partial L}{\partial a_i} = -\frac{\partial L}{\partial c_i}
$$

This means that the gradient for the `Negate` operator is simply the negation of the output gradient `out_grad`.

### Code Implementation

```python
class Negate(TensorOp):
    def compute(self, a):
        # Compute the element-wise negation of input tensor a
        return array_api.negative(a)
  
    def gradient(self, out_grad, node):
        # Return the negation of the output gradient
        return -out_grad
```

## Transpose

The `Transpose` operator can swap the dimensions of a tensor according to the specified `axes` parameter. If `axes` is not specified, by default, it swaps the last two dimensions. For example, if the input is a matrix (2-dimensional tensor), the transpose operation will swap its rows and columns.

### Gradient Computation

For the transpose operation, the gradient computation is relatively straightforward. The gradient must be transposed back to ensure that the gradient of each element returns to its original position. Mathematically, if a tensor of shape $ m \times n $ is transposed to a new tensor of shape $ n \times m $, then the gradient will also transpose back from $ n \times m $ to $ m \times n $.

If the transpose operation is represented by a permutation matrix $ P $ such that $ B = PA $, then the gradient of $ A $ with respect to $ B $, $ \frac{\partial L}{\partial A} $, can be obtained through $ P^T \frac{\partial L}{\partial B} $, where $ L $ is the loss function.

### Code Implementation

```python
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # Determine the order of the axes for transposition
        axes_to_use = self.axes or list(range(a.ndim))[::-1]
        return array_api.transpose(a, axes=axes_to_use)

    def gradient(self, out_grad, node):
        # Compute the gradient of the transpose, i.e., the reverse transpose operation
        transposed_out_grad = self.compute(out_grad.cached_data)
        return Tensor(transposed_out_grad)
```

## 结果

![f2.png](https://p0-bytetech-private.bytetech.info/tos-cn-i-93o7bpds8j/fa12e47aecf94e84b57bb467a99409df~tplv-93o7bpds8j-compressed.awebp?policy=eyJ2bSI6MiwiY2siOiJieXRldGVjaCJ9&rk3s=5aaa0ea2&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1700035117&x-orig-sign=QZ0JYjSiiT6XF4SRa4EQpg4Vf1c%3D)

# 总结

This blog post has completed the forward computation and gradient propagation logic for some operators. Of course, lab1 contains some other exercises, so let's meet in the next blog post. Goodbye for now 👋.
