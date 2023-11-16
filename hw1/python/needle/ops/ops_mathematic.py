"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

# NOTE: we will import array_api as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return Tensor(out_grad * self.scalar * array_api.power(node, self.scalar - 1))


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        a, b = node.inputs  # Assuming 'node' is a tuple containing (a, b)
        grad_a = out_grad / b
        grad_b = -out_grad * a / array_api.power(b, 2)
        return Tensor(grad_a), Tensor(grad_b)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad * ( 1 / self.scalar)


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        # 如果没有提供axes，我们创建一个默认的，将最后两个轴颠倒的axes
        if self.axes is None:
            axes_to_use = list(range(a.ndim))
            axes_to_use[-2], axes_to_use[-1] = axes_to_use[-1], axes_to_use[-2]
        else:
            # 根据提供的轴创建新的轴顺序
            axes_to_use = list(range(a.ndim))
            axes_to_use[self.axes[0]], axes_to_use[self.axes[1]] = axes_to_use[self.axes[1]], axes_to_use[self.axes[0]]

        # 使用确定的轴编号执行转置
        return array_api.transpose(a, axes=axes_to_use)


    def gradient(self, out_grad, node):
        # 使用逆转置顺序执行转置
        transposed_out_grad = self.compute(out_grad.cached_data)

        return Tensor(transposed_out_grad)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        if self.shape is None:
            return a
        print(f"把{a.shape}reshape成{self.shape}")
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        # 获取输入张量的形状
        input_shape = node.inputs[0].cached_data.shape

        grad = array_api.reshape(out_grad.cached_data, input_shape)
        return Tensor(grad)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, shape=self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape  # 获取输入张量的形状
        output_shape = self.shape  # 获取广播后的形状

        # 初始化梯度形状为全1，这将用于确定哪些维度上需要求和
        grad_shape = [1] * len(output_shape)

        # 对于输入张量的每一个维度，如果它没有被广播（即形状大小不为1），
        # 那么我们在该维度上的梯度形状应该与输入形状相同
        for i, (input_size, output_size) in enumerate(zip(input_shape, output_shape)):
            if input_size != 1:
                grad_shape[i] = input_size

        # 我们将 out_grad 求和以便使其形状与输入张量的形状相匹配
        grad = array_api.sum(out_grad.cached_data, axis=tuple(i for i, size in enumerate(grad_shape) if size == 1))
        
        # 确保梯度形状正确
        grad = array_api.reshape(grad, input_shape)

        return Tensor(grad)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            return array_api.sum(a)
        
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        input_value = node.inputs[0]

        if self.axes is not None:
            output_shape = [1 if axis in self.axes else size for axis, size in enumerate(input_value.shape)]
            grad = array_api.broadcast_to(array_api.reshape(out_grad.cached_data, output_shape), input_value.shape)
        else:
            # 创建一个与input_value形状相同的张量，每个元素都是out_grad
            grad = array_api.full(input_value.shape, out_grad.cached_data)

        print(f"Summation: 输入梯度是{out_grad}, 输出梯度是{grad}, 输入是{input_value}")
        return Tensor(grad)


        

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # 检查a和b是否为标量，如果是，则使用标量乘法
        if isinstance(a, (float, int)) and not isinstance(b, (float, int)):
            return array_api.multiply(a, b)
        elif isinstance(b, (float, int)) and not isinstance(a, (float, int)):
            return array_api.multiply(b, a)
        else:
            # 否则执行矩阵乘法
            return array_api.matmul(a, b)

    @staticmethod
    def match_shape(grad, original_shape):
        """
        Adjust the shape of gradient to match the original shape of the tensor.
        If grad has more dimensions than the original, sum along the extra axes.
        If grad has fewer dimensions, unsqueeze dimensions at the front.
        """
        # Sum extra dimensions if grad has more dimensions than original
        while grad.ndim > len(original_shape):
            grad = grad.sum(axis=0)

        # Add dimensions of size 1 if grad has fewer dimensions than original
        while grad.ndim < len(original_shape):
            grad = np.expand_dims(grad, axis=0)

        return grad

    def gradient(self, out_grad, node):
        a, b = node.inputs

        # Compute gradient for a
        print(f"原始的out_grad.cached_data的shape{out_grad.cached_data}, b的shape{b.cached_data.shape}")
        grad_a = self.compute(out_grad.cached_data, transpose(b).cached_data)
        print(f"原始的grad_a的shape{grad_a.shape}, a的shape{a.shape}")
        grad_a = self.match_shape(grad_a, a.shape)
        print(f"改造后的grad_a的shape{grad_a.shape}, a的shape{a.shape}")

        # Compute gradient for b
        grad_b = self.compute(transpose(a).cached_data, out_grad.cached_data)
        print(f"原始的grad_b的shape{grad_b.shape}, b的shape{b.shape}")
        grad_b = self.match_shape(grad_b, b.shape)
        print(f"改造后的grad_b的shape{grad_b.shape}, b的shape{b.shape}")
        return Tensor(grad_a), Tensor(grad_b)




def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return array_api.negative(a)
        
    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        # 获取前向传播时输入的值
        input_value = node.inputs[0]
        # 计算梯度
        grad = out_grad / input_value
        return grad


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(0, a.data)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


if __name__ ==  "__main__":
    print(PowerScalar(2).compute(array_api.array([1, 2, 3])))