import os
import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/guide/create_op?hl=zh-cn
lib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cmake-build-debug/libtfop.dylib')
my_ops = tf.load_op_library(lib_path)


def dynamic_partition_softmax(input_tensor, indices):
    return my_ops.dynamic_partition_softmax(input_tensor=input_tensor, indices=indices)


@tf.RegisterGradient("DynamicPartitionSoftmax")
def _dynamic_partition_softmax_grad(op, grad):
    output_tensor = op.outputs[0]
    indices = op.inputs[1]
    grad_out = my_ops.dynamic_partition_softmax_grad(
        input_tensor=output_tensor, grad_tensor=grad, indices=indices)

    return grad_out, None


def graph_reduce(src, dst, adjs, reduce_method='mean'):
    assert reduce_method in ("sum", "mean")
    return my_ops.graph_reduce(reduce_method=reduce_method, src_tensor=src,
                               dst_tensor=dst, adj_tensor=adjs)


@tf.RegisterGradient("GraphReduce")
def _graph_reduce_grad(op, grad):
    reduce_method = op.get_attr('reduce_method')
    adjs = op.inputs[2]
    grad_out = my_ops.graph_reduce_grad(
        reduce_method=reduce_method, grad_tensor=grad, adj_tensor=adjs)

    return None, grad_out, None


def softmax(array):
    ips = np.exp(np.array(array, dtype=np.float32))
    sum_value = ips.sum()
    return list(ips / sum_value)


def softmax_grad(array, grad):
    softmax_out = softmax(array)
    grad = np.array(grad, dtype=np.float32)

    return list((grad - np.dot(softmax_out, grad)) * softmax_out)


class MyLibTest(tf.test.TestCase):
    def testGraphReduce(self):
        with self.test_session():
            src = tf.constant([1.2, 0.5, 3.1, 2.2, 4.3, 7.6], dtype=tf.float32)
            # dst = tf.constant([1.2, 0.5, 3.1, 2.2, 4.3, 7.6, 5.1, 7.3, 1.5], dtype=tf.float32)
            dst = tf.constant([[1.2, 4.5], [0.5, 4.2], [3.1, 12],
                               [2.2, 4.9], [4.3, 5.5],
                               [7.6, 5.9], [5.1, 6.1], [7.3, 5.6], [1.5, 3.1]],
                              dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 0], [1, 1],
                                [2, 0],
                                [3, 0], [3, 1],
                                [5, 0]], dtype=tf.int32)
            reduce_out = graph_reduce(src, dst, adjs)
            print(reduce_out.eval())

    def testGraphReduceGrad(self):
        with self.test_session():
            # grad_inputs = tf.constant([1.2, 0.5, 3.1], dtype=tf.float32)
            grad_inputs = tf.constant([[1.2, 4.5], [0.5, 4.2], [3.1, 12]], dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 0], [1, 1],
                                [2, 0], [2, 1], [2, 2], [2, 3]], dtype=tf.int32)
            grad_out = my_ops.graph_reduce_grad(reduce_method="sum",
                                                grad_tensor=grad_inputs, adj_tensor=adjs)
            print(grad_out.eval())

    def testDynamicPartitionSoftmax(self):
        with self.test_session():
            inputs = tf.constant([1.2, 0.4, 3.1, 2.2, 4.3, 7.6, 5.1, 7.3, 1.5], dtype=tf.float32)
            indices = tf.constant([0, 0, 0, 1, 1, 2, 3, 3, 5], dtype=tf.int32)
            softmax_out = list(dynamic_partition_softmax(inputs, indices).eval())
            softmax_out_np = softmax([1.2, 0.4, 3.1]) + softmax([2.2, 4.3]) + \
                             softmax([7.6]) + softmax([5.1, 7.3]) + softmax([1.5])

            self.assertAllEqual(softmax_out, softmax_out_np)

    def testDynamicPartitionSoftmaxGrad(self):
        with self.test_session():
            inputs = tf.constant([1.2, 0.4, 3.1, 2.2, 4.3, 7.6, 5.1, 7.3, 1.5], dtype=tf.float32)
            indices = tf.constant([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=tf.int32)
            grad = tf.constant([1.6, 1.4, 5.1, 1.2, 1.3, 2.6, 4.1, 5.3, 1.3], dtype=tf.float32)

            softmax_out = dynamic_partition_softmax(inputs, indices)
            grad_out = list(my_ops.dynamic_partition_softmax_grad(softmax_out, grad, indices).eval())
            softmax_grad_np = softmax_grad([1.2, 0.4, 3.1], [1.6, 1.4, 5.1]) + \
                              softmax_grad([2.2, 4.3], [1.2, 1.3]) + \
                              softmax_grad([7.6, 5.1, 7.3, 1.5], [2.6, 4.1, 5.3, 1.3])

            self.assertAllEqual(grad_out, softmax_grad_np)


if __name__ == "__main__":
    tf.test.main()
