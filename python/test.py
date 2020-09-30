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


def graph_reduce(src, dst, adjs, weights=None, need_weight_grad=True, reduce_method='mean'):
    assert reduce_method in ("sum", "mean")
    if weights is None:
        return my_ops.graph_reduce(reduce_method=reduce_method, src_tensor=src,
                                   dst_tensor=dst, adj_tensor=adjs)
    else:
        return my_ops.graph_reduce_with_weight(reduce_method=reduce_method, src_tensor=src,
                                               dst_tensor=dst, adj_tensor=adjs, weight_tensor=weights)


@tf.RegisterGradient("GraphReduce")
def _graph_reduce_grad(op, grad):
    reduce_method = op.get_attr('reduce_method')
    adjs = op.inputs[2]
    grad_out = my_ops.graph_reduce_grad(
        reduce_method=reduce_method, grad_tensor=grad, adj_tensor=adjs)

    return None, grad_out, None


@tf.RegisterGradient("GraphReduceWithWeight")
def _graph_reduce_with_weight_grad(op, grad):
    reduce_method = op.get_attr('reduce_method')
    dst_tensor = op.inputs[2]
    adjs = op.inputs[3]
    weights = op.inputs[4]
    dst_grad, weight_grad = my_ops.graph_reduce_with_weight_grad(
        reduce_method=reduce_method, grad_tensor=grad, dst_tensor=dst_tensor,
        adj_tensor=adjs, weight_tensor=weights)

    return None, dst_grad, None, weight_grad


def graph_reorder(node_lists):
    unique_nodes, indices = my_ops.graph_reorder(input_list=node_lists)
    return unique_nodes, indices


def softmax(array):
    ips = np.exp(np.array(array, dtype=np.float32))
    sum_value = ips.sum()
    return list(ips / sum_value)


def softmax_grad(array, grad):
    softmax_out = softmax(array)
    grad = np.array(grad, dtype=np.float32)

    return list((grad - np.dot(softmax_out, grad)) * softmax_out)


class MyLibTest(tf.test.TestCase):
    def testGraphReorder(self):
        with self.test_session():
            nodes1 = tf.constant([1, 2, 3, 4], dtype=tf.int64)
            nodes2 = tf.constant([3, 4, 5, 6, 7], dtype=tf.int64)
            nodes3 = tf.constant([9, 3, 4, 5, 8, 9], dtype=tf.int64)
            nodes4 = tf.constant([15, 21, 44, 78, 32], dtype=tf.int64)

            unique_nodes, indices = graph_reorder([nodes1, nodes2, nodes3, nodes4])
            self.assertAllEqual(nodes1, tf.gather(unique_nodes, indices[0]))
            print(tf.gather(unique_nodes, indices[0]).eval())
            self.assertAllEqual(nodes2, tf.gather(unique_nodes, indices[1]))
            print(tf.gather(unique_nodes, indices[1]).eval())
            self.assertAllEqual(nodes3, tf.gather(unique_nodes, indices[2]))
            print(tf.gather(unique_nodes, indices[2]).eval())
            self.assertAllEqual(nodes4, tf.gather(unique_nodes, indices[3]))
            print(tf.gather(unique_nodes, indices[3]).eval())

    def testGraphReduce1(self):
        with self.test_session():
            src = tf.constant([1.2, 3.1, 1.3, 5.3, 2.3, 4.8], dtype=tf.float32)
            dst = tf.constant([1.2, 0.5, 3.1, 2.2, 4.3, 7.6, 5.1, 4.1],
                              dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 3], [1, 4],
                                [2, 7],
                                [3, 6], [3, 3],
                                [5, 5]], dtype=tf.int32)
            reduce_out = graph_reduce(src, dst, adjs, reduce_method="sum")
            print(reduce_out.eval())

    def testGraphReduceGrad1(self):
        with self.test_session():
            grad = tf.constant([1.2, 3.1, 1.3, 5.3, 2.3, 4.8], dtype=tf.float32)
            dst = tf.constant([1.2, 0.5, 3.1, 2.2, 4.3, 7.6, 5.1, 4.1],
                              dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 3], [1, 4],
                                [2, 7],
                                [3, 6], [3, 3],
                                [5, 5]], dtype=tf.int32)
            reduce_grad_out = my_ops.graph_reduce_grad(grad_tensor=grad,
                                                       dst_tensor=dst, adj_tensor=adjs,
                                                       reduce_method="sum")
            print(reduce_grad_out.eval())

    def testGraphReduce2(self):
        with self.test_session():
            src = tf.constant([[1.2, 0.5],
                               [3.1, 2.2],
                               [1.3, 7.2],
                               [5.3, 3.6],
                               [2.3, 7.9],
                               [4.8, 10.6]], dtype=tf.float32)
            dst = tf.constant([[1.2, 4.5], [0.5, 4.2], [3.1, 12],
                               [2.2, 4.9], [4.3, 5.5],
                               [7.6, 5.9], [5.1, 6.1], [4.1, 9.1]],
                              dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 3], [1, 4],
                                [2, 7],
                                [3, 6], [3, 3],
                                [5, 5]], dtype=tf.int32)
            reduce_out = graph_reduce(src, dst, adjs)
            print(reduce_out.eval())

    def testGraphReduceGrad2(self):
        with self.test_session():
            grad = tf.constant([[1.2, 0.5],
                                [3.1, 2.2],
                                [1.3, 7.2],
                                [5.3, 3.6],
                                [2.3, 7.9],
                                [4.8, 10.6]], dtype=tf.float32)
            dst = tf.constant([[1.2, 4.5], [0.5, 4.2], [3.1, 12],
                               [2.2, 4.9], [4.3, 5.5],
                               [7.6, 5.9], [5.1, 6.1], [4.1, 9.1]],
                              dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 3], [1, 4],
                                [2, 7],
                                [3, 6], [3, 3],
                                [5, 5]], dtype=tf.int32)
            reduce_grad_out = my_ops.graph_reduce_grad(grad_tensor=grad,
                                                       dst_tensor=dst, adj_tensor=adjs,
                                                       reduce_method="mean")
            print(reduce_grad_out.eval())

    def testGraphReduceWithWeight1(self):
        with self.test_session():
            src = tf.constant([1.2, 3.1, 1.3, 5.3, 2.3, 4.8], dtype=tf.float32)
            dst = tf.constant([1.2, 0.5, 3.1, 2.2, 4.3, 7.6, 5.1, 4.1],
                              dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 3], [1, 4],
                                [2, 7],
                                [3, 6], [3, 3],
                                [5, 5]], dtype=tf.int32)
            weight = tf.constant([6.8, 5.4, 3.6, 2.7, 1.3, 4.4, 8.4, 1.7, 3.3], dtype=tf.float32)
            reduce_out = graph_reduce(src, dst, adjs, weight, reduce_method="sum")
            print(reduce_out.eval())

    def testGraphReduceWithWeightGrad1(self):
        with self.test_session():
            grad = tf.constant([1.2, 3.1, 1.3, 5.3, 2.3, 4.8], dtype=tf.float32)
            dst = tf.constant([1.2, 0.5, 3.1, 2.2, 4.3, 7.6, 5.1, 4.1],
                              dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 3], [1, 4],
                                [2, 7],
                                [3, 6], [3, 3],
                                [5, 5]], dtype=tf.int32)
            weight = tf.constant([6.8, 5.4, 3.6, 2.7, 1.3, 4.4, 8.4, 1.7, 3.3], dtype=tf.float32)
            dst_grad, weight_grad = my_ops.graph_reduce_with_weight_grad(grad_tensor=grad,
                                                                         dst_tensor=dst, adj_tensor=adjs,
                                                                         weight_tensor=weight,
                                                                         reduce_method="sum")
            print(dst_grad.eval(), weight_grad.eval())

    def testGraphReduceWithWeight2(self):
        with self.test_session():
            src = tf.constant([[1.2, 3.2],
                               [3.1, 5.3],
                               [1.3, 9.5],
                               [5.3, 9.4],
                               [2.3, 8.1],
                               [4.8, 7.4]], dtype=tf.float32)
            dst = tf.constant([[1.2, 8.1], [0.5, 3.2], [3.1, 1.2],
                               [2.2, 1.1], [4.3, 4.5], [2.3, 7.6], [2.6, 5.1], [4.1, 4.9]],
                              dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 3], [1, 4],
                                [2, 7],
                                [3, 6], [3, 3],
                                [5, 5]], dtype=tf.int32)
            weight = tf.constant([6.8, 5.4, 3.6, 2.7, 1.3, 4.4, 8.4, 1.7, 3.3], dtype=tf.float32)
            reduce_out = graph_reduce(src, dst, adjs, weight, reduce_method="mean")
            print(reduce_out.eval())

    def testGraphReduceWithWeightGrad2(self):
        with self.test_session():
            grad = tf.constant([[1.2, 3.2],
                               [3.1, 5.3],
                               [1.3, 9.5],
                               [5.3, 9.4],
                               [2.3, 8.1],
                               [4.8, 7.4]], dtype=tf.float32)
            dst = tf.constant([[1.2, 8.1], [0.5, 3.2], [3.1, 1.2],
                               [2.2, 1.1], [4.3, 4.5], [2.3, 7.6], [2.6, 5.1], [4.1, 4.9]],
                              dtype=tf.float32)
            adjs = tf.constant([[0, 0], [0, 1], [0, 2],
                                [1, 3], [1, 4],
                                [2, 7],
                                [3, 6], [3, 3],
                                [5, 5]], dtype=tf.int32)
            weight = tf.constant([6.8, 5.4, 3.6, 2.7, 1.3, 4.4, 8.4, 1.7, 3.3], dtype=tf.float32)
            dst_grad, weight_grad = my_ops.graph_reduce_with_weight_grad(grad_tensor=grad,
                                                                         dst_tensor=dst, adj_tensor=adjs,
                                                                         weight_tensor=weight,
                                                                         reduce_method="sum")
            print(dst_grad.eval(), weight_grad.eval())

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
