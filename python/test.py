import os
import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/guide/create_op?hl=zh-cn
lib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cmake-build-debug/libtfop.dylib')
my_ops = tf.load_op_library(lib_path)

print(dir(my_ops))


ps = my_ops.get_ps_handle(key_dtype=tf.int64, value_dtype=tf.float32)
keys = tf.constant([1, 2, 5, 7], dtype=tf.int64)
values = tf.constant([1.2, 3.4, 5.6, 7.8], dtype=tf.float32)
default_value = tf.constant([0.0, 1.0, 0.0, 1.0], dtype=tf.float32)
no_op = my_ops.ps_push(byte_ps_shard=ps, keys=keys, values=values)

# print(no_op)
with tf.control_dependencies([no_op]):
    out = my_ops.ps_pull(byte_ps_shard=ps, keys=keys, default_value=default_value)

with tf.Session() as sess:
    ot = sess.run(out)
    print(ot)
