import os
import tensorflow as tf

# https://www.tensorflow.org/guide/create_op?hl=zh-cn
lob_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cmake-build-debug/libtfop.dylib')
zero_out_module = tf.load_op_library(lob_path)


class ZeroOutTest(tf.test.TestCase):
    def testZeroOut(self):
        with self.test_session():
            result = zero_out_module.zero_out([5, 4, 3, 2, 1])
            self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])


if __name__ == "__main__":
    tf.test.main()
