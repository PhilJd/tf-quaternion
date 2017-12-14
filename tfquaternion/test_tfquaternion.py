# Copyright Philipp Jund (jundp@cs.uni-freiburg.de) 2017. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import tensorflow as tf
import tfquaternion as tfq


class TfquaternionTest(tf.test.TestCase):
    ''' Tests for the module functions '''
    def test_point_to_quaternion(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test_from_rotation_matrix(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test_multiply(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test_divide(self):
        # raise NotImplementedError("Test not implemented")
        pass


class QuaternionTest(tf.test.TestCase):
    ''' Tests for the member functions of class Quaternion '''

    def test___init__(self):
        ref = tfq.Quaternion([1.0, 0.0, 0.0, 0.0])
        val = (1, 0, 0, 0)
        variable = tf.Variable(val)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            # from variable
            self.assertAllEqual(tfq.Quaternion(variable).eval(), ref.eval())
            # from constant
            self.assertAllEqual(tfq.Quaternion(tf.constant(val)).eval(), ref.eval())
            # from np.array
            self.assertAllEqual(tfq.Quaternion(np.array(val)).eval(), ref.eval())
            # from list
            self.assertAllEqual(tfq.Quaternion([1.0, 0.0, 0.0, 0.0]).eval(), ref.eval())
            self.assertAllEqual(tfq.Quaternion([1, 0, 0, 0], dtype=tf.float32).eval(), ref.eval())
            # from tuple
            self.assertAllEqual(tfq.Quaternion(val, dtype=tf.float32).eval(), ref.eval())
            # unknown type
            self.assertRaises(ValueError, tfq.Quaternion, [[1, 0]])
            self.assertRaises(TypeError, tfq.Quaternion, {1, 2 ,3 , 4})

    def test_value(self):
        val = [1.0, 2.0, 3.0, 4.0]
        q = tfq.Quaternion(val)
        with self.test_session():
            self.assertAllEqual(q.value().eval(), tf.constant(val).eval())
        

    def test__quaternions_to_tensors(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test___add__(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test___sub__(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test___mul__(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test___imul__(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test___div__(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test___idiv__(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test___repr__(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test_inverse(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test_normalized(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test_as_rotation_matrix(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test__validate_shape(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test__validate_type(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test__norm(self):
        with self.test_session():
            self.assertEqual(tfq.Quaternion((1, 2, 3, 4))._norm().eval(), 30.0)
            self.assertEqual(tfq.Quaternion((-1, -2, -3, -4))._norm().eval(), 30.0)
            self.assertEqual(tfq.Quaternion((0, 0, 0, 0))._norm().eval(), 0.0)

    def test__abs(self):
        delta = 0.00001
        with self.test_session(use_gpu=True):
            self.assertAlmostEqual(tfq.Quaternion((1, 2, 3, 4))._abs().eval(),
                                   5.47722, delta=delta)
            self.assertAlmostEqual(tfq.Quaternion((-1, -2, -3, -4))._abs().eval(),
                                   5.47722, delta=delta)
            self.assertEqual(tfq.Quaternion((0, 0, 0, 0))._abs().eval(), 0.0)


class AdditionalQuaternionTests(tf.test.TestCase):

    def test_tfequal(self):
        a = tfq.Quaternion((1, 0, 0, 0))
        b = tfq.Quaternion((1, 0, 0, 0))
        c = tfq.Quaternion((1, 0, 1, 0))
        with self.test_session():
            self.assertTrue(all(tf.equal(a.value(), b.value()).eval()))
            self.assertFalse(all(tf.equal(a.value(), c.value()).eval()))


if __name__ == "__main__":
  tf.test.main()
