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
#
#
#
#
#
# AutoEvalTestCase
# =======
# The AutoEvalTestCase automatically evaluates tensor-like objects
# such as tf.Variables, tf.Tensor, tf.SparseTensor etc. This is motivated by
# the fact that most tests are concerned with the values of the tensors and not
# the tf.Tensor object. Inheriting from this class removes the need for calling
# `.eval()` for each tensor, being easier on the eyes.
#
#
#  Example:
#  =======
#  class MyModuleTest(AutoEvalTestCase):
#
#      def test_myfunction():
#          gt = np.array([[1., 1.], [1., 1.]], dtype=tf.float32)
#          with self.test_session():
#              self.assertEqual(tf.ones((2, 2)), gt)
#              # you can still use the vanilla tf.test.TestCase when required
#              tf.test.TestCase().assertEqual(tf.ones((2, 2)).eval(), gt)
#


import tensorflow as tf


class AutoEvalTestCase(tf.test.TestCase):
    """ Unit test class that auto-evaluates tf.Tensors in assertions. """

    @staticmethod
    def _is_tensor(x):
        """ Returns true if x is tensor-like, including tfq.Quaternion. """
        return isinstance(x, (tf.Tensor, tf.SparseTensor, tf.Variable,
                              tfq.Quaternion))

    @staticmethod
    def _auto_eval(x):
        """ Evaluates x if it is a Tensor-like object otherwise returns x. """
        return x.eval() if AutoEvalTestCase._is_tensor(x) else x

    def assertEqual(self, a, b, msg=None):
        sup = super(AutoEvalTestCase, self)
        return sup.assertEqual(self._auto_eval(a), self._auto_eval(b), msg=msg)

    def assertAllEqual(self, a, b):
        sup = super(AutoEvalTestCase, self)
        return sup.assertAllEqual(self._auto_eval(a), self._auto_eval(b))

    def assertAlmostEqual(self, a, b, places=None, msg=None, delta=None):
        sup = super(AutoEvalTestCase, self)
        return sup.assertAlmostEqual(self._auto_eval(a), self._auto_eval(b),
                                     places=places, msg=msg, delta=delta)

    def assertAllClose(self, a, b, *args, **kwargs):
        sup = super(AutoEvalTestCase, self)
        return sup.assertAllClose(self._auto_eval(a), self._auto_eval(b),
                                  *args, **kwargs)

    def assertTrue(self, a, msg=None):
        sup = super(AutoEvalTestCase, self)
        return sup.assertTrue(self._auto_eval(a), msg)

    def assertFalse(self, a, msg=None):
        sup = super(AutoEvalTestCase, self)
        return sup.assertFalse(self._auto_eval(a), msg)

    def assertAllFalse(self, a):
        """ Checks for a given boolean np.array if all values are False. """
        a = self._GetNdArray(self._auto_eval(a))
        return self.assertFalse(a.any())

    def assertAllTrue(self, a):
        """ Checks for a given boolean np.array if all values are True. """
        a = self._GetNdArray(self._auto_eval(a))
        return self.assertTrue(a.all())
