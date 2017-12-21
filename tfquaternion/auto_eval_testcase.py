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

import tensorflow as tf
import tfquaternion as tfq


class AutoEvalTestCase(tf.test.TestCase):
    """ Unit test class that auto-evaluates tf.Tensors in assertions. """

    @staticmethod
    def _is_tensor(x):
        """ Returns true if x is tensor-like, including tfq.Quaternion. """
        return isinstance(x, (tf.Tensor, tf.SparseTensor,
                              tf.Variable, tfq.Quaternion))

    @staticmethod
    def _auto_eval(x):
        """ Evaluates x if it is a Tensor-like object otherwise returns x. """
        return x.eval() if AutoEvalTestCase._is_tensor(x) else x

    def assertEqual(self, a, b, msg=None):
        return super().assertEqual(self._auto_eval(a), self._auto_eval(b), msg=msg)

    def assertAllEqual(self, a, b):
        return super().assertAllEqual(self._auto_eval(a), self._auto_eval(b))

    def assertAlmostEqual(self, a, b, places=None, msg=None, delta=None):
        return super().assertAlmostEqual(self._auto_eval(a),
                                         self._auto_eval(b),
                                         places=places, msg=msg, delta=delta)

    def assertAllClose(self, a, b, *args, **kwargs):
        return super().assertAllClose(self._auto_eval(a), self._auto_eval(b),
                                      *args, **kwargs)

    def assertTrue(self, a, msg=None):
        return super().assertTrue(self._auto_eval(a), msg)

    def assertFalse(self, a, msg=None):
        return super().assertFalse(self._auto_eval(a), msg)

    def assertAllFalse(self, a):
        """ Checks for a given boolean np.array if all values are False. """
        a = self._GetNdArray(self._auto_eval(a))
        return self.assertFalse(a.any())

    def assertAllTrue(self, a):
        """ Checks for a given boolean np.array if all values are True. """
        a = self._GetNdArray(self._auto_eval(a))
        return self.assertTrue(a.all())
