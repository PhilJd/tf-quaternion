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
# These tests mostly follow moble's quaternion tests (under MIT license):
# https://github.com/moble/quaternion/blob/master/test/test_quaternion.py

from __future__ import division
import itertools
import numpy as np
import tensorflow as tf
import tfquaternion as tfq
from auto_eval_testcase import AutoEvalTestCase


# ____________________________________________________________________________
#                    functions to create testing quaternions
def get_quaternions():
    """ Returns two np.arrays of testing quaternions as np.array and
        Quaternion.
    """
    q_nan1 = np.array([np.nan, 0., 0., 0.], dtype=np.float32)
    q_inf1 = np.array([np.inf, 0., 0., 0.], dtype=np.float32)
    q_minf1 = np.array([-np.inf, 0., 0., 0.], dtype=np.float32)
    q_0 = np.array([0., 0., 0., 0.], dtype=np.float32)
    q_1 = np.array([1., 0., 0., 0.], dtype=np.float32)
    x = np.array([0., 1., 0., 0.], dtype=np.float32)
    y = np.array([0., 0., 1., 0.], dtype=np.float32)
    z = np.array([0., 0., 0., 1.], dtype=np.float32)
    q = np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float32)
    qneg = np.array([-1.1, -2.2, -3.3, -4.4], dtype=np.float32)
    qbar = np.array([1.1, -2.2, -3.3, -4.4], dtype=np.float32)
    qnormalized = np.array([0.18257418583505537115232326093360,
                            0.36514837167011074230464652186720,
                            0.54772255750516611345696978280080,
                            0.73029674334022148460929304373440],
                            dtype=np.float32)
    qlog = np.array([1.7959088706354, 0.515190292664085,
                     0.772785438996128, 1.03038058532817], dtype=np.float32)
    qexp = np.array([2.81211398529184, -0.392521193481878,
                     -0.588781790222817, -0.785042386963756], dtype=np.float32)
    qmultidim = np.array([[1.1,  2.2, 3.3, 4.4], [1.1,  2.2, 3.3, 4.4]],
                         dtype=np.float32)
    np_quats = np.array([q_nan1, q_inf1, q_minf1, q_0, q_1, x, y, z, q, qneg,
                         qbar, qnormalized, qlog, qexp, qmultidim],
                         dtype=object)
    tf_quats = np.array([tfq.Quaternion(q_np) for q_np in np_quats])
    return np_quats, tf_quats


_qs, _ = get_quaternions()
N = len(_qs)
# lists of indices
(Q_NAN1, Q_INF1, Q_MINF1, Q_0, Q_1, X, Y, Z, Q,
 QNEG, QBAR, QNORMALIZED, QLOG, QEXP, QMULTIDIM) = range(N)
QS_ZERO = [i for i in range(N) if not np.flatnonzero(_qs[i]).any()]
QS_NONZERO = [i for i in range(N) if np.flatnonzero(_qs[i]).any()]
QS_NAN = [i for i in range(N) if np.isnan(_qs[i]).any()]
QS_NONNAN = [i for i in range(N) if not np.isnan(_qs[i]).any()]
QS_NONNANNONZERO = [i for i in range(N) if (not np.isnan(_qs[i]).any()
                                            and np.flatnonzero(_qs[i]).any())]
QS_INF = [i for i in range(N) if np.isinf(_qs[i]).any()]
QS_NONINF = [i for i in range(N) if not np.isinf(_qs[i]).any()]
QS_NONINFNONZERO = [i for i in range(N) if (not np.isinf(_qs[i]).any()
                                            and np.flatnonzero(_qs[i]).any())]
QS_FINITE = [i for i in range(N) if np.isfinite(_qs[i]).all()]
QS_FINITENONMULTI = [i for i in range(N) if np.isfinite(_qs[i]).all()
                                            and _qs[i].shape == (4,)]
QS_NONFINITE = [i for i in range(N) if not np.isfinite(_qs[i]).all()]
QS_FINITENONZERO = [i for i in range(N) if (np.isfinite(_qs[i]).all()
                                            and np.flatnonzero(_qs[i]).any())]

# ____________________________________________________________________________
#
class TfquaternionTest(AutoEvalTestCase):
    """ Tests for the module functions """

    def test_vector3d_to_quaternion(self):
         vec = np.array([1, 2, 3], dtype=np.float32)
         result =  np.array([0, 1, 2, 3], dtype=np.float32)
         quat = tfq.vector3d_to_quaternion(vec)
         with self.test_session():
            self.assertAllEqual(quat, result)
            # repeat stacks to obtain multidimensional quaternions
            for i in range(3):
                vec = np.stack([vec, vec])
                quat = tfq.vector3d_to_quaternion(vec)
                result = np.stack([result, result])
                self.assertAllEqual(quat, result)

    def test_quaternion_multiply(self):
        qs_np, qs_tf = get_quaternions()
        with self.test_session() as sess:
            for q in qs_tf[QS_FINITE]:
                self.assertAllEqual(q * qs_tf[Q_1], q)
                self.assertAllEqual(q * 1.0, q)
                self.assertAllEqual(1.0 * q, q)
                self.assertAllEqual(0.0 * q, np.zeros(q.shape))
                self.assertAllEqual(0.0 * q, q * 0.0)
            # Check multiplication with scalar
            for s in [-3., -2.3, -1.2, -1., 0., 1.0, 1.2, 2.3, 3.]:
                for q in qs_tf[QS_FINITE]:
                    q_w, q_x,  q_y,  q_z = tf.unstack(q, axis=-1)
                    s_times_q = [s * q_w, s * q_x, s * q_y, s * q_z]
                    s_times_q = tfq.Quaternion(tf.stack(s_times_q, axis=-1))
                    self.assertAllEqual(q * s, s_times_q)
                    self.assertAllEqual(s * q, q * s)
            # Check linearity (use placeholders to speed this up)
            pl1, pl2, pl3 = [tf.placeholder(tf.float32, (None, 4)) for i in range(3)]
            q1 , q2, q3 = [tfq.Quaternion(pl) for pl in [pl1, pl2, pl3]]
            ops = [q1 * (q2 + q3), (q1 * q2) + (q1 * q3),  # check 1
                   (q1 + q2) * q3, (q1 * q3) + (q2 * q3)]  # check 2
            triplets = itertools.permutations(qs_np[QS_FINITENONMULTI], 3)
            for q1np, q2np, q3np in triplets:
                q1np = q1np.reshape(1, 4) if q1np.shape == (4,) else q1np
                q2np = q2np.reshape(1, 4) if q2np.shape == (4,) else q2np
                q3np = q3np.reshape(1, 4) if q3np.shape == (4,) else q3np
                fd = { pl1: q1np, pl2: q2np, pl3: q3np }
                result = sess.run(ops, feed_dict=fd)
                # checks q1 * (q2 + q3) == (q1 * q2) + (q1 * q3),
                self.assertAllClose(result[0], result[1])
                # checks (q1 + q2) * q3 == (q1 * q3) + (q2 * q3)
                self.assertAllClose(result[2], result[3])
            # Check the multiplication table
            for q in [qs_tf[Q_1], qs_tf[X], qs_tf[Y], qs_tf[Z]]:
                self.assertAllEqual(qs_tf[Q_1] * q, q)
                self.assertAllEqual(q * qs_tf[Q_1], q)
            self.assertAllEqual(qs_tf[X] * qs_tf[X], -qs_tf[Q_1])
            self.assertAllEqual(qs_tf[X] * qs_tf[Y], qs_tf[Z])
            self.assertAllEqual(qs_tf[X] * qs_tf[Z], -qs_tf[Y])
            self.assertAllEqual(qs_tf[Y] * qs_tf[X], -qs_tf[Z])
            self.assertAllEqual(qs_tf[Y] * qs_tf[Y], -qs_tf[Q_1])
            self.assertAllEqual(qs_tf[Y] * qs_tf[Z], qs_tf[X])
            self.assertAllEqual(qs_tf[Z] * qs_tf[X], qs_tf[Y])
            self.assertAllEqual(qs_tf[Z] * qs_tf[Y], -qs_tf[X])
            self.assertAllEqual(qs_tf[Z] * qs_tf[Z], -qs_tf[Q_1])
            self.assertAllEqual(qs_tf[Z] * qs_tf[Z], -qs_tf[Q_1])

    def test_quaternion_divide(self):
        qs_np, qs_tf = get_quaternions()
        with self.test_session() as sess:
            # Check scalar division
            for q in qs_tf[QS_FINITENONZERO]:
                # use + np.zeros to broadcast qs_np[Q_1] to the multidim shape
                self.assertAllClose(q / q, qs_np[Q_1] + np.zeros(q.shape))
                self.assertRaises(TypeError, tfq.quaternion_divide, 1, q)
                self.assertAllClose(1.0 / q, q.inverse())
                self.assertAllClose([1.0] / q, q.inverse())
                self.assertAllClose(0.0 / q, qs_np[Q_0] + np.zeros(q.shape))
                for s in [-3., -2.3, -1.2, -1., 0., 1.0, 1.2, 2.3, 3.]:
                    self.assertAllClose(s / q, s * (q.inverse()))
            for q in qs_tf[QS_NONNAN]:
                self.assertAllClose(q / 1.0, q)
                for s in [-3., -2.3, -1.2, -1., 0., 1.0, 1.2, 2.3, 3.]:
                    # use np.array(1) to allow division by zero
                    gt = q * (np.array(1.0) / s).astype(np.float32)
                    self.assertAllClose(q / s, gt)

            # Check linearity
            pl1, pl2, pl3 = [tf.placeholder(tf.float32, (None, 4)) for i in range(3)]
            q1 , q2, q3 = [tfq.Quaternion(pl) for pl in [pl1, pl2, pl3]]
            ops = [(q1 + q2) / q3, (q1 / q3) + (q2 / q3)]
            for q1np, q2np in itertools.permutations(qs_np[QS_FINITE], 2):
                for q3np in qs_np[QS_FINITENONZERO]:
                    q1np = q1np.reshape(1, 4) if q1np.shape == (4,) else q1np
                    q2np = q2np.reshape(1, 4) if q2np.shape == (4,) else q2np
                    q3np = q3np.reshape(1, 4) if q3np.shape == (4,) else q3np
                    fd = {pl1: q1np, pl2: q2np, pl3: q3np}
                    a, b = sess.run(ops, feed_dict=fd)
                    # checks (q1 + q2) / q3 == (q1 / q3) + (q2 / q3)
                    self.assertAllClose(a, b)

            # Check the multiplication table
            for q in [qs_tf[Q_1], qs_tf[X], qs_tf[Y], qs_tf[Z]]:
                self.assertAllClose(qs_tf[Q_1] / q, q.conj())
                self.assertAllClose(q / qs_tf[Q_1], q)
            self.assertAllClose(qs_tf[X] / qs_tf[X], qs_tf[Q_1])
            self.assertAllClose(qs_tf[X] / qs_tf[Y], -qs_tf[Z])
            self.assertAllClose(qs_tf[X] / qs_tf[Z], qs_tf[Y])
            self.assertAllClose(qs_tf[Y] / qs_tf[X], qs_tf[Z])
            self.assertAllClose(qs_tf[Y] / qs_tf[Y], qs_tf[Q_1])
            self.assertAllClose(qs_tf[Y] / qs_tf[Z], -qs_tf[X])
            self.assertAllClose(qs_tf[Z] / qs_tf[X], -qs_tf[Y])
            self.assertAllClose(qs_tf[Z] / qs_tf[Y], qs_tf[X])
            self.assertAllClose(qs_tf[Z] / qs_tf[Z], qs_tf[Q_1])

    def test_quaternion_conjugate(self):
        qs_np, qs_tf = get_quaternions()
        with self.test_session():
            self.assertAllEqual(qs_tf[Q].conjugate(), qs_tf[QBAR])
            for q in qs_tf[QS_NONNAN]:
                self.assertAllEqual(q.conjugate(), q.conj())
                self.assertAllEqual(q.conjugate().conjugate(), q)
                cw, cx, cy, cz = tf.unstack(q.conjugate(), axis=-1)
                qw, qx, qy, qz = tf.unstack(q, axis=-1)
                self.assertAllEqual(cw, qw)
                self.assertAllEqual(cx, -qx)
                self.assertAllEqual(cy, -qy)
                self.assertAllEqual(cz, -qz)

    def test_rotate_vector_by_quaternion(self):
        v = [3., 4., 5.]
        qs_np, qs_tf = get_quaternions()
        with self.test_session():
            for q in qs_tf[QS_FINITENONZERO]:
                rotated = q * tfq.vector3d_to_quaternion(v) * q.inverse()
                rotated = tfq.quaternion_to_vector3d(rotated)
                self.assertAllClose(tfq.rotate_vector_by_quaternion(q, v),
                                    rotated)

    def test_quaternion_to_tensor(self):
        ref = tfq.Quaternion([1.0, 0.0, 0.0, 0.0])
        with self.test_session():
            self.assertAllEqual(type(tf.convert_to_tensor(ref)), tf.Tensor)
            # reduce sum internally calls tf.convert_to_tensor
            self.assertAllEqual(tf.reduce_sum(ref), 1.0)

class QuaternionTest(AutoEvalTestCase):
    """ Tests for the member functions of class Quaternion """

    def test___init__(self):
        ref = tfq.Quaternion([1.0, 0.0, 0.0, 0.0])
        val = (1.0, 0.0, 0.0, 0.0)
        variable = tf.Variable(val)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            #default constructor
            self.assertAllEqual(tfq.Quaternion(), ref)
            # from variable
            self.assertAllEqual(tfq.Quaternion(variable), ref)
            # from constant
            self.assertAllEqual(tfq.Quaternion(tf.constant(val)), ref)
            # from np.array
            self.assertAllEqual(tfq.Quaternion(np.array(val)), ref)
            # from list
            self.assertAllEqual(tfq.Quaternion([1.0, 0.0, 0.0, 0.0]), ref)
            self.assertAllEqual(tfq.Quaternion([1, 0, 0, 0]), ref)
            # from tuple
            self.assertAllEqual(tfq.Quaternion(val), ref)
            # from Quaternion
            self.assertAllEqual(tfq.Quaternion(ref), ref)
            # wrong shape
            self.assertRaises(ValueError, tfq.Quaternion, [[1, 0]])
            # wrong type
            self.assertRaises(TypeError, tfq.Quaternion, val, dtype=tf.int32)
            self.assertRaises(TypeError, tfq.Quaternion, [1, 0, 0, 0],
                              dtype=tf.int32)

    def test_value(self):
        val = [1.0, 2.0, 3.0, 4.0]
        q = tfq.Quaternion(val)
        with self.test_session():
            self.assertAllEqual(q, tf.constant(val))


    def test___add__(self):
        a = tfq.Quaternion([1, 2, 3, 4])
        b = tfq.Quaternion([5, 6, 7, 8])
        result = np.array([6, 8, 10, 12], dtype=np.float32)
        with self.test_session():
            self.assertAllEqual(a + b, result)

    def test___sub__(self):
        a = tfq.Quaternion([1, 2, 3, 4])
        b = tfq.Quaternion([5, 6, 7, 8])
        result = np.array([-4, -4, -4, -4], dtype=np.float32)
        with self.test_session():
            self.assertAllEqual(a - b, result)

    def test___rmul__(self):
        qs_np, qs_tf = get_quaternions()
        # Note that quaternion multiplication is only applied if both
        # multiplicands are quaternions. Also note that multiplying
        # by a np.array returns one Quaternion for each entry in the np.array,
        # therefore we don't test this here
        with self.test_session():
            self.assertAllEqual(qs_tf[X] * qs_tf[X], -qs_tf[Q_1])
            self.assertAllEqual(qs_tf[X] * 2.0, [0.0, 2.0, 0.0, 0.0])
            self.assertAllClose(qs_np[X] * qs_tf[X], [0.0, 1.0, 0.0, 0.0])
            self.assertAllClose(qs_np[Q] * qs_tf[Q], qs_np[Q] * qs_np[Q])

    def test___rdiv__(self):
        qs_np, qs_tf = get_quaternions()
        with self.test_session():
            self.assertAllClose(qs_tf[X] / qs_tf[X], qs_tf[Q_1])
            self.assertAllClose(qs_tf[X] / 2.0, [0.0, 0.5, 0.0, 0.0])

    def test___lt__(self):
        ref = tf.constant(np.arange(4), dtype=tf.float32)
        ref2 = tf.constant(np.arange(2, 6), dtype=tf.float32)
        ref3 = tf.constant(np.arange(-2, 2), dtype=tf.float32)
        ref_q = tfq.Quaternion(ref)
        with self.test_session():
            self.assertAllFalse(ref < ref)
            self.assertAllTrue(ref < ref2)
            self.assertAllTrue(ref3 < ref)

    def test___le__(self):
        ref = tf.constant(np.arange(4), dtype=tf.float32)
        ref2 = tf.constant(np.arange(2, 6), dtype=tf.float32)
        ref3 = tf.constant(np.arange(-2, 2), dtype=tf.float32)
        ref_q = tfq.Quaternion(ref)
        with self.test_session():
            self.assertAllTrue(ref <= ref)
            self.assertAllTrue(ref <= ref2)
            self.assertAllTrue(ref3 <= ref)

    def test___eq__(self):
        ref = tf.constant(np.arange(4), dtype=tf.float32)
        ref2 = tf.constant(np.arange(2, 6), dtype=tf.float32)
        ref_q = tfq.Quaternion(ref)
        with self.test_session():
            self.assertAllTrue(ref_q == ref)
            self.assertAllTrue(ref_q == ref_q)
            self.assertAllTrue(ref_q == tfq.Quaternion(ref))
            self.assertAllFalse(ref_q == ref2)

    def test___ne__(self):
        ref = tf.constant(np.arange(4), dtype=tf.float32)
        ref2 = tf.constant(np.arange(2, 6), dtype=tf.float32)
        ref_q = tfq.Quaternion(ref)
        with self.test_session():
            self.assertAllFalse(ref_q != ref)
            self.assertAllFalse(ref_q != ref_q)
            self.assertAllFalse(ref_q != tfq.Quaternion(ref))
            self.assertAllTrue(ref_q != ref2)

    def test___gt__(self):
        ref = tf.constant(np.arange(4), dtype=tf.float32)
        ref2 = tf.constant(np.arange(2, 6), dtype=tf.float32)
        ref3 = tf.constant(np.arange(-2, 2), dtype=tf.float32)
        ref_q = tfq.Quaternion(ref)
        with self.test_session():
            self.assertAllFalse(ref > ref)
            self.assertAllFalse(ref > ref2)
            self.assertAllTrue(ref > ref3)

    def test___ge__(self):
        ref = tf.constant(np.arange(4), dtype=tf.float32)
        ref2 = tf.constant(np.arange(2, 6), dtype=tf.float32)
        ref3 = tf.constant(np.arange(-2, 2), dtype=tf.float32)
        ref_q = tfq.Quaternion(ref)
        with self.test_session():
            self.assertAllTrue(ref >= ref)
            self.assertAllFalse(ref >= ref2)
            self.assertAllTrue(ref >= ref3)

    def test_inverse(self):
        qs_np, qs_tf = get_quaternions()
        with self.test_session() as sess:
            for q in qs_tf[QS_FINITENONZERO]:
                gt = qs_np[Q_1] + np.zeros(q.shape)
                self.assertAllClose(q * q.inverse(), gt)

    def test_normalized(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test_as_rotation_matrix(self):
        # raise NotImplementedError("Test not implemented")
        pass

    def test__validate_shape(self):
        self.assertRaises(ValueError, tfq.Quaternion, [1,2,3])

    def test__validate_type(self):
        self.assertRaises(TypeError, tfq.Quaternion, dtype=tf.int32)

    def test_norm(self):
        with self.test_session():
            self.assertAllEqual(tfq.Quaternion((1, 2, 3, 4)).norm(), [30.0])
            self.assertAllEqual(tfq.Quaternion((-1, -2, -3, -4)).norm(), [30.0])
            self.assertAllEqual(tfq.Quaternion((0, 0, 0, 0)).norm(), [0.0])

    def test_abs(self):
        delta = 0.00001
        with self.test_session(use_gpu=True):
            self.assertAlmostEqual(tfq.Quaternion((1, 2, 3, 4)).abs(),
                                   5.47722, delta=delta)
            self.assertAlmostEqual(tfq.Quaternion((-1, -2, -3, -4)).abs(),
                                   5.47722, delta=delta)
            self.assertAllEqual(tfq.Quaternion((0, 0, 0, 0)).abs(), 0.0)


if __name__ == "__main__":
    tf.test.main()
