import tensorflow as tf
import tfQuaternion as tfq


class TfquaternionTest(tf.TestCase):
    ''' Tests for the module functions '''
    def test_point_to_quaternion():
        raise NotImplementedError("Test not implemented")

    def test_from_rotation_matrix():
        raise NotImplementedError("Test not implemented")

    def test_multiply(a, b):
        raise NotImplementedError("Test not implemented")

    def test_divide(a, b):
        raise NotImplementedError("Test not implemented")


class QuaternionTest(tf.TestCase):
    ''' Tests for the class Quaternion '''

    def test___init__(self):
        raise NotImplementedError("Test not implemented")

    def test_value(self):
        raise NotImplementedError("Test not implemented")

    def test__quaternions_to_tensors(quats):
        raise NotImplementedError("Test not implemented")

    def test___add__(a, b):
        raise NotImplementedError("Test not implemented")

    def test___sub__(self):
        raise NotImplementedError("Test not implemented")

    def test___mul__(a, b):
        raise NotImplementedError("Test not implemented")

    def test___imul__(self, other):
        raise NotImplementedError("Test not implemented")

    def test___div__(a, b):
        raise NotImplementedError("Test not implemented")

    def test___idiv__(self, other):
        raise NotImplementedError("Test not implemented")

    def test___repr__(self):
        raise NotImplementedError("Test not implemented")

    def test_inverse(self):
        raise NotImplementedError("Test not implemented")

    def test_normalized(self):
        raise NotImplementedError("Test not implemented")

    def test_as_rotation_matrix(self):
        raise NotImplementedError("Test not implemented")

    def test_diag(a, b):
        raise NotImplementedError("Test not implemented")

    def test_tr_add(a, b, c, d):
        raise NotImplementedError("Test not implemented")

    def test_tr_sub(a, b, c, d):
        raise NotImplementedError("Test not implemented")

    def test__validate_shape(x):
        raise NotImplementedError("Test not implemented")

    def test__validate_type(x):
        raise NotImplementedError("Test not implemented")

    def test__norm(self):
        raise NotImplementedError("Test not implemented")

    def test__abs(self):
        raise NotImplementedError("Test not implemented")
