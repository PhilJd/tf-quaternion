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

"""

This small library implements quaternion operations in tensorflow.
All operations are differentiable.

"""
import tensorflow as tf


# ____________________________________________________________________________
#                     Quaternion module functions
def scope_wrapper(func, *args, **kwargs):
    """Create a tf name scope around the function with its name."""
    def scoped_func(*args, **kwargs):
        with tf.name_scope("quaternion_{}".format(func.__name__)):
            return func(*args, **kwargs)
    return scoped_func


@scope_wrapper
def vector3d_to_quaternion(x):
    """Convert a tensor of 3D vectors to a quaternion.

    Prepends a 0 to the last dimension, i.e. [[1,2,3]] -> [[0,1,2,3]].

    Args:
        x: A `tf.Tensor` of rank R, the last dimension must be 3.

    Returns:
        A `Quaternion` of Rank R with the last dimension being 4.

    Raises:
        ValueError, if the last dimension of x is not 3.
    """
    x = tf.convert_to_tensor(x)
    if x.shape[-1] != 3:
        raise ValueError("The last dimension of x must be 3.")
    Quaternion.validate_type(x)
    return Quaternion(tf.pad(x, (len(x.shape) - 1) * [[0, 0]] + [[1, 0]]))


@scope_wrapper
def quaternion_to_vector3d(q):
    """Remove the w component(s) of quaternion(s) q."""
    return q.value()[..., 1:]


@scope_wrapper
def _prepare_tensor_for_div_mul(x):
    """Prepare the tensor x for division/multiplication.

    This function
    a) converts x to a tensor if necessary,
    b) prepends a 0 in the last dimension if the last dimension is 3,
    c) validates the type and shape.
    """
    x = tf.convert_to_tensor(x)
    if x.shape[-1] == 3:
        x = vector3d_to_quaternion(x)
    Quaternion.validate_shape(x)
    Quaternion.validate_type(x)
    return x


@scope_wrapper
def quaternion_multiply(a, b):
    """Multiply two quaternion tensors.

    Note that this differs from tf.multiply and is not commutative.

    Args:
        a, b: A `tf.Tensor` with shape (..., 4).

    Returns:
        A `Quaternion`.
    """
    a = _prepare_tensor_for_div_mul(a)
    b = _prepare_tensor_for_div_mul(b)
    w1, x1, y1, z1 = tf.split(a, num_or_size_splits=4, axis=-1)
    w2, x2, y2, z2 = tf.split(b, num_or_size_splits=4, axis=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return Quaternion(tf.concat(values=[w, x, y, z], axis=-1))


@scope_wrapper
def quaternion_divide(a, b):
    """Divide tensor `a` by quaternion tensor `b`. `a` may be a scalar value.

    Args:
        a: A scalar or `tf.Tensor` with shape (..., 4).
        b: A `tf.Tensor` with shape (..., 4).

    Returns:
        A `Quaternion`.
    """
    a = tf.convert_to_tensor(a)
    if a.shape == () or a.shape[-1] == 1:  # scalar
        return Quaternion(tf.multiply(a, b.conj()) / Quaternion(b).norm())
    bnorm = Quaternion(b).norm()
    w1, x1, y1, z1 = tf.split(a, num_or_size_splits=4, axis=-1)
    w2, x2, y2, z2 = tf.split(b, num_or_size_splits=4, axis=-1)
    w = (w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2) / bnorm
    x = (-w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2) / bnorm
    y = (-w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2) / bnorm
    z = (-w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2) / bnorm
    return Quaternion(tf.concat(values=[w, x, y, z], axis=-1))


@scope_wrapper
def quaternion_conjugate(q):
    """Compute the conjugate of q, i.e. [q.w, -q.x, -q.y, -q.z]."""
    return Quaternion(tf.multiply(q, [1.0, -1.0, -1.0, -1.0]))


@scope_wrapper
def rotate_vector_by_quaternion(q, v, q_ndims=None, v_ndims=None):
    """Rotate a vector (or tensor with last dimension of 3) by q.

    This function computes v' = q * v * conjugate(q) but faster.
    Fast version can be found here:
    https://blog.molecular-matters.com/2013/05/24/a-faster-quaternion-vector-multiplication/

    Args:
        q: A `Quaternion` or `tf.Tensor` with shape (..., 4)
        v: A `tf.Tensor` with shape (..., 3)
        q_ndims: The number of dimensions of q. Only necessary to specify if
            the shape of q is unknown.
        v_ndims: The number of dimensions of v. Only necessary to specify if
            the shape of v is unknown.

    Returns: A `tf.Tensor` with the broadcasted shape of v and q.
    """
    v = tf.convert_to_tensor(v)
    q = q.normalized()
    w = q.value()[..., 0]
    q_xyz = q.value()[..., 1:]
    # Broadcast shapes. Todo(phil): Prepare a pull request which adds
    # broadcasting support to tf.cross
    if q_xyz.shape.ndims is not None:
        q_ndims = q_xyz.shape.ndims
    if v.shape.ndims is not None:
        v_ndims = v.shape.ndims
    for _ in range(v_ndims - q_ndims):
        q_xyz = tf.expand_dims(q_xyz, axis=0)
    for _ in range(q_ndims - v_ndims):
        v = tf.expand_dims(v, axis=0) + tf.zeros_like(q_xyz)
    q_xyz += tf.zeros_like(v)
    v += tf.zeros_like(q_xyz)
    t = 2 * tf.cross(q_xyz, v)
    return v + tf.expand_dims(w, axis=-1) * t + tf.cross(q_xyz, t)


# ____________________________________________________________________________
#                      The quaternion class
class Quaternion(object):
    """A multidimensional quaternion. The API resembles that of tf.Variable."""

    # When trying to scale the components of the Quaternion individually
    # by right-multiplying a tf.Quaternion with a 4-dimensional np.array a, the
    # default numpy behaviour is to call `Quaternion.__rmul__(i)` for each
    # element i in a, resulting in 4 tfq.Quaternions instead of one.
    # Setting __array_priority__ = 1000 fixes this. (For further reference see
    # https://stackoverflow.com/questions/40694380/forcing-multiplication-to-use-rmul-instead-of-numpy-array-mul-or-byp)
    __array_priority__ = 1000

    def __init__(self, wxyz=(1, 0, 0, 0), dtype=tf.float32, name=None):
        """The quaternion constructor.

        Args:
            wxyz: The values for w, x, y, z, a `tf.Tensor` with shape (..., 4).
                Note that quaternions only support floating point numbers.
                Defaults to (1.0, 0.0, 0.0, 0.0)
            dtype: The type used for the quaternion, must be a floating point
                number, i.e. one of tf.float16, tf.float32, tf.float64.
            name: An optional name for the tensor.

        Returns:
            A Quaternion.

        Raises:
            ValueError, if wxyz is a `tf.Tensor` and the tensors dtype differs
                from the given dtype.
            ValueError, if the last dimension of wxyz is not 4.
            TypeError, if dtype is not a float.
        """
        self._q = tf.convert_to_tensor(wxyz, dtype=dtype, name=name)
        self.name = name if name else ""
        self.validate_type(self._q)
        self.validate_shape(self._q)  # check that shape is (..., 4)

    def value(self):
        """The `Tensor` which holds the value of the quaternion.

        Note that this does not return a reference, so you can not alter the
        quaternion through this.
        """
        return self._q

    def eval(self, session=None):
        """In a session, computes and returns the value of this quaternion."""
        return self._q.eval(session=session)

    def _ref(self):
        return self._q._ref()

    @property
    def dtype(self):
        """The `DType` of this quaternion."""
        return self._q.dtype

    @property
    def op(self):
        """The `Operation` of this quaternion."""
        return self._q.op

    @property
    def graph(self):
        """The `Graph` of this quaternion."""
        return self._q.graph

    @property
    def shape(self):
        """The `TensorShape` of the variable. Is always [..., 4].

        Returns:
          A `TensorShape`.
        """
        return self._q.get_shape()

    def get_shape(self):
        """An Alias of Quaternion.shape."""
        return self.shape

    def _as_graph_element(self):
        """Conversion function for Graph.as_graph_element()."""
        return self._q

    def __add__(self, other):
        return Quaternion(tf.add(self._q, tf.convert_to_tensor(other)))

    def __radd__(self, other):
        return Quaternion(tf.add(tf.convert_to_tensor(other), self._q))

    def __sub__(self, other):
        return Quaternion(tf.subtract(self._q, tf.convert_to_tensor(other)))

    def __rsub__(self, other):
        return Quaternion(tf.subtract(tf.convert_to_tensor(other), self._q))

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            return quaternion_multiply(self, other)
        return Quaternion(tf.multiply(self._q, tf.convert_to_tensor(other)))

    def __rmul__(self, other):
        # This is only called when __mul__ fails, so 'other' can not
        # be a Quaternion.
        return Quaternion(tf.multiply(self._q, tf.convert_to_tensor(other)))

    def __div__(self, other):
        if isinstance(other, Quaternion):
            return quaternion_divide(self, other)
        return tf.divide(self._q, tf.convert_to_tensor(other))

    def __rdiv__(self, other):
        if (isinstance(other, Quaternion) or
                tf.convert_to_tensor(other).shape == () or
                tf.convert_to_tensor(other).shape[-1] == 1):  # scalar
            return quaternion_divide(other, self)
        return tf.divide(tf.convert_to_tensor(other), self._q)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __neg__(self):
        return Quaternion(-1 * self._q)

    # rich comparisons
    def __lt__(self, other):  # <
        return tf.less(self._q, other)

    def __le__(self, other):  # <=
        return tf.less_equal(self._q, other)

    def __eq__(self, other):  # ==
        return tf.equal(self._q, other)

    def __ne__(self, other):  # !=
        return tf.not_equal(self._q, other)

    def __gt__(self, other):  # >
        return tf.greater(self._q, other)

    def __ge__(self, other):  # >=
        return tf.greater_equal(self._q, other)

    def __repr__(self):
        return "<tfq.Quaternion '{}' ({})>".format(self.name,
                                                   self._q.__repr__()[1:-1])

    @scope_wrapper
    def conjugate(self):
        """Compute the conjugate of self.q, i.e. [w, -x, -y, -z]."""
        return quaternion_conjugate(self)

    def conj(self):
        """Compute the conjugate of self.q, i.e. [w, -x, -y, -z].

        Alias for Quaternion.conjugate().
        """
        return quaternion_conjugate(self)

    @scope_wrapper
    def inverse(self):
        """Compute the inverse of the quaternion, i.e. q.conjugate / q.norm."""
        return Quaternion(tf.convert_to_tensor(self.conjugate()) / self.norm())

    @scope_wrapper
    def normalized(self):
        """Compute the normalized quaternion."""
        return Quaternion(tf.divide(self._q, self.abs()))

    @scope_wrapper
    def as_rotation_matrix(self):
        """Calculate the corresponding rotation matrix.

        See
        http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/

        Returns:
            A `tf.Tensor` with R+1 dimensions and
            shape [d_1, ..., d_(R-1), 3, 3], the rotation matrix
        """
        # helper functions
        def diag(a, b):  # computes the diagonal entries,  1 - 2*a**2 - 2*b**2
            return 1 - 2 * tf.pow(a, 2) - 2 * tf.pow(b, 2)

        def tr_add(a, b, c, d):  # computes triangle entries with addition
            return 2 * a * b + 2 * c * d

        def tr_sub(a, b, c, d):  # computes triangle entries with subtraction
            return 2 * a * b - 2 * c * d

        w, x, y, z = tf.unstack(self.normalized().value(), num=4, axis=-1)
        m = [[diag(y, z), tr_sub(x, y, z, w), tr_add(x, z, y, w)],
             [tr_add(x, y, z, w), diag(x, z), tr_sub(y, z, x, w)],
             [tr_sub(x, z, y, w), tr_add(y, z, x, w), diag(x, y)]]
        return tf.stack([tf.stack(m[i], axis=-1) for i in range(3)], axis=-2)

    @staticmethod
    def validate_shape(x):
        """Raise a value error if x.shape ist not (..., 4)."""
        error_msg = ("Can't create a quaternion from a tensor with shape {}."
                     "The last dimension must be 4.")
        # Check is performed during graph construction. If your dimension
        # is unknown, tf.reshape(x, (-1, 4)) might work.
        if x.shape[-1] != 4:
            raise ValueError(error_msg.format(x.shape))

    @staticmethod
    def validate_type(x):
        """Raise a type error if the dtype of x is not float."""
        if not x.dtype.is_floating:
            raise TypeError("Quaternion: dtype must be one of float16/32/64.")

    @scope_wrapper
    def norm(self, keepdims=True):
        """Return the norm of the quaternion."""
        return tf.reduce_sum(tf.square(self._q), axis=-1, keep_dims=keepdims)

    @scope_wrapper
    def abs(self, keepdims=True):
        """Return the square root of the norm of the quaternion."""
        return tf.sqrt(self.norm(keepdims))


# ____________________________________________________________________________
#                          quaternion to tensor conversion
def quaternion_to_tensor(x, dtype=None, name=None, as_ref=None):
    """Convert a Quaternion to a `tf.Tensor`."""
    # Todo(phil): handle as_ref correctly
    return tf.convert_to_tensor(x.value(), dtype, name)


tf.register_tensor_conversion_function(Quaternion, quaternion_to_tensor,
                                       priority=100)
