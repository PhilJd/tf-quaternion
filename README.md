# Tensorflow Quaternion
An implementation of quaternions for tensorflow. Fully differentiable. Licensed under Apache 2.0 License.


**Note: This project is currently in alpha status. Some functions have not even been tested yet.**


The tfquaternion module provides an implementation of quaternions as a tensorflow graph.
The quaternion value can either be represented as `tf.Tensor` or `tf.Variable`.
As all operations are derivable, the module can be used to optimize a rotation of
points in 3D space, given that a `tf.Variable` is used to represent the value.
Other awesome features are:
- Operations are scoped, so they appear nice and clean in your tensorboard graph.
- Operators are implemented.

### Installation

To install the git version as development version run:
```
git clone https://github.com/PhilJd/tf-quaternion.git
cd tf-quaternion
pip install -e .
```
The -e option only links the working copy to the python site-packages,
so to upgrade, you only need to run `git pull`.


### Usage
`tfquaternion` offers module functions for the basic quaternion arithmetic
operations as well as a `Quaternion` class which supports the relevant magic
methods. This is similar to the tensorflow API, e.g. `tfq.quaternion_multiply`
vs. `tf.multiply` and `tfq.Quaternion` vs `tf.Tensor`. Note that all functions
starting with `tf.quaternion_...` assume that it's arguments are `tf.Tensor`s
(or `tfq.Quaternion`s) that can be casted to `tfq.Quaternion`, i.e. the shape
must be (..., 4).

Before getting started, an important note on the division:
This library resembles the division behaviour of
(mobles quaternion)[https://github.com/moble/quaternion/]. While in
general the division operator is not defined (from the notation q1/q2 one can
not conclude if q1/q2 = q1 * q2^-1 or q1/q2 = q2^-1 * q1), we follow moble's
implementation, i.e.  `tfq.quaternion_divide` and `Quaternion.__truediv__`
compute `q1/q2 = q1 * 1/q2`.


#### Example
A simple rotation by a quaternion can look like this:
```
>>> import tfquaternion as tfq
>>> import tensorflow as tf
>>> s = tf.Session()
>>> points = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
>>> quat = tfq.Quaternion([0, 1, 0, 0])  rotate by 180 degrees around x axis
>>> s.run(tf.matmul(quat.as_rotation_matrix(), points))
array([[ 1.,  0.,  0.],
       [ 0., -1.,  0.],
       [ 0.,  0., -1.]], dtype=float32)

```

#### API

##### class Quaternion
The usage of the `*`-Operator depends on the multiplier. If the multiplier is a
Quaternion, quaternion multiplication is performed while multiplication with
a tf.Tensor uses tf.multiply. The behaviour of division is similar, except if
the dividend is a scalar, then the inverse of the quaternion is computed.
```
tfq.Quaternion([1, 0, 0, 0]) * tfq.Quaternion([0, 4, 0, 0])
>>> tfq.Quaternion([0, 4, 0, 0)
tfq.Quaternion([1, 0, 0, 0]) * tf.Tensor([0, 4, 0, 0])
>>> tf.Quaternion([0, 0, 0, 0)
```