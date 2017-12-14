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
Let's take a look at a simple rotation:
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

If you'd like to have a certain feature please check the ToDo file first before opening an issue.
