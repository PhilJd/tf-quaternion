# Tensorflow Quaternion
An implementation of quaternions for and written in tensorflow. Fully derivable. Licensed under MIT License.


Note: This project is currently in beta status. 

The tfquaternion module provides an implementation of quaternions as a tensorflow graph. As all operations are derivable, the module can also be used to optimize a rotation of points in 3D space.

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

ToDo: Add an example of optimization + images.
