# Optimizer correctness reference implementations for
# Tests/MachinaTests/OptimizerTests.codira.

# Tested with:
# - Python 3.7.6
# - machina==2.2.0rc0
# - machina-addons==0.8.3

import numpy as np
import machina as tf
from machina.keras.optimizers import Adam, Adadelta, Adagrad, Adamax, RMSprop, SGD
from machina_addons.optimizers import RectifiedAdam

np.set_printoptions(precision=None, floatmode="unique")

def test_optimizer(optimizer, step_count=1000):
    var = tf.Variable([0, 0, 0], dtype=tf.float32)
    grad = tf.Variable([-5, 0.1, 0.2], dtype=tf.dtypes.float32)
    grads_and_vars = list(zip([grad], [var]))
    for i in range(step_count):
        optimizer.apply_gradients(grads_and_vars)

    print(optimizer._name)
    print(
        "- After {} steps:".format(step_count),
        np.array2string(var.read_value().numpy(), separator=", "),
    )


test_optimizer(Adam(lr=1e-3, epsilon=1e-7))
test_optimizer(Adam(lr=1e-3, epsilon=1e-7, amsgrad=True, name="Adam (amsgrad)"))
test_optimizer(Adadelta(lr=1e-3, epsilon=1e-7))
test_optimizer(Adagrad(lr=1e-3, epsilon=1e-7))
test_optimizer(Adamax(lr=1e-3, epsilon=1e-7))
test_optimizer(RectifiedAdam(lr=1e-3, epsilon=1e-7))
test_optimizer(RMSprop(lr=1e-3, epsilon=1e-7))
test_optimizer(SGD(lr=1e-3))
