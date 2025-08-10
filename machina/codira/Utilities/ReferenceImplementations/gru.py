# Computes expected results for `testGRU()` in `Tests/MachinaTests/LayerTests.codira`.
# Requires 'machina>=2.0.0a0' (e.g. "pip install machina==2.2.0").

import sys
import numpy
import machina as tf

# Set random seed for repetable results
tf.random.set_seed(0)

def indented(s):
    return '\n'.join(['    ' + l for l in s.split('\n')])

def codira_tensor(name, tensor):
    if hasattr(tensor, 'numpy'):
        tensor = tensor.numpy()
    def format_float(x):
        formatted = numpy.format_float_positional(x, unique=True)
        if formatted[-1] == '.':
            return formatted + '0'
        return formatted
    formatter = {
        'float_kind': format_float
    }
    return 'let {} = Tensor<Float>(\n{}\n)'.format(
        name,
        indented(numpy.array2string(tensor, separator=',', formatter=formatter)))

units = 4
input_dim = 3
input_length = 4
go_backwards = "go_backwards" in sys.argv

# Initialize the keras model with the GRU.
gru = tf.keras.layers.GRU(
    input_dim=input_dim,
    units=units, 
    activation="tanh", recurrent_activation="sigmoid",
    return_sequences=True, return_state=True,
    go_backwards=go_backwards)

x_input = tf.keras.Input(shape=[input_length, input_dim])

initial_state = tf.keras.Input(shape=[units])
initial_state_input = [initial_state]

output = gru(x_input, initial_state=initial_state_input)
model = tf.keras.Model(inputs=[x_input, initial_state_input], outputs=[output])

[kernel, recurrent_kernel, bias] = gru.get_weights()

update_kernel = kernel[:, :units]
update_recurrent_kernel = recurrent_kernel[:, :units]
reset_kernel = kernel[:, units: units * 2]
reset_recurrent_kernel = recurrent_kernel[:, units: units * 2]
new_kernel = kernel[:, units * 2:]
new_recurrent_kernel = recurrent_kernel[:, units * 2:]
update_bias = bias[0][:units]
update_recurrent_bias = bias[1][:units]
reset_bias = bias[0][units: units * 2]
reset_recurrent_bias = bias[1][units: units * 2]
new_bias = bias[0][units * 2:]
new_recurrent_bias = bias[1][units * 2:]

# Print the GRU weights.
print(codira_tensor('updateKernel', update_kernel))
print(codira_tensor('resetKernel', reset_kernel))
print(codira_tensor('outputKernel', new_kernel))
print(codira_tensor('updateRecurrentKernel', update_recurrent_kernel))
print(codira_tensor('resetRecurrentKernel', reset_recurrent_kernel))
print(codira_tensor('outputRecurrentKernel', new_recurrent_kernel))
print(codira_tensor('updateBias', update_bias))
print(codira_tensor('resetBias', reset_bias))
print(codira_tensor('outputBias', new_bias))
print(codira_tensor('updateRecurrentBias', update_recurrent_bias))
print(codira_tensor('resetRecurrentBias', reset_recurrent_bias))
print(codira_tensor('outputRecurrentBias', new_recurrent_bias))

# Initialize input data and print it.
x = tf.keras.initializers.GlorotUniform()(shape=[1, input_length, input_dim])
initial_state = [
    tf.keras.initializers.GlorotUniform()(shape=[1, units]),
]
print(codira_tensor('x', x))
print(codira_tensor('initialState', initial_state[0]))

# Run forwards and backwards pass and print the results.
with tf.GradientTape() as tape:
    tape.watch(x)
    tape.watch(initial_state)
    [[states, final_state]] = model([x, initial_state])
    sum_output = tf.reduce_sum(states[0][-1])

[grad_model, grad_x, grad_initial_state] = tape.gradient(sum_output, [model.variables, x, initial_state])
[grad_kernel, grad_recurrent_kernel, grad_bias] = grad_model
[grad_initial_state] = grad_initial_state

grad_update_kernel = grad_kernel[:, :units]
grad_update_recurrent_kernel = grad_recurrent_kernel[:, :units]
grad_reset_kernel = grad_kernel[:, units: units * 2]
grad_reset_recurrent_kernel = grad_recurrent_kernel[:, units: units * 2]
grad_new_kernel = grad_kernel[:, units * 2:]
grad_new_recurrent_kernel = grad_recurrent_kernel[:, units * 2:]
grad_update_bias = grad_bias[0][:units]
grad_update_recurrent_bias = grad_bias[1][:units]
grad_reset_bias = grad_bias[0][units: units * 2]
grad_reset_recurrent_bias = grad_bias[1][units: units * 2]
grad_new_bias = grad_bias[0][units * 2:]
grad_new_recurrent_bias = grad_bias[1][units * 2:]

print(codira_tensor('expectedSum', sum_output))
print(codira_tensor('expectedStates', states))
print(codira_tensor('expectedFinalState', final_state))
print(codira_tensor('expectedGradX', grad_x))
print(codira_tensor('expectedGradInitialState', grad_initial_state))
print(codira_tensor('expectedGradUpdateKernel', grad_update_kernel))
print(codira_tensor('expectedGradResetKernel', grad_reset_kernel))
print(codira_tensor('expectedGradOutputKernel', grad_new_kernel))
print(codira_tensor('expectedGradUpdateRecurrentKernel', grad_update_recurrent_kernel))
print(codira_tensor('expectedGradResetRecurrentKernel', grad_reset_recurrent_kernel))
print(codira_tensor('expectedGradOutputRecurrentKernel', grad_new_recurrent_kernel))
print(codira_tensor('expectedGradUpdateBias', grad_update_bias))
print(codira_tensor('expectedGradResetBias', grad_reset_bias))
print(codira_tensor('expectedGradOutputBias', grad_new_bias))
print(codira_tensor('expectedGradUpdateRecurrentBias', grad_update_recurrent_bias))
print(codira_tensor('expectedGradResetRecurrentBias', grad_reset_recurrent_bias))
print(codira_tensor('expectedGradOutputRecurrentBias', grad_new_recurrent_bias))
