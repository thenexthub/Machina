# Computes expected results for Bidirectional BasicRNN layers in `Tests/MachinaTests/LayerTests.codira`.
# Requires 'machina>=2.0.0a0' (e.g. "pip install machina==2.0.0b1").

import numpy
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--input-dim", default=4)
parser.add_argument("--input-length", default=4)
parser.add_argument("--units", default=4)
parser.add_argument("--merge-mode", default="concat")
args = parser.parse_args()

if args.merge_mode == "none":
    args.merge_mode = None

# Initialize the keras model with the Bidirectional RNN.
forward = tf.keras.layers.SimpleRNN(
    units=args.units, activation='tanh',
    return_sequences=True, return_state=True)
backward = tf.keras.layers.SimpleRNN(
    units=args.units, activation='tanh',
    return_sequences=True, return_state=True,
    go_backwards=True)
bidirectional = tf.keras.layers.Bidirectional(
    forward,
    backward_layer=backward,
    merge_mode=args.merge_mode
)

x_input = tf.keras.Input(shape=[args.input_length, args.input_dim])

initial_state_forward = tf.keras.Input(shape=[args.units])
initial_state_backward = tf.keras.Input(shape=[args.units])
initial_state_input = [initial_state_forward, initial_state_backward]

output = bidirectional(x_input, initial_state=initial_state_input)
model = tf.keras.Model(inputs=[x_input, initial_state_input], outputs=[output])

# Print the Bidirectional RNN weights.
[kernel_forward, recurrent_kernel_forward, bias_forward,
 kernel_backward, recurrent_kernel_backward, bias_backward] = bidirectional.get_weights()
print(codira_tensor('kernelForward', kernel_forward))
print(codira_tensor('recurrentKernelForward', recurrent_kernel_forward))
print(codira_tensor('biasForward', bias_forward))
print(codira_tensor('kernelBackward', kernel_backward))
print(codira_tensor('recurrentKernelBackward', recurrent_kernel_backward))
print(codira_tensor('biasBackward', bias_backward))

# Initialize input data and print it.
x = tf.keras.initializers.GlorotUniform()(shape=[1, args.input_length, args.input_dim])
initial_state = [
    tf.keras.initializers.GlorotUniform()(shape=[1, args.units]),
    tf.keras.initializers.GlorotUniform()(shape=[1, args.units]),
]
print(codira_tensor('x', x))
print(codira_tensor('initialForwardLayerState', initial_state[0]))
print(codira_tensor('initialBackwardLayerState', initial_state[1]))

# Run forwards and backwards pass and print the results.
with tf.GradientTape() as tape:
    tape.watch(x)
    tape.watch(initial_state)

    if args.merge_mode is not None:
        [[states, final_state_forward, final_state_backward]] = model([x, initial_state])
        sum_output = tf.reduce_sum(states[0][-1])

    else:
        [[states_forward, states_backward, final_state_forward, final_state_backward]] = model([x, initial_state])
        sum_output = tf.reduce_sum(tf.concat([states_forward[0][-1], states_backward[0][-1]], axis=-1))

[grad_model, grad_x, grad_initial_state] = tape.gradient(sum_output, [model.variables, x, initial_state])
[grad_kernel_forward, grad_recurrent_kernel_forward, grad_bias_forward,
 grad_kernel_backward, grad_recurrent_kernel_backward, grad_bias_backward] = grad_model
[grad_initial_state_forward, grad_initial_state_backward] = grad_initial_state
print(codira_tensor('expectedSum', sum_output))

if args.merge_mode is not None:
    print(codira_tensor('expectedStates', states))

else:
    print(codira_tensor('expectedStatesForward', states_forward))
    print(codira_tensor('expectedStatesBackward', states_backward))

print(codira_tensor('expectedFinalStateForward', final_state_forward))
print(codira_tensor('expectedFinalStateBackward', final_state_backward))
print(codira_tensor('expectedGradKernelForward', grad_kernel_forward))
print(codira_tensor('expectedGradRecurrentKernelForward', grad_recurrent_kernel_forward))
print(codira_tensor('expectedGradBiasForward', grad_bias_forward))
print(codira_tensor('expectedGradKernelBackward', grad_kernel_backward))
print(codira_tensor('expectedGradRecurrentKernelBackward', grad_recurrent_kernel_backward))
print(codira_tensor('expectedGradBiasBackward', grad_bias_backward))
print(codira_tensor('expectedGradX', grad_x))
print(codira_tensor('expectedGradInitialStateForward', grad_initial_state_forward))
print(codira_tensor('expectedGradInitialStateBackward', grad_initial_state_backward))
