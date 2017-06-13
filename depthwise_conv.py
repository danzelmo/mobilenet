# Modified by Devin Anzelmo 2017
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras import layers as l

from tensorflow.contrib.keras import constraints
from tensorflow.contrib.keras import initializers
from tensorflow.contrib.keras import regularizers

from tensorflow.contrib.keras.python.keras.engine import InputSpec
from tensorflow.contrib.keras.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.contrib.keras.python.keras.utils import conv_utils

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn


# Modified from seperable_conv2d in keras tensorflow backend
def depthwise_conv2d(x,
                     depthwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
    """2D convolution with separable filters.
    Arguments:
        x: input tensor
        depthwise_kernel: convolution kernel for the depthwise convolution.
        strides: strides tuple (length 2).
        padding: padding mode, "valid" or "same".
        data_format: data format, "channels_first" or "channels_last".
        dilation_rate: tuple of integers,
            dilation rates for the depthwise convolution.
  
    Returns:
        Output tensor.
  
    Raises:
        ValueError: if `data_format` is neither `channels_last` or
        `channels_first`.
    """
    if data_format is None:
        data_format = K.image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format ' + str(data_format))

    x = K._preprocess_conv2d_input(x, data_format)
    padding = K._preprocess_padding(padding)
    strides = (1,) + strides + (1,)

    x = nn.depthwise_conv2d(
        x,
        depthwise_kernel,
        strides=strides,
        padding=padding,
        rate=dilation_rate)
    return K._postprocess_conv2d_output(x, data_format)


# Adapted SeperableConv2D keras layer for depthwise convolution. 
class DepthWiseConv2D(l.Conv2D):
    """Depthwise spatial convolution layer
    
    depthwise spatial convolution
    (which acts on each input channel separately)
    
    output channels. The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.
  
    Arguments:
        filters: Integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution).
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filterss_in * depth_multiplier`.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix.
        bias_initializer: Initializer for the bias vector.
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix.
        bias_constraint: Constraint function applied to the bias vector.
  
    Input shape:
        4D tensor with shape:
        `(batch, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, rows, cols, channels)` if data_format='channels_last'.
  
    Output shape:
        4D tensor with shape:
        `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 depth_multiplier=1,
                 activation=None,
                 use_bias=False,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        super(DepthWiseConv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if len(input_shape) < 4:
            raise ValueError('Inputs to `SeparableConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`SeparableConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1],
                                  input_dim, self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        outputs = depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            data_format=self.data_format,
            strides=self.strides,
            padding=self.padding)

        if self.bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
        else:
            rows = input_shape[1]
            cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding, self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding, self.strides[1])
        if self.data_format == 'channels_first':
            return tensor_shape.TensorShape(
                [input_shape[0], self.filters, rows, cols])
        else:
            return tensor_shape.TensorShape(
                [input_shape[0], rows, cols, self.filters])

    def get_config(self):
        config = super(DepthWiseConv2D, self).get_config()
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(
            self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(
            self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(
            self.depthwise_constraint)
        return config


# add this to custom objects for restoring model save files
get_custom_objects().update({'DepthWiseConv2D': DepthWiseConv2D})
