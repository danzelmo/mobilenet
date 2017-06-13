from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras import layers as l
from . import depthwise_conv as dw


def conv2_bn(x, filts, k=3, stride=1, rate=1, name=None, pad='same'):
    x = l.Conv2D(filts, (k, k),
                 strides=(stride, stride),
                 dilation_rate=rate,
                 padding=pad,
                 name=name)(x)

    x = l.BatchNormalization(name=name + '_bn')(x)
    x = l.Activation('relu', name=name + '_relu')(x)
    return x


def depth2_bn(x, filts, stride=1, multiplier=1, k=3, name=None, pad='same'):
    x = dw.DepthWiseConv2D(filts, (k, k),
                           strides=(stride, stride),
                           depth_multiplier=multiplier,
                           padding=pad,
                           use_bias=False,
                           name=name + '_dw')(x)

    x = l.BatchNormalization(name=name + '_bn')(x)
    x = l.Activation('relu', name=name + '_relu')(x)
    return x

# Seperable conv2d with batchnormalization after depthwise as in mobilenet paper
def seperable_2d(x, filts, stride=1, name=None, pad='same'):
    x = depth2_bn(x, filts,
                  multiplier=1,
                  stride=stride,
                  k=3, name=name,
                  pad=pad)

    x = conv2_bn(x, filts, k=1, name=name + '_1x1')

    return x
    
# The mobilenet feature extractor and classification head
def mobilenet_base(inputs):
    x = conv2_bn(inputs, 32, stride=2, name='conv1')
    x = seperable_2d(x, 32, name='sep1')
    x = seperable_2d(x, 64, 2, name='sep2')
    x = seperable_2d(x, 128, name='sep3')
    x = seperable_2d(x, 128, 2, name='sep4')
    x = seperable_2d(x, 256, name='sep5')
    x = seperable_2d(x, 256, 2, name='sep6')

    for i in range(5):
        x = seperable_2d(x, 512, name='sep' + str(i + 7))

    x = seperable_2d(x, 512, 2, name='sep12')
    x = seperable_2d(x, 1024, name='sep13')
    return x


def mobilenet_clf_head(inputs, num_classes):
    x = mobilenet_base(inputs)
    x = l.GlobalAvgPool2D(name='avg_pool1')(x)
    x = l.Dense(num_classes, activation='softmax', name='dense_pred')(x)
    model = Model(inputs, x)
    return model
