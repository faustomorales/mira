import io
import logging
import configparser
from collections import defaultdict
from functools import reduce

import numpy as np
from keras import backend as K
from keras import layers as keras_layers
from keras.models import Model
from keras.regularizers import l2

log = logging.getLogger(__name__)


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def convert_darknet_weights(
    config_path: str,
    weights_path: str,
    output_path: str,
    weights_only: bool=False
):
    """Convert DarkNet weights to Keras weights. This function
    taken directly from:
    https://github.com/qqwweee/keras-yolo3/blob/master/convert.py

    """
    def unique_config_sections(config_file):
        """Convert all config sections to have unique names. Adds unique suffixes
        to config sections for compability with configparser.
        """
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(config_file) as fin:
            for line in fin:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream

    assert config_path.endswith('.cfg'), '{} is not a .cfg file'.format(config_path)  # noqa: E501
    assert weights_path.endswith('.weights'), '{} is not a .weights file'.format(weights_path)  # noqa: E501
    assert output_path.endswith('.h5'), 'output path {} is not a .h5 file'.format(output_path)  # noqa: E501

    log.info('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(
        shape=(3, ), dtype='int32', buffer=weights_file.read(12))
    if (major*10+minor) >= 2 and major < 1000 and minor < 1000:
        seen = np.ndarray(
            shape=(1,),
            dtype='int64',
            buffer=weights_file.read(8)
        )
    else:
        seen = np.ndarray(
            shape=(1,),
            dtype='int32',
            buffer=weights_file.read(4)
        )
    log.info(
        'Weights header: {0}, {1}, {2}, {3}'.format(
            major,
            minor,
            revision,
            seen
        )
    )

    log.info('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    log.info('Creating Keras model.')
    index = {
        'input': 0,
        'zeropadding2d': 0,
        'convolution2d': 0,
        'concatenate': 0,
        'maxpooling2d': 0,
        'add': 0,
        'upsampling2d': 0,
        'batchnormalization': 0,
        'leakyrelu': 0
    }
    input_layer = keras_layers.Input(
        shape=(None, None, 3),
        name='input_{0}'.format(index['input'])
    )
    index['input'] += 1
    prev_layer = input_layer
    all_layers = []

    if 'net_0' in cfg_parser.sections():
        weight_decay = float(cfg_parser['net_0']['decay'])
    else:
        weight_decay = 5e-4

    count = 0
    out_index = []
    for section in cfg_parser.sections():
        log.info('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)

            weights_shape = (size, size, prev_layer_shape[-1], filters)
            darknet_w_shape = (filters, weights_shape[2], size, size)
            weights_size = np.product(weights_shape)
            log.info(
                'conv2d (activation: {0}, weights: {1}, bn: {2})'.format(
                    str(activation),
                    str(weights_shape),
                    str(batch_normalize)
                )
            )
            conv_bias = np.ndarray(
                shape=(filters, ),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            count += filters

            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                count += 3 * filters

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]

            conv_weights = np.ndarray(
                shape=darknet_w_shape,
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [
                conv_weights, conv_bias
            ]

            # Handle activation.
            act_fn = None
            if activation == 'leaky':
                pass  # Add advanced activation later.
            elif activation != 'linear':
                raise ValueError(
                    'Unknown activation function `{}` in section {}'.format(
                        activation, section))

            # Create Conv2D layer
            if stride > 1:
                # Darknet uses left and top padding instead of 'same' mode
                prev_layer = keras_layers.ZeroPadding2D(
                    ((1, 0), (1, 0)),
                    name='zeropadding2d_{0}'.format(index['zeropadding2d'])
                )(prev_layer)
                index['zeropadding2d'] += 1
            conv_layer = keras_layers.Conv2D(
                filters=filters,
                kernel_size=(size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                activation=act_fn,
                padding=padding,
                name='convolution2d_{0}'.format(index['convolution2d'])
            )(prev_layer)
            index['convolution2d'] += 1

            if batch_normalize:
                conv_layer = keras_layers.BatchNormalization(
                    weights=bn_weight_list,
                    name='batchnormalization_{0}'.format(
                        index['batchnormalization']
                    )
                )(conv_layer)
                index['batchnormalization'] += 1
            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = keras_layers.LeakyReLU(
                    alpha=0.1,
                    name='leakyrelu_{0}'.format(index['leakyrelu'])
                )(prev_layer)
                index['leakyrelu'] += 1
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                log.info(
                    'Concatenating route layers: ' + ','.join(
                        [str(l) for l in layers]
                    )
                )
                concatenate_layer = keras_layers.Concatenate(
                    name='concatenate_{0}'.format(index['concatenate'])
                )(layers)
                index['concatenate'] += 1
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                skip_layer = layers[0]  # only one layer to route
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                keras_layers.MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same',
                    name='maxpooling2d_{0}'.format(index['maxpooling2d'])
                )(prev_layer)
            )
            index['maxpooling2d'] += 1
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            shortcut_index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            all_layers.append(keras_layers.Add(
                name='add_{0}'.format(index['add'])
            )([all_layers[shortcut_index], prev_layer]))
            index['add'] += 1
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            all_layers.append(
                keras_layers.UpSampling2D(
                    stride,
                    name='upsampling2d_{0}'.format(index['upsampling2d'])
                )(prev_layer)
            )
            index['upsampling2d'] += 1
            prev_layer = all_layers[-1]

        elif section.startswith('yolo'):
            out_index.append(len(all_layers)-1)
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass

        else:
            raise ValueError(
                'Unsupported section header type: {}'.format(section))

    # Create and save model.
    if len(out_index) == 0:
        out_index.append(len(all_layers)-1)
    model = Model(
        inputs=input_layer,
        outputs=[all_layers[i] for i in out_index]
    )
    if weights_only:
        model.save_weights('{}'.format(output_path))
        log.info('Saved weights to {0}'.format(output_path))
    else:
        model.save('{}'.format(output_path))
        log.info('Saved model to {0}'.format(output_path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) / 4
    weights_file.close()
    log.info(
        'Read {} of {} from Darknet weights'.format(
            count,
            count + remaining_weights
        )
    )
    if remaining_weights > 0:
        log.warning('{0} unused weights'.format(remaining_weights))
    return model
