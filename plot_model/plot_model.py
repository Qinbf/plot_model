
"""Utilities related to model visualization."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import re
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
try:
    from tensorflow.python.keras.engine.functional import Functional as Functional_or_Network
except ModuleNotFoundError:
    from tensorflow.python.keras.engine.network import Network as Functional_or_Network


try:
  # pydot-ng is a fork of pydot that is better maintained.
  import pydot_ng as pydot
except ImportError:
  # pydotplus is an improved version of pydot
  try:
    import pydotplus as pydot
  except ImportError:
    # Fall back on pydot if necessary.
    try:
      import pydot
    except ImportError:
      pydot = None


def check_pydot():
  """Returns True if PyDot and Graphviz are available."""
  if pydot is None:
    return False
  try:
    # Attempt to create an image of a blank graph
    # to check the pydot/graphviz installation.
    pydot.Dot.create(pydot.Dot())
    return True
  except (OSError, pydot.InvocationException):
    return False


def is_wrapped_model(layer):
  from tensorflow.python.keras.layers import wrappers
  return (isinstance(layer, wrappers.Wrapper) and
          isinstance(layer.layer, Functional_or_Network))


def add_edge(dot, src, dst, output_shape=None):
  if not dot.get_edge(src, dst):
    if output_shape:
      dot.add_edge(pydot.Edge(src, dst, label=output_shape))
    else:
      dot.add_edge(pydot.Edge(src, dst))

def model_to_dot(model,
                 show_shapes=False,
                 show_layer_names=True,
                 rankdir='TB',
                 expand_nested=False,
                 dpi=96,
                 style=0,
                 color=True,
                 subgraph=False):
  """Convert a Keras model to dot format.

  Arguments:
    model: A Keras model instance.
    show_shapes: whether to display shape information.
    show_layer_names: whether to display layer names.
    rankdir: `rankdir` argument passed to PyDot,
        a string specifying the format of the plot:
        'TB' creates a vertical plot;
        'LR' creates a horizontal plot.
    expand_nested: whether to expand nested models into clusters.
    dpi: Dots per inch.
    style: value 0,1.
    color: whether to display color.
    subgraph: whether to return a `pydot.Cluster` instance.

  Returns:
    A `pydot.Dot` instance representing the Keras model or
    a `pydot.Cluster` instance representing nested model if
    `subgraph=True`.

  Raises:
    ImportError: if graphviz or pydot are not available.
  """
  from tensorflow.python.keras.layers import wrappers
  from tensorflow.python.keras.engine import sequential

  if not check_pydot():
    if 'IPython.core.magics.namespace' in sys.modules:
      # We don't raise an exception here in order to avoid crashing notebook
      # tests where graphviz is not available.
      print('Failed to import pydot. You must install pydot'
            ' and graphviz for `pydotprint` to work.')
      return
    else:
      raise ImportError('Failed to import pydot. You must install pydot'
                        ' and graphviz for `pydotprint` to work.')

  if subgraph:
    dot = pydot.Cluster(style='dashed', graph_name=model.name)
    dot.set('label', model.name)
    dot.set('labeljust', 'l')
  else:
    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    dot.set('concentrate', True)
    dot.set('dpi', dpi)
    dot.set_node_defaults(shape='record')

  sub_n_first_node = {}
  sub_n_last_node = {}
  sub_w_first_node = {}
  sub_w_last_node = {}

  if not model._is_graph_network:
    node = pydot.Node(str(id(model)), label=model.name)
    dot.add_node(node)
    return dot
  elif isinstance(model, sequential.Sequential):
    if not model.built:
      model.build()
  layers = model._layers
  num_layers = len(layers)

  # Create graph nodes.
  for i, layer in enumerate(layers):
    layer_id = str(id(layer))

    # Append a wrapped layer's label to node's label, if it exists.
    layer_name = layer.name
    class_name = layer.__class__.__name__
    class_name_lower = class_name.lower()
    config = 0
    try:
        config = layer.get_config()
    except:
        pass

    if isinstance(layer, wrappers.Wrapper):
      if expand_nested and isinstance(layer.layer, Functional_or_Network):
        submodel_wrapper = model_to_dot(layer.layer, show_shapes,
                                        show_layer_names, rankdir,
                                        expand_nested,
                                        subgraph=True)
        # sub_w : submodel_wrapper
        sub_w_nodes = submodel_wrapper.get_nodes()
        sub_w_first_node[layer.layer.name] = sub_w_nodes[0]
        sub_w_last_node[layer.layer.name] = sub_w_nodes[-1]
        dot.add_subgraph(submodel_wrapper)
      else:
        layer_name = '{}({})'.format(layer_name, layer.layer.name)
        child_class_name = layer.layer.__class__.__name__
        class_name = '{}({})'.format(class_name, child_class_name)

    if expand_nested and isinstance(layer, Functional_or_Network):
      submodel_not_wrapper = model_to_dot(layer, show_shapes,
                                          show_layer_names, rankdir,
                                          expand_nested,
                                          subgraph=True)
      # sub_n : submodel_not_wrapper
      sub_n_nodes = submodel_not_wrapper.get_nodes()
      sub_n_first_node[layer.name] = sub_n_nodes[0]
      sub_n_last_node[layer.name] = sub_n_nodes[-1]
      dot.add_subgraph(submodel_not_wrapper)

    # Create node's label.

    if show_layer_names:
      label = '{}: {}'.format(layer_name, class_name)
      inputs = re.compile('input')
      if inputs.findall(class_name_lower):
        pass
      else:
        if config != 0:
          conv = re.compile('conv')
          if conv.findall(class_name_lower):
            label = '{}:{},{}|kernel:{}  strides:{}'.format(layer_name,
                                                            class_name,
                                                            config.get('padding', 'na'),
                                                            config.get('kernel_size', 'na'),
                                                            config.get('strides', 'na'))
          pool = re.compile('pool')
          if pool.findall(class_name_lower) and class_name_lower[:6]!='global':
            label = '{}:{},{}|kernel:{}  strides:{}'.format(layer_name,
                                                            class_name,
                                                            config.get('padding', 'na'),
                                                            config.get('pool_size', 'na'),
                                                            config.get('strides', 'na'))
          activation = re.compile('activation')
          if activation.findall(class_name_lower):
            label = '{}:{}|{}'.format(layer_name,
                                      class_name,
                                      config['activation'])
          dropout = re.compile('dropout') 
          if dropout.findall(class_name_lower):
            label = '{}:{}|{}'.format(layer_name,
                                      class_name,
                                      config['rate'])
          dense = re.compile('dense') 
          if dense.findall(class_name_lower):
            label = '{}:{}|{}'.format(layer_name,
                                      class_name,
                                      config['activation'])

    else:
      label = '{}'.format(class_name)
      inputs = re.compile('input')
      if inputs.findall(class_name_lower):
        pass
      else:
        if config != 0:
          conv = re.compile('conv')
          if conv.findall(class_name_lower):
            label = '{},{}|kernel:{}  strides:{}'.format(class_name,
                                                         config.get('padding', 'na'),
                                                         config.get('kernel_size', 'na'),
                                                         config.get('strides', 'na'))
          pool = re.compile('pool')
          if pool.findall(class_name_lower) and class_name_lower[:6]!='global':
            label = '{},{}|kernel:{}  strides:{}'.format(class_name,
                                                         config.get('padding', 'na'),
                                                         config.get('pool_size', 'na'),
                                                         config.get('strides', 'na'))
          activation = re.compile('activation')
          if activation.findall(class_name_lower):
            label = '{}|{}'.format(class_name,
                                   config['activation']) 
          dropout = re.compile('dropout') 
          if dropout.findall(class_name_lower):
            label = '{}|{}'.format(class_name,
                                   config['rate'])
          dense = re.compile('dense') 
          if dense.findall(class_name_lower):
            label = '{}|{}'.format(class_name,
                                   config['activation'])

    # Rebuild the label as a table including input/output shapes.
    if show_shapes:

      def format_shape(shape):
        return str(shape).replace(str(None), '?')

      try:
        outputlabels = format_shape(layer.output_shape)
      except AttributeError:
        outputlabels = '?'
      if hasattr(layer, 'input_shape'):
        inputlabels = format_shape(layer.input_shape)
      elif hasattr(layer, 'input_shapes'):
        inputlabels = ', '.join(
            [format_shape(ishape) for ishape in layer.input_shapes])
      else:
        inputlabels = '?'

      if style == 0:
        inputs = re.compile('input')
        if inputs.findall(class_name_lower):
          label = '{%s}|{input:}|{%s}' % (label,
                                        inputlabels)
        else:                  
          for i,node in enumerate(layer._inbound_nodes):
            for outbound_layer in nest.flatten(node.outbound_layer):
              if outbound_layer.outbound_nodes == []:
                label = '{%s}|{output:}|{%s}' % (label,
                                            outputlabels)
              else:
                label = '{%s}' % (label)
      elif style == 1:
        label = '{%s}|{input:|output:}|{{%s}|{%s}}' % (label,
                                                      inputlabels,
                                                      outputlabels)
 

    if not expand_nested or not isinstance(layer, Functional_or_Network):
      if color == True:
        inputs = re.compile('input')
        conv = re.compile('conv')
        pool = re.compile('pool')
        normalization = re.compile('normalization')
        activation = re.compile('activation')
        dropout = re.compile('dropout') 
        dense = re.compile('dense') 
        padding = re.compile('padding')
        concatenate = re.compile('concatenate')
        rnn = re.compile('rnn')
        lstm = re.compile('lstm')
        gru = re.compile('gru')
        bidirectional = re.compile('bidirectional')
        if inputs.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label,  fillcolor='deeppink', style="filled")
        elif conv.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label, fillcolor='cyan', style="filled")
        elif pool.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label, fillcolor='chartreuse', style="filled")
        elif normalization.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label, fillcolor='dodgerblue1', style="filled")
        elif activation.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label, fillcolor='pink', style="filled")
        elif dropout.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label, fillcolor='darkorange', style="filled")
        elif dense.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label, fillcolor='darkorchid1', style="filled")
        elif padding.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label, fillcolor='beige', style="filled")
        elif concatenate.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label, fillcolor='tomato', style="filled")
        elif rnn.findall(class_name_lower) or lstm.findall(class_name_lower) or gru.findall(class_name_lower) or bidirectional.findall(class_name_lower):
          node = pydot.Node(layer_id, label=label, fillcolor='yellow1', style="filled")
        else:
          node = pydot.Node(layer_id, label=label, fillcolor='gold', style="filled")
      else:
        node = pydot.Node(layer_id, label=label)
      dot.add_node(node)

  # Connect nodes with edges.
  for j, layer in enumerate(layers):
    # print(layer)
    # print(layer.output_shape)
    def format_shape(shape):
        return str(shape).replace(str(None), '?')

    layer_id = str(id(layer))
    for i, node in enumerate(layer._inbound_nodes):
      node_key = layer.name + '_ib-' + str(i)
      if node_key in model._network_nodes:
        for inbound_layer in nest.flatten(node.inbound_layers):
          inbound_layer_id = str(id(inbound_layer))
          if not expand_nested:
            assert dot.get_node(inbound_layer_id)
            assert dot.get_node(layer_id)
            if style == 0:
              try:
                add_edge(dot, inbound_layer_id, layer_id, format_shape(inbound_layer.output_shape))
              except:
                add_edge(dot, inbound_layer_id, layer_id, '?')
            elif style == 1:
              add_edge(dot, inbound_layer_id, layer_id)
          else:
            # if inbound_layer is not Model or wrapped Model
            if (not isinstance(inbound_layer, Functional_or_Network) and
                not is_wrapped_model(inbound_layer)):
              # if current layer is not Model or wrapped Model
              if (not isinstance(layer, Functional_or_Network) and
                  not is_wrapped_model(layer)):
                assert dot.get_node(inbound_layer_id)
                assert dot.get_node(layer_id)
                if style == 0:
                  try:
                    add_edge(dot, inbound_layer_id, layer_id, format_shape(inbound_layer.output_shape))
                  except:
                    add_edge(dot, inbound_layer_id, layer_id, '?')
                elif style == 1:
                  add_edge(dot, inbound_layer_id, layer_id)
              # if current layer is Model
              elif isinstance(layer, Functional_or_Network):
                if style == 0:
                  add_edge(dot, inbound_layer_id,
                          sub_n_first_node[layer.name].get_name(),
                          format_shape(inbound_layer.output_shape))
                elif style == 1:
                  add_edge(dot, inbound_layer_id,
                          sub_n_first_node[layer.name].get_name())
              # if current layer is wrapped Model
              elif is_wrapped_model(layer):
                if style == 0:
                  try:
                    add_edge(dot, inbound_layer_id, layer_id, format_shape(inbound_layer.output_shape))
                  except:
                    add_edge(dot, inbound_layer_id, layer_id, '?')
                  name = sub_w_first_node[layer.layer.name].get_name()
                  add_edge(dot, layer_id, name, format_shape(layer.output_shape))
                elif style == 1:
                  add_edge(dot, inbound_layer_id, layer_id)
                  name = sub_w_first_node[layer.layer.name].get_name()
                  add_edge(dot, layer_id, name)
            # if inbound_layer is Model
            elif isinstance(inbound_layer, Functional_or_Network):
              name = sub_n_last_node[inbound_layer.name].get_name()
              if isinstance(layer, Functional_or_Network):
                output_name = sub_n_first_node[layer.name].get_name()
                if style == 0:
                  try:
                    add_edge(dot, name, output_name, format_shape(layer.output_shape))
                  except:
                    add_edge(dot, name, output_name, '?')
                elif style == 1:
                  add_edge(dot, name, output_name)
              else:
                if style == 0:
                  try:
                    add_edge(dot, name, layer_id, format_shape(layer.output_shape))
                  except:
                    add_edge(dot, name, layer_id, '?')
                elif style == 1:
                  add_edge(dot, name, layer_id)
            # if inbound_layer is wrapped Model
            elif is_wrapped_model(inbound_layer):
              inbound_layer_name = inbound_layer.layer.name
              if style == 0:
                try:
                  add_edge(dot,
                          sub_w_last_node[inbound_layer_name].get_name(),
                          layer_id,
                          format_shape(inbound_layer.output_shape))
                except:
                  add_edge(dot,
                          sub_w_last_node[inbound_layer_name].get_name(),
                          layer_id,
                          '?')
              elif style == 1:
                add_edge(dot,
                        sub_w_last_node[inbound_layer_name].get_name(),
                        layer_id)
         
  return dot


def plot_model(model,
               to_file='model.png',
               show_shapes=True,
               show_layer_names=False,
               rankdir='TB',
               expand_nested=False,
               style = 0,
               color = True,
               dpi=96):
  """Converts a Keras model to dot format and save to a file.

  Example:

  >>> import tensorflow as tf
  >>> input = tf.keras.Input(shape=(100,), dtype='int32', name='input')
  >>> x = tf.keras.layers.Embedding(output_dim=512, input_dim=10000, input_length=100)(input)
  >>> x = tf.keras.layers.LSTM(32)(x)
  >>> x = tf.keras.layers.Dense(64, activation='relu')(x)
  >>> x = tf.keras.layers.Dense(64, activation='relu')(x)
  >>> x = tf.keras.layers.Dense(64, activation='relu')(x)
  >>> output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
  >>> model = tf.keras.Model(inputs=[input], outputs=[output])
  >>> dot_img_file = '/tmp/model_1.png'
  >>> plot_model(model, to_file=dot_img_file, show_shapes=True)
  <IPython.core.display.Image object>

  Arguments:
    model: A Keras model instance
    to_file: File name of the plot image.
    show_shapes: whether to display shape information.
    show_layer_names: whether to display layer names.
    rankdir: `rankdir` argument passed to PyDot,
        a string specifying the format of the plot:
        'TB' creates a vertical plot;
        'LR' creates a horizontal plot.
    expand_nested: Whether to expand nested models into clusters.
    style: value 0,1.
    color: whether to display color.
    dpi: Dots per inch.

  Returns:
    A Jupyter notebook Image object if Jupyter is installed.
    This enables in-line display of the model plots in notebooks.
  """
  assert(style == 0 or style == 1)
  dot = model_to_dot(model,
                     show_shapes=show_shapes,
                     show_layer_names=show_layer_names,
                     rankdir=rankdir,
                     expand_nested=expand_nested,
                     style = style,
                     color = color,
                     dpi=dpi)
  if dot is None:
    return
  _, extension = os.path.splitext(to_file)
  if not extension:
    extension = 'png'
  else:
    extension = extension[1:]
  # Save image to disk.
  dot.write(to_file, format=extension)
  # Return the image as a Jupyter Image object, to be displayed in-line.
  # Note that we cannot easily detect whether the code is running in a
  # notebook, and thus we always return the Image if Jupyter is available.
  try:
    from IPython import display
    return display.Image(filename=to_file)
  except ImportError:
    pass
