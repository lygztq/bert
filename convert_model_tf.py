from run_classifier import FLAGS
import tensorflow as tf
import modeling
import os

# model used: L12 H768 A12

bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
tf.gfile.MakeDirs(FLAGS.output_dir)

with tf.Graph().as_default() as graph_old:
    # input nodes
    input_ids = tf.placeholder(tf.int32, shape=[None, None], name="input_ids")
    input_mask = tf.placeholder(tf.int32, shape=[None, None], name="input_mask")
    token_type_ids = tf.placeholder(tf.int32, shape=[None, None], name="token_type_ids")
    
    model = modeling.BertModel(bert_config, False, input_ids, input_mask, token_type_ids)

    # output nodes
    output_layer = model.get_pooled_output()
    one = tf.ones_like(output_layer)
    output_node = tf.math.add(output_layer, one, name="output_node")
    # output_node = tf.identity(output_layer, name="output_node")

    isess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(isess, FLAGS.init_checkpoint)

    constant_graph = tf.graph_util.convert_variables_to_constants(isess, graph_old.as_graph_def(add_shapes=True), ["output_node"])
    constant_graph = tf.graph_util.remove_training_nodes(constant_graph)
    with tf.gfile.GFile(os.path.join(FLAGS.output_dir, "bert.pb"), mode="wb") as f:
        f.write(constant_graph.SerializeToString())

