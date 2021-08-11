# import tensorflow as tf
# import numpy as np

# with tf.Graph().as_default() as graph:
#     x1 = tf.constant(
#         [[0, 1,  2,  3],
#         [4, 5,  6,  7],
#         [8, 9, 10, 11]],
#         dtype=tf.float32,
#         name="x1")

#     max_loop = tf.shape(x1)[0]

#     output_ta = tf.TensorArray(
#         dtype=tf.float32,
#         size=max_loop+1,
#         dynamic_size=False,
#         clear_after_read=False,
#         infer_shape=True)

#     init_v = tf.zeros(tf.shape(x1)[1], tf.float32, name="init_v")
#     output_ta = output_ta.write(0, init_v)

#     def cumsum(i, buff_ta):
#         next_elem = tf.nn.embedding_lookup(x1, i)
#         pre_elem = buff_ta.read(i)
#         sum = pre_elem + next_elem
#         i_next = i + 1
#         buff_ta = buff_ta.write(i_next, sum)
#         return i_next, buff_ta

#     _, final_buff = tf.while_loop(
#         cond=lambda i, _: tf.less(i, max_loop),
#         body=cumsum,
#         loop_vars=(tf.constant(0, dtype=tf.int32), output_ta))

#     out = final_buff.stack()

#     isess = tf.InteractiveSession()
#     with tf.gfile.GFile("./convert_model/dev_test.pb", mode="wb") as f:
#         f.write(graph.as_graph_def().SerializeToString())

# import tensorflow as tf

# model_path = "./convert_model/dev_test.pb"


# with tf.gfile.GFile(model_path, "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#     graph = tf.import_graph_def(graph_def, name="")

# for node in graph_def.node:
#     print("Node: {}:\n------".format(node.name))

#     print("\n----Inputs----")
#     for idx, n_name in enumerate(node.input):
#         print("input node {}: {}".format(idx, n_name))
    
#     print("\n----Attrs----")
#     for name, value in node.attr.items():
#         print("{}: {}".format(name, value))

#     print("\n##############################################n")

from tvm.relay.ty import Any
from tvm.relay import const, var, Tuple
from tvm.relay.op import concatenate, shape_of, strided_slice

a = [3, 5, Any()]
b = var("b", shape=a, dtype="float32")
c = shape_of(b)
print(c)
d = [strided_slice(c, begin=[i], end=[i+1], strides=[1]) for i in range(len(a))]
print(d)
