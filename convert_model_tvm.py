import argparse
import os
import tensorflow as tf
import tvm
import tvm.relay.testing.tf as tf_testing
from tvm import relay
import numpy as np
from time import time
import ctypes

_cudart = ctypes.CDLL('libcudart.so')

def cu_prof_start():
  ret = _cudart.cudaProfilerStart()
  if ret != 0:
    raise Exception('cudaProfilerStart() returned %d' % ret)

def cu_prof_stop():
  ret = _cudart.cudaProfilerStop()
  if ret != 0:
    raise Exception('cudaProfilerStop() returned %d' % ret)

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./convert_model",
                    help="dir path to input tf-protobuf model")
parser.add_argument("--output_dir", type=str, default="./output_model",
                    help="dir path to store output model and libs")
parser.add_argument("--update", action="store_true")
parser.add_argument("--nvprof", action="store_true", help="Use --profile-from-start off here")
parser.add_argument("--mt_ver", action="store_true", help="use mt inner version")
parser.add_argument("--print_mod", action="store_true", help="output relay ir module as text file")
parser.add_argument("--num_trials", type=int, default=1, help="number of trials, return the mean result")
args = parser.parse_args()

# configure paths
model_name = "bert.pb"
model_path = os.path.join(args.model_dir, model_name)
lib_output_path = os.path.join(args.output_dir, "relay_vm_model_lib.so")
code_output_path = os.path.join(args.output_dir, "relay_vm_model_code.ro")

if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

have_cached = os.path.exists(lib_output_path) and os.path.exists(code_output_path)

# configure devices

# target = tvm.target.Target("llvm")
# dev = tvm.cpu(0)

target = tvm.target.Target("cuda -libs=cublas,cudnn")
if args.mt_ver:
    dev = tvm.gpu(0)
    print("MT version, use gpu device name instead of cuda.")
else:
    dev = tvm.cuda(0)

# convert and compile, if already had one, use the cached one.
if not have_cached or args.update:
    print("Convert model from beginning...")
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        
    mod, params = relay.frontend.from_tensorflow(graph_def, outputs=["output_node"])

    if args.print_mod:
        with open(os.path.join(args.output_dir, "raw_ir_module.txt"), 'w') as f:
            f.write(str(mod))

    print("Tensorflow protobuf imported to relay frontend.")

    with tvm.transform.PassContext(opt_level=3):
        exec = relay.backend.vm.compile(mod, target, params=params)
        if args.print_mod:
            from dlctools.transforms import instrument_stats, bytecode_easyread
            import json
            with open(os.path.join(args.output_dir, "vm_bytecode_stats.txt"), 'w') as f:
                stats = instrument_stats(exec.bytecode, sort=True)
                num_insts = 0
                for _, v in stats.items():
                    num_insts += v
                stats = {k: "{}, {:.4f}".format(v, v / num_insts * 100) for k, v in stats.items()}
                json.dump(stats, f, indent=4)
            with open(os.path.join(args.output_dir, "vm_bytecode.txt"), 'w') as f:
                f.write(bytecode_easyread(exec.bytecode))
        code, lib = exec.save()

    lib.export_library(lib_output_path)
    with open(code_output_path, "wb") as fo:
        fo.write(code)
else:
    print("Load cached model...")
    lib = tvm.runtime.load_module(lib_output_path)
    code = bytearray(open(code_output_path, "rb").read())
    exec = tvm.runtime.vm.Executable.load_exec(code, lib)

# run
batch_size = 64
seq_len = 128
word_table_size = 256

input_id_data = np.random.randint(word_table_size, size=(batch_size, seq_len), dtype=np.int32)
input_mask_data = np.ones_like(input_id_data)
token_type_ids_data = np.zeros_like(input_id_data)

def relay_run(inputs, exec_, dev_):
    if args.mt_ver:
        inputs = list(map(lambda i: tvm.nd.array(i, ctx=dev_), inputs))
    else:
        inputs = list(map(lambda i: tvm.nd.array(i, device=dev_), inputs))

    des_vm = tvm.runtime.vm.VirtualMachine(exec_, dev_)
    
    # warmup
    _ = des_vm.run(*inputs)

    comp_time = 0
    if args.nvprof:
        cu_prof_start()
    for _ in range(args.num_trials):
        s_time = time()
        res = des_vm.run(*inputs)
        comp_time += time() - s_time
    comp_time /= args.num_trials

    if args.mt_ver:
        res_np = res.asnumpy()
    else:
        res_np = res.numpy()
    return res_np, comp_time

def tf_run(inputs):
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    
    input_ids_node = graph.get_tensor_by_name("input_ids:0")
    input_mask_node = graph.get_tensor_by_name("input_mask:0")
    token_type_ids_node = graph.get_tensor_by_name("token_type_ids:0")
    output_node = graph.get_tensor_by_name("output_node:0")

    with graph.as_default():
        with graph.device("/device:GPU:0"):
            with tf.Session() as sess:
                # warmup
                _ = sess.run(output_node,
                    feed_dict={input_ids_node: inputs[0],
                               input_mask_node: inputs[1],
                               token_type_ids_node: inputs[2]})
                if args.nvprof:
                    cu_prof_start()
                comp_time = 0
                out = None
                for _ in range(args.num_trials):
                    s_time = time()
                    out = sess.run(output_node,
                        feed_dict={input_ids_node: inputs[0],
                                input_mask_node: inputs[1],
                                token_type_ids_node: inputs[2]})
                    comp_time += time() - s_time
                comp_time /= args.num_trials

    return out, comp_time

relay_res, relay_time = relay_run([input_id_data, input_mask_data, token_type_ids_data], exec, dev)
tf_res, tf_time = tf_run([input_id_data, input_mask_data, token_type_ids_data])

err = (np.abs(relay_res - tf_res) / tf_res).mean() * 100
# err, tf_time = 0, 0

print("Err rate: {:.4f} (%), num_trials: {}, tf time: {:.4f} (s), relay time: {:.4f} (s)".format(err, args.num_trials, tf_time, relay_time))
