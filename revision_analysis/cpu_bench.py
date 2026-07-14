import os, sys, time, json, gc
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
PROJ = 'd:/PhD/Experiments/classify_polyp'
sys.path.insert(0, PROJ)
os.chdir(PROJ)
import numpy as np
import tensorflow as tf
from finetune.models import FT_REGISTRY
from finetune.config_finetune import CHECKPOINTS_DIR
from finetune.train_finetune import _load_weights_by_name
from PIL import Image
import json as _json
SP = 'C:/Users/yrdde/AppData/Local/Temp/claude/d--PhD-Experiments-classify-polyp/d5982003-bfbd-41f4-9d83-acfeabbb7c31/scratchpad'

def get_flops(model, size):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    conc = tf.function(lambda x: model(x)).get_concrete_function(tf.TensorSpec([1, size, size, 3], tf.float32))
    (frozen, graph_def) = convert_variables_to_constants_v2_as_graph(conc)
    with tf.Graph().as_default() as g:
        tf.graph_util.import_graph_def(graph_def, name='')
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        opts['output'] = 'none'
        flops = tf.compat.v1.profiler.profile(g, options=opts)
    return flops.total_float_ops

def bench(model_key, rich, size, ckpt):
    (model, _) = FT_REGISTRY[model_key](image_size=size, rich_head=rich, backbone_weights='imagenet')
    model.trainable = True
    try:
        model.load_weights(ckpt)
    except Exception:
        _load_weights_by_name(model, ckpt)
    params = model.count_params()
    try:
        flops = get_flops(model, size)
    except Exception as e:
        flops = None
        print('flops err', str(e)[:80])
    x1 = tf.random.normal([1, size, size, 3])
    for _ in range(5):
        model(x1, training=False)
    N = 60
    t = []
    for _ in range(N):
        s = time.perf_counter()
        model(x1, training=False)
        t.append(time.perf_counter() - s)
    lat_ms = float(np.median(t) * 1000)
    x32 = tf.random.normal([32, size, size, 3])
    for _ in range(3):
        model(x32, training=False)
    s = time.perf_counter()
    for _ in range(10):
        model(x32, training=False)
    thr = 10 * 32 / (time.perf_counter() - s)
    import glob
    imgs = glob.glob('data/Polyp/*.jpg')[:40]
    tp = []
    for p in imgs:
        s = time.perf_counter()
        im = Image.open(p).convert('RGB').resize((size, size), Image.BICUBIC)
        arr = np.asarray(im, np.float32) / 255.0
        tp.append(time.perf_counter() - s)
    prep_ms = float(np.median(tp) * 1000)
    size_mb = os.path.getsize(ckpt) / 1000000.0
    r = dict(model=model_key, rich_head=rich, input=size, params=int(params), flops_G=round(flops / 1000000000.0, 3) if flops else None, cpu_latency_ms_bs1=round(lat_ms, 2), cpu_throughput_img_s_bs32=round(thr, 1), preprocess_ms=round(prep_ms, 3), ckpt_size_mb=round(size_mb, 2))
    print(_json.dumps(r), flush=True)
    tf.keras.backend.clear_session()
    gc.collect()
    return r
res = []
res.append(bench('efficientnet_b1', False, 240, os.path.join(CHECKPOINTS_DIR, 'efficientnet_b1_full_run1_best.h5')))
res.append(bench('mobilenetv3_small', True, 256, os.path.join(CHECKPOINTS_DIR, 'mobilenetv3_small_full_run1_best.h5')))
json.dump(res, open(os.path.join(SP, 'cpu_bench.json'), 'w'), indent=2)
print('SAVED cpu_bench.json', flush=True)
