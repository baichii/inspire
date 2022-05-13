"""
@创建日期 ：2022/5/8
@修改日期 ：2022/5/8
@作者 ：jzj
@功能 ：
"""

import numpy as np
import tensorflow as tf


def to_tensor(arr):
    """默认 float32"""
    return tf.convert_to_tensor(arr)


def softmax(x):
    x = np.exp((x - np.max(x, axis=-1, keepdims=True)))
    return x / x.sum(axis=-1, keepdims=True)


def map_r(x, callback_fn=None):
    if isinstance(x, (list, tuple, set)):
        return type(x)(map_r(xx, callback_fn) for xx in x)
    elif isinstance(x, dict):
        return type(x)((key, map_r(xx, callback_fn))for key, xx in x.items())
    return callback_fn(x) if callback_fn is not None else None


def cprofiler(func, key_word="tottime", save_path=None):
    """性能测试, fixme: 源生方法保存的为二进制文件"""
    import pstats
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    func()
    profiler.disable()
    if key_word:
        stats = pstats.Stats(profiler).sort_stats(key_word)
    stats.print_stats()
    stats.dump_stats(save_path)
