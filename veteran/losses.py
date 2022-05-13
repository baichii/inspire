"""
@创建日期 ：2022/5/6
@修改日期 ：2022/5/9
@作者 ：jzj
@功能 ：获取target value和advantage
       注意，新格式下，数据由[batch_size, xxx] 扩展为 [batch_size, time_steps, xxx]
"""

import tensorflow as tf
from collections import deque


def monte_carlo(values, returns, *args):
    return returns, returns - values


def temporal_difference(values, returns, rewards, lmb, gamma):
    """td lambda"""
    target_values = deque([returns[:, -1]])
    for i in range(values.shape[1] - 2, -1, -1):
        reward = rewards[:, i] if rewards is not None else 0
        value = values[:, i+1]
        target_values.appendleft(reward + gamma * ((1 - lmb) * value) + lmb * target_values[0])
    target_values = tf.stack(tuple(target_values), axis=1)
    return target_values, target_values - values


def upgo(values, returns, rewards, lmb, gamma):
    """up go, fixme: cartpole无法收敛"""
    target_values = deque([returns[:, -1]])
    for i in range(values.shape[1] - 2, -1, -1):
        reward = rewards[:, i] if rewards is not None else 0
        value = values[:, i+1]
        target_values.append(reward + gamma * tf.math.maximum(value, (1-lmb) * value + lmb * target_values[0]))
    target_values = tf.stack(tuple(target_values), axis=1)
    return target_values, target_values - values


def vtrace():
    """fixme: read paper, vtrace和impala绑定,sad"""
    pass


def compute_target(algorithm, values, returns, rewards, lmb, gamma):
    """默认基线为mc returns"""
    if values is None:
        return returns, returns
    elif algorithm == "MC":
        return monte_carlo(values, returns)
    elif algorithm == "TD":
        return temporal_difference(values, returns, rewards, lmb, gamma)
    elif algorithm == "UPGO":
        return upgo(values, returns, rewards, lmb, gamma)
    else:
        raise NotImplementedError(f"Not support {algorithm}")
