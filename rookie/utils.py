"""
@创建日期 ：2022/4/26
@修改日期 ：2022/4/26
@作者 ：jzj
@功能 ：
"""

import gym
import numpy as np
import tensorflow as tf
from gym import wrappers
import flappy_bird_gym


def make_video(env, model, save_path):
    env = wrappers.Monitor(env, save_path, force=True)
    rewards = 0
    steps = 0
    done = False
    state = env.reset()
    while not done:
        outputs = model(np.atleast_2d(state))
        action = np.argmax(outputs["action_value"])
        state, reward, done, _ = env.step(action)
        rewards += reward
        steps += 1
    print(rewards)


def state_standard(state):
    """旋转 + 标准化"""
    # state = np.rot90(state)[::-1]
    state = state.astype("float32") / 128 - 1
    return state


def make_video(env, model, save_path):
    env = gym.wrappers.Monitor(env, save_path, force=True)
    rewards = 0
    steps = 0
    done = False
    state = env.reset()
    while not done:
        state_normal = state_standard(state)
        outputs = model.inference(state_normal)
        action = np.argmax(outputs["policy"])
        state, reward, done, info = env.step(action)
        rewards += reward
        steps += 1
    return rewards, info

def demo():
    env = gym.make("CartPole-v0")
    model = tf.keras.models.load_model("/Users/jiangzhenjie/Desktop/models")
    save_path = "/Users/jiangzhenjie/Desktop/gp/videos"
    make_video(env, model, save_path)


def count_parameters(model):
    total_parameters = 0
    for variable in model.trainable_variables:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters
    print(total_parameters)


if __name__ == '__main__':
    make_video()
