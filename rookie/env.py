"""
@创建日期 ：2022/4/25
@修改日期 ：2022/4/25
@作者 ：jzj
@功能 ：环境生成
"""

import gym
import flappy_bird_gym


def make_env(id, **kwargs):
    env = gym.make(id, **kwargs)
    return env
