"""
@创建日期 ：2022/4/25
@修改日期 ：2022/4/25
@作者 ：jzj
@功能 ：环境生成
"""

import cv2
import gym
import numpy as np
import flappy_bird_gym
from flappy_bird_gym import FlappyBirdEnvRGB, FlappyBirdEnvSimple

RegisterCache = {}


def register_env(id, obj):
    RegisterCache[id] = obj


def make_env(id, **kwargs):
    obj = RegisterCache.get(id, None)
    if obj:
        return obj(**kwargs)
    else:
        return gym.make(id, **kwargs)


class FlappyBirdEnvSimple_C(FlappyBirdEnvSimple):
    """修改奖励"""
    def __init__(self, **kwargs):
        super(FlappyBirdEnvSimple_C, self).__init__(**kwargs)
        self.score_temp = 0

    def reset(self):
        self.score_temp = 0
        obs = super(FlappyBirdEnvSimple_C, self).reset()
        return obs

    def step(self, action):
        obs, reward, done, info = super(FlappyBirdEnvSimple_C, self).step(action)
        return obs, self._process_reward(done, info), done, info

    def _process_reward(self, done, info):
        if not done:
            if self.score_temp == info["score"]:
                reward_ = 0.01
            else:
                reward_ = 1.
                self.score_temp += 1
        else:
            reward_ = -1
        return reward_


class FlappyBirdEnvRGB_C(FlappyBirdEnvRGB):
    """自定义环境，修改奖励、添加图像预处理"""
    def __init__(self, size=(84, 84), **kwargs):
        self.size = size
        super(FlappyBirdEnvRGB_C, self).__init__(**kwargs)
        self.score_temp = 0

    def reset(self):
        self.score_temp = 0
        obs = super(FlappyBirdEnvRGB_C, self).reset()
        return self._process_obs(obs)

    def step(self, action):
        obs, reward, done, info = super(FlappyBirdEnvRGB_C, self).step(action)
        return self._process_obs(obs), self._process_reward(done, info), done, info

    def _process_reward(self, done, info):
        if not done:
            if self.score_temp == info["score"]:
                reward_ = 0.01
            else:
                reward_ = 1.
                self.score_temp += 1
        else:
            reward_ = -1
        return reward_

    def _process_obs(self, image):
        image = cv2.resize(image, self.size)
        image = np.rot90(np.fliplr(image))
        image = image / 255.
        return image


register_env("FlappyBird-C-v0", FlappyBirdEnvSimple_C)
register_env("FlappyBird-rgb-C-v0", FlappyBirdEnvRGB_C)
