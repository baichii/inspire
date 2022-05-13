"""
@创建日期 ：2022/4/25
@修改日期 ：2022/5/8
@作者 ：jzj
@功能 ：veteran ac基线generator,from rookie.generator
       在wrapper之外，移除所有对tf的依赖，以ndarray，list为核心
       fixme: 只有部分模块支持rnn的burn_in_steps
"""

import random

import numpy as np

from veteran.utils import softmax, to_tensor


class Generator:
    """for actor-critic，取消了对returns的标准化"""
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def generate(self, model, args):
        moments = []
        keys = ["observation", "selected_prob", "action", "value", "reward", "return"]
        observation = self.env.reset()
        done = False
        accumulated_reward = 0

        hidden = {}  # todo: 支持RNN
        legal_actions = list(range(self.env.action_space.n))  # fixme: 兼容问题，当前环境不返回行动list

        while not done:
            moment = {key: None for key in keys}
            moment["observation"] = observation

            outputs = model.inference(observation)
            prob = softmax(outputs["policy"])
            action = random.choices(legal_actions, weights=prob)[0]
            observation, reward, done, info = self.env.step(action)
            accumulated_reward += reward

            moment["selected_prob"] = prob[action]
            moment["action"] = action
            moment["value"] = outputs["value"][0]
            moment["reward"] = reward
            moments.append(moment)

        ret = 0
        rets = []
        for i, m in reversed(list(enumerate(moments))):
            ret = m["reward"] + self.args["gamma"] * ret
            rets.append(ret)
            moments[i]["return"] = ret

        # fixme: 标准化，移动到sampler
        if False:
            rets = ((rets - np.mean(rets)) / (np.std(rets) + np.finfo(np.float32).eps.item()).tolist())[::-1]
            for i, m in reversed(list(enumerate(moments))):
                moments[i]["return"] = rets[i]

        episode = {"args": args, "steps": len(moments), "reward": accumulated_reward, "moment": moments}
        return episode

    def execute(self, model, args):
        return self.generate(model, args)


class Sampler:
    """采样模块，以episode为存储核心，采样基于episode和forward step进行"""
    def __init__(self, episodes, args):
        self.episodes = episodes
        self.args = args
        self.selector_iter = self._selector()

    def batch(self):
        episodes = next(self.selector_iter)
        batch = make_batch(episodes, self.args)
        return batch

    def _selector(self):
        while True:
            yield [self.select_episode() for _ in range(self.args["batch_size"])]

    def select_episode(self):
        while True:
            ep_idx = random.randrange(len(self.episodes))
            # 设置接受率，一个丐版优先采样
            accept_rate = 1 - (len(self.episodes)-1 - ep_idx) / self.args["max_episodes"]
            if random.random() < accept_rate:
                break
        ep = self.episodes[ep_idx]
        turn_candidates = 1 + max(0, ep["steps"] - self.args["forward_steps"])
        train_start = random.randrange(turn_candidates)
        st = max(0, train_start - self.args["burn_in_step"])
        ed = min(train_start + self.args["forward_steps"], ep["steps"])
        ep_minimum = {
            "args": ep["args"], "moment": ep["moment"][st:ed],
            "train_start": train_start, "start": st, "end": ed,  "total": ep["steps"]
        }
        return ep_minimum

    def is_ok(self):
        if len(self.episodes) < self.args["min_episodes"]:
            return False
        return True


def make_batch(episodes, args):
    """
    todo: 对RNN的支持，使用burn_in_steps以及额外参数来对潜在缺失的历史帧进行pad
    return 格式为tf.tensor float32，堆叠格式为[b,t,xxx]
    """
    observations, datum = [], []

    for ep in episodes:
        moments = ep["moment"]
        obs = np.asarray([m["observation"] for m in moments], dtype="float32")
        prob = np.asarray([[m["selected_prob"]] for m in moments], dtype="float32")
        action = np.asarray([[m["action"]] for m in moments], dtype="int64")
        value = np.asarray([[m["value"]] for m in moments], dtype="float32")
        reward = np.asarray([[m["reward"]] for m in moments], dtype="float32")
        ret = np.asarray([[m["return"]] for m in moments], dtype="float32")

        # 如果采样序列长度小于forward_steps，填充
        if len(ep["moment"]) < args["forward_steps"]:
            pad_len = args["forward_steps"] - (ep["end"] - ep["train_start"])
            obs = np.pad(obs, pad_width=[(0, pad_len)] + [(0, 0)] * (len(obs.shape)-1), mode="constant", constant_values=0)
            prob = np.pad(prob, pad_width=((0, pad_len), (0, 0)), mode="constant", constant_values=1)
            action = np.pad(action, pad_width=((0, pad_len), (0, 0)), mode="constant", constant_values=0)
            value = np.pad(value, pad_width=((0, pad_len), (0, 0)), mode="constant", constant_values=0)
            reward = np.pad(reward, pad_width=((0, pad_len), (0, 0)), mode="constant", constant_values=0)
            ret = np.pad(ret, pad_width=((0, pad_len), (0, 0)), mode="constant", constant_values=0)

        observations.append(obs)
        datum.append((prob, action, value, reward, ret))

    observations = to_tensor(np.asarray(observations))
    prob, action, value, reward, ret = [to_tensor(np.asarray(val)) for val in zip(*datum)]

    return {
        "observation": observations,
        "selected_prob": prob,
        "action": action,
        "value": value,
        "reward": reward,
        "return": ret}


