"""
@创建日期 ：2022/4/25
@修改日期 ：2022/5/5
@作者 ：jzj
@功能 ：数据生成(No hidden, No player), 生成一个episode的数据，输出一个字典
       注意，不同算法需要输出的keys不一样， 但提取逻辑相同(如果有)
       比如dqn需要next-observation, ac需要values，通过参数调度吧
"""

import numpy as np
import tensorflow as tf


class GeneratorDQN:
    """更新return的方法都集成到generator中"""
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.epsilon = args["epsilon"]
        self.epsilon_decay = args["epsilon_decay"]

    def generate(self, model, args):
        moments = []
        keys = ["observation", "value", "action", "reward", "value_next", "return"]

        state = self.env.reset()
        rewards = 0
        done = False
        while not done:
            moment = {key: None for key in keys}
            prev_state = state
            outputs = model.inference(prev_state)
            value = outputs["value"]
            action = np.argmax(value)
            if np.random.random() < self.epsilon:
                action = np.random.choice([0, 1])
                self.epsilon *= self.epsilon_decay
            state, reward, done, _ = self.env.step(action)

            outputs_next = model.inference(state)
            value_next = outputs_next["value"][0, np.argmax(outputs["value"])].numpy()
            rewards += reward
            if done:
                reward = -10
                value_next = 0
            moment["observation"] = prev_state
            moment["action"] = action
            moment["reward"] = reward
            moment["value_next"] = value_next
            moments.append(moment)

        for i, m in reversed(list(enumerate(moments))):
            discounted_return = m["reward"] + self.args["gamma"] * m["value_next"]
            moments[i]["return"] = discounted_return

        episode = {"args": args, "step": len(moments), "rewards": rewards, "moments": moments}
        return episode

    def execute(self, model, args):
        return self.generate(model, args)


class GeneratorA2C:
    """for actor-critic"""
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def generate(self, model, args):
        moments = []
        keys = ["observation", "selected_prob", "action", "value", "reward", "return"]
        state = self.env.reset()
        done = False

        while not done:
            moment = {key: None for key in keys}
            prev_state = state
            outputs = model.inference(state)
            policy, value = outputs["policy"], outputs["value"]

            action = tf.random.categorical(policy, 1)[0, 0]
            prob = tf.math.softmax(policy)

            state, reward, done, _ = self.env.step(action.numpy())
            moment["observation"] = prev_state
            moment["selected_prob"] = prob[0][action]
            moment["action"] = action
            moment["value"] = value[0]
            moment["reward"] = reward
            moments.append(moment)

        ret = 0
        rets = []
        for i, m in reversed(list(enumerate(moments))):
            ret = m["reward"] + self.args["gamma"] * ret
            rets.append(ret)
            moments[i]["return"] = ret

        # fixme: 标准化
        if False:
            rets = ((rets - np.mean(rets)) / (np.std(rets) + np.finfo(np.float32).eps.item()).tolist())[::-1]
            for i, m in reversed(list(enumerate(moments))):
                moments[i]["return"] = rets[i]

        episode = {"step": len(moments), "moments": moments}
        return episode

    def execute(self, model, args=None):
        return self.generate(model, args)
