"""
@创建日期 ：2022/4/26
@修改日期 ：2022/4/26
@作者 ：jzj
@功能 ：目标模型运行n次，评估奖励，对于没有随机性的结果，1次就好
       fixme: tf.keras.models 多进程pickle问题
"""


import numpy as np
import multiprocessing as mp
from rookie.env import make_env


class Evaluator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def evaluate_dqn(self, model, args):
        state = self.env.reset()
        done = False
        accumulated_rewards = 0
        while not done:
            outputs = model.inference(state)
            action = np.argmax(outputs["value"])
            state, reward, done, _ = self.env.step(action)
            accumulated_rewards += reward
        return accumulated_rewards

    def evaluate_a2c(self, model, args):
        state = self.env.reset()
        done = False
        accumulated_rewards = 0
        while not done:
            outputs = model.inference(state)
            action = np.argmax(outputs["policy"])
            state, reward, done, _ = self.env.step(action)
            accumulated_rewards += reward
        return accumulated_rewards

    def execute(self, model, args):
        if self.args["mode"] == "dqn":
            return self.evaluate_dqn(model, args)
        elif self.args["mode"] == "a2c":
            return self.evaluate_a2c(model, args)


def evaluation(model, args, num_processes):
    """多进程评估"""
    env = make_env(**args["env_args"])
    evaluator = Evaluator(env, args)
    if args["eval"]["num_simulation"] == 1 or num_processes == 1:
        cache = []
        for _ in range(args["eval"]["num_simulation"]):
            env.reset()
            reward = evaluator.execute(model, args)
            cache.append(reward)
    else:
        raise TypeError("can't pickle weak ref objects")
        # cache = mp.Manager().list()
        # pool = mp.Pool(processes=num_processes)
        # res = pool.starmap(executor.execute, [(model, args) for _ in range(num_simulation)])
        # pool.close()
        # pool.join()

    reward = sum(cache)/len(cache)
    return reward


