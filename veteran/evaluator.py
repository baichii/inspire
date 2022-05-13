"""
@创建日期 ：2022/5/10
@修改日期 ：2022/5/10
@作者 ：jzj
@功能 ：from rookie.evaluator
"""

import numpy as np
from veteran.env import make_env


class Evaluator:
    def __init__(self, env, args):
        self.env = env
        self.args = args

    def evaluate(self, model, args):
        state = self.env.reset()
        done = False
        accumulated_reward = 0
        actions = []
        while not done:
            outputs = model.inference(state)
            action = np.argmax(outputs["policy"])
            state, reward, done, info = self.env.step(action)
            accumulated_reward += reward
            actions.append(action)
        print("actions: ", actions)
        return accumulated_reward

    def execute(self, model, args):
        return self.evaluate(model, args)


def evaluation(model, args, num_processes):
    """单进程评估"""
    env = make_env(**args["env_args"])
    evaluator = Evaluator(env, args)
    if args["eval_args"]["num_simulation"] == 1 or num_processes == 1:
        cache = []
        for _ in range(args["eval_args"]["num_simulation"]):
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

    reward = sum(cache)/args["eval_args"]["num_simulation"]
    return reward
