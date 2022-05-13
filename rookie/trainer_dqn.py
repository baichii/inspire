"""
@创建日期 ：2022/4/26
@修改日期 ：2022/5/5
@作者 ：jzj
@功能 ：trainer for dqn
       dqn都支持 replaybuffer
       fixme: logger，模型命名
"""

import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
import tensorflow.keras as keras
from rookie.env import make_env
from rookie.evaluator import evaluation
from rookie.generator import GeneratorDQN
from rookie.models import make_model, ModelWrapper


def make_batch(experiences):
    observations = np.asarray([experience["observation"] for experience in experiences])
    actions = np.asarray([experience["action"] for experience in experiences])
    rewards = np.asarray([experience["reward"] for experience in experiences])
    values_next = np.asarray([experience["value_next"] for experience in experiences])
    returns = np.asarray([experience["return"] for experience in experiences])

    actions = np.expand_dims(actions, 1)
    rewards = np.expand_dims(rewards, 1)
    values_next = np.expand_dims(values_next, 1)
    returns = np.expand_dims(returns, 1)

    return {"observations": observations,
            "actions": actions,
            "rewards": rewards,
            "values_next": values_next,
            "returns": returns}


class Sampler:
    """采样模块"""
    def __init__(self, episodes, args):
        self.episodes = episodes
        self.args = args
        self.selector = self._selector()

    def batch(self):
        experience = next(self.selector)
        return make_batch(experience)

    def _selector(self):
        while True:
            yield random.sample(self.episodes, k=self.args["batch_size"])

    def is_ok(self):
        if len(self.episodes) < self.args["min_episodes"]:
            return False
        return True


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.episodes = deque(maxlen=args["max_episodes"])
        self.sampler = Sampler(self.episodes, args)

        self.optimizer = keras.optimizers.Adam(learning_rate=args["learning_rate"])
        self.steps = 0

    def train(self):
        if self.sampler.is_ok():
            batch = self.sampler.batch()
            with tf.GradientTape() as tape:
                outputs = self.model(batch["observations"])
                selected_value = tf.gather(outputs["value"], batch["actions"], axis=-1, batch_dims=1)
                loss = tf.math.reduce_mean(tf.square(batch["returns"] - selected_value))
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss
        else:
            return 0


class Learner:
    """
    包含：数据采集 -> 采样 -> 训练 -> 控制（日志，回调）
    """
    def __init__(self, args):
        self.args = args
        self.env = make_env(**args["env_args"])
        self.model = make_model(args["model_id"], args["model_args"])
        self.wrapper_model = ModelWrapper(self.model)

        # 数据生成
        self.generator = GeneratorDQN(self.env, self.args["train_args"])

        # 训练
        self.trainer = Trainer(self.wrapper_model, args["train_args"])
        self.best_reward = 0

    def feed_episode(self, episode):
        self.trainer.episodes.extend(episode["moments"])

    def run(self):
        n_step = 0
        while n_step < self.args["train_args"]["num_step"]:
            episode = self.generator.execute(self.wrapper_model, args=None)
            self.feed_episode(episode)
            loss = self.trainer.train()
            n_step += 1
            if n_step % 100 == 0:
                cur_reward = evaluation(self.wrapper_model, self.args, num_processes=1)
                print(f"n_step: {n_step}, loss: {loss:.8f}, num_simulation: {self.args['eval']['num_simulation']}, "
                      f"reward: {cur_reward:.2f}")
                if cur_reward > self.best_reward:
                    self.best_reward = cur_reward
                    self.wrapper_model.save(os.path.join(self.args["eval"]["save_dir"], "best_models"))
                self.wrapper_model.save(os.path.join(self.args["eval"]["save_dir"], "models"))


def demo1():
    """cartpole dqn"""
    args = {
        "env_args": {"id": "CartPole-v0"},
        "mode": "dqn",
        "model_id": "cartpole_dqn",
        "model_args": {"action_dim": 2, "hidden_dims": [128, 128]},
        "train_args": {
            "min_episodes": 512,
            "max_episodes": 10000,
            "batch_size": 512,
            "learning_rate": 0.01,
            "num_step": 10000,
            "gamma": 0.99,
            "epsilon": 0,
            "epsilon_decay": 0
        },
        "eval": {
            "num_simulation": 10,
            "save_dir": "./models",
        }
    }
    learner = Learner(args)
    learner.run()


def demo2():
    """flappybird simple dqn"""
    args = {
        "env_args": {"id": "FlappyBird-v0"},
        "mode": "dqn",
        "model_id": "flappybirdsimple_dqn",
        "model_args": {},
        "train_args": {
            "min_episodes": 512,
            "max_episodes": 20000,
            "batch_size": 512,
            "learning_rate": 0.001,
            "decay": 0.00001,
            "num_step": 50000,
            "gamma": 0.99,
            "epsilon": 0.3,
            "epsilon_decay": 0.9975
        },
        "eval": {
            "num_simulation": 10,
            "save_dir": "./models",
        }
    }
    learner = Learner(args)
    learner.run()


if __name__ == '__main__':
    demo1()
    # demo2()
