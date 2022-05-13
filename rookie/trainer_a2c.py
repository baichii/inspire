"""
@创建日期 ：2022/5/4
@修改日期 ：2022/5/4
@作者 ：jzj
@功能 ：trainer for a2c
       fixme: replaybuffer bug
"""

import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
import tensorflow.keras as keras
from rookie.env import make_env
from rookie.evaluator import evaluation
from rookie.generator import GeneratorA2C
from rookie.models import ModelWrapper, make_model


def compute_loss(action_probs, values, returns):
    """计算actor-critic loss"""
    advantage = returns - values
    action_probs = tf.clip_by_value(action_probs, 1e-10, 1.)
    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    # huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    # critic_loss = huber_loss(values, returns)
    critic_loss = 0.5 * tf.reduce_sum(tf.pow(advantage, 2))
    return actor_loss + critic_loss


def make_batch(experiences):
    observations = np.asarray([experience["observation"] for experience in experiences])
    actions = np.asarray([experience["action"] for experience in experiences])
    returns = np.asarray([experience["return"] for experience in experiences])

    actions = np.expand_dims(actions, 1)
    returns = np.expand_dims(returns, 1)
    return {"observation": observations, "action": actions, "return": returns}


class SamplerV1:
    """采样模块，支持episode"""
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


class SamplerV2:
    """ 仅使用当前episode的所有数据"""
    def __init__(self, episodes, args):
        self.episodes = episodes
        self.args = args
        self.selector = self._selector()

    def batch(self):
        experience = next(self.selector)
        return make_batch(experience)

    def _selector(self):
        while True:
            yield self.episodes

    def is_ok(self):
        return True


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args

        if not args["use_replay_buffer"]:
            self.episodes = []
            self.sampler = SamplerV2(self.episodes, args)
        else:
            self.episodes = deque(maxlen=args["max_episodes"])
            self.sampler = SamplerV1(self.episodes, args)

        self.optimizer = keras.optimizers.Adam(learning_rate=args["learning_rate"])
        self.steps = 0

    def train(self):
        if self.sampler.is_ok():
            batch = self.sampler.batch()
            with tf.GradientTape() as tape:
                outputs = self.model(batch["observation"])
                policy, values = outputs["policy"], outputs["value"]
                prob = tf.math.softmax(policy, axis=-1)
                selected_prob = tf.gather(prob, batch["action"], axis=-1, batch_dims=1)
                loss = compute_loss(selected_prob, values, batch["return"])
            variables = self.model.trainable_variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return loss
        else:
            return tf.constant(0.)


class Learner:
    """
    包含：数据采集 -> 采样 -> 训练 -> 控制（日志，回调）
    参数格式
    参考trainer
    """
    def __init__(self, args):
        self.args = args
        self.env = make_env(**args["env_args"])
        self.model = make_model(args["model_id"], args["model_args"])
        self.wrapper_model = ModelWrapper(self.model)

        # 数据生成
        self.generator = GeneratorA2C(self.env, self.args["train_args"])

        # 训练
        self.trainer = Trainer(self.wrapper_model, args["train_args"])
        self.best_reward = 0

    def feed_episode(self, episode):
        """v2"""
        if not self.args["train_args"]["use_replay_buffer"]:
            self.trainer.episodes.clear()
            self.trainer.episodes.extend(episode["moments"])
        else:
            self.trainer.episodes.extend(episode["moments"])

    def run(self):
        n_step = 0
        while n_step < self.args["train_args"]["num_step"]:
            episode = self.generator.execute(self.wrapper_model)
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
    """cartpole a2c"""
    args = {
        "env_args": {"id": "CartPole-v0"},
        "mode": "a2c",
        "model_id": "cartpole_a2c",
        "model_args": {},
        "train_args": {
            "use_replay_buffer": True,
            "min_episodes": 512,
            "max_episodes": 10000,
            "batch_size": 512,
            "learning_rate": 0.01,
            "decay": 0.00001,
            "num_step": 1000,
            "gamma": 0.99,
        },
        "eval": {
            "num_simulation": 10,
            "save_dir": "./models",
        }
    }
    learner = Learner(args)
    learner.run()


def demo2():
    """flappybird simple a2c"""
    args = {
        "env_args": {"id": "FlappyBird-v0"},
        "mode": "a2c",
        "model_id": "flappybirdsimple_a2c",
        "model_args": {},
        "train_args": {
            "use_replay_buffer": True,
            "min_episodes": 512,
            "max_episodes": 10000,
            "batch_size": 512,
            "learning_rate": 0.01,
            "decay": 0.00001,
            "num_step": 50000,
            "gamma": 0.99,
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
