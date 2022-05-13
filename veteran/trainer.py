"""
@创建日期 ：2022/5/8
@修改日期 ：2022/5/9
@作者 ：jzj
@功能 ：a2c ppo baseline from rookie.trainer_a2c
       1、添加ppo作为基线算法
       2、重写采样策略
       3、为了在未来支持RNN，数据格式由[batch, obs_im] -> [batch, time_step, obs_dim]
       4、模型封装又learner ->trainer
       5、不在支持非replay buffer场景
"""

import os
import psutil
import datetime
import numpy as np
from collections import deque
import tensorflow as tf
import tensorflow.keras as keras
from veteran.env import make_env
from veteran.evaluator import evaluation
from veteran.losses import compute_target
from veteran.generator import Generator, Sampler
from veteran.models import make_model, ModelWrapper


def compute_loss(batch, model, args, eps=1e-10):
    """ppo version"""
    outputs = forward_prediction(model, batch, args)
    target_values, target_advantages = compute_target(args["policy_target"], batch["value"], batch["return"],
                                                      batch["reward"], args["lmb"], args["gamma"])
    log_selected_policies_cur = tf.gather(tf.math.log_softmax(outputs["policy"]),  batch["action"], axis=-1, batch_dims=-1)
    log_selected_policies_old = tf.math.log(tf.clip_by_value(batch["selected_prob"], eps, 1.0))

    rhos = tf.exp(log_selected_policies_cur - log_selected_policies_old)
    min_advantage = tf.where(target_advantages > 0,
                             (1 + args["clip_ratio"]) * target_advantages,
                             (1 - args["clip_ratio"]) * target_advantages)
    actor_loss = -tf.reduce_mean(tf.math.minimum(rhos * target_advantages, min_advantage))
    critic_loss = 0.5 * tf.reduce_mean(tf.pow(outputs["value"] - batch["return"], 2))

    return {"actor": actor_loss,  "critic": critic_loss, "total": actor_loss + critic_loss}


def compute_loss_v2(batch, model, args, eps=1e-10):
    """
    ppo version new
    1、添加 target advantages 标准化
    2、参考ppo baseline 修改min_advantage的计算逻辑
    """
    outputs = forward_prediction(model, batch, args)
    target_values, target_advantages = compute_target(args["policy_target"], batch["value"], batch["return"],
                                                      batch["reward"], args["lmb"], args["gamma"])
    target_advantages = (target_advantages - np.mean(target_advantages)) / np.std(target_advantages)
    dist_crossentropy = -tf.reduce_mean(tf.math.log_softmax(outputs["policy"]) * tf.math.softmax(outputs["policy"]))

    log_selected_policies_cur = tf.gather(tf.math.log_softmax(outputs["policy"]),  batch["action"], axis=-1, batch_dims=-1)
    log_selected_policies_old = tf.math.log(tf.clip_by_value(batch["selected_prob"], eps, 1.0))

    rhos = tf.exp(log_selected_policies_cur - log_selected_policies_old)

    min_advantage = tf.clip_by_value(rhos, 1 - args["clip_ratio"], 1 + args["clip_ratio"]) * target_advantages

    actor_loss = -tf.reduce_mean(tf.math.minimum(rhos * target_advantages, min_advantage))
    critic_loss = 0.5 * tf.reduce_mean(tf.pow(outputs["value"] - batch["return"], 2))
    total_loss = actor_loss + critic_loss - dist_crossentropy * args["entropy_coeff"]

    return {"actor": actor_loss,  "critic": critic_loss, "total": total_loss}


def forward_prediction(model, batch, args):
    """
    将前向推理从trainer中抽象出来，
    fixme: 1、统一trainer和evaluator的前向 2、支持hidden
    """
    observations = batch["observation"]
    outputs = model(observations)
    return outputs


class Trainer:
    def __init__(self, model, args):
        self.episodes = deque(maxlen=args["max_episodes"])
        self.model = model
        self.args = args
        self.default_lr = 1e-5
        self.data_cnt = self.args["batch_size"] * self.args["forward_steps"]
        lr = args.get("learning_rate", None) or self.default_lr * self.data_cnt
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.sampler = Sampler(self.episodes, args)
        self.steps = 0
        self.wrapped_model = ModelWrapper(self.model)
        self.trained_model = self.wrapped_model

    def train(self):
        if self.sampler.is_ok():
            batch = self.sampler.batch()
            with tf.GradientTape() as tape:
                losses = compute_loss_v2(batch, self.trained_model, self.args)
            variables = self.trained_model.trainable_variables
            gradients = tape.gradient(losses["total"], variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
            return losses
        else:
            return {}


class Learner:
    """
    包含：数据采集 -> 采样 -> 训练 -> 控制（日志，回调）
    参数格式
    参考trainer
    """
    def __init__(self, args, net=None):
        self.args = args
        self.env = make_env(**args["env_args"])
        self.model = net if net else make_model(args["model_id"], args["model_args"])

        self.generator = Generator(self.env, self.args["train_args"])
        self.trainer = Trainer(self.model, args["train_args"])
        self.wrapped_model = ModelWrapper(self.model)
        self.best_reward = 0
        self.cur_step = 0

        # tensorboard
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(os.path.join(args["eval_args"]["workdir"], current_time))

    def feed_episode(self, episode):
        """添加内存占用保护"""
        men_limit = 95
        mem_percent = psutil.virtual_memory().percent
        mem_safe = mem_percent < men_limit
        max_episodes = self.args["train_args"]["max_episodes"] if mem_safe else int(len(self.trainer.episodes) * men_limit / mem_percent)
        while len(self.trainer.episodes) > max_episodes:
            self.trainer.episodes.popleft()
        self.trainer.episodes.append(episode)

    def model_path(self):
        return os.path.join(self.args["eval_args"]["workdir"], str(self.cur_step))

    def model_path_best(self):
        return os.path.join(self.args["eval_args"]["workdir"], "best")

    def evaluation(self):
        """
        模型评估，模型保存
        注意：保存的模型一律为原始模型，而非封装模型，在tf下尤其要注意区别
        """
        if self.cur_step % self.args["eval_args"]["eval_interval"] == 0:
            cur_reward = evaluation(self.wrapped_model, self.args, num_processes=1)
            with self.summary_writer.as_default():
                tf.summary.scalar("eval/reward", cur_reward, step=self.cur_step)
            if cur_reward > self.best_reward:
                self.best_reward = cur_reward
                self.model.save(self.model_path_best())
            self.model.save(self.model_path())


    def run(self):
        while self.cur_step < self.args["train_args"]["epochs"]:
            episode = self.generator.execute(self.wrapped_model, self.args["train_args"])
            self.feed_episode(episode)
            losses = self.trainer.train()
            self.cur_step += 1
            if self.cur_step % self.args["eval_args"]["eval_interval"] == 0:
                print(f"step: {self.cur_step}, loss: {losses.get('total', 0)}, reward: {episode['reward']:.4f}, "
                      f"steps:{episode['steps']}")

            with self.summary_writer.as_default():
                tf.summary.scalar("train/reward", episode["reward"], step=self.cur_step)
                tf.summary.scalar("train/steps", episode["steps"], step=self.cur_step)
                tf.summary.scalar("loss/total", losses.get("total", 0), step=self.cur_step)
                tf.summary.scalar("loss/actor", losses.get("actor", 0), step=self.cur_step)
                tf.summary.scalar("loss/critic", losses.get("critic", 0), step=self.cur_step)
            self.evaluation()


def demo():
    """
    flappybird a2c ppo baseline, 默认参数已收敛

    """
    args = {
        # "env_args": {"id": "CartPole-v0"},
        # "model_id": "cartpole_a2c",
        "env_args": {"id": "FlappyBird-C-v0"},
        "model_id": "flappybirdsimple_a2c",
        "model_args": {},
        "train_args": {
            "min_episodes": 32,
            "max_episodes": 2000,
            "gamma": 0.99,
            "lmb": 0.97,
            "clip_ratio": 0.2,
            "learning_rate": 0.0003,
            "batch_size": 32,
            "forward_steps": 128,
            "decay": 0.00001,
            "epochs": 1000,
            "policy_target": "TD",
            "entropy_coeff": 0.,
            "advantage_standardize": True,
            "burn_in_step": 0  # for RNN
        },
        "eval_args": {
            "eval_interval": 50,
            "num_simulation": 1,
            "workdir": "./workdir"}
    }
    learner = Learner(args)
    learner.run()


if __name__ == '__main__':
    demo()
