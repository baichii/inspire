"""
@创建日期 ：2022/5/10
@修改日期 ：2022/5/11
@作者 ：jzj
@功能 ：executor for flappy bird rgb
"""

import tensorflow as tf
from tensorflow.keras import models, layers
from veteran.trainer import Learner
from veteran.models import register_model


class FlappyBirdRGBA2C(models.Model):
    """fixme: 暂不支持自定义参数"""
    def __init__(self, actor_dim=2, critic_dim=1):
        super(FlappyBirdRGBA2C, self).__init__()
        self.conv1 = layers.Conv2D(16, 8, 4, activation="relu")
        self.conv2 = layers.Conv2D(32, 4, 2, activation="relu")
        self.conv3 = layers.Conv2D(64, 3, 2, padding="same", activation="relu")
        self.fc = layers.Dense(256)
        self.actor = layers.Dense(actor_dim)
        self.critic = layers.Dense(critic_dim)

    def call(self, inputs):
        batch_size = list(inputs.shape[:-3])
        batch_size[0] = -1
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = tf.reshape(x, (*batch_size, 1600))
        x = self.fc(x)
        return {"policy": self.actor(x), "value": self.critic(x)}


register_model(id="flappy_bird_rgb_a2c", obj=FlappyBirdRGBA2C)


def main():
    args = {
        "env_args": {"id": "FlappyBird-rgb-C-v0"},
        "model_id": "flappy_bird_rgb_a2c",
        "model_args": {},
        "train_args": {
            "min_episodes": 16,
            "max_episodes": 2000,
            "gamma": 0.99,
            "lmb": 0.97,
            "clip_ratio": 0.2,
            "learning_rate": 0.0003,
            "batch_size": 16,
            "forward_steps": 64,
            "decay": 0.00001,
            "epochs": 10,
            "policy_target": "TD",
            "entropy_coeff": 0.001,
            "advantage_standardize": True,
            "burn_in_step": 0  # for RNN
        },
        "eval_args": {
            "eval_interval": 1,
            "num_simulation": 1,
            "workdir": "./workdir"}
    }
    learner = Learner(args)
    learner.run()


if __name__ == '__main__':
    main()

