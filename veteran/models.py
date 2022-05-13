"""
@创建日期 ：2022/5/6
@修改日期 ：2022/5/6
@作者 ：jzj
@功能 ：from rookie，添加数据转换和对hidden的支持
"""

import tensorflow as tf
from tensorflow.keras import models, layers
from veteran.utils import map_r


class ModelWrapperV2(models.Model):
    """fixme: dev版本，支持hidden，一些数据存储格式暂不明确"""
    def __init__(self, model):
        super(ModelWrapperV2, self).__init__()
        self.model = model

    def init_hidden(self, batch_size=None):
        if hasattr(self.model, "init_hidden"):
            if batch_size is None:
                hidden = self.model.init_hidden([])
                return hidden
            else:
                return self.model.init_hidden(batch_size)
        return None

    def call(self, *args, **kwargs):
        return self.model(*args, *kwargs)

    def inference(self, x, hidden, **kwargs):
        """fixme: inference需要添加格式转换 """
        if hasattr(self.model, "inference"):
            return self.model.inference(x, hidden)

        x = tf.expand_dims(x, 0)
        outputs = self.call(x, hidden, **kwargs)
        return outputs


class ModelWrapper(models.Model):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def call(self, inputs):
        return self.model(inputs)

    def inference(self, x):
        if hasattr(self.model, "inference"):
            return self.model.inference(x)

        x = tf.expand_dims(x, 0)
        outputs = self.call(x)
        return map_r(outputs, lambda o: o.numpy().squeeze(0))


class CartPoleA2C(tf.keras.Model):
    def __init__(self, num_action=2, num_hidden_units=128):
        super(CartPoleA2C, self).__init__()
        self.common = layers.Dense(num_hidden_units, activation=None)
        self.activation = layers.ReLU()
        self.actor = layers.Dense(num_action)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor):
        x = self.common(inputs)
        x = self.activation(x)
        return {"policy": self.actor(x), "value": self.critic(x)}



class FlappyBirdSimpleA2C(models.Model):
    def __init__(self, policy_dim=2, value_dim=1, hidden_dims=[32, 64]):
        super(FlappyBirdSimpleA2C, self).__init__()
        self.input_layers = layers.InputLayer(input_shape=(2,))
        self.hidden_layers = []
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(layers.Dense(hidden_dim, activation="tanh",))

        self.policy_head = layers.Dense(policy_dim)
        self.value_head = layers.Dense(value_dim)

    def call(self, inputs):
        x = self.input_layers(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return {"policy": policy, "value": value}


RegisterCache = {"cartpole_a2c": CartPoleA2C,
                 "flappybirdsimple_a2c": FlappyBirdSimpleA2C}


def make_model(id, args):
    model_obj = RegisterCache.get(id, None)
    if model_obj:
        return model_obj(**args)
    else:
        raise NotImplementedError("model id: {} not found in register cache".format(id))


def register_model(id, obj):
    RegisterCache[id] = obj
