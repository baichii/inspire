"""
@创建日期 ：2022/4/25
@修改日期 ：2022/4/26
@作者 ：jzj
@功能 ：模型库，输出统一以字典格式
       dqn 输出 value
       a2c 输出 policy value
       fixme: 可能会抽象为参数构建的模式，不确定
"""


from typing import List
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models


def make_model(id, args):
    if id == "cartpole_dqn":
        return CartPoleDQN(**args)
    elif id == "cartpole_a2c":
        return CartPoleA2C(**args)
    elif id == "flappybirdsimple_dqn":
        return FlappyBirdSimpleDqn(**args)
    elif id == "flappybirdsimple_a2c":
        return FlappyBirdSimpleA2C(**args)
    else:
        raise NotImplementedError

class ModelWrapper(models.Model):
    """fixme: dev ing"""
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
        return outputs


# CartPole DQN
class CartPoleDQN(models.Model):
    def __init__(self, action_dim, hidden_dims: List):
        super(CartPoleDQN, self).__init__()
        self.input_layers = layers.InputLayer(input_shape=(4,))
        self.hidden_layers = []
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(layers.Dense(hidden_dim, activation="tanh"))
        self.output_layer = layers.Dense(action_dim)

    def call(self, inputs):
        x = self.input_layers(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return {"value": x}


# CartPole A2C
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


# FlappyBirdRGB A2C
class ConvBlock(layers.Layer):
    def __init__(self, filter, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = layers.Conv2D(filter, kernel_size, stride, padding="same")
        self.bn = layers.BatchNormalization()
        self.activation = layers.ReLU()

    def call(self, inputs):
        return self.activation(self.bn(self.conv(inputs)))


class ResidualBlock(layers.Layer):
    def __init__(self, filter, kernel_size, stride, squeeze_factor, se=False):
        """fixme: 添加Se支持"""
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvBlock(filter//squeeze_factor, kernel_size, stride)
        self.conv_block2 = ConvBlock(filter, kernel_size, stride)
        self.short_cut = ConvBlock(filter, 1)
        self.output_bn = layers.BatchNormalization()
        self.output_ac = layers.ReLU()

    def call(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = x + self.short_cut(inputs)
        x = self.output_ac(self.output_bn(x))
        return x


class PolicyHead(layers.Layer):
    def __init__(self, policy_dim):
        super(PolicyHead, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=3, strides=1, padding="same")
        self.bn = layers.BatchNormalization()
        self.dense = layers.Dense(policy_dim)

    def call(self, inputs):
        b, h, w, c = inputs.shape
        x = self.bn(self.conv(inputs))
        x = tf.reshape(x, (-1, h*w))
        x = self.dense(x)
        return x


class ValueHead(layers.Layer):
    def __init__(self, value_dim):
        super(ValueHead, self).__init__()
        self.conv = layers.Conv2D(1, kernel_size=3, strides=1, padding="same")
        self.bn = layers.BatchNormalization()
        self.dense = layers.Dense(value_dim)

    def call(self, inputs):
        b, h, w, c = inputs.shape
        x = self.bn(self.conv(inputs))
        x = tf.reshape(x, (-1, h*w))
        x = self.dense(x)
        return x


class FlappyBirdA2C(models.Model):
    """
    简单模型，注意policy输出的是logit值为非概率
    """
    def __init__(self, filters=[32, 64, 128], blocks=[2, 2, 4]):
        super(FlappyBirdA2C, self).__init__()
        self.conv1 = layers.Conv2D(32, 5, 2, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.ReLU()
        self.pool1 = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")

        self.middle_layers = []

        for filter, block in zip(filters, blocks):
            for n in range(block):
                self.middle_layers.append(ResidualBlock(filter, 3, 1, 4))
            self.middle_layers.append(layers.MaxPooling2D(pool_size=3, strides=2, padding="same"))

        self.policy_head = PolicyHead(policy_dim=2)
        self.value_head = ValueHead(value_dim=1)

    def call(self, inputs):
        x = self.pool1(self.ac1(self.bn1(self.conv1(inputs))))
        for layer in self.middle_layers[:-1]:
            x = layer(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return {"policy": policy, "value": value}


# FlappyBirdSimple A2C
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


# FlappyBirdSimple DQN
class FlappyBirdSimpleDqn(models.Model):
    def __init__(self, value_dim=2, hidden_dims=[256, 256]):
        super(FlappyBirdSimpleDqn, self).__init__()
        self.input_layers = layers.InputLayer(input_shape=(2,))
        self.hidden_layers = []
        for hidden_dim in hidden_dims:
            self.hidden_layers.append(layers.Dense(hidden_dim, activation="tanh"))

        self.value_head = layers.Dense(value_dim)

    def call(self, inputs):
        x = self.input_layers(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        value = self.value_head(x)
        return {"value": value}
