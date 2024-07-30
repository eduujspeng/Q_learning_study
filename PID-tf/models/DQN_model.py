from tensorflow.keras import models, layers, optimizers  # type: ignore # 从TensorFlow导入相关模块

def build_model(state_size, action_size):  # 定义构建模型的函数
    model = models.Sequential()  # 创建一个顺序模型
    model.add(layers.Dense(32, input_dim=state_size, activation='relu'))  # 添加第一层全连接层，激活函数为ReLU
    model.add(layers.Dense(32, activation='relu'))  # 添加第二层全连接层，激活函数为ReLU
    model.add(layers.Dense(action_size, activation=None))  # 添加输出层，激活函数为线性函数
    model.compile(loss='mse', optimizer=optimizers.Adam(lr=0.001))  # 编译模型，损失函数为均方误差，优化器为Adam
    return model  # 返回构建的模型


if __name__ == '__main__':
    model = build_model(3, 101)

    import numpy as np
    state = np.array([50, 50, 50])
    state = state[np.newaxis]
    print(state, state.shape)

    action_np = model(state).numpy()
    print(action_np.shape)
    print(np.argmax(action_np[0]))
