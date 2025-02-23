import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
import time
import cv2
from gym.wrappers import TimeLimit

# ===========================
# 1️⃣ 配置超参数
# ===========================
GAMMA = 0.99  # 折扣因子
LR = 1e-4  # 学习率
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 30000  # 让 ε 下降更慢，防止过早收敛
BATCH_SIZE = 64
MEMORY_SIZE = 50000
TARGET_UPDATE = 1000
MAX_STEPS = 5000
NUM_EPISODES = 1000
SAVE_PATH = './train.model'

# ===========================
# 2️⃣ 自定义 Frame Skip
# ===========================
class CustomFrameSkip(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

# ===========================
# 3️⃣ 创建环境
# ===========================
def make_env(episode):
    env = gym.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomFrameSkip(env, skip=4)
    env = TimeLimit(env, max_episode_steps=2000)
    env.metadata['render_fps'] = 9999  # 解除 FPS 限制
    return env

# ===========================
# 4️⃣ Q-Learning算法
# ===========================
class QLearning:
    def __init__(self, action_space, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=30000, learning_rate=LR, gamma=GAMMA):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.action_space = action_space
        self.epsilon = epsilon_start  # 在初始化时根据epsilon_start初始化epsilon
        self.q_table = np.zeros((84, 84, 4, action_space))  # 状态空间为(84, 84, 4)，动作空间大小为action_space
        self.steps_done = 0

    def select_action(self, state):
        # 将 state 转换为整数类型
        state_idx = tuple(np.floor(state * 10).astype(int))  # 将每个维度的状态值乘以10并转为整数
        # 确保通道维度索引不超过3
        state_idx = (state_idx[0], state_idx[1], np.clip(state_idx[2], 0, 3))  # 使第三维度的索引不超过3
        if random.random() < self.epsilon:
            # 随机选择动作，确保动作在有效范围内
            action = random.choice(range(self.action_space))  # 动作范围应该是从 0 到 action_space - 1
        else:
            # 使用Q表选择动作
            action = np.argmax(self.q_table[state_idx[0], state_idx[1], state_idx[2]])

        # 确保动作在合法范围内
        action = np.clip(action, 0, self.action_space - 1)  # 确保动作在有效范围内
        return action

    def update_q_table(self, state, action, reward, next_state, done):
        # 离散化处理状态（确保索引为整数）
        state_idx = tuple(np.floor(state * 10).astype(int))  # 将每个维度的状态值乘以10并转为整数
        next_state_idx = tuple(np.floor(next_state * 10).astype(int))  # 同样处理next_state

        # 使用np.clip确保索引不会超出 q_table 的维度范围
        state_idx = tuple(np.clip(i, 0, self.q_table.shape[dim] - 1) for dim, i in enumerate(state_idx))
        next_state_idx = tuple(np.clip(i, 0, self.q_table.shape[dim] - 1) for dim, i in enumerate(next_state_idx))

        # 选择下一状态的最佳动作，并确保它在有效的动作空间范围内
        best_next_action = np.argmax(self.q_table[next_state_idx[0], next_state_idx[1], next_state_idx[2]])
        best_next_action = np.clip(best_next_action, 0, self.q_table.shape[3] - 1)  # 确保动作索引在有效范围内

        # 更新Q表
        td_target = reward + self.gamma * self.q_table[next_state_idx[0], next_state_idx[1], next_state_idx[2], best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state_idx[0], state_idx[1], state_idx[2], action]
        self.q_table[state_idx[0], state_idx[1], state_idx[2], action] += self.learning_rate * td_error

        # 逐步减少epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon_start - self.steps_done / self.epsilon_decay)




# ===========================
# 5️⃣ 训练部分
# ===========================
def train():
    env = make_env(0)
    agent = QLearning(action_space=env.action_space.n)  # 创建Q-learning代理

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False
        frames = deque(maxlen=4)

        for _ in range(4):
            frames.append(preprocess(state))
        state = np.stack(frames, axis=0)

        for step in range(1, MAX_STEPS + 1):
            env.render()  # 显示游戏界面
            action = agent.select_action(state)  # 选择动作
            next_state, reward, done, info = env.step(action)
            reward = compute_reward(info, step)  # 计算奖励

            frames.append(preprocess(next_state))
            next_state = np.stack(frames, axis=0)
            agent.update_q_table(state, action, reward, next_state, done)  # 更新Q表
            state = next_state
            total_reward += reward

            if done:
                break

        agent.steps_done += 1  # 递增步骤
        print(f"Episode {episode}/{NUM_EPISODES} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.4f}")

    env.close()  # 关闭环境，释放资源



# ===========================
# 6️⃣ 奖励函数（鼓励 AI 通过关卡）
# ===========================
def compute_reward(info, step):
    reward = 0

    # 根据玩家的前进距离奖励（鼓励玩家前进）
    distance_reward = (info['x_pos'] - info.get('prev_x_pos', 0)) * 0.1
    info['prev_x_pos'] = info['x_pos']  # 更新当前位置
    reward += distance_reward

    # 对跳跃行为给予奖励（防止 AI 停留在原地）
    if info.get('jumping', False):  # 假设通过 'jumping' 标志可以确认跳跃
        reward += 0.2  # 每次跳跃奖励一定值

    # 对通过关卡给予奖励
    if info.get('flag_get', False):  # 如果通过关卡（触摸旗帜）
        reward += 1000  # 获得过关奖励

    # 对掉入陷阱惩罚
    if info.get('life_lost', False):  # 如果失去生命（掉入陷阱或被敌人击中）
        reward -= 500  # 失去生命的惩罚

    # 对避免死亡给予奖励
    if info.get('enemy_nearby', False):  # 假设有标志判断敌人是否靠近
        reward -= 100  # 如果敌人靠近惩罚，以鼓励躲避敌人

    reward -= 0.02  # 小的惩罚，鼓励代理尽快行动

    return reward





# ===========================
# 7️⃣ 图像预处理
# ===========================
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return np.array(obs, dtype=np.float32) / 255.0

# ===========================
# 8️⃣ 运行训练
# ===========================
if __name__ == '__main__':
    train()
