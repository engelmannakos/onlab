import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import DQN

env = gym.make('navi_env:navi-env-v0')

train = True
new_train = True
model_name = "distance_matters"
steps = 10000

class TensorboardCallback(BaseCallback):

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

if train:
    if new_train:
        model = DQN(
            "MultiInputPolicy",
            env,
            learning_rate = 0.0003,
            buffer_size=1024,
            learning_starts = 0,
            target_update_interval=3000,
            exploration_final_eps=0,
            exploration_fraction=0.95,
            verbose=1,
            tensorboard_log="F:/navi_dqn/tensor/manhattan_end")

        model.learn(
            total_timesteps=steps,
            callback = TensorboardCallback(),
            log_interval=4,
            tb_log_name=model_name)

        model.save(model_name)


    else:
        model = DQN.load(model_name)
        model.set_env(env)

        model.learn(
            total_timesteps=steps,
            callback = TensorboardCallback(),
            log_interval=4,
            tb_log_name=model_name)

        model.save(model_name)

else:
    model = DQN.load(model_name)
    model.set_env(env)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()
