import gym_super_mario_bros as bros
import os
import torch.cuda
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers import GrayScaleObservation as GSO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from matplotlib import pyplot as plt

checkpoint_dir = "./train/"
logs_dir = "./logs/"

# Create game environment
env = bros.make('SuperMarioBros-v0')
# Simplifies the controls
env = JoypadSpace(env, SIMPLE_MOVEMENT)
# Greyscale to reduce processing requirements allowing for faster performance
env = GSO(env, True)
# Wrap with the dummy env
env = DummyVecEnv([lambda: env])
# Stack the frames, so we have "memory" so to speak
env = VecFrameStack(env, 5, channels_order='last')

# For saving progress
class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

callback = TrainAndLoggingCallback(check_freq=25000, save_path=checkpoint_dir)

# The actual AI model
#model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=logs_dir, learning_rate=0.000001, n_steps=1024)

model = PPO.load('./train/best_model_100000.zip', env=env, tensorboard_log=logs_dir, learning_rate=0.000001, n_steps=1024, print_system_info=True)
#model.learn(total_timesteps=1000000, callback=callback)


state = env.reset()

while True:
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)
    env.render()
'''#Create flag, restart or not
done = True

# Loop through each frame in the game
for step in range(100000):
    if done:
        # start the game
        env.reset()
    # do random actions
    state, reward, done, info = env.step(env.action_space.sample())
    # show the game
    env.render()
env.close()'''