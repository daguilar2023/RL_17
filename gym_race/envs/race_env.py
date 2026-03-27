import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D

class RaceEnv(gym.Env):
    metadata = {'render_modes' : ['human'], 'render_fps' : 30}
    def __init__(
        self,
        render_mode="human",
        observation_mode="discrete",
        action_mode="classic",
        reward_mode="sparse",
        race_mode=0,
    ):
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.reward_mode = reward_mode
        self.race_mode = race_mode

        self.action_space = spaces.Discrete(4 if self.action_mode == "extended" else 3)
        if self.observation_mode == "continuous":
            self.observation_space = spaces.Box(
                low=np.zeros(5, dtype=np.float32),
                high=np.ones(5, dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                np.array([0, 0, 0, 0, 0]),
                np.array([10, 10, 10, 10, 10]),
                dtype=int,
            )
        self.is_view = True
        self.pyrace = PyRace2D(
            self.is_view,
            mode=self.race_mode,
            observation_mode=self.observation_mode,
            action_mode=self.action_mode,
            reward_mode=self.reward_mode,
        )
        self.memory = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        del self.pyrace
        self.is_view = True
        self.msgs=[]
        self.pyrace = PyRace2D(
            self.is_view,
            mode=self.race_mode,
            observation_mode=self.observation_mode,
            action_mode=self.action_mode,
            reward_mode=self.reward_mode,
        )
        obs = self.pyrace.observe()
        return np.asarray(obs, dtype=self.observation_space.dtype), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe()
        return np.asarray(obs, dtype=self.observation_space.dtype), reward, done, False, {'dist':self.pyrace.car.distance, 'check':self.pyrace.car.current_check, 'crash': not self.pyrace.car.is_alive}

    # def render(self, close=False , msgs=[], **kwargs): # gymnasium.render() does not accept other keyword arguments
    def render(self): # gymnasium.render() does not accept other keyword arguments
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
