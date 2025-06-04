import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register


class HeteroHighDimMECEnvironmentv3(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_users=10,
        max_bw=20,
        max_power=10,
        max_comp=100,
        max_steps=10,
        k_factor=2.0,           
        min_distance=5.0,        
        max_distance=10.0,      
        path_loss_exp=2.0        
    ):
        super().__init__()
        self.num_users = num_users
        self.max_bw = max_bw
        self.max_power = max_power
        self.max_comp = max_comp
        self.max_steps = max_steps

        # Rician channel parameter
        self.k_factor = k_factor

        # Log distance path loss parameters
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.path_loss_exp = path_loss_exp

        
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_users,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_users * 5,),
            dtype=np.float32
        )

        # reward weight matrix definition
        self.alpha = np.array(
            [[1.0, 0.1, 0.1, 0.1, 0.2, 0.2, 1.0]
             for _ in range(self.num_users)]
        )
        self.step_count = 0

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

         # ================== Reset your state here ================== ==============

        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        type_onehot = np.eye(3)[self.user_type]
        # stack features
        feats = np.stack([
            self.user_demand,
            self.latency_req,
            self.channel_quality,
            self.user_pos,
            self.user_compute
        ], axis=1)
        obs = np.concatenate([feats, type_onehot], axis=1).flatten()
        return obs.astype(np.float32)

    def step(self, action):

        action = (action.reshape(self.num_users, 5) + 1.0) / 2.0


        # ================== Define your reward here ================== ============
        reward = np.random.uniform(0.1, 1)

        # update mobility
        self.user_pos = np.clip(
            self.user_pos + np.random.normal(0.0, 0.01, size=self.num_users),
            0.0, 1.0
        )

        self.step_count += 1
        done = self.step_count >= self.max_steps

        obs = self._get_obs()
        return obs, reward, done, False, info

    def render(self, mode="human"):
        print(f"[Step {self.step_count}]")


# register the environment
register(
    id="HeteroHighDimMEC-v3",
    entry_point="envs.v3:HeteroHighDimMECEnvironmentv3"
)


if __name__ == "__main__":
    env = HeteroHighDimMECEnvironmentv3()
    obs, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        env.render()
    # print("Episode finished:", info)

