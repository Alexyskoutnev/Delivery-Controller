from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo



if __name__ == "__main__":
    config = {"epochs": 10000, "lr": 1e-3, 'potholes': 0, 'traffic_jams': 0
            , 'map_size': (5, 5), 'render_mode': None, 'epoch_print_cnt': 500}
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
    input_size, output_size = env.observation_dim, env.action_dim
    ray.init()
    breakpoint()
    algo = ppo.PPO(env=env, config={"env_config": {}, })
