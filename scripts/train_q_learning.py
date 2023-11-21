import torch

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.algo.qlearning import QAgent

def train(env, agent, config):
    for epoch in range(config['epochs']):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.select_action(torch.tensor(state, dtype=torch.float32).view(1, -1))
            next_state, reward, done, _ = env.step(action)
            agent.update(
                torch.tensor(state, dtype=torch.float32).view(1, -1),
                action,
                reward,
                torch.tensor(next_state, dtype=torch.float32).view(1, -1),
                done
            )
            state = next_state
            total_reward += reward
            
            if config['render_mode'] == 'human':
                pass

        print(f"[{epoch}] : {total_reward}")
        if epoch % config['epoch_print_cnt'] == 0 and epoch != 0:
            print(f"Episode {epoch}, Total reward: {total_reward}")

if __name__ == "__main__":
    config = {"epochs": 10000, "lr": 1e-3, 'potholes': 0, 'traffic_jams': 0
            , 'map_size': (5, 5), 'render_mode': None, 'epoch_print_cnt': 500}
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
    input_size, output_size = env.observation_dim, env.action_dim
    agent = QAgent(input_size, output_size)
    train(env, agent, config)
