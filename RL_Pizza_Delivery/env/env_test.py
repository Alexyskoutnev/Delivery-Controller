import numpy as np
import torch
from tensorboardX import SummaryWriter
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer

class ENV(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size
        self.action_size = 4
        self.action_dim = 4
        self.grid = np.zeros((grid_size, grid_size))
        self.start = np.array([0, 0], dtype=np.float32)
        self.observation_dim = 2
        self.state = self.start
        self.goal = (grid_size -1, grid_size - 1)
        self.obstacles = [(2, 2),]
        self.obstacles = []
    
    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        start = np.array([0,0])
        self.state = start
        for obstacle in self.obstacles:
            self.grid[obstacle[0]][obstacle[1]] = -1
        return np.array(start, dtype=np.float32)

    def step(self, action):
        next_state = self.state

        if action == 0 and self.state[0] > 0:
            next_state[0] -= 1  # Move up
        elif action == 1 and self.state[0] < self.grid_size - 1:
            next_state[0] += 1  # Move down
        elif action == 2 and self.state[1] > 0:
            next_state[1] -= 1  # Move left
        elif action == 3 and self.state[1] < self.grid_size - 1:
            next_state[1] += 1  # Move right

        reward = -1  # Move cost
        if tuple(next_state) in self.obstacles:
            reward = -50  # Penalty for hitting an obstacle
            next_state = self.state

        done = False
        if tuple(next_state) == self.goal:
            reward = self.grid_size * 15  # Goal reached
            done = True
        
        self.state = next_state

        return next_state, reward, done, {}

if __name__ == "__main__":
    # Training parameters
    grid_size = 5
    state_size = grid_size * grid_size
    action_size = 4
    learning_rate = 0.0001
    gamma = 0.99
    epsilon_decay = 0.95
    epsilon_min = 0.01
    episodes = 1000
    max_moves = grid_size * 10
    GAMMA = 0.99
    BATCH_SIZE = 32
    REPLAY_SIZE = 10000
    LEARNING_RATE = 1e-4
    SYNC_TARGET_FRAMES = 1000
    REPLAY_START_SIZE = 10000

    EPSILON_DECAY_LAST_FRAME = 150000
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.01

    env = ENV(grid_size)
    input_size, output_size = env.observation_dim, env.action_dim
    # agent = QAgent(input_size, output_size, buffer, lr=learning_rate, gamma=gamma)
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = QAgent(input_size, output_size, buffer, lr=learning_rate, gamma=gamma)

    frame_idx = 1

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                      frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), m_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_m_reward is None or best_m_reward < m_reward:
                torch.save(net.state_dict(), args.env +
                           "-best_%.0f.dat" % m_reward)
                if best_m_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_m_reward, m_reward))
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())



    # for episode in range(episodes):
    #     state = env.reset()
    #     total_reward = 0.0
    #     moves = 0
    #     epsilon = 0.2
    #     while moves < max_moves:
    #         epsilon = max(epsilon_min, epsilon * epsilon_decay)
    #         # print(f"[{moves}], Epsilon [{epsilon}]")
    #         action = agent.select_action(torch.tensor(state, dtype=torch.float32), epsilon)
    #         next_state, reward, done, _ = env.step(action)
    #         agent.update(torch.tensor(state, dtype=torch.float32), action, reward, torch.tensor(next_state, dtype=torch.float32), done)
    #         state = next_state
    #         total_reward += reward
    #         moves += 1
    #         # print(f"state : [{state}], action : [{action}], next state: [{next_state}], reward: [{reward}]")
    #         if done:
    #             break
    #     print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
    # # Test the trained agent
    # state = env.reset()
    # path = [state.tolist()]
    # agent.eval = True
    # moves = 0
    # eval_reward = 0
    # while moves < max_moves:
    #     action = agent.select_action(torch.tensor(state, dtype=torch.float32))
    #     next_state, reward, done, _ = env.step(action)
    #     path.append(next_state.tolist())
    #     state = next_state
    #     moves += 1
    #     eval_reward += reward
    #     if done:
    #         break

    print("Optimal Path:", path)
    print(f"Eval reward: {eval_reward}")
    