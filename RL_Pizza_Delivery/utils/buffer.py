
import collections
import numpy as np

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states, dtype=np.float32), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states, dtype=np.float32)

class PPOBuffer(object):
    def __init__(self) -> None:
        pass

    # def add(self, )

if __name__ == "__main__":
    buffer =  ExperienceBuffer(10)
    sample_experience = ([1], 1, 1, [2], False)
    buffer.append(sample_experience)
    batch_size = 1