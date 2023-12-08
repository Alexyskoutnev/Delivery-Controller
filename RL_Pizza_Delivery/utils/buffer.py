
import collections
import numpy as np

Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward',
                               'done', 'new_state'])

class ExperienceBuffer:
    """
    A replay buffer to store and sample experiences for training a reinforcement learning agent.

    Args:
        capacity (int): The maximum capacity of the buffer.

    Attributes:
        buffer (collections.deque): A deque data structure to store experiences.
    """
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    
    def __len__(self):
        """
        Get the current number of experiences in the buffer.

        Returns:
            int: Number of experiences in the buffer.
        """
        return len(self.buffer)
    
    def append(self, experience):
        """
        Append a new experience to the buffer.

        Args:
            experience (Experience): The experience to be added to the buffer.
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: States of the sampled experiences.
                - np.ndarray: Actions taken in the sampled experiences.
                - np.ndarray: Rewards received in the sampled experiences.
                - np.ndarray: 'Done' flags indicating the end of episodes.
                - np.ndarray: Next states of the sampled experiences.
        """
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
            zip(*[self.buffer[idx] for idx in indices])
        return np.array(states, dtype=np.float32), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states, dtype=np.float32)