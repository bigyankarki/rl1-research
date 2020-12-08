from environments.base_environment import BaseEnvironment, OneHotEnv
import numpy as np

class RiverSwim(OneHotEnv):
    """Implements the environment for an RLGlue environment
    Note:
        env_init, env_start, env_step, env_cleanup, and env_message are required
        methods.
    """

    def __init__(self):
        super().__init__()

        self.num_states = 6
        self.num_actions = 2

    def env_init(self, env_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
        """

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions, self.num_states))

        for s in range(self.num_states):

            if s == 0:
                self.P[s, 0, s] = 1
                self.P[s, 1, s] = 0.1
                self.P[s, 1, s + 1] = 0.9
                self.R[s, 0, s] = 5. / 1000.
            elif s == self.num_states - 1:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.05
                self.P[s, 1, s] = 0.95
                self.R[s, 1, s] = 1
            else:
                self.P[s, 0, s - 1] = 1
                self.P[s, 1, s - 1] = 0.05
                self.P[s, 1, s] = 0.05
                self.P[s, 1, s + 1] = 0.9

        self.random_seed = env_info.get('random_seed', 22)
        self.rand_generator = np.random.RandomState(self.random_seed)

        self.start_state = self.rand_generator.choice(self.num_states)
        self.reward_obs_term = [0.0, None, False]