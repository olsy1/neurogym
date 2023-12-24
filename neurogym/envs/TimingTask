import neurogym as ngym
from neurogym import spaces
import numpy as np

class TimingTask(ngym.TrialEnv):
    """
    The agent must wait for a specific duration before performing an action 
    to receive a reward. Two different inputs signal different waiting periods.
    """
    def __init__(self, dt=100, rewards=None, timing=None):
        super().__init__(dt=dt)

        self.rewards = {'incorrect': -0.1, 'correct': +1.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'input1': 1500,  # 1.5 seconds in milliseconds
            'input2': 3000,  # 3 seconds in milliseconds
        }
        if timing:
            self.timing.update(timing)

        # Observation space: two inputs
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,),
                                            dtype=np.float32)

        # Action space: 0 (no action), 1 (perform action)
        self.action_space = spaces.Discrete(2)

    def _new_trial(self, **kwargs):
        # Randomly choose between input 1 and input 2
        input_type = self.rng.choice(['input1', 'input2'])
        trial = {'input_type': input_type}

        self.add_period('input')
        self.add_period('wait', duration=self.timing[input_type])
        self.add_period('reward', duration=self.timing[input_type] * 2)

        # Set observation based on input type
        ob = np.zeros(self.observation_space.shape)
        ob[0 if input_type == 'input1' else 1] = 1
        self.set_ob(ob, 'input')

        return trial

    def _step(self, action):
        new_trial = False
        reward = 0

        # Check if action is performed during the reward period
        if self.in_period('reward') and action == 1:
            # Correct action
            reward = self.rewards['correct']
            new_trial = True
        elif self.in_period('wait') and action == 1:
            # Incorrect action
            reward = self.rewards['incorrect']
            new_trial = True

        return self.ob_now, reward, new_trial, {'new_trial': new_trial}