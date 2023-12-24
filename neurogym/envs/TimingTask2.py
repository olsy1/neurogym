import numpy as np
import neurogym as ngym
from neurogym import spaces


class TimingTask(ngym.TrialEnv):
    def __init__(self, dt=100, rewards=None, timing=None, sigma=1):
        super().__init__(dt=dt)
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards: correct timing, incorrect timing, no action
        self.rewards = {'correct': +1., 'fail': 0., 'no_action': 0.}
        if rewards:
            self.rewards.update(rewards)

        # Timing for the different stages of the task
        self.timing = {
            'stimulus': 6000,  # Maximum duration of the stimulus period
            'decision': 100}  # Decision period
        if timing:
            self.timing.update(timing)

        # Define observations and action spaces
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)  # Two possible actions

    def _new_trial(self, **kwargs):
        trial = {
            'type': self.rng.choice([1, 2]),  # Type of trial (1 or 2)
            'correct_action_timing': 3000 if trial['type'] == 1 else 6000  # Correct timing for action
        }
        trial.update(kwargs)

        self.add_period('stimulus', duration=self.timing['stimulus'])
        self.add_period('decision', duration=self.timing['decision'])

        # Set the stimulus for the trial type
        stim = np.zeros(self.observation_space.shape)
        stim[trial['type'] - 1] = 1
        self.add_ob(stim, 'stimulus')

        self.set_groundtruth(trial['correct_action_timing'], 'decision')

        return trial

    def _step(self, action):
        new_trial = False
        reward = 0
        current_time = self.t - self.start_t['decision']
        gt = self.gt_now

        if self.in_period('decision'):
            new_trial = True
            if current_time >= gt and action != 0:  # Correct timing and action
                reward = self.rewards['correct']
            elif current_time < gt and action != 0:  # Action taken too early
                reward = self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    # Instantiate and test the task
    env = TimingTask()
    trial = env.new_trial()
    print('Trial info', trial)
    env.reset()
    total_reward = 0
    for t in range(trial['stimulus'] + trial['decision']):
        ob, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward
        if info['new_trial']:
            break
    print('Total reward obtained:', total_reward)