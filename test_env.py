#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:46:43 2019

@author: gryang
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

import gym
import neurogym


def test_run(env_name):
    kwargs = {'dt': 100}
    env = gym.make(env_name, **kwargs)
    env.reset()
    for stp in range(100):
        action = env.action_space.sample()
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        if done:
            env.reset()


def test_run_all():
    from neurogym import all_tasks
    for env_name in all_tasks.keys():
        try:
            test_run(env_name)
            print('Success at running env: {:s}'.format(env_name))
        except BaseException as e:
            print('Failure at running env: {:s}'.format(env_name))
            print(e)


def test_plot(env_name):
    kwargs = {'dt': 100}
    env = gym.make(env_name, **kwargs)

    env.reset()
    observations = []
    for stp in range(100):
        if np.mod(stp, 2) == 0:
            action = 0
        else:
            action = 0
        state, rew, done, info = env.step(action)  # env.action_space.sample())
        observations.append(state)

        # print(state)
        # print(info)
        # print(rew)
        # print(info)
    obs = np.array(observations)
    plt.figure()
    plt.imshow(obs.T, aspect='auto')
    plt.show()


if __name__ == '__main__':
    test_run_all()