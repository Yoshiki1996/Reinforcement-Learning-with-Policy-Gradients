#!/usr/bin/env python3

import gym
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import *

import sys

env = gym.make('CartPole-v0')

RNG_SEED = 1
tf.set_random_seed(RNG_SEED)
env.seed(RNG_SEED)

alpha = 0.001
gamma = 0.99

w_init = xavier_initializer(uniform=False)
b_init = tf.constant_initializer(0.1)

try:
    output_units = env.action_space.shape[0]
except AttributeError:
    output_units = env.action_space.n

input_shape = env.observation_space.shape[0]
NUM_INPUT_FEATURES = 4
x = tf.placeholder(tf.float32, shape=(None, NUM_INPUT_FEATURES), name='x')
y = tf.placeholder(tf.float32, shape=(None, output_units), name='y')

out = fully_connected(inputs=x,
                      num_outputs=output_units,
                      activation_fn=tf.nn.softmax,
                      weights_initializer=w_init,
                      weights_regularizer=None,
                      biases_initializer=b_init,
                      scope='fc')

all_vars = tf.global_variables()

pi = tf.contrib.distributions.Bernoulli(p=out, name='pi')
pi_sample = pi.sample()
log_pi = pi.log_prob(y, name='log_pi')

Returns = tf.placeholder(tf.float32, name='Returns')
optimizer = tf.train.GradientDescentOptimizer(alpha)
train_op = optimizer.minimize(-1.0 * Returns * log_pi)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

MEMORY = 25
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

track_returns = []
for ep in range(16384):  #16384
    obs = env.reset()

    G = 0
    ep_states = []
    ep_actions = []
    ep_rewards = [0]
    done = False
    t = 0
    I = 1
    while not done:
        ep_states.append(obs)
        env.render()
        action = sess.run([pi_sample], feed_dict={x: [obs]})[0][0]
        # print(sess.run([pi_sample], feed_dict={x: [obs]}))
        # print(action)
        ep_actions.append(action)
        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward * I)
        G += reward * I
        I *= gamma

        t += 1
        if t >= MAX_STEPS:
            break

    returns = np.array([G - np.cumsum(ep_rewards[:-1])]).T
    index = ep % MEMORY

    _ = sess.run([train_op], feed_dict={x: np.array(ep_states),
                                        y: np.array(ep_actions),
                                        Returns: returns})

    track_returns.append(G)
    track_returns = track_returns[-MEMORY:]
    mean_return = np.mean(track_returns)
    print("Episode {} finished after {} steps with return {}".format(ep, t, G))
    print("Mean return over the last {} episodes is {}".format(MEMORY, mean_return))

    # with tf.variable_scope('fc', reuse=True): print(sess.run(tf.get_variable("weights")))
    # with tf.variable_scope("mus", reuse=True):
    #     print("incoming weights for the mu's from the first hidden unit:", sess.run(tf.get_variable("weights"))[0,:])


sess.close()
