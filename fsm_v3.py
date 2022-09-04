# Script to test a pre-trained model
# Written by Matthew Yee-King
# MIT license 
# https://mit-license.org/

import sys
import os
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
import random
import time 

# env_name = "gym_gs:BreakwallNoFrameskip-v1" 
# model_file = "./pre-trained/mac_hard_breakwall/gym_gs:BreakwallNoFrameskip-v1_20211018-114642_5424"

env_name = "SpaceInvadersNoFrameskip-v4"
model_file = "openaigym/pre-trained/aero_atari_spaceinvaders/SpaceInvadersNoFrameskip-v4_20220627-112743_13"

def create_env(env_name, seed=42):
    try:
        # Use the Baseline Atari environment because of Deepmind helper functions
        env = make_atari(env_name)
        # Warp the frames, grey scale, stake four frame and scale to smaller ratio
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        print("Loaded gym")
        env.seed(seed)
        return env
    except:
        print("Failed to make gym env", env_name)
        return None

def run_sim(env, frame_count):
    state = np.array(env.reset())
    total_reward = 0
    action = 4
    for i in range(frame_count):
        env.render('None')
        # Take best action
        if i % 40 == 0:
            if action == 4:
                action = 5
            else:
                action  = 4
        # action = 4
        # action = keras.backend.argmax(action_values[0]).numpy()
        state, reward, done, _ = env.step(action)
        state =  np.array(state)
        total_reward += reward
        if done:
            print("Game over at frame", i, "rew", total_reward)
            env.reset()
            #break
        #time.sleep(0.1)
    print("Sim ended : rew is ", total_reward)
    tries.append(total_reward)
    total_reward = 0


def main(env_name,frame_count=1000, seed=0):
    env = create_env(env_name=env_name, seed=seed)
    assert env is not None, "Failed to make env " + env_name
    # model = create_q_model(num_actions=env.action_space.n)
    # model_testfile = model_file + ".data-00000-of-00001"
    # assert os.path.exists(model_testfile), "Failed to load model: " + model_testfile
    # print("Model weights look loadable", model_testfile)
    # model.load_weights(model_file)
    # print("Model loaded weights - starting sim")
    print("Starting sim")
    run_sim(env, frame_count)


# main(env_name, frame_count=1000)
tries = []
for i in range(100):
    main(env_name, frame_count=1000, seed=i)
print(sum(tries)/len(tries))
