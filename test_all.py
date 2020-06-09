import pandas as pd
import gym
import numpy as np
from argparse import ArgumentParser
import os

from stable_baselines.sac.policies import MlpPolicy as SACPolicy
from stable_baselines.ddpg.policies import MlpPolicy as  DDPGPolicy
from stable_baselines.td3.policies import MlpPolicy as TD3Policy
from stable_baselines.deepq.policies import MlpPolicy as DQNPolicy
from stable_baselines.common.policies import MlpPolicy

from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import SAC, PPO2, DDPG, TD3, DQN, TRPO, A2C, ACKTR, ACER

import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from data_utils import load_data, save_data

train_data, test_data = load_data()

total_timesteps = 2000
color = ['r', '#1b2966', 'g', '#74d3dc', 'y', '#7354f4', 'm', '#7e84f3', 'k','#fcc500', "#c0003b"]
cols = ["SAC", "DDPG", "TD3", "PPO2", "TRPO", "A2C", "ACKTR"]

def run(Model, Policy, gamma):
    env = gym.make('Stock-v0')
    env._init_data(train_data)

    if gamma != 0:
        n_actions = env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(gamma) * np.ones(n_actions))
        model = Model(Policy, env, verbose=1, action_noise=action_noise)
        model.gamma = 0.2
    else:
        model = Model(Policy, env, verbose=1)

    model.learn(total_timesteps=total_timesteps, log_interval=10)

    print("test model")
    env = gym.make('TestStock-v0')
    env._init_data(test_data)
    obs = env.reset()

    for _ in range(686):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)

    return env.asset_memory


# sac = run(SAC, SACPolicy, 0.5)
# ddpg = run(DDPG, DDPGPolicy, 0.5)
# td3 = run(TD3, TD3Policy, 0.1)
# ppo2 = run(PPO2, MlpPolicy, 0)
# trpo = run(TRPO, MlpPolicy, 0)
# a2c = run(A2C, MlpPolicy, 0)
# acktr = run(ACKTR, MlpPolicy, 0)

def run_multiple(name, Model, Policy, gamma):
    res = []
    for _ in range(10):
        sac = run(Model, Policy, gamma)
        res.append(sac)
    
    avg = np.mean(res, axis=0)
    res.append(avg)

    df = pd.DataFrame(np.stack(res, axis=-1), columns=["sac_%i"%i for i in range(11)])
    df.to_csv(name + ".csv")
    return avg

def run_all_multiple(path=""):
    sac = run_multiple(path+"SAC", SAC, SACPolicy, 0.5)
    ddpg = run_multiple(path+"DDPG", DDPG, DDPGPolicy, 0.5)
    td3 = run_multiple(path+"TD3", TD3, TD3Policy, 0.5)
    ppo2 = run_multiple(path+"PPO2", PPO2, MlpPolicy, 0)
    trpo = run_multiple(path+"TRPO", TRPO, MlpPolicy, 0)
    a2c = run_multiple(path+"A2C", A2C, MlpPolicy, 0)
    acktr = run_multiple(path+"ACKTR", ACKTR, MlpPolicy, 0)

    output_data = [sac, ddpg, td3, ppo2, trpo, a2c, acktr]
    color = ['r', 'b', 'g', 'c', 'y', 'm', 'k']
    for d, l, c in zip(output_data, color, cols):
        plt.plot(d, l, label=c)
        
    plt.legend()
    plt.savefig('comparison_mean_%i.png' % total_timesteps)
    plt.close()

    output = np.stack(output_data, axis=-1)
    df = pd.DataFrame(output, columns=cols)
    df.to_csv("prediction_mean_%i.csv" % total_timesteps)


def load_avg(path):
    data = []
    files = sorted(os.listdir(path))
    names = []
    for p in files:
        names.append(p.rstrip(".csv"))
        df = pd.read_csv(os.path.join(path, p)) 
        avg = df["sac_10"].to_numpy()
        data.append(avg)
    
    for i,(d, c) in enumerate(zip(data, names)):
        plt.plot(d, color[i], label=c)
    
    plt.legend()
    plt.savefig('comparison_mean.png')
    plt.close()

    data = np.stack(data, axis=-1)
    df = pd.DataFrame(data, columns=names)
    df.to_csv("prediction_mean.csv")



def draw_comparison(path, title):
    df = pd.read_csv(path)
    start = datetime.strptime("2016-01-01", "%Y-%m-%d")
    length = df.shape[0]
    times = [(start + timedelta(days=i+1)) for i in range(length)]
    for i, x in enumerate(cols):
        plt.plot(times, df[x].to_numpy(), color=color[i], label=x)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Balance")
    plt.title(title)
    plt.savefig(path + ".png")
    plt.close()


# draw_comparison("results_multiples_10000/prediction_mean_10000.csv", "Prediction After Learning 10000 Iterations")
# draw_comparison("results_multiples_8000/prediction_mean_8000.csv", "Prediction After Learning 8000 Iterations")
# draw_comparison("results_multiples_6000/prediction_mean_6000.csv", "Prediction After Learning 6000 Iterations")
# draw_comparison("results_multiple_4000/prediction_mean_4000.csv", "Prediction After Learning 4000 Iterations")
# draw_comparison("results_multiples_2000/prediction_mean.csv", "Prediction After Learning 2000 Iterations")


def draw_multiple_times(path, iterations, method):
    
    df = pd.read_csv(path)
    start = datetime.strptime("2016-01-01", "%Y-%m-%d")
    length = df.shape[0]
    columns = df.columns
    times = [(start + timedelta(days=i+1)) for i in range(length)]
    for i, x in enumerate(columns[1:-1]):
        plt.plot(times, df[x].to_numpy(), color=color[i], label="trained time %i"%i)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Balance")
    plt.title("Prediction results of %s after %i iterations" % (method, iterations))
    plt.savefig(path + ".png")
    plt.close()


# if iterations > 6000:
#         path = "results_multiples_%i/all/results_multiple_%i%s.csv" % (iterations, iterations, method)
#     else:
#         path = "results_multiples_%i/all/%s.csv" % (iterations, method)

# for x in cols:
#     draw_multiple_times(10000, x)

run_multiple("SAC_2000", SAC, SACPolicy, 0.5)
# draw_multiple_times("SAC_2000.csv", 2000, "SAC")
# draw_multiple_times("SAC_10000_gamma0.5.csv", 10000, "SAC")