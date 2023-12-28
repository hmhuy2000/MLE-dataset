import argparse
import re

import h5py
import torch
import gym

from huggingface_sb3 import load_from_hub
from sb3_contrib import TQC

import numpy as np
from torch import nn
import os

import gym
from stable_baselines3.common.monitor import Monitor

import gym
import panda_gym

import numpy as np

def get_reset_data():
    data = dict(
        observations = [],
        next_observations = [],
        actions = [],
        rewards = [],
        terminals = [],
        timeouts = [],
        logprobs = [],
        qpos = [],
        qvel = []
    )
    return data

def rollout(policy, env_name, max_path, num_data,min_return,max_return,device):
    env = gym.make(env_name)
    print(env.action_space)
    # env = check_and_normalize_box_actions(env)
    data = get_reset_data()
    traj_data = get_reset_data()
    arr_return = []
    arr_len = []
    _returns = 0
    t = 0 
    done = False
    s = env.reset()
    while len(data['rewards']) < num_data:
        a,_ = policy.predict(
            s,  # type: ignore[arg-type]
            state=None,
            episode_start=None,
            deterministic=False,
        )
        a = a.squeeze()
        try:
            ns, rew, done, infos = env.step(a)
        except:
            print('lost connection')
            env.close()
            env = gym.make(env_name)
            s = env.reset()
            traj_data = get_reset_data()
            t = 0
            arr_return.append(_returns)
            _returns = 0
            continue

        _returns += rew

        t += 1
        timeout = False
        terminal = False
        if t == max_path:
            timeout = True
        elif done:
            terminal = True


        traj_data['observations'].append(s)
        traj_data['actions'].append(a)
        traj_data['next_observations'].append(ns)
        traj_data['rewards'].append(rew)
        traj_data['terminals'].append(terminal)
        traj_data['timeouts'].append(timeout)
        traj_data['logprobs'].append(0.0)

        s = ns
        if terminal or timeout:
            if (max_return> _returns > min_return):
                arr_return.append(_returns)
                arr_len.append(t)
                print('Len=%d, R=%f ± %f, Q1=%f, Q2=%f, Q3=%f. Progress:%d/%d' % (np.mean(arr_len), np.mean(arr_return),np.std(arr_return),np.percentile(arr_return, 25),
                                                                            np.percentile(arr_return, 50),np.percentile(arr_return, 75), len(data['rewards']), num_data),
                    end='\r')
                for k in data:
                    data[k].extend(traj_data[k])
            # else:
            #     print(_returns)
            traj_data = get_reset_data()
            s = env.reset()
            t = 0
            _returns = 0
    
    new_data = dict(
        observations=np.array(data['observations']).astype(np.float32),
        actions=np.array(data['actions']).astype(np.float32),
        next_observations=np.array(data['next_observations']).astype(np.float32),
        rewards=np.array(data['rewards']).astype(np.float32),
        terminals=np.array(data['terminals']).astype(bool),
        timeouts=np.array(data['timeouts']).astype(bool)
    )
    new_data['infos/action_log_probs'] = np.array(data['logprobs']).astype(np.float32)

    for k in new_data:
        new_data[k] = new_data[k][:num_data]
    print('Finished trajectory: Len=%d, R=%f ± %f, Q1=%f, Q2=%f, Q3=%f. Progress:%d/%d' % (np.mean(arr_len),np.mean(arr_return),np.std(arr_return),np.percentile(arr_return, 25),
                                                                            np.percentile(arr_return, 50),np.percentile(arr_return, 75), len(data['rewards']), num_data))
    return new_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PandaPickAndPlace-v1')#['HalfCheetah-v3','Walker2d-v3','Swimmer-v3','Humanoid-v3','Hopper-v3']
    parser.add_argument('--pklfile', type=str, default=None)
    parser.add_argument('--max_path', type=int, default=1000)
    parser.add_argument('--num_data', type=int, default=200000)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sample_env = gym.make(args.env)
    state_shape=sample_env.observation_space.shape
    action_shape=sample_env.action_space.shape
    device = 'cuda'
    sample_env.close()
    
    
    if (args.env == 'HalfCheetah-v3'):
        levels = [0.0,0.2,0.4,0.6,0.8] 
        min_arr = [-100000,-100000,-100000,-100000,-100000]
        max_arr = [12500,10000,7000,5000,4000]
    elif (args.env == 'Humanoid-v3'): 
        levels = [0.0,0.3,0.5,0.6,0.7] 
        min_arr = [8000,5000,3500,2000,0]
        max_arr = [8500,7000,5000,3500,2000]
    elif (args.env == 'Walker2d-v3'):
        levels = [0.0,0.3,0.5,0.7,0.9] 
        min_arr = [4300,2500,1500,1000,0]
        max_arr = [5000,3500,2500,1500,1000]
    elif (args.env == 'Hopper-v3'):
        levels = [0.0,0.6,0.6,0.8,0.9] 
        min_arr = [3700,2500,1500,1000,0]
        max_arr = [3800,3000,2000,1500,1000]
    elif (args.env == 'PandaPickAndPlace-v1'):
        levels = [0.0] 
        min_arr = [-10000]
        max_arr = [1000000]
    else:
        raise
    
    print('-'*50)
    print(args.env,levels)
    print(min_arr)
    print(max_arr)
    print('-'*50)
    
    for id in range(len(levels)):
        level = levels[id]
        min_return = min_arr[id]
        max_return = max_arr[id]
        print(level,min_return,max_return)
        
        checkpoint = load_from_hub(
            repo_id=f"sb3/tqc-{args.env}",
            filename=f"tqc-{args.env}.zip",
        )
        model = TQC.load(checkpoint,env=sample_env)
        model.actor.noise = level
        model.actor.eval()
        
        data = rollout(model, args.env, max_path=args.max_path, num_data=args.num_data,
            min_return=min_return,max_return=max_return,device=device)
        save_path = f'./buffers/final_{args.env}'
        save_file = f'{save_path}/{id}.hdf5'
        print()
        print(f'save at{save_file}')
        print('-'*50)
        
        os.makedirs(save_path,exist_ok=True)
        hfile = h5py.File(save_file, 'w')
        for k in data:
            hfile.create_dataset(k, data=data[k], compression='gzip')
        hfile.close()

