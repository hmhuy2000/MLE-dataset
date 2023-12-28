import gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import TD3
from sb3_contrib import TQC
from stable_baselines3.common.evaluation import evaluate_policy

# Retrieve the model from the hub
## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
## filename = name of the model zip file from the repository
checkpoint = load_from_hub(
	repo_id="sb3/tqc-HalfCheetah-v3",
	filename="tqc-HalfCheetah-v3.zip",
)
model = TQC.load(checkpoint)
model.actor.noise = 0.8
# Evaluate the agent and watch it
eval_env = gym.make("HalfCheetah-v3")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, render=False, n_eval_episodes=10, deterministic=False, warn=False
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")