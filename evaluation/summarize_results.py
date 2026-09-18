import pandas as pd

df = pd.read_csv("./data/evaluation_100.csv")

print("=== Reward Model Metrics ===")
print(f"Mean reward SFT : {df['reward_sft'].mean():.3f}")
print(f"Mean reward PPO : {df['reward_ppo'].mean():.3f}")
print(f"Mean difference : {df['reward_diff'].mean():+.3f}")
print(f"RM win rate     : {(df['reward_diff'] > 0).mean()*100:.1f}%")
print(f"RM tie rate     : {(df['reward_diff'] == 0).mean()*100:.1f}%")
print(f"RM loss rate    : {(df['reward_diff'] < 0).mean()*100:.1f}%")