import gymnasium as gym
import metaworld

# Choose a benchmark and task
env_name = 'reach-v3'
benchmark = 'MT1'  # could be MT10, MT50 later

# Create the environment
env = gym.make(f'Meta-World/{benchmark}', env_name=env_name, seed=42, render_mode='rgb_array')

# Reset environment
obs, info = env.reset()
done = False
frame_count = 0

while not done:
    # Sample random action
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Render frame
    frame = env.render()  # RGB array
    frame_count += 1
    
    # Print frame info
    print(f"Frame {frame_count}: shape = {frame.shape}, dtype = {frame.dtype}")

env.close()
print("Episode finished.")
