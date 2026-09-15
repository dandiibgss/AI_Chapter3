# Example 3.23 OpenAI Gym CartPole
# https://gym.openai.com/docs/
# https://gym.openai.com/envs/CartPole-v0/
try:
    import gymnasium as gym
except ImportError:
    import gym

try:
    env = gym.make('CartPole-v1', render_mode='human')
except Exception:
    try:
        env = gym.make('CartPole-v0', render_mode='human')
    except Exception:
        env = gym.make('CartPole-v0')

for i_episode in range(20):
    reset_ret = env.reset()
    observation = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret

    for t in range(100):
        try:
            env.render()
        except Exception:
            pass

        print(observation)
        action = env.action_space.sample()

        step_ret = env.step(action)
        if len(step_ret) == 5:
            observation, reward, terminated, truncated, info = step_ret
            done = terminated or truncated
        else:
            observation, reward, done, info = step_ret

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()