from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)

### Let's checkout how the trained agent play !!
## begin the episode
state = env.reset()

# initialize the sampled reward
samp_reward = 0

while True:
    # agent selects an action
    action = agent.select_action(state)

    # agent performs the selected action
    next_state, reward, done, _ = env.step(action)

    # render environment 
    env.render()

    # agent performs internal updates based on sampled experience
    agent.step(state, action, reward, next_state, done)

    # update the sampled reward
    samp_reward += reward

    # update the state (s <- s') to next time step
    state = next_state

    if done:
        print(f'Total reward : {samp_reward}')
        break
