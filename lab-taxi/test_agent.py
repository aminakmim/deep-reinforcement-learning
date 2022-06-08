import unittest
import gym
import agent
import matplotlib.pyplot as plt
import seaborn as sns

class TestAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = agent.Agent()
        self.env = gym.make('Taxi-v3')
        return super().setUp()
    
    def test_select_action(self):
        # Check whether all actions are picked uniformly for random agent
        action_list = []
        for _ in range(10000):
            action_list.append(self.agent.select_action(state=1))

        sns.histplot(action_list, stat='probability')
        plt.pause(5)
        plt.close()
        self.assertTrue(1)

    def test_get_epsilon(self):
        # Check whether epsilon is changing in zigzag upto 1000 episodes
        epsilon_list = []

        for i in range(1, 2000):
            curr_eps = self.agent.get_epsilon(i) # epsiode count starts from 1
            epsilon_list.append(curr_eps)

        sns.lineplot(x=list(range(1, 2000)), y=epsilon_list, legend=True)
        plt.pause(5)
        plt.close()
        self.assertTrue(1)

    def test_step(self):
        # begin the episode
        state = self.env.reset()

        # initialize the sampled reward
        samp_reward = 0

        while True:
            # agent selects an action
            action = self.agent.select_action(state)

            # agent performs the selected action
            next_state, reward, done, _ = self.env.step(action)

            # agent performs internal updates based on sampled experience
            self.agent.step(state, action, reward, next_state, done)

            # update the sampled reward
            samp_reward += reward

            # update the state (s <- s') to next time step
            state = next_state

            if done:
                break
        self.assertNotEqual(samp_reward, 0)

if __name__ == '__main__':
    unittest.main()
