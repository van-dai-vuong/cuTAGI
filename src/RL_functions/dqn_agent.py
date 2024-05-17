# Replay memory
from collections import namedtuple, deque
import random
import matplotlib.pyplot as plt
import numpy as np
import math

from src.RL_functions.helpers import *
from src.gym_kalman.env_revised_rewards import *

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
from matplotlib import gridspec
from itertools import count

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN_agent():
    def __init__(self, n_observations, n_actions, device='cpu', LR=0.01, GAMMA=0.999):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.device = device
        self.GAMMA = GAMMA
        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        self.episode_rewards = []

    def select_action(self, state, greedy=False, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=1000):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if not greedy:
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return self.policy_net(state).max(1).indices.view(1, 1)
            else:
                sample2 = random.random()
                if sample2 < 1/100:
                    # select action 1
                    return torch.tensor([[1]], device=self.device, dtype=torch.long)
                else:
                    return torch.tensor([[0]], device=self.device, dtype=torch.long)
                # # # random action
                # return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

    def plot_rewards(self, show_result=False, ylim=None):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.plot(durations_t.numpy())
        plt.ylim(ylim[0], ylim[1])

        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model(self, BATCH_SIZE):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes, step_look_back, num_steps_per_episode, \
              components, time_step, hyperparameters, x_init, \
              abnormal_ts_percentage, anomaly_range, \
              batchsize, TAU, plot_samples=False, learning_curve_ylim = [4000, 6000]):
        if torch.cuda.is_available():
            num_episodes = num_episodes
        else:
            num_episodes = num_episodes

        track_intervention_taken_times = np.zeros(num_episodes)

        for i_episode in range(num_episodes):
            anm_pos1 = np.random.randint(step_look_back, int(num_steps_per_episode/2))
            anm_pos2 = np.random.randint(int(num_steps_per_episode/2) + step_look_back, num_steps_per_episode)

            sample = random.random()
            if sample < abnormal_ts_percentage:
                syn_ts = generate_time_series(components = components,\
                                      time_step = time_step, \
                                      hyperparameters = hyperparameters,\
                                      num_steps = num_steps_per_episode, \
                                      x_init = x_init,\
                                      insert_anomaly = True,\
                                      anomaly_timesteps = [anm_pos1, anm_pos2],\
                                      anomaly_LT = [np.random.uniform(anomaly_range[0], anomaly_range[1]), np.random.uniform(anomaly_range[0], anomaly_range[1])]
                                      )
                anomaly_injected = True
            else:
                syn_ts = generate_time_series(components = components,\
                                      time_step = time_step, \
                                      hyperparameters = hyperparameters,\
                                      num_steps = num_steps_per_episode, \
                                      x_init = x_init,\
                                      insert_anomaly = False,
                                      )
                anomaly_injected = False

            # Set the dataset for the environment
            i_ts_datasets = {
                'measurement': syn_ts['y'],
                'timestamps': syn_ts['timesteps'],
                'time_step': time_step,
                'components': components,
                'hyperparameters': hyperparameters,
                'initial_states': x_init,
            }
            env = KalmanInterventionEnv(render_mode=None, time_series_datasets=i_ts_datasets, step_look_back = step_look_back, hyperparameters=hyperparameters, smoothing_length=0)

            state, info = env.reset(mode = 'train')
            # state = np.hstack((state['KF_hidden_states'], intervention_taken))
            state = state['KF_hidden_states']
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Nomalize the state
            # Compute the stationary standard deviation for AR
            AR_std_stationary = np.sqrt(hyperparameters['ar']['process_error_var']/(1-hyperparameters['ar']['phi']**2))
            LA_var_stationary = hyperparameters['autoregressive_acceleration']['LA_process_error_var']/(1-hyperparameters['autoregressive_acceleration']['phi']**2)
            if step_look_back == 64:
                seg_len = 8
            state = normalize_tensor_two_parts(state, 0, np.sqrt(LA_var_stationary), \
                                            0, AR_std_stationary, seg_len)

            total_reward_one_episode = 0
            for t in count():
                action = self.select_action(state)

                observation, reward, terminated, truncated, _ = env.step(action.item())

                if action.item() == 1:
                    track_intervention_taken_times[i_episode] += 1
                    intervention_taken = True

                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated
                total_reward_one_episode += reward

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation['KF_hidden_states'],\
                            dtype=torch.float32, device=self.device).unsqueeze(0)
                    next_state = normalize_tensor_two_parts(next_state, 0, np.sqrt(LA_var_stationary),\
                                                            0, AR_std_stationary, seg_len)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model(batchsize)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_rewards.append(total_reward_one_episode)
                    self.plot_rewards(ylim = learning_curve_ylim)
                    break

            if plot_samples:
                if track_intervention_taken_times[i_episode] == 2:
                    # Plot prediction
                    fig = plt.figure(figsize=(20, 9))
                    gs = gridspec.GridSpec(4, 1)
                    ax0 = plt.subplot(gs[0])
                    ax1 = plt.subplot(gs[1])
                    ax2 = plt.subplot(gs[2])
                    ax3 = plt.subplot(gs[3])

                    ax0.plot(syn_ts['timesteps'], syn_ts['y'], label='True')
                    # plot the standard deviation of the prediction
                    ax0.plot(syn_ts['timesteps'], info['y_pred']['mu'], label='Predicted')
                    ax0.set_title('Predicted vs True')
                    ax0.set_ylabel('y')
                    if anomaly_injected:
                        anomaly_pos = syn_ts['timesteps'][anm_pos1]
                        ax0.axvline(x=anomaly_pos, color='gray', linestyle='--')
                        anomaly_pos = syn_ts['timesteps'][anm_pos2]
                        ax0.axvline(x=anomaly_pos, color='gray', linestyle='--')
                    ax0.legend()

                    ax1.plot(syn_ts['timesteps'], info['hidden_states']['mu'][:,2], label='LA')
                    ax1.fill_between(syn_ts['timesteps'], info['hidden_states']['mu'][:,2] - np.sqrt(info['hidden_states']['var'][:,2,2]),\
                                        info['hidden_states']['mu'][:,2] + np.sqrt(info['hidden_states']['var'][:,2,2]), color='gray', alpha=0.2)
                    ax1.set_ylabel('LA')

                    ax2.plot(syn_ts['timesteps'], info['hidden_states']['mu'][:,3], label='LA')
                    ax2.fill_between(syn_ts['timesteps'], info['hidden_states']['mu'][:,3] - np.sqrt(info['hidden_states']['var'][:,3,3]),\
                                        info['hidden_states']['mu'][:,3] + np.sqrt(info['hidden_states']['var'][:,3,3]), color='gray', alpha=0.2)
                    ax2.set_ylabel('PD')

                    AR_var_stationary = hyperparameters['ar']['process_error_var']/(1-hyperparameters['ar']['phi']**2)
                    ax3.fill_between(syn_ts['timesteps'], np.zeros_like(syn_ts['timesteps'])-3*np.sqrt(AR_var_stationary), np.zeros_like(syn_ts['timesteps'])+3*np.sqrt(AR_var_stationary), color='red', alpha=0.1)
                    ax3.plot(syn_ts['timesteps'], info['hidden_states']['mu'][:,-1], label='AR')
                    ax3.fill_between(syn_ts['timesteps'], info['hidden_states']['mu'][:,-1] - np.sqrt(info['hidden_states']['var'][:,-1,-1]),\
                                        info['hidden_states']['mu'][:,-1] + np.sqrt(info['hidden_states']['var'][:,-1,-1]), color='gray', alpha=0.2)
                    ax3.set_ylabel('AR')
                    plt.show()

        print('Complete')
        self.plot_rewards(show_result=True, ylim=learning_curve_ylim)
        # plot a bar chart of the number of interventions taken
        plt.figure(2)
        plt.bar(np.arange(num_episodes)+1, track_intervention_taken_times)
        plt.title('Number of interventions taken')
        plt.xlabel('Episode')
        plt.ylabel('Number of interventions taken')
        plt.show()
        plt.ioff()
