from collections import defaultdict

import numpy as np
import tensorflow as tf
from .base_sampler import BaseSampler


class SimpleSampler(BaseSampler):
    def __init__(self, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._current_observation = None
        self._total_samples = 0
        self._frequence_exploration = 50
        self._cadidate_nums = 20
        self._lamda = 1.0

    def _process_observations(self,
                              observation,
                              action,
                              reward,
                              terminal,
                              next_observation,
                              std,
                              info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'stds': [std],
            'infos': info,
        }

        return processed_observation

    def sample(self, disturb=False, fake_env=None, Qs = None):
        if self._current_observation is None:
            self._current_observation = self.env.reset()

        if disturb and self._total_samples % self._frequence_exploration==0:
            def disturb_actions(action, low, high, candidates_num=10):
                action = np.array(action)
                low = np.array(low)
                high = np.array(high)
                std = np.diag((high-low)/4.0)
                noise = np.random.multivariate_normal(np.zeros_like(action), std, candidates_num-1)
                #noise = np.random.normal(size=(candidates_num-1,action.shape[0])) * ((high-low)/2)
                
                base_actions = np.tile(action, (candidates_num-1, 1))
                augmented_actions = base_actions + noise
                all_actions = np.row_stack((augmented_actions, [action]))
                all_actions = np.clip(all_actions, low, high)
                return all_actions

            action = self.policy.actions_np([
                self.env.convert_to_active_observation(
                    self._current_observation)[None]
                ])[0]

            candidates_num = self._cadidate_nums
            lamda_uncertainty = self._lamda
            candidate_actions = disturb_actions(action, self.action_low, self.action_high, candidates_num).astype(np.float32)
            current_obs = np.tile(self._current_observation, (candidates_num, 1)).astype(np.float32)
            _, _, _, info = fake_env.step(current_obs, candidate_actions, deterministic=False)
            model_uncertainty = info['dev']

            with tf.keras.backend.get_session().as_default():
                cand_acts = tf.convert_to_tensor(candidate_actions, tf.float32, name='candidate_actions')
                #cur_obs = tf.convert_to_tensor(current_obs, tf.float32, name='current_obs')
                Qs_values = tuple(Q([current_obs, candidate_actions]) for Q in Qs)
                min_Q = tf.reduce_min(Qs_values, axis=0).eval()
                
            ucb_value = min_Q.squeeze() + lamda_uncertainty * model_uncertainty.squeeze()
            optimal_action_index = np.argmax(ucb_value)
            action = candidate_actions[optimal_action_index]
            #print(f'*********************************************************{optimal_action_index}*********************************************************')
        else:
            action = self.policy.actions_np([
                self.env.convert_to_active_observation(
                    self._current_observation)[None]
                ])[0]

        next_observation, reward, terminal, info = self.env.step(action)
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_observations(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            std=0,
            info=info,
        )

        for key, value in processed_sample.items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = {
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            }
            self.pool.add_path(last_path)
            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self._current_observation = None
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(
            batch_size, observation_keys=observation_keys, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })

        return diagnostics
