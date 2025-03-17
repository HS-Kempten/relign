import numpy as np
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
)

import os
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization


class EvaluationCallback(EvalCallback):

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                assert isinstance(episode_rewards, list)
                assert isinstance(episode_lengths, list)
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    **kwargs,
                )

            mean_reward, _ = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, _ = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)
            self.last_mean_ep_length = float(mean_ep_length)

            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = float(mean_reward)
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training



class Threshold(object):
    """
    A threshold wrapper that behaves essentially like a number, which is used in the callback and
    the evaluations.
    """

    def adjust(self, factor):
        self.threshold = self.threshold * factor

    def __init__(self, threshold):
        self.threshold = threshold

    def __eq__(self, other):
        return self.threshold == other

    def __lt__(self, other):
        return self.threshold < other

    def __le__(self, other):
        return self.threshold <= other

    def __gt__(self, other):
        return self.threshold > other

    def __ge__(self, other):
        return self.threshold >= other

    def __ne__(self, other):
        return not(self.__eq__(other))

    def __repr__(self):
        return str(self.threshold)

    def __mul__(self, other):
        if isinstance(other, Threshold):
            other = other.threshold
        return Threshold(self.threshold * other)

    def __format__(self, format_spec):
        if not format_spec:
            return str(self.threshold)
        return format(self.threshold, format_spec)


class CLCallbackSteps(BaseCallback):
    """

    Args:
        look_back (int): Number of last evaluations that should be used to check if task is solved
        factor (float): Factor with which the threshold is multiplied to get new threshold
    """
    parent: EvaluationCallback

    def __init__(self, threshold, factor=0.8, look_back=5, max_steps=20, lr_schedule_rat=0.8):
        super().__init__(verbose=0)

        self.max_steps = max_steps
        self.threshold = threshold
        self.look_back = look_back
        self.factor = factor

        self.rewards = []
        self.last_adjustment = 0
        self.without_update = 0

        self.lr_schedule_rat = lr_schedule_rat

    def _init_callback(self) -> None:
        self.init_lr = self.parent.model.learning_rate

    def _on_step(self) -> bool:
        self.rewards.append(self.parent.last_mean_ep_length)

        self.without_update += 1
        if self.n_calls >= self.last_adjustment + self.look_back:

            # Check if task is solved
            if np.mean(self.rewards[-self.look_back:]) <= self.max_steps:
                self.threshold.adjust(self.factor)
                self.last_adjustment = self.n_calls
                self.parent.best_mean_reward = -np.inf
                self.without_update = 1
                self.parent.model.learning_rate = self.init_lr

            if self.without_update % (3 * self.look_back) == 0:
                lr = self.parent.model.learning_rate
                self.parent.model.learning_rate = lr * self.lr_schedule_rat
                self.parent.model._setup_lr_schedule()

        self.logger.record("eval/score_goal_threshold", self.threshold.threshold)
        return True
