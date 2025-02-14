from stable_baselines3.common.callbacks import BaseCallback

from src.model.training.scoring import score_model


class AtariCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rewards = dict(num_timesteps=[], rewards=[])
        self.evaluations = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        self.evaluations.append({"num_timesteps": self.num_timesteps, "reward": 0})

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.rewards["num_timesteps"].append(self.num_timesteps)
        if "rewards" in self.locals:
            self.rewards["rewards"].append(self.locals["rewards"].item())
        elif "reward" in self.locals:
            self.rewards["rewards"].append(self.locals["reward"])
        else:
            print(self.locals)
            raise ValueError("No reward found in locals")
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class ThreadTheNeedleCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.rewards = dict(num_timesteps=[], rewards=[])
        self.evaluations = []

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        score = score_model(self.model)
        score["num_timesteps"] = self.num_timesteps
        self.evaluations.append(score)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.rewards["num_timesteps"].append(self.num_timesteps)
        if "rewards" in self.locals:
            self.rewards["rewards"].append(self.locals["rewards"].item())
        elif "reward" in self.locals:
            self.rewards["rewards"].append(self.locals["reward"])
        else:
            print(self.locals)
            raise ValueError("No reward found in locals")

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        score = score_model(self.model)
        score["num_timesteps"] = self.num_timesteps
        self.evaluations.append(score)
