"""
RSL-RL PPO configurations for Jetson Dog.

Based on NVIDIA Isaac Lab 3.0 Unitree Go2 PPO.
"""

from isaaclab.utils import configclass

from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2FlatPPORunnerCfg,
)


# ============================================================
# STAGE 1: STANDING PPO
# ============================================================

@configclass
class JetsonDogStandPPORunnerCfg(
    UnitreeGo2FlatPPORunnerCfg
):
    """
    PPO configuration for standing training.
    """

    def __post_init__(self):

        super().__post_init__()

        # Experiment
        self.experiment_name = "jetson_dog_stand"

        # Training
        self.num_steps_per_env = 24

        self.max_iterations = 1000

        self.save_interval = 50

        # Actor network
        self.actor.hidden_dims = [
            128,
            128,
            128,
        ]

        self.actor.activation = "elu"

        self.actor.obs_normalization = False

        # Critic network
        self.critic.hidden_dims = [
            128,
            128,
            128,
        ]

        self.critic.activation = "elu"

        self.critic.obs_normalization = False

        # PPO
        self.algorithm.learning_rate = 1.0e-3

        self.algorithm.gamma = 0.99

        self.algorithm.lam = 0.95

        self.algorithm.clip_param = 0.2

        self.algorithm.entropy_coef = 0.01

        self.algorithm.num_learning_epochs = 5

        self.algorithm.num_mini_batches = 4

        self.algorithm.desired_kl = 0.01

        self.algorithm.max_grad_norm = 1.0


# ============================================================
# STAGE 2: WALKING PPO
# ============================================================

@configclass
class JetsonDogWalkPPORunnerCfg(
    UnitreeGo2FlatPPORunnerCfg
):
    """
    PPO configuration for walking training.
    """

    def __post_init__(self):

        super().__post_init__()

        # Experiment
        self.experiment_name = "jetson_dog_walk"

        # Training
        self.num_steps_per_env = 24

        self.max_iterations = 2000

        self.save_interval = 50

        # Actor network
        self.actor.hidden_dims = [
            128,
            128,
            128,
        ]

        self.actor.activation = "elu"

        self.actor.obs_normalization = False

        # Critic network
        self.critic.hidden_dims = [
            128,
            128,
            128,
        ]

        self.critic.activation = "elu"

        self.critic.obs_normalization = False

        # PPO
        self.algorithm.learning_rate = 5.0e-4

        self.algorithm.gamma = 0.99

        self.algorithm.lam = 0.95

        self.algorithm.clip_param = 0.2

        self.algorithm.entropy_coef = 0.01

        self.algorithm.num_learning_epochs = 5

        self.algorithm.num_mini_batches = 4

        self.algorithm.desired_kl = 0.01

        self.algorithm.max_grad_norm = 1.0