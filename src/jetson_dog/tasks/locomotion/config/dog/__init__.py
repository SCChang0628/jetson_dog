"""
Register Jetson Dog training environments.
"""

import gymnasium as gym

from . import agents


# ============================================================
# STANDING ENVIRONMENT
# ============================================================

gym.register(

    id="Jetson-Dog-Stand",

    entry_point="isaaclab.envs:ManagerBasedRLEnv",

    disable_env_checker=True,

    kwargs={

        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "JetsonDogStandEnvCfg"
        ),

        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "JetsonDogStandPPORunnerCfg"
        ),

        "default_agent": "rsl_rl",
    },
)


# ============================================================
# WALKING ENVIRONMENT
# ============================================================

gym.register(

    id="Jetson-Dog-Walk",

    entry_point="isaaclab.envs:ManagerBasedRLEnv",

    disable_env_checker=True,

    kwargs={

        "env_cfg_entry_point": (
            f"{__name__}.env_cfg:"
            "JetsonDogWalkEnvCfg"
        ),

        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "JetsonDogWalkPPORunnerCfg"
        ),

        "default_agent": "rsl_rl",
    },
)