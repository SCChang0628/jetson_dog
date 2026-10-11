"""
Test Jetson Dog Gymnasium registrations.
"""

import gymnasium as gym

import jetson_dog.tasks.locomotion.config.dog


def test_standing_registration():

    spec = gym.spec("Jetson-Dog-Stand")

    assert spec.entry_point == (
        "isaaclab.envs:ManagerBasedRLEnv"
    )

    assert spec.kwargs["env_cfg_entry_point"].endswith(
        "JetsonDogStandEnvCfg"
    )

    assert spec.kwargs["default_agent"] == "rsl_rl"


def test_walking_registration():

    spec = gym.spec("Jetson-Dog-Walk")

    assert spec.entry_point == (
        "isaaclab.envs:ManagerBasedRLEnv"
    )

    assert spec.kwargs["env_cfg_entry_point"].endswith(
        "JetsonDogWalkEnvCfg"
    )

    assert spec.kwargs["default_agent"] == "rsl_rl"


def test_ppo_registration():

    stand = gym.spec("Jetson-Dog-Stand")

    walk = gym.spec("Jetson-Dog-Walk")

    assert "rsl_rl_cfg_entry_point" in stand.kwargs

    assert "rsl_rl_cfg_entry_point" in walk.kwargs