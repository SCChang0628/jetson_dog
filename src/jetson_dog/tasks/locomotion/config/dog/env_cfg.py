"""
Jetson Dog locomotion environments.

Isaac Lab 3.0
Isaac Sim 6.1.0
Python 3.12

Environments:
    JetsonDogStandEnvCfg
    JetsonDogWalkEnvCfg

Based on NVIDIA Unitree Go2 flat velocity task.
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass, replace

from isaaclab_tasks.core.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2Rewards,
)

from jetson_dog.assets.dog import (
    DOG_CFG,
    STANDING_HEIGHT,
)

from jetson_dog.tasks.locomotion.mdp.rewards import (
    joint_posture_l2,
)


# ============================================================
# CUSTOM REWARDS
# ============================================================

@configclass
class JetsonDogRewards(UnitreeGo2Rewards):
    """
    Extend NVIDIA Go2 rewards with posture tracking.
    """

    joint_posture = RewTerm(
        func=joint_posture_l2,
        weight=-0.1,
    )


# ============================================================
# BASE ENVIRONMENT
# ============================================================

@configclass
class JetsonDogBaseEnvCfg(UnitreeGo2FlatEnvCfg):
    """
    Shared configuration for standing and walking.
    """

    rewards: JetsonDogRewards = JetsonDogRewards()

    def __post_init__(self):

        super().__post_init__()

        # ----------------------------------------------------
        # PHYSICS BACKEND
        # ----------------------------------------------------

        # Isaac Lab 3.0 Go2 Flat defaults to Newton.
        # Explicitly use Isaac Sim PhysX for this USD model.

        self.sim.physics.default = (
            self.sim.physics.isaacsim_physx
        )

        self.sim.use_newton_actuators = False

        # ----------------------------------------------------
        # ROBOT
        # ----------------------------------------------------

        self.scene.robot = replace(
            DOG_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
        )

        # ----------------------------------------------------
        # TERRAIN
        # ----------------------------------------------------

        self.scene.terrain.terrain_type = "plane"

        self.scene.terrain.terrain_generator = None

        self.scene.height_scanner = None

        # ----------------------------------------------------
        # OBSERVATIONS
        # ----------------------------------------------------

        self.observations.policy.height_scan = None

        # ----------------------------------------------------
        # SCENE
        # ----------------------------------------------------

        # Start with fewer environments.
        # Increase after verifying stability.

        self.scene.num_envs = 256

        self.scene.env_spacing = 1.5

        # ----------------------------------------------------
        # SIMULATION
        # ----------------------------------------------------

        # Physics: 200 Hz
        self.sim.dt = 0.005

        # Control: 50 Hz
        self.decimation = 4

        self.sim.render_interval = self.decimation

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        self.actions.joint_pos.scale = 0.15

        # ----------------------------------------------------
        # CONTACT SENSORS
        # ----------------------------------------------------

        self.scene.contact_forces.prim_path = (
            "{ENV_REGEX_NS}/Robot/[^/]*"
        )

        self.scene.contact_forces.update_period = (
            self.sim.dt
        )

        # ----------------------------------------------------
        # REWARDS
        # ----------------------------------------------------

        # Base height target

        self.rewards.base_height_l2.params[
            "target_height"
        ] = STANDING_HEIGHT

        self.rewards.base_height_l2.params[
            "sensor_cfg"
        ] = None

        # Foot name mapping

        self.rewards.feet_air_time.params[
            "sensor_cfg"
        ].body_names = ".*_foot"

        # ----------------------------------------------------
        # TERMINATIONS
        # ----------------------------------------------------

        # Your base body is named base_link, not base.

        self.terminations.base_contact.params[
            "sensor_cfg"
        ].body_names = "base_link"

        # ----------------------------------------------------
        # EVENTS
        # ----------------------------------------------------

        # Disable mass / COM randomization until the
        # actual robot mass properties are validated.

        self.events.base_external_force_torque = None

        self.events.add_base_mass = None

        self.events.base_com = None

        # Disable random pushes initially.

        self.events.push_robot = None

        # Disable terrain curriculum.

        self.curriculum.terrain_levels = None

        # ----------------------------------------------------
        # EPISODE
        # ----------------------------------------------------

        self.episode_length_s = 20.0


# ============================================================
# STAGE 1: STANDING
# ============================================================

@configclass
class JetsonDogStandEnvCfg(JetsonDogBaseEnvCfg):
    """
    Stage 1:

    Learn to remain standing with zero commanded
    linear and angular velocity.
    """

    def __post_init__(self):

        super().__post_init__()

        # ----------------------------------------------------
        # COMMANDS
        # ----------------------------------------------------

        self.commands.base_velocity.heading_command = False

        self.commands.base_velocity.rel_heading_envs = 0.0

        self.commands.base_velocity.rel_standing_envs = 1.0

        self.commands.base_velocity.ranges.lin_vel_x = (
            0.0, 0.0
        )

        self.commands.base_velocity.ranges.lin_vel_y = (
            0.0, 0.0
        )

        self.commands.base_velocity.ranges.ang_vel_z = (
            0.0, 0.0
        )

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        self.actions.joint_pos.scale = 0.15

        # ----------------------------------------------------
        # REWARDS
        # ----------------------------------------------------

        # Track zero forward/lateral velocity.

        self.rewards.track_lin_vel_xy_exp.weight = 2.0

        # Track zero yaw velocity.

        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Keep the body upright.

        self.rewards.flat_orientation_l2.weight = -2.5

        # Maintain body height.

        self.rewards.base_height_l2.weight = -20.0

        # Maintain standing joint posture.

        self.rewards.joint_posture.weight = -0.3

        # Reduce vertical movement.

        self.rewards.lin_vel_z_l2.weight = -2.0

        # Reduce roll/pitch angular motion.

        self.rewards.ang_vel_xy_l2.weight = -0.1

        # Reduce sudden action changes.

        self.rewards.action_rate_l2.weight = -0.02

        # Reduce excessive torques.

        self.rewards.dof_torques_l2.weight = -0.0002

        # No foot air-time reward for standing.

        self.rewards.feet_air_time = None

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        self.events.reset_base.params[
            "pose_range"
        ] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (-0.05, 0.05),
        }

        self.events.reset_base.params[
            "velocity_range"
        ] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.02, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.05, 0.05),
        }

        # Keep joints close to the initial standing pose.

        self.events.reset_robot_joints.params[
            "position_range"
        ] = (0.97, 1.03)


# ============================================================
# STAGE 2: WALKING
# ============================================================

@configclass
class JetsonDogWalkEnvCfg(JetsonDogBaseEnvCfg):
    """
    Stage 2:

    Learn velocity tracking after standing training.

    Commands:
        Forward
        Backward
        Turning
    """

    def __post_init__(self):

        super().__post_init__()

        # ----------------------------------------------------
        # COMMANDS
        # ----------------------------------------------------

        self.commands.base_velocity.heading_command = False

        self.commands.base_velocity.rel_heading_envs = 0.0

        self.commands.base_velocity.rel_standing_envs = 0.1

        self.commands.base_velocity.ranges.lin_vel_x = (
            -0.3, 0.3
        )

        self.commands.base_velocity.ranges.lin_vel_y = (
            -0.1, 0.1
        )

        self.commands.base_velocity.ranges.ang_vel_z = (
            -0.6, 0.6
        )

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        self.actions.joint_pos.scale = 0.25

        # ----------------------------------------------------
        # REWARDS
        # ----------------------------------------------------

        # Main objective: velocity tracking.

        self.rewards.track_lin_vel_xy_exp.weight = 2.0

        self.rewards.track_ang_vel_z_exp.weight = 1.0

        # Body stability.

        self.rewards.flat_orientation_l2.weight = -2.5

        # Body height.

        self.rewards.base_height_l2.weight = -10.0

        # Allow joint movement during walking.

        self.rewards.joint_posture.weight = -0.1

        # Motion penalties.

        self.rewards.lin_vel_z_l2.weight = -2.0

        self.rewards.ang_vel_xy_l2.weight = -0.1

        self.rewards.dof_torques_l2.weight = -0.0002

        self.rewards.dof_acc_l2.weight = -2.5e-7

        self.rewards.action_rate_l2.weight = -0.005

        # Encourage feet to lift during movement.

        self.rewards.feet_air_time.weight = 0.02

        # ----------------------------------------------------
        # RESET
        # ----------------------------------------------------

        self.events.reset_base.params[
            "pose_range"
        ] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        }

        self.events.reset_base.params[
            "velocity_range"
        ] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.05, 0.05),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }

        self.events.reset_robot_joints.params[
            "position_range"
        ] = (0.9, 1.1)