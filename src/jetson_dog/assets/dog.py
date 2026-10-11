"""
Jetson Dog robot configuration.

Isaac Lab: release/3.0.0
Isaac Sim: 6.1.0
Python: 3.12

Robot:
    12-DOF quadruped

Joint groups:
    4 shoulders
    4 thighs
    4 calves
"""

from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import DCMotorCfg

from isaaclab_physx.sim.schemas import (
    PhysxArticulationCfg,
    PhysxRigidBodyCfg,
)

from jetson_dog.assets import JETSON_DOG_ASSETS_DIR


# ============================================================
# USD MODEL
# ============================================================

USD_PATH = Path(
    JETSON_DOG_ASSETS_DIR
) / "Dog_v1.usda"


# ============================================================
# JOINT NAMES
# ============================================================

JOINT_NAMES = [
    "FL_shoulder_Joint",
    "RL_shoulder_Joint",
    "FR_shoulder_Joint",
    "RR_shoulder_Joint",

    "FL_thigh_Joint",
    "RL_thigh_Joint",
    "FR_thigh_Joint",
    "RR_thigh_Joint",

    "FL_calves_Joint",
    "RL_calves_Joint",
    "FR_calves_Joint",
    "RR_calves_Joint",
]


# ============================================================
# DEFAULT STANDING JOINT POSITIONS
# ============================================================

# Values from your Isaac Sim Action Graph.
# Unit: radians

DEFAULT_JOINT_POS = {

    # Shoulders
    ".*_shoulder_Joint": 0.0,

    # Thighs
    ".*_thigh_Joint": 0.4,

    # Calves
    ".*_calves_Joint": -0.8,
}


# ============================================================
# ROBOT PARAMETERS
# ============================================================

# Initial estimated base height.
#
# Your dog has approximately 0.088 m thigh and calf links.
# Its height is much smaller than Unitree Go2.
#
# This value must be verified in simulation.

STANDING_HEIGHT = 0.18


# Initial actuator limits based on the USDA.
#
# These are provisional limits and must eventually be
# verified against your actual servo specifications.

SHOULDER_TORQUE = 2.94
THIGH_TORQUE = 2.94
CALF_TORQUE = 1.96

MAX_JOINT_VELOCITY = 5.0


# Initial PD gains.
#
# These are conservative starting values, not validated
# real-world servo parameters.

JOINT_STIFFNESS = 5.0
JOINT_DAMPING = 0.1


# ============================================================
# ARTICULATION CONFIGURATION
# ============================================================

DOG_CFG = ArticulationCfg(

    prim_path="{ENV_REGEX_NS}/Robot",

    # Your USDA places ArticulationRootAPI on base_link.
    articulation_root_prim_path="/base_link",

    spawn=sim_utils.UsdFileCfg(

        usd_path=str(USD_PATH),

        activate_contact_sensors=True,

        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),

        articulation_props=[
            PhysxArticulationCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ],
    ),

    # --------------------------------------------------------
    # INITIAL STATE
    # --------------------------------------------------------

    init_state=ArticulationCfg.InitialStateCfg(

        pos=(
            0.0,
            0.0,
            STANDING_HEIGHT,
        ),

        joint_pos=DEFAULT_JOINT_POS,

        joint_vel={
            ".*": 0.0,
        },
    ),

    soft_joint_pos_limit_factor=0.9,

    # --------------------------------------------------------
    # ACTUATORS
    # --------------------------------------------------------

    actuators={

        # Shoulder motors
        "shoulders": DCMotorCfg(

            joint_names_expr=[
                ".*_shoulder_Joint",
            ],

            actuator_effort_limit=SHOULDER_TORQUE,

            saturation_effort=SHOULDER_TORQUE,

            actuator_velocity_limit=MAX_JOINT_VELOCITY,

            stiffness=JOINT_STIFFNESS,

            damping=JOINT_DAMPING,

            friction=0.0,
        ),

        # Thigh motors
        "thighs": DCMotorCfg(

            joint_names_expr=[
                ".*_thigh_Joint",
            ],

            actuator_effort_limit=THIGH_TORQUE,

            saturation_effort=THIGH_TORQUE,

            actuator_velocity_limit=MAX_JOINT_VELOCITY,

            stiffness=JOINT_STIFFNESS,

            damping=JOINT_DAMPING,

            friction=0.0,
        ),

        # Calf motors
        "calves": DCMotorCfg(

            joint_names_expr=[
                ".*_calves_Joint",
            ],

            actuator_effort_limit=CALF_TORQUE,

            saturation_effort=CALF_TORQUE,

            actuator_velocity_limit=MAX_JOINT_VELOCITY,

            stiffness=JOINT_STIFFNESS,

            damping=JOINT_DAMPING,

            friction=0.0,
        ),
    },
)