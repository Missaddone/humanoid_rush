import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


ROBOTS_DIR = os.path.dirname(os.path.abspath(__file__))
HUMANOID_USD_PATH = os.path.join(ROBOTS_DIR, "humanoid.usd")
HUMANOID_28_USD_PATH = os.path.join(ROBOTS_DIR, "humanoid_28.usd")
HUMANOID_FROMXML_USD_PATH = os.path.join(ROBOTS_DIR, "humanoid_from_xml.usd")

HUMANIUD_28_CONFIG = ArticulationCfg(
    
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=HUMANOID_28_USD_PATH,
        # usd_path=HUMANOID_FROMXML_USD_PATH,
        # usd_path=HUMANOID_USD_PATH,
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Humanoid28/humanoid_28.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=None,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.2),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
            velocity_limit_sim={".*": 100.0},
        ),
    },
)
