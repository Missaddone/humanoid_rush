# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
from dataclasses import MISSING

from humanoid_rush.robots.humanoid_28 import HUMANIUD_28_CONFIG

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from isaaclab.sensors import RayCasterCfg, patterns


MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../motions")


@configclass
class HumanoidRushEnvCfg(DirectRLEnvCfg):
    # 这两个数指的是什么含义？
    # 构建的buffer shape是(num_envs, num_amp_observations, amp_observation_space)
    # 帧的长度，和单帧长度
    num_amp_observations = 2
    amp_observation_space = 81

    early_termination = True
    termination_height = 0.5

    # 给定的参考速度
    target_speed_mean = 1.5
    target_speed_min = 1.5
    target_speed_max = 1.5
    velocity_tracking_sigma = 0.25

    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk_mimickit_rootframe_loop2.npz")
    # motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
    reference_body = "torso"
    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
        在轨迹中随机采样一个时间开始
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
        在轨迹的起始点开始（可能有多段轨迹）
    """


    # 高度图
    height_scanner_enable = False
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso", # 以torso为中心
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.2, # 每0.2m一个采样点
            size=(3.0, 2.0) # 以torso为中心 3.0m x 2.0m的局部网格
        ),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )
    
    height_dim = 16 * 11

    # 启用地形
    terrain_enable = False

    # env
    decimation = 2
    episode_length_s = 10.0
    # - spaces definition
    action_space = 28
    state_space = 0

    if height_scanner_enable:
        observation_space = 82 + height_dim
    else:
        observation_space = 82

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG.replace(num_rows=5, num_cols=5, curriculum=False),
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = HUMANIUD_28_CONFIG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                velocity_limit_sim={
                    ".*": 100.0,
                },
            ),
        },
    )


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # viewer
    # viewer = ViewerCfg(
    #     eye=(2.0, 2.0, 1.0),
    #     lookat=(0.0, 0.0, 0.0),
    #     origin_type="asset_root",
    #     asset_name="robot",
    #     env_index=0,
    # )

    # custom parameters/scales


    # - reset states/conditions
