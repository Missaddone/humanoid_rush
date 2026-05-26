# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, quat_apply, quat_apply_inverse

from .humanoid_rush_env_cfg import HumanoidRushEnvCfg
from humanoid_rush.motions.motion_loader import MotionLoader
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.sensors import RayCaster


def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


class HumanoidRushEnv(DirectRLEnv):
    cfg: HumanoidRushEnvCfg

    def __init__(self, cfg: HumanoidRushEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        # 这些属性来自哪里？是urdf文件定义的？还是isaac本身
        joint_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        joint_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (joint_upper_limits + joint_lower_limits)
        self.action_scale = joint_upper_limits - joint_lower_limits

        # 加载动作文件
        self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        # 参考速度
        self.target_speed = sample_uniform(
            self.cfg.target_speed_min,
            self.cfg.target_speed_max,
            (self.num_envs,),
            device=self.device
        )

        # 定义一个箭头
        self.visualization_markers = define_markers()
        self.actual_marker_offset = torch.tensor((0.0, 0.0, 0.6), device=self.device)
        self.target_marker_offset = torch.tensor((0.0, 0.0, 0.65), device=self.device)

        # key body indexs
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        # 参考刚体（torso）在isaac中的索引
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        # 关键刚体在isaac中的索引
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        # isaac中关节对于motion里的索引
        self.motion_joint_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        # 参考刚体在motion里的索引
        # 取[0]是为了将list转成单个index
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        # 关键刚体在motion里的索引
        self.motion_key_body_index = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observation and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size, ))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        # 创建高度图
        if self.cfg.height_scanner_enable:
            self.height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self.height_scanner

        if self.cfg.terrain_enable:
            # add rough terrain
            self.cfg.terrain.num_envs = self.scene.cfg.num_envs
            self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
            self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        else:
            # add ground plane
            spawn_ground_plane(
                prim_path="/World/ground",
                cfg=GroundPlaneCfg(
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1.0,
                        dynamic_friction=1.0,
                        restitution=0.0,
                    ),
                ),
            )

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 这里说明网络输出的是归一化的，但是其他工程呢》这是由什么决定的？网络结构？
        traget_position = self.actions * self.action_scale + self.action_offset
        self.robot.set_joint_position_target(traget_position)

    def _get_observations(self) -> dict:

        policy_obs = compute_policy_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
            self.target_speed,
        )

        # 启动高度图
        if self.cfg.height_scanner_enable:
            ray_hits_w = self.height_scanner.data.ray_hits_w
            root_pos_w = self.robot.data.body_pos_w[:, self.ref_body_index]
            # 获取高度差
            height_scan = root_pos_w[:, 2].unsqueeze(1) - ray_hits_w[..., 2]
            # 处理非法字符
            height_scan = torch.nan_to_num(height_scan, nan=0.0, posinf=0.0, neginf=0.0)
            # 进行限幅
            height_scan = torch.clamp(height_scan, -2.0, 2.0)

            policy_obs = torch.cat((policy_obs, height_scan), dim=-1)

        amp_obs = compute_AMP_obs_modified(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.robot.data.body_pos_w[:, self.ref_body_index],
            self.robot.data.body_quat_w[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # updata AMP observation history
        # 将observation_buffer 全部往后移动一格
        for i in reversed(range(self.cfg.num_amp_observations -1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # 将最新的observation插入开头
        self.amp_observation_buffer[:, 0] = amp_obs.clone()

        # 我印象中extra是一个传递额外信息的zidian
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        observations = {"policy": policy_obs}

        # 更新箭头
        self._visualize_markers()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        root_quat_w = self.robot.data.body_quat_w[:, self.ref_body_index]
        root_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.ref_body_index]
        root_lin_vel_b = quat_apply_inverse(root_quat_w, root_lin_vel_w)

        forward_vel = root_lin_vel_b[:, 0]
        lateral_vel = root_lin_vel_b[:, 1]
        vertical_vel = root_lin_vel_b[:, 2]

        forward_error = forward_vel - self.target_speed
        tracking_error = torch.square(forward_error) + torch.square(lateral_vel) + torch.square(vertical_vel)
        return torch.exp(-tracking_error / self.cfg.velocity_tracking_sigma)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # 躯干z轴高度过低终止
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        # 把指定环境中的执行器状态和外力/外力矩缓存清零，恢复到“无外部扰动”的初始状态
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 重新采样参考速度
        self.target_speed[env_ids] = sample_uniform(
            self.cfg.target_speed_min,
            self.cfg.target_speed_max,
            (len(env_ids),),
            device=self.device
        )

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        # 以random开头
        elif self.cfg.reset_strategy.startswith("random"):
            # 是否存在start字段
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Ubnkown reset strategy: {self.cfg.reset_strategy}")
        
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)



    # reset策略为默认状态
    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        # 需要加上每个并行环境的原点坐标作为偏移
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel
    
    # reset策略为随机
    def _reset_strategy_random(self, env_ids: torch.Tensor, start: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # smaple random motion times (or zeros if start is True)
        num_samples = env_ids.shape[0]
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)

        # sample randon motions
        (
            joint_positions,
            joint_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
        food_body_indexes = self._motion_loader.get_body_index(["right_foot", "left_foot"])

        root_state = self.robot.data.default_root_state[env_ids].clone()

        # x/y 放到对应 env origin
        root_state[:, 0:3] = self.scene.env_origins[env_ids]

        # 根据 root-frame motion 的最低脚高度，把 root 抬到地面上
        min_foot_z = body_positions[:, food_body_indexes, 2].amin(dim=1)
        root_state[:, 2] = 1.1 # 抬高一点，避免碰到地面

        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]

        # joint state
        joint_pos = joint_positions[:, self.motion_joint_indexes]
        joint_vel = joint_velocities[:, self.motion_joint_indexes]

        # updata AMP observation
        amp_observations = self.collect_reference_motions(num_samples, times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

        return root_state, joint_pos, joint_vel
        
    # 这个函数就是创建数据集输入，给discriminator
    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            joint_positions,
            joint_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)

        # compute AMP observation#
        # 这里必须调用同一个计算函数
        amp_observation = compute_AMP_obs_modified(
            joint_positions[:, self.motion_joint_indexes],
            joint_velocities[:, self.motion_joint_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_index]
        )
        return amp_observation.view(-1, self.amp_observation_size)
    
    def _visualize_markers(self):
        marker_locations = self.robot.data.root_pos_w
        forward_marker_orientations = self.robot.data.root_quat_w
        command_marker_orientations = self.robot.data.root_quat_w

        root_lin_vel_b = quat_apply_inverse(
            self.robot.data.root_quat_w,
            self.robot.data.root_lin_vel_w,
        )

        actual_speed = root_lin_vel_b[:, 0].clamp(min=0.0)
        target_speed = self.target_speed.clamp(min=0.0)

        actual_length = (actual_speed / self.cfg.target_speed_mean).clamp(0.2, 2.0)
        target_length = (target_speed / self.cfg.target_speed_mean).clamp(0.2, 2.0)

        width = 0.15
        actual_scales = torch.stack(
            (actual_length, torch.full_like(actual_length, width), torch.full_like(actual_length, width)),
            dim=-1,
        )
        target_scales = torch.stack(
            (target_length, torch.full_like(target_length, width), torch.full_like(target_length, width)),
            dim=-1,
        )

        actual_marker_locations = marker_locations + self.actual_marker_offset
        target_marker_locations = marker_locations + self.target_marker_offset
        loc = torch.vstack((actual_marker_locations, target_marker_locations))

        rots = torch.vstack((forward_marker_orientations, command_marker_orientations))
        scales = torch.vstack((actual_scales, target_scales))

        all_envs = torch.arange(self.num_envs, device=self.device)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

        self.visualization_markers.visualize(
            loc,
            rots,
            scales=scales,
            marker_indices=indices,
        )

@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

@torch.jit.script
def quaternion_to_tangent_and_normal_inverse(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply_inverse(q, ref_tangent)
    normal = quat_apply_inverse(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward


@torch.jit.script
def compute_policy_obs(
    joint_positions: torch.Tensor,
    joint_velocities: torch.Tensor,
    root_positions_w: torch.Tensor,
    root_rotations_w: torch.Tensor,
    root_linear_velocities_w: torch.Tensor,
    root_angular_velocities_w: torch.Tensor,
    key_body_positions_w: torch.Tensor, 
    target_speed:  torch.Tensor,
) -> torch.Tensor:
    # base坐标系的root线速度和角速度
    root_linear_velocities_b = quat_apply_inverse(root_rotations_w, root_linear_velocities_w)
    root_angular_velocities_b = quat_apply_inverse(root_rotations_w, root_angular_velocities_w)


    #
    key_body_positions_rel_w = key_body_positions_w - root_positions_w.unsqueeze(-2)
    num_key_bodies = key_body_positions_rel_w.shape[1]
    root_rotations_expanded = root_rotations_w.unsqueeze(1).expand(-1, num_key_bodies, -1)
    key_body_positions_b = quat_apply_inverse(
        root_rotations_expanded.reshape(-1, 4),
        key_body_positions_rel_w.reshape(-1, 3),
    ).view(key_body_positions_rel_w.shape)

    obs = torch.cat(
        (
            joint_positions,
            joint_velocities,
            root_positions_w[:, 2:3], 
            quaternion_to_tangent_and_normal_inverse(root_rotations_w),
            root_linear_velocities_b,
            root_angular_velocities_b,
            key_body_positions_b.view(key_body_positions_b.shape[0], -1),
            target_speed.unsqueeze(-1),
        ),
        dim=-1,
    )

    return obs


@torch.jit.script
def compute_AMP_obs(
    joint_positions: torch.Tensor,
    joint_velocities: torch.Tensor,
    root_positions_w: torch.Tensor,
    root_rotations_w: torch.Tensor,
    root_linear_velocities_w: torch.Tensor,
    root_angular_velocities_w: torch.Tensor,
    key_body_positions_w: torch.Tensor, 
) -> torch.Tensor:
    obs = torch.cat(
        (
            joint_positions,
            joint_velocities,
            root_positions_w[:, 2:3], #只去z轴高度，并保持二维shaoe
            quaternion_to_tangent_and_normal(root_rotations_w),
            root_linear_velocities_w,
            root_angular_velocities_w,
            (key_body_positions_w -root_positions_w.unsqueeze(-2)).view(key_body_positions_w.shape[0], -1), # key body相对torso在世界坐标系下的位置，并拉平成二维
        ),
        dim=-1,
    )
    return obs

@torch.jit.script
def compute_AMP_obs_modified(
    joint_positions: torch.Tensor,
    joint_velocities: torch.Tensor,
    root_positions_w: torch.Tensor,
    root_rotations_w: torch.Tensor,
    root_linear_velocities_w: torch.Tensor,
    root_angular_velocities_w: torch.Tensor,
    key_body_positions_w: torch.Tensor,
) -> torch.Tensor:
    # root-frame velocity
    root_linear_velocities_b = quat_apply_inverse(root_rotations_w, root_linear_velocities_w)
    root_angular_velocities_b = quat_apply_inverse(root_rotations_w, root_angular_velocities_w)

    # key body positions in root frame
    num_envs = key_body_positions_w.shape[0]
    num_key_bodies = key_body_positions_w.shape[1]

    key_body_positions_b = key_body_positions_w - root_positions_w.unsqueeze(1)
    key_body_positions_b = key_body_positions_b.reshape(num_envs * num_key_bodies, 3)

    root_rotations_expand = root_rotations_w.unsqueeze(1).repeat(1, num_key_bodies, 1)
    root_rotations_expand = root_rotations_expand.reshape(num_envs * num_key_bodies, 4)

    key_body_positions_b = quat_apply_inverse(root_rotations_expand, key_body_positions_b)
    key_body_positions_b = key_body_positions_b.reshape(num_envs, num_key_bodies * 3)

    # keep old AMP dimension, but remove world-frame root height/orientation information
    root_height_obs = torch.zeros_like(root_positions_w[:, 2:3])

    root_rotation_obs = torch.zeros((num_envs, 6), device=root_rotations_w.device)
    root_rotation_obs[:, 0] = 1.0
    root_rotation_obs[:, 5] = 1.0

    obs = torch.cat(
        (
            joint_positions,
            joint_velocities,
            root_height_obs,
            root_rotation_obs,
            root_linear_velocities_b,
            root_angular_velocities_b,
            key_body_positions_b,
        ),
        dim=-1,
    )
    return obs
