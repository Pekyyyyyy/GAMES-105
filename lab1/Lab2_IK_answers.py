import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
max_iter = 100
alpha = 0.1

def forward_kinematics(joint_name, joint_parent, joint_orientations, joint_initial_position, forward_path):
    joint_position = joint_initial_position.copy()

    for joint_idx in forward_path:
        parent_idx = joint_parent[joint_idx]
        parent_orientation = R.from_quat(joint_orientations[parent_idx])
        bone_len = joint_initial_position[joint_idx] - joint_initial_position[parent_idx]
        joint_new_position = parent_orientation.apply(bone_len) + joint_position[parent_idx]
        joint_position[joint_idx] = joint_new_position

    return joint_position, joint_orientations

def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path_index, path_joint_name, forward_path, inverse_path = meta_data.get_path_from_root_to_end()

    # 使用scipy求Jacobian矩阵
    def residuals(new_orientations):
        # 更新关节朝向
        for i, idx in enumerate(forward_path):
            joint_orientations[idx] = new_orientations[i]
        
        fk_positions, _ = forward_kinematics(meta_data.joint_name, meta_data.joint_parent, joint_orientations, meta_data.joint_initial_position, forward_path)
        
        end_effector_pos = fk_positions[meta_data.joint_name.index(meta_data.end_joint)]
        return end_effector_pos - target_pose
    
    
    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations