import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
from scipy.optimize._numdiff import approx_derivative

def part1_inverse_kinematics_quat(meta_data, joint_positions, joint_orientations, target_pose):
    ik_joint_path, _, _, _ = meta_data.get_path_from_root_to_end()
    joint_parents = meta_data.joint_parent

    joint_offsets = []
    for idx in range(len(joint_positions)):
        if idx == 0:
            joint_offsets.append(np.zeros(3))
        else:
            p = joint_parents[idx]
            joint_offsets.append(meta_data.joint_initial_position[idx] - meta_data.joint_initial_position[p])
    joint_offsets_t = torch.tensor(np.array(joint_offsets), dtype=torch.float32)

    local_quats_data = []
    for ik_idx in range(len(ik_joint_path)):
        idx = ik_joint_path[ik_idx]
        parent_idx = joint_parents[idx]
        curr_q = R.from_quat(joint_orientations[idx])
        
        if parent_idx == -1:
            local_quats_data.append(curr_q.as_quat())
        else:
            parent_q_inv = R.from_quat(joint_orientations[parent_idx]).inv()
            local_quats_data.append((parent_q_inv * curr_q).as_quat())
            
    local_quats_var = torch.tensor(np.array(local_quats_data), dtype=torch.float32, requires_grad=True)
    target_pose_t = torch.tensor(target_pose, dtype=torch.float32)

    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([x, y, z, w])

    # 辅助函数：四元数旋转向量 (Batch支持)
    def quat_apply(q, v):
        xyz = q[:3]
        w = q[3]
        t = 2 * torch.cross(xyz, v)
        return v + w * t + torch.cross(xyz, t)
        
    def quat_inv(q):
        # 四元数逆（单位四元数的逆等于共轭）：q_inv = [-x, -y, -z, w]
        q_inv = q.clone()
        q_inv[0] = -q_inv[0]
        q_inv[1] = -q_inv[1]
        q_inv[2] = -q_inv[2]
        return q_inv

    optimizer = torch.optim.Adam([local_quats_var], lr=0.1) # 步长可以大一点
    
    root_idx = ik_joint_path[0]
    if joint_parents[root_idx] == -1:
        root_parent_global_q = torch.tensor([0, 0, 0, 1], dtype=torch.float32) # Identity
    else:
        root_parent_global_q = torch.tensor(joint_orientations[joint_parents[root_idx]], dtype=torch.float32)

    for epoch in range(100):
        optimizer.zero_grad()
        
        # 在每次 Forward 前，对优化变量进行归一化！
        normalized_quats = torch.nn.functional.normalize(local_quats_var, p=2, dim=1)
        
        root_local_q = normalized_quats[0]
        curr_global_q = quat_mul(root_parent_global_q, root_local_q)
        curr_global_pos = torch.tensor(joint_positions[root_idx], dtype=torch.float32)
        
        for i in range(len(ik_joint_path) - 1):
            idx = ik_joint_path[i]
            next_idx = ik_joint_path[i + 1]
            
            if joint_parents[idx] == next_idx:
                # 逆向 (Child -> Parent)
                local_q = normalized_quats[i] # Current node local
                next_global_q = quat_mul(curr_global_q, quat_inv(local_q))
                
                offset = joint_offsets_t[idx]
                next_global_pos = curr_global_pos - quat_apply(next_global_q, offset)
                
            else:
                # 正向 (Parent -> Child)
                local_q = normalized_quats[i+1] 
                next_global_q = quat_mul(curr_global_q, local_q)
                
                offset = joint_offsets_t[next_idx]
                next_global_pos = curr_global_pos + quat_apply(curr_global_q, offset)
            
            curr_global_pos = next_global_pos
            curr_global_q = next_global_q
            
        loss = torch.norm(curr_global_pos - target_pose_t)
        loss.backward()
        optimizer.step()
        
        if loss.item() < 1e-4:
            break
    final_quats = torch.nn.functional.normalize(local_quats_var, p=2, dim=1).detach().numpy()
    
    # Update Root Global
    if joint_parents[root_idx] == -1:
        joint_orientations[root_idx] = final_quats[0]
    else:
        p_q = R.from_quat(joint_orientations[joint_parents[root_idx]])
        l_q = R.from_quat(final_quats[0])
        joint_orientations[root_idx] = (p_q * l_q).as_quat()
        
    # 后续传播逻辑同之前的代码...
    
    return joint_positions, joint_orientations

def part1_inverse_kinematics_CCD(meta_data, joint_positions, joint_orientations, target_pose, max_angle = 0.1):
    """
    CCD (Cyclic Coordinate Descent) 方法计算逆运动学
    输入: 
        meta_data: 包含关节信息的元数据
        joint_positions: 当前的关节位置，shape为(M, 3)
        joint_orientations: 当前的关节朝向，shape为(M, 4)，四元数形式
        target_pose: 目标位置，shape为(3,)
    输出:
        joint_positions: 计算得到的关节位置
        joint_orientations: 计算得到的关节朝向
    """
    ik_joint_path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_parents = meta_data.joint_parent
    
    # 计算所有关节的偏移量
    joint_offsets = []
    for idx in range(len(joint_positions)):
        if idx == 0:
            joint_offsets.append(np.zeros(3))
        else:
            p = joint_parents[idx]
            joint_offsets.append(meta_data.joint_initial_position[idx] - meta_data.joint_initial_position[p])
    end_effector_idx = ik_joint_path[-1]

    # 计算所有关节的旋转量
    joint_local_oriens = []
    for idx in range(len(joint_orientations)):
        joint_orientation = R.from_quat(joint_orientations[idx])
        if idx == 0:
            joint_local_oriens.append(joint_orientation.as_quat())
        else:
            p = joint_parents[idx]
            p_orientation_inv = R.from_quat(joint_orientations[p]).inv()
            joint_local_oriens.append((p_orientation_inv * joint_orientation).as_quat())

    max_iterations = 20
    threshold = 1e-4

    curr_positions = joint_positions.copy()
    curr_orientations = joint_orientations.copy()
    # CCD主循环
    for epoch in range(max_iterations):
        
        # IK链迭代，
        for i in range(len(ik_joint_path) - 2, -1, -1):
            end_effector_pos = curr_positions[end_effector_idx]

            joint_idx = ik_joint_path[i]
            joint_pos = curr_positions[joint_idx]
            vec_to_effector = end_effector_pos - joint_pos
            vec_to_target = target_pose - joint_pos
            
            len_to_effector = np.linalg.norm(vec_to_effector)
            len_to_target = np.linalg.norm(vec_to_target)
            
            if len_to_effector < 1e-6 or len_to_target < 1e-6:
                print(f"Zero Division：len to effector:{len_to_effector} or target:{len_to_target}")
                continue
            
            vec_to_effector_norm = vec_to_effector / len_to_effector
            vec_to_target_norm = vec_to_target / len_to_target
            
            # 计算旋转轴和旋转角
            rotation_axis = np.cross(vec_to_effector_norm, vec_to_target_norm)
            rotation_axis_len = np.linalg.norm(rotation_axis)
            if rotation_axis_len < 1e-6:
                print(f"Zero Division：rotation_axis_len")
                continue
            rotation_axis = rotation_axis / rotation_axis_len
            rotation_angle = np.arccos(np.clip(np.dot(vec_to_effector_norm, vec_to_target_norm), -1.0, 1.0))
            
            # 限制旋转弧度
            rotation_angle = np.clip(rotation_angle, -max_angle, max_angle)
            joint_rotation = R.from_rotvec(rotation_axis * rotation_angle)
            
            curr_global_rot = R.from_quat(curr_orientations[joint_idx])
            curr_orientations[joint_idx] = (joint_rotation * curr_global_rot).as_quat()
            joint_local_oriens[joint_idx] = (joint_rotation * R.from_quat(joint_local_oriens[joint_idx])).as_quat()
            # 更新后续节点，后续节点均绕关节旋转
            for j in range(i + 1, len(ik_joint_path)):
                child_idx = ik_joint_path[j]
                prev_idx = ik_joint_path[j - 1]
                
                # 判断路径方向：正向还是逆向
                if joint_parents[child_idx] == prev_idx:
                    # 正向：
                    parent_orient = R.from_quat(curr_orientations[prev_idx])
                    parent_pos = curr_positions[prev_idx]
                    off = joint_offsets[child_idx]
                    curr_positions[child_idx] = parent_pos + parent_orient.apply(off)
                    curr_orientations[child_idx] = (parent_orient * R.from_quat(joint_local_oriens[child_idx])).as_quat()
                else:
                    # 逆向：prev_idx是子节点，child_idx是其父节点
                    pass
                    prev_orient = R.from_quat(curr_orientations[prev_idx])
                    prev_pos = curr_positions[prev_idx]
                    off = joint_offsets[prev_idx]  # 获取prev相对于其父节点的偏移
                    
                    # 反向推导：parent_global = child_global * child_local.inv()
                    child_orient = prev_orient * R.from_quat(joint_local_oriens[prev_idx]).inv()
                    curr_positions[child_idx] = prev_pos - child_orient.apply(off)
                    curr_orientations[child_idx] = child_orient.as_quat()
                
        
        end_effector_pos = curr_positions[end_effector_idx]
        error = np.linalg.norm(end_effector_pos - target_pose)
        
        print(f"epcoh：{epoch}, end_effector_pos:{end_effector_pos}")
        if error < threshold:
            print(f"error:{error} < threshold:{threshold}")
            break
    
    # 更新所有关节的位置和朝向
    joint_positions = curr_positions
    joint_orientations = curr_orientations
    
    for idx in range(len(meta_data.joint_name)):
        if idx in ik_joint_path or idx == 0:
            continue
        parent_idx = joint_parents[idx]
        parent_orient = R.from_quat(joint_orientations[parent_idx])
        off = joint_offsets[idx]
        parent_pos = joint_positions[parent_idx]
        joint_positions[idx] = parent_pos + parent_orient.apply(off)
        joint_orientations[idx] = (R.from_quat(joint_local_oriens[idx]) * parent_orient).as_quat()

    return joint_positions, joint_orientations


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose, max_rad = 0.1):
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
    ik_joint_path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    # 计算Offsets
    joint_parents = meta_data.joint_parent
    joint_offsets = []
    for idx in range(len(joint_positions)):
        if idx == 0:
            joint_offsets.append(np.zeros(3))
            continue
        joint_offsets.append(meta_data.joint_initial_position[idx] - meta_data.joint_initial_position[joint_parents[idx]])
    joint_offsets_t = torch.tensor(np.array(joint_offsets), dtype=torch.float32)

    # 计算IK链局部旋转向量矩阵
    local_eulers = []
    for ik_idx in range(len(ik_joint_path)):
        joint_idx = ik_joint_path[ik_idx]
        parent_idx = joint_parents[joint_idx]
        curr_q = R.from_quat(joint_orientations[joint_idx])
        if parent_idx == -1:
            local_eulers.append(curr_q.as_euler("XYZ", degrees=False))
        else:
            parent_q_inv = R.from_quat(joint_orientations[parent_idx]).inv()
            local_eulers.append((parent_q_inv * curr_q).as_euler("XYZ", degrees=False))            
    local_eulers_t = torch.tensor(np.array(local_eulers), dtype=torch.float32, requires_grad=True)
    target_pose_t = torch.tensor(target_pose, dtype=torch.float32)

    def euler_to_mat(angle_vec):
        x, y, z = angle_vec[0], angle_vec[1], angle_vec[2]
        cx, sx = torch.cos(x), torch.sin(x)
        cy, sy = torch.cos(y), torch.sin(y)
        cz, sz = torch.cos(z), torch.sin(z)
        
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        
        rx = torch.stack([torch.stack([ones, zeros, zeros]), torch.stack([zeros, cx, -sx]), torch.stack([zeros, sx, cx])])
        ry = torch.stack([torch.stack([cy, zeros, sy]), torch.stack([zeros, ones, zeros]), torch.stack([-sy, zeros, cy])])
        rz = torch.stack([torch.stack([cz, -sz, zeros]), torch.stack([sz, cz, zeros]), torch.stack([zeros, zeros, ones])])
        
        return rz @ ry @ rx

    max_iter = 10
    alpha = 0.1
    threshold = 1e-4
    # Adam优化器实现梯度下降
    optimizer = torch.optim.Adam([local_eulers_t], lr=alpha)
    for epoch in range(max_iter):
        optimizer.zero_grad()
        root_idx = ik_joint_path[0]
        
        # 从根节点开始进行前向运动学计算
        # 根节点的位置保持不变
        curr_global_pos = torch.tensor(joint_positions[root_idx], dtype=torch.float32)
        # 根节点的全局旋转由其优化后的局部旋转决定
        curr_global_rot = euler_to_mat(local_eulers_t[0])
        
        # 从根节点开始沿IK链进行FK
        for ik_idx in range(len(ik_joint_path) - 1):
            idx = ik_joint_path[ik_idx]
            next_idx = ik_joint_path[ik_idx + 1]
            
            # 判断方向：如果下一个节点是当前节点的父节点，则为逆向；否则为正向
            if joint_parents[idx] == next_idx:
                # 逆向：从子节点指向父节点
                r_local_rot = euler_to_mat(local_eulers_t[ik_idx])
                next_global_rot = curr_global_rot @ r_local_rot.t()
                curr_offset = joint_offsets_t[idx]
                next_global_pos = curr_global_pos - next_global_rot @ curr_offset
            else:
                # 正向：从父节点指向子节点  
                r_local_rot = euler_to_mat(local_eulers_t[ik_idx + 1])
                next_global_rot = curr_global_rot @ r_local_rot
                curr_offset = joint_offsets_t[next_idx]
                next_global_pos = curr_global_pos + curr_global_rot @ curr_offset
            
            curr_global_pos = next_global_pos
            curr_global_rot = next_global_rot
        
        end_effector_pos = curr_global_pos
        print(f"第{epoch}次迭代位置{end_effector_pos}, 目标位置：{target_pose_t}")
        delta = torch.norm(end_effector_pos - target_pose_t)
        delta.backward()
        optimizer.step()

        if delta.item() < threshold:
            print(f"收敛于{epoch}次迭代")
            break

    # 更新所有关节的Pos/Rot，因为IK可能改变了局部旋转，所有Joint都要更新
    # 这里锁定了IK的Root节点位置
    final_local_eulers = local_eulers_t.detach().numpy()
    
    # 第一步：沿着IK链更新节点，从root_idx开始向末端传播（复用FK的计算逻辑）
    root_idx = ik_joint_path[0]
    joint_parents = meta_data.joint_parent
    
    # 更新root节点的朝向（位置锁定）
    root_local_quat = R.from_euler("XYZ", final_local_eulers[0], degrees=False)
    joint_orientations[root_idx] = root_local_quat.as_quat()
    
    # 从根节点开始沿IK链进行位置和朝向的更新（与FK逻辑一致）    
    curr_pos = joint_positions[root_idx].copy()
    curr_orient = root_local_quat
    
    for ik_idx in range(len(ik_joint_path) - 1):
        curr_idx = ik_joint_path[ik_idx]
        next_idx = ik_joint_path[ik_idx + 1]
        
        # 判断方向
        if joint_parents[curr_idx] == next_idx:
            # 逆向：从子节点回到父节点
            curr_local_quat = R.from_euler("XYZ", final_local_eulers[ik_idx], degrees=False)
            next_orient = curr_orient * curr_local_quat.inv()
            next_pos = curr_pos - next_orient.apply(joint_offsets[curr_idx])
        else:
            # 正向：从父节点到子节点
            next_local_quat = R.from_euler("XYZ", final_local_eulers[ik_idx + 1], degrees=False)
            next_orient = curr_orient * next_local_quat
            next_pos = curr_pos + curr_orient.apply(joint_offsets[next_idx])
        
        # 更新下一个节点
        joint_positions[next_idx] = next_pos
        joint_orientations[next_idx] = next_orient.as_quat()
        
        curr_pos = next_pos
        curr_orient = next_orient
    
    # 第二步：从IK链上的每个节点向外扩展，更新其他所有节点
    visited = set(ik_joint_path)
    queue = list(ik_joint_path)
    
    while queue:
        idx = queue.pop(0)
        parent_orient = R.from_quat(joint_orientations[idx])
        
        # 找到idx的所有子节点
        for child_idx in range(len(joint_positions)):
            if joint_parents[child_idx] == idx and child_idx not in visited:
                visited.add(child_idx)
                queue.append(child_idx)
                
                # 计算子节点的局部旋转
                curr_q = R.from_quat(joint_orientations[child_idx])
                parent_q_inv = parent_orient.inv()
                local_rot = parent_q_inv * curr_q
                
                # 更新子节点
                joint_orientations[child_idx] = (parent_orient * local_rot).as_quat()
                joint_positions[child_idx] = joint_positions[idx] + parent_orient.apply(joint_offsets[child_idx])
    
    
    # 打印手位置
    end_effector_idx = ik_joint_path[-1]
    print(f"手部位置：{joint_positions[end_effector_idx]}")
    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    root_position = joint_positions[0]
    target_pos = [root_position[0] + relative_x, target_height, root_position[2] + relative_z]
    # joint_positions, joint_orientations = part1_inverse_kinematics_CCD(meta_data, joint_positions, joint_orientations, target_pos, max_angle = 0.2)
    joint_positions, joint_orientations = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pos)
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    joint_positions, joint_orientations = (meta_data, joint_positions, joint_orientations, left_target_pose)
    # joint_positions, joint_orientations = part1_inverse_kinematics_CCD(meta_data, joint_positions, joint_orientations, right_target_pose)
    return joint_positions, joint_orientations