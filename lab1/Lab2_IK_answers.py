import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
from scipy.optimize._numdiff import approx_derivative

# [0.6036, 1.2825, 0.0000]
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
        
        return rx @ ry @ rz

def create_virtual_joint_path(meta_data, joint_offsets: np.ndarray, joint_local_rotations: np.ndarray, root_joint_orientation):
    """
    创建虚拟链条，以ik_joint_path[0]为root节点，计算local_rot
    处理IK路径中可能包含逆向连接的情况（如从脚→腰→手）
    
    输出：
        virtual_offsets: 虚拟链条中相邻节点间的偏移
        virtual_local_rots: 虚拟链条中每个节点相对于前一个节点的局部旋转
    """
    ik_joint_path, _, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_parents = meta_data.joint_parent
    
    virtual_offsets = []
    virtual_local_eulers = []

    # 虚拟链条以ik_joint_path[0]为起点
    for i in range(len(ik_joint_path)):
        if i == 0:
            # 根节点的偏移为0
            virtual_offsets.append(np.zeros(3))
            # 根节点的局部旋转就是其全局旋转
            virtual_local_eulers.append(root_joint_orientation.as_euler("XYZ"))
        else:
            curr_idx = ik_joint_path[i]
            prev_idx = ik_joint_path[i - 1]
            
            if joint_parents[curr_idx] == prev_idx:
                # 正向：curr是prev的子节点
                virtual_offsets.append(joint_offsets[curr_idx].copy())
                virtual_local_eulers.append(joint_local_rotations[curr_idx].as_euler("XYZ"))
            else:
                # 逆向：curr是prev的父节点
                virtual_offsets.append(-joint_offsets[prev_idx].copy())
                # 虚拟局部旋转是实际旋转的逆
                prev_rot = R.from_quat(joint_local_rotations[prev_idx].copy())
                virtual_local_eulers.append(prev_rot.inv().as_euler("XYZ"))
    
    return virtual_offsets, virtual_local_eulers


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    ik_joint_path, _, _, _ = meta_data.get_path_from_root_to_end()
    joint_parents = meta_data.joint_parent
    
    original_orientations = joint_orientations.copy()
    # 存储所有关节的局部旋转
    local_rotations = [] 
    for idx in range(len(joint_positions)):
        p = joint_parents[idx]
        curr_q = R.from_quat(original_orientations[idx])
        if p == -1:
            local_rotations.append(curr_q)
        else:
            parent_q = R.from_quat(original_orientations[p])
            local_rotations.append(parent_q.inv() * curr_q)

    # 计算所有关节的偏移量
    joint_offsets = []
    for idx in range(len(joint_positions)):
        if idx == 0:
            joint_offsets.append(np.zeros(3))
        else:
            p = joint_parents[idx]
            joint_offsets.append(meta_data.joint_initial_position[idx] - meta_data.joint_initial_position[p])
    chain_offsets, chain_local_eulers = create_virtual_joint_path(meta_data, joint_offsets, local_rotations, R.from_quat(joint_orientations[ik_joint_path[0]]))

    chain_offsets_t = torch.tensor(np.array(chain_offsets), dtype=torch.float32)
    chain_local_eulers_t = torch.tensor(np.array(chain_local_eulers), dtype=torch.float32, requires_grad=True)
    target_pose_t = torch.tensor(target_pose, dtype=torch.float32)

    max_iter = 30
    alpha = 0.01
    threshold = 1e-4
    optimizer = torch.optim.Adam([chain_local_eulers_t], lr=alpha)
    for epoch in range(max_iter):
        optimizer.zero_grad()
        root_idx = ik_joint_path[0]
        curr_global_pos = torch.tensor(joint_positions[root_idx], dtype=torch.float32)
        curr_global_rot = euler_to_mat(chain_local_eulers_t[0])

        # 跳过根节点，计算效应器位置
        for ik_idx in range(1, len(ik_joint_path)):
            idx = ik_joint_path[ik_idx]
            r_local_rot = euler_to_mat(chain_local_eulers_t[ik_idx])
            curr_global_pos = curr_global_pos + curr_global_rot @ chain_offsets_t[ik_idx]
            curr_global_rot = curr_global_rot @ r_local_rot
        
        error = torch.norm(curr_global_pos - target_pose_t)
        if error < threshold:
            print(f"error{error} < threshold{threshold}")
        print(f"End Effector Pos:{str(curr_global_pos)}, error:{str(error)}.")
        error.backward()
        optimizer.step()

    # 重建节点
    final_eulers = chain_local_eulers_t.detach().cpu().numpy()
    
    root_idx = ik_joint_path[0]
    
    root_local_quat = R.from_euler("XYZ", final_eulers[0], degrees=False)
    joint_orientations[root_idx] = root_local_quat.as_quat()
    curr_pos = joint_positions[root_idx].copy()
    curr_orient = root_local_quat
    
    # 遍历虚拟链条，从第1个节点开始
    for i in range(1, len(ik_joint_path)):
        curr_idx = ik_joint_path[i]
        prev_idx = ik_joint_path[i - 1]
        
        # 获取虚拟链条上的局部旋转和偏移
        local_euler = final_eulers[i]
        local_quat = R.from_euler("XYZ", local_euler, degrees=False)
        offset = chain_offsets[i]
        
        curr_pos = curr_pos + curr_orient.apply(offset)
        curr_orient = curr_orient * local_quat
        
        joint_positions[curr_idx] = curr_pos
        joint_orientations[curr_idx] = curr_orient.as_quat()
    
    # 第二步：更新所有非IK链上的节点
    for idx in range(len(joint_positions)):
        p = joint_parents[idx]
        
        if p == -1:
            continue  # Root已经处理过
        if idx in ik_joint_path:
            continue  # IK链上的节点已经处理过
        
        # 非IK链节点：保持Local不变，跟随父节点
        parent_pos = joint_positions[p]
        parent_rot = R.from_quat(joint_orientations[p])
        local_r = local_rotations[idx]
        
        joint_orientations[idx] = (parent_rot * local_r).as_quat()
        offset = meta_data.joint_initial_position[idx] - meta_data.joint_initial_position[p]
        joint_positions[idx] = parent_pos + parent_rot.apply(offset)

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


def part1_inverse_kinematics_gradient(meta_data, joint_positions, joint_orientations, target_pose, max_rad = 0.1):
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
    使用两骨骼IK算法
    """
    # Two Bone IK
    ik_joint_path, _, _, _ = meta_data.get_path_from_root_to_end()
    root_id = ik_joint_path[0]
    mid_id = ik_joint_path[1]
    effector_id = ik_joint_path[2]

    pos_target = np.array([joint_positions[0][0] + relative_x, target_height, joint_positions[0][2] + relative_z])
    pos_root = joint_positions[root_id].copy()
    pos_mid = joint_positions[mid_id].copy()
    pos_effector = joint_positions[effector_id].copy()

    # 计算骨骼长度
    root_bone_len = np.linalg.norm(pos_mid - pos_root)
    mid_bone_len = np.linalg.norm(pos_effector - pos_mid)
    
    # 从root指向target的向量
    vec_root_target = pos_target - pos_root
    root_target_len = np.linalg.norm(vec_root_target)
    
    max_len = root_bone_len + mid_bone_len
    eps = 1e-6
    
    if root_target_len > max_len - eps:
        # 如果目标不可达，拉伸到最大长度的位置
        if root_target_len > eps:
            pos_target = pos_root + (vec_root_target / root_target_len) * (max_len - eps)
            vec_root_target = pos_target - pos_root
            root_target_len = max_len - eps
        else:
            root_target_len = eps
    elif root_target_len < eps:
        # 避免重合导致的除零错误
        root_target_len = eps

    root_effector_len = np.linalg.norm(pos_effector - pos_root)
    
    # Pole Vector约束
    pole_vector = np.array([0., -1., 0.])
    
    # 计算mid节点旋转
    cos_mid_rad = (root_bone_len ** 2 + mid_bone_len ** 2 - root_target_len ** 2) / (2 * root_bone_len * mid_bone_len)
    cos_mid_rad = np.clip(cos_mid_rad, -1.0, 1.0)
    mid_rad = np.arccos(cos_mid_rad)
    
    cos_cur_rad = (root_bone_len ** 2 + mid_bone_len ** 2 - root_effector_len ** 2) / (2 * root_bone_len * mid_bone_len)
    cos_cur_rad = np.clip(cos_cur_rad, -1.0, 1.0)
    cur_rad = np.arccos(cos_cur_rad)
    
    delta = mid_rad - cur_rad
    
    # 使用pole vector作为旋转轴约束
    # 旋转轴应该是root指向target方向与pole vector的叉积
    root_target_dir = vec_root_target / root_target_len
    rotation_axis = np.cross(root_target_dir, pole_vector)
    rotation_axis_len = np.linalg.norm(rotation_axis)
    
    if rotation_axis_len > eps:
        rotation_axis = rotation_axis / rotation_axis_len
        mid_rot = R.from_rotvec(delta * rotation_axis)
        
        joint_orientations[mid_id] = (mid_rot * R.from_quat(joint_orientations[mid_id])).as_quat()
        joint_orientations[effector_id] = (mid_rot * R.from_quat(joint_orientations[effector_id])).as_quat()
        
        effector_offset = pos_effector - pos_mid
        pos_effector = pos_mid + mid_rot.apply(effector_offset)
    
    # 旋转root节点，对齐target方向
    vec_root_effector = pos_effector - pos_root
    vec_root_effector_norm = vec_root_effector / (np.linalg.norm(vec_root_effector) + eps)
    vec_root_target_norm = vec_root_target / root_target_len
    
    # 计算旋转角度和轴
    cos_angle = np.clip(np.dot(vec_root_effector_norm, vec_root_target_norm), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    axis = np.cross(vec_root_effector_norm, vec_root_target_norm)
    axis_len = np.linalg.norm(axis)
    
    if axis_len > eps and angle > eps:
        axis = axis / axis_len
        root_rot = R.from_rotvec(axis * angle)
        
        joint_orientations[root_id] = (root_rot * R.from_quat(joint_orientations[root_id])).as_quat()
        joint_orientations[mid_id] = (root_rot * R.from_quat(joint_orientations[mid_id])).as_quat()
        joint_orientations[effector_id] = (root_rot * R.from_quat(joint_orientations[effector_id])).as_quat()
        
        joint_positions[mid_id] = pos_root + root_rot.apply(pos_mid - pos_root)
        joint_positions[effector_id] = pos_root + root_rot.apply(pos_effector - pos_root)

    else:
        joint_positions[effector_id] = pos_target.copy()
    
    effector_rot = R.from_quat(joint_orientations[effector_id])
    for i in range(len(joint_positions)):
        if meta_data.joint_parent[i] == effector_id:
            offset = meta_data.joint_initial_position[i] - meta_data.joint_initial_position[effector_id]
            joint_positions[i] = joint_positions[effector_id] + effector_rot.apply(offset)
            
            # 旋转也跟随
            joint_orientations[i] = joint_orientations[effector_id]
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    joint_positions, joint_orientations = (meta_data, joint_positions, joint_orientations, left_target_pose)
    # joint_positions, joint_orientations = part1_inverse_kinematics_CCD(meta_data, joint_positions, joint_orientations, right_target_pose)
    return joint_positions, joint_orientations