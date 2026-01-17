import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_stack = []
    joint_name = []
    joint_parent = []
    joint_offset = []
    current_joint = -1

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        is_hierarchy = False
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('HIERARCHY'):
                is_hierarchy = True
                continue
            if is_hierarchy:
                if line.startswith('MOTION'):
                    print("Finished reading hierarchy.")
                    break

                if line.startswith('ROOT') or line.startswith('JOINT'):
                    name = line.split()[1]
                    joint_name.append(name)

                    parent = joint_stack[-1] if joint_stack else -1
                    joint_parent.append(parent)

                    current_joint = len(joint_name) - 1
                    joint_stack.append(current_joint)

                elif line.startswith('End Site'):
                    name = joint_name[joint_stack[-1]] + '_end'
                    joint_name.append(name)
                    joint_parent.append(joint_stack[-1])

                    current_joint = len(joint_name) - 1
                    joint_stack.append(current_joint)

                elif line.startswith('OFFSET'):
                    offset_values = np.array(line.split()[1:4], dtype=float)
                    joint_offset.append(offset_values)

                elif line.startswith('}'):
                    joint_stack.pop()
    
    joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    frame_motion_data = motion_data[frame_id]
    joint_positions = np.zeros((len(joint_name), 3))
    joint_orientations = np.zeros((len(joint_name), 4))

    channel_index = 0

    for i in range(len(joint_name)):
        if joint_parent[i] == -1:
            # 处理根节点
            joint_orientations[i] = R.from_euler('XYZ', frame_motion_data[channel_index+3:channel_index+6], degrees=True).as_quat()
            joint_positions[i] = frame_motion_data[channel_index:channel_index+3]
            channel_index += 6

        else:
            parent_index = joint_parent[i]
            local_offset = joint_offset[i]
            if joint_name[i].endswith('_end'):
                local_rotation = R.from_quat([0, 0, 0, 1]).as_quat()
            else:
                local_rotation = R.from_euler('XYZ', frame_motion_data[channel_index:channel_index+3], degrees=True).as_quat()
                channel_index += 3
            
            # 计算全局位置和方向
            parent_orientation = joint_orientations[parent_index]
            joint_positions[i] = joint_positions[parent_index] + R.from_quat(parent_orientation).apply(local_offset)
            joint_orientations[i] = (R.from_quat(parent_orientation) * R.from_quat(local_rotation)).as_quat()

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    # 读取骨骼信息
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)

    # 建立从A-pose到T-pose的关节名称映射，_end关节需要跳过正确处理Channel数
    T_joint_name_to_joint_index = {}
    A_joint_name_to_joint_index = {}
    index = 0
    for name in T_joint_name:
        if not name.endswith('_end'):
            T_joint_name_to_joint_index[name] = index
            index += 1
    index = 0
    for name in A_joint_name:
        if not name.endswith('_end'):
            A_joint_name_to_joint_index[name] = index
            index += 1

    A_motion_data = load_motion_data(A_pose_bvh_path)
    num_frames = A_motion_data.shape[0]
    motion_data = np.zeros_like(A_motion_data)
    
    # 逐帧A-pose中骨骼数据到T-pose骨骼结构中
    for i in range(num_frames):
        A_frame_motion = A_motion_data[i]
        T_frame_motion = np.zeros_like(A_frame_motion)

        for j, name in enumerate(A_joint_name_to_joint_index.keys()):
            T_joint_index = T_joint_name_to_joint_index.get(name)
            A_joint_index = A_joint_name_to_joint_index.get(name)

            if (T_joint_index is None) or (A_joint_index is None):
                raise ValueError(f"Joint name {name} not found in BVH files.")
            
            r_original = R.from_euler('XYZ', A_frame_motion[3 + A_joint_index*3 : 6 + (A_joint_index)*3], degrees=True)
            r_quaternion = r_original.as_quat()
            if name == 'lShoulder':
                T_frame_motion[3 + 3 * T_joint_index:6 + 3 * T_joint_index] = R.from_quat(R.from_euler('XYZ', [0, 0, 45], degrees=True).as_quat() * r_quaternion).as_euler('XYZ', degrees=True)
            elif name == 'rShoulder':
                T_frame_motion[3 + 3 * T_joint_index:6 + 3 * T_joint_index]= R.from_quat(R.from_euler('XYZ', [0, 0, -45], degrees=True).as_quat() * r_quaternion).as_euler('XYZ', degrees=True)
            elif T_joint_index == 0:
                T_frame_motion[0:6] = A_frame_motion[0:6]
            else:
                T_frame_motion[3 + 3 * T_joint_index:6 + 3 * T_joint_index] = r_original.as_euler('XYZ', degrees=True)

        motion_data[i] = T_frame_motion

    return motion_data
