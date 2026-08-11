import argparse

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

XML = "src/assets/robots/unitree_g1/xmls/g1.xml"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Convert a GMR-retargeted G1 motion (root_pos/root_rot/dof_pos, "
      "GMR's native fps) into mjlab's tracking npz format (joint_pos/vel, "
      "body_pos/quat/lin_vel/ang_vel_w, resampled to the env control rate)."
  )
  parser.add_argument(
      "--src", required=True, help="Path to the GMR output npz (root_pos, root_rot xyzw, dof_pos, fps)."
  )
  parser.add_argument(
      "--dst", required=True, help="Path to write the converted mjlab tracking npz to."
  )
  parser.add_argument(
      "--output-fps",
      type=int,
      default=50,
      help="Must match the env control rate (1 / (sim.timestep * decimation)); default 50 for G1 tracking.",
  )
  return parser.parse_args()


args = parse_args()
SRC = args.src
DST = args.dst
OUTPUT_FPS = args.output_fps

BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

src = np.load(SRC)
input_fps = int(src["fps"])
root_pos_in = src["root_pos"].astype(np.float64)
root_rot_in = src["root_rot"].astype(np.float64)  # xyzw
joint_pos_in = src["dof_pos"].astype(np.float64)

# --- Resample from input_fps (GMR retarget rate) to OUTPUT_FPS (env control rate). ---
# MotionCommand advances one motion frame per control step, so the stored frame
# rate must equal the control rate or the motion plays back at the wrong speed.
n_in = joint_pos_in.shape[0]
duration = (n_in - 1) / input_fps
t_in = np.arange(n_in) / input_fps
t_out = np.arange(0, duration, 1.0 / OUTPUT_FPS)

root_pos = np.stack(
    [np.interp(t_out, t_in, root_pos_in[:, i]) for i in range(3)], axis=1
)
joint_pos = np.stack(
    [np.interp(t_out, t_in, joint_pos_in[:, i]) for i in range(joint_pos_in.shape[1])],
    axis=1,
)
slerp = Slerp(t_in, R.from_quat(root_rot_in))
root_rot = slerp(np.clip(t_out, t_in[0], t_in[-1])).as_quat()  # xyzw

fps = OUTPUT_FPS

model = mujoco.MjModel.from_xml_path(XML)
data = mujoco.MjData(model)

print("nq:", model.nq)
print("nv:", model.nv)
print(f"resampled {n_in} frames @ {input_fps}fps -> {joint_pos.shape[0]} frames @ {fps}fps")

body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in BODY_NAMES]
print("body ids:", body_ids)

T = joint_pos.shape[0]
B = model.nbody

body_pos_w_full = np.zeros((T, B, 3), dtype=np.float32)
body_quat_w_full = np.zeros((T, B, 4), dtype=np.float32)

joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) * fps
joint_vel[0] = joint_vel[1]

# qpos for floating-base humanoid:
# [root xyz 3] + [root quat wxyz 4] + [29 joint angles]
for t in range(T):
    qpos = np.zeros(model.nq, dtype=np.float64)

    qpos[0:3] = root_pos[t]

    quat = root_rot[t]
    # GMR/root_rot is xyzw. Convert to wxyz for MuJoCo.
    if abs(np.linalg.norm(quat) - 1.0) > 1e-2:
        quat = quat / (np.linalg.norm(quat) + 1e-8)

    qpos[3:7] = np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)

    n = min(joint_pos.shape[1], model.nq - 7)
    qpos[7:7+n] = joint_pos[t, :n]

    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    body_pos_w_full[t] = data.xpos.astype(np.float32)
    # MuJoCo xquat is wxyz
    body_quat_w_full[t] = data.xquat.astype(np.float32)

body_lin_vel_w_full = np.zeros_like(body_pos_w_full)
body_lin_vel_w_full[1:] = (body_pos_w_full[1:] - body_pos_w_full[:-1]) * fps
body_lin_vel_w_full[0] = body_lin_vel_w_full[1]

body_ang_vel_w_full = np.zeros_like(body_pos_w_full)
for b in range(B):
    xyzw_b = body_quat_w_full[:, b, [1, 2, 3, 0]].astype(np.float64)
    rot_b = R.from_quat(xyzw_b)
    rel = rot_b[1:] * rot_b[:-1].inv()
    body_ang_vel_w_full[1:, b, :] = rel.as_rotvec() * fps
body_ang_vel_w_full[0] = body_ang_vel_w_full[1]

# Drop the MuJoCo "world" body (index 0). MotionLoader indexes bodies using
# mjlab's entity-relative convention (Entity.body_names excludes world), so the
# saved arrays must match that convention -- same layout as dance1_subject2.npz.
body_pos_w = body_pos_w_full[:, 1:]
body_quat_w = body_quat_w_full[:, 1:]
body_lin_vel_w = body_lin_vel_w_full[:, 1:]
body_ang_vel_w = body_ang_vel_w_full[:, 1:]

np.savez(
    DST,
    fps=np.array([fps], dtype=np.float64),
    joint_pos=joint_pos.astype(np.float32),
    joint_vel=joint_vel.astype(np.float32),
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
)

print("Saved:", DST)
print("joint_pos:", joint_pos.shape)
print("body_pos_w:", body_pos_w.shape, "(world body dropped)")
print("body_quat_w:", body_quat_w.shape)
