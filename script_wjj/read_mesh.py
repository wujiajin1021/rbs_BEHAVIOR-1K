import h5py
import numpy as np
from pxr import Usd, UsdGeom

h5_path = "/home/ps/BEHAVIOR-1K/episode_00000010_new.h5"
target_prim_path = "/World/scene_0/controllable__r1pro__robot_r1/left_arm_link5/visuals"

# 1) 从 H5 找到 usd_path / link_name / prim_name
with h5py.File(h5_path, "r") as f:
    prims = f["data"]["demo_0"]["prims"]
    usd_path = link_name = prim_name = None
    for k in prims.keys():
        g = prims[k]
        if str(g.attrs.get("prim_path", "")) == target_prim_path:
            usd_path = str(g.attrs["usd_path"])
            link_name = str(g.attrs["link_name"]).split(":")[-1]   # robot_r1:left_arm_link5 -> left_arm_link5
            prim_name = str(g.attrs["prim_name"])                  # visuals
            break

if usd_path is None:
    raise RuntimeError("target prim not found in h5")

# 2) 打开 USD，定位到 .../left_arm_link5/visuals 子树
stage = Usd.Stage.Open(usd_path)
root = None
suffix = f"/{link_name}/{prim_name}"
for p in stage.Traverse():
    if p.GetPath().pathString.endswith(suffix):
        root = p
        break
if root is None:
    raise RuntimeError(f"cannot find {suffix} in {usd_path}")

# 3) 提取该子树下所有 Mesh 的顶点和面
all_mesh = []
for p in Usd.PrimRange(root):
    if p.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(p)
        V = np.asarray(m.GetPointsAttr().Get(), dtype=np.float32)             # (Nv, 3)
        F_counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
        F_idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        all_mesh.append({
            "mesh_path": p.GetPath().pathString,
            "vertices": V,
            "face_vertex_counts": F_counts,
            "face_vertex_indices": F_idx,
        })

print("mesh num:", len(all_mesh))
for i, m in enumerate(all_mesh[:5]):
    print(i, m["mesh_path"], m["vertices"].shape, m["face_vertex_counts"].shape)