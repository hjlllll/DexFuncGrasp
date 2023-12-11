# Annotation system

This folder is for our Annotation system, we use TeachNet mapping human hand to ShadowHand and collect functional dexterous hand grasp. Other dexterous hands collection which use directly angle mapping from ShadowHand are also provided.

## Dependencies

### HardWare
follow the realsense website and install realsense

```bash
two RGB cameras ===== our frame_shape = [720, 1280]
one realsense camera ==== we use Inter SR305

```

### Common Packages

```bash
conda create -n annotation python==3.8.13
conda activate annotation

# here install pytorch with cuda

```

### Grasp pose collection

First, create a folder, for example, named ***/Grasp_Pose ***.

Then, run, which **--idx** means the id of category, and -**-instance** means which object to be grasped, **--cam_1** and **--cam_2** means the ids of them:

```bash
python shadow_dataset_human_shadow_add_issacgym_system_pytorch3d_mesh_new_dataset.py --idx 0 --instance 0 --cam_1 6 --cam_2 4
```

Then the grasp pose are saved in dir ***Grasp_Pose/***.

#### Using IsaacGym to verify meanwhile.

We read the grasp pose file from ***Grasp_Pose/***. and sent to IsaacGym to verify at the same time, success grasps will be saved in dir ***Refine_Pose/***.

```bash
cd..
cd IsaacGym-DexFuncGrasp
python grasp_gym_runtime_white_new_data.py --pipeline cpu --grasp_mode dynamic --idx 0 --instance 0
```