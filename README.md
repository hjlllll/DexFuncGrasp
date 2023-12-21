<p align="center">
    <h2 align="center">DexFuncGrasp: A Robotic Dexterous Functional Grasp Dataset constructed 
      from a Cost-Effective Real-Simulation Annotation System (AAAI2024)</h2>
    <p align="center">Jinglue Hang, Xiangbo Lin&dagger;, Tianqiang Zhu, Xuanheng Li, Rina Wu, Xiaohong Ma and Yi Sun;<br />
    Dalian University of Technology<br />
    &dagger; corresponding author<br />
    <a href='https://hjlllll.github.io/DFG/'>project page</a> 

</p>



## Contents

1. [Abstract](#abstract)

2. [Grasp pose collection](#grasp-pose-collection)

3. [Grasp Transfer for Dataset Extension](#grasp-transfer-for-dataset-extension)

4. [DFG Dataset](#dfg-dataset)

5. [DexFuncGraspNet](#dexfuncgraspnet)

6. [Simulation Experiment](#simulation-experiment)

7. [Acknowledgments](#acknowledgments)

8. [Citation](#citation)

9. [License](#license)

## Abstract
<div align=center>
<img src="pic/dataset.png" width="640px">

</div>


Robot grasp dataset is the basis of designing the robot’s grasp generation model. Compared with the building grasp dataset for Low-DOF grippers, it is harder for High-DOF dexterous robot hand. Most current datasets meet the needs of generating stable grasps, but they are not suitable for dexterous hands to complete human-like functional grasp, such as grasp the handle of a cup or pressing the button of a flashlight, so as to enable robots to complete subsequent functional manipulation action autonomously, and there is no dataset with functional grasp pose annotations at present. This paper develops a unique Cost-Effective Real-Simulation Annotation System by leveraging natural hand’s actions. The system is able to capture a functional grasp of a dexterous hand in a simulated environment assisted by human demonstration in real world. By using this system, dexterous grasp data can be collected efficiently as well as cost-effective. Finally, we construct the first dexterous functional grasp dataset with rich pose annotations. A Functional Grasp Synthesis Model is also provided to validate the effectiveness of the proposed system and dataset.


## Download 
#### If you want to complete this work, you can download these: (choose optional)
- [Isaac Gym](https://drive.google.com/file/d/1Imk8_GPQ75mYkBS76k2PgHdsZxEHoog-/view?usp=drive_link) preview 4.0 (3.0)
- [Obj_Data](https://drive.google.com/file/d/1pOJcw1g1dAuC1KwPKKP3nVGtXr39YwmN/view?usp=drive_link) 
- [VRC-Dataset](https://drive.google.com/file/d/1FJ5-4uw-7dPvpWVau7ChYrn93y0m92C6/view?usp=drive_link)
- [Pretrained VRCNET Model](https://drive.google.com/file/d/1benvNLM9HmLDlu4Y-AF3LAEn-T0LGRYk/view?usp=drive_link)
- [DFG-Dataset](https://drive.google.com/file/d/1QVZh5OMwcZJtQJOob_Q5kzr-hBODYChy/view?usp=drive_link) 
- [Pretrained DexFuncGraspNet Model](https://drive.google.com/file/d/1ytPVvy6u9KeFsI8zaoT3SaTTsKUHUi_j/view?usp=drive_link) 
- [Baseline-Results](https://drive.google.com/file/d/1fWPR87JSEWzQoTuI-zCkjdaCT95jIQvA/view?usp=drive_link)
## Enviorment
#### Three conda env
- ***annotate*** for - [Grasp pose collection](#grasp-pose-collection), [Grasp Transfer for Dataset Extension](#grasp-transfer-for-dataset-extension)
, and [DexFuncGraspNet](#dexfuncgraspnet).
- ***vrcnet*** for - [VRCNET](#vrcnet) 
    - Follow the instruction from [vrcnet-project](https://github.com/paul007pl/VRCNet) to create conda env.
- ***dexfuncgrasp*** for - [DexFuncGraspNet](#dexfuncgraspnet) - [CVAE](#cvae).
    
## Grasp pose collection

### Cost-effective Annotaion System

<div align=center>
<img src="pic/method-pipeline2.png" width="840px">
</div>

- Our Annotation system: we use [TeachNet](https://github.com/TAMS-Group/TeachNet_Teleoperation) mapping human hand to ShadowHand and collect functional dexterous hand grasp. Other dexterous hands collection which use directly angle mapping from ShadowHand are also provided.

### HardWare
- follow the realsense website and install realsense

```bash
two RGB cameras ===== our frame_shape = [720, 1280]
one realsense camera ==== we use Inter SR305

```

### Dependencies

- Ubuntu 20.04 (optional)

- Python 3.8

- PyTorch 1.10.1

- Numpy 1.22.0

- mediapipe 0.8.11

- [pytorch-kinematics](https://github.com/PKU-EPIC/DexGraspNet/tree/main/thirdparty/pytorch_kinematics/pytorch_kinematics)  0.3.0 

- [Isaac Gym](https://drive.google.com/file/d/1Imk8_GPQ75mYkBS76k2PgHdsZxEHoog-/view?usp=drive_link) preview 4.0 (3.0)

- CUDA 11.1

### Common Packages

```bash
conda create -n annotate python==3.8.13
conda activate annotate

# Install pytorch with cuda
pip install torch==1.10.1 torchvision==0.11.2 ## or using offical code from pytorch website
pip install numpy==1.22.0
cd Annotation/
cd pytorch_kinematics/ #need download from up link
pip install -e.
cd ..
pip install -r requirement.txt

# Install IsaacGym : 
# download from up link and put in to folder Annotation/
cd IsaacGym/python/
pip install -e .
export LD_LIBRARY_PATH=/home/your/path/to/anaconda3/envs/annotate/lib
```

### Process steps

- Download [Isaac Gym](https://drive.google.com/file/d/1Imk8_GPQ75mYkBS76k2PgHdsZxEHoog-/view?usp=drive_link) preview 4.0 (3.0)
    ```bash
    |-- Annotation
        |-- IsaacGym
    ```
- Download [Obj_Data](https://drive.google.com/file/d/1pOJcw1g1dAuC1KwPKKP3nVGtXr39YwmN/view?usp=drive_link) 
    ```bash
    |-- Annotation
        |-- IsaacGym
            |-- assets
                |-- urdf
                    |-- off
                        |-- Obj_Data
                        |-- Obj_Data_urdf
    ```
- Set the cameras in real as shown in the figure.

- Follow the instruction from [handpose3d](https://github.com/TemugeB/handpose3d), get the camera_paremeters folder, or use mine.

- Create a folder, for example, named ***/Grasp_Pose***.

- Run .py, which **--idx** means the id of category, and -**-instance** means which object to be grasped, **--cam_1** and **--cam_2** means the ids of them:

```bash
python shadow_dataset_human_shadow_add_issacgym_system_pytorch3d_mesh_new_dataset.py --idx 0 --instance 0 --cam_1 6 --cam_2 4
```


#### Using IsaacGym to verify meanwhile (open an another terminal at the same time).

- We read the grasp pose file from ***Grasp_Pose/***. and sent to IsaacGym to verify **at the same time**, success grasps and collected success rate will be saved in dir ***/Tink_Grasp_Transfer/Dataset/Grasps/***.

```bash
cd..
cd IsaacGym/python
python grasp_gym_runtime_white_new_data.py --pipeline cpu --grasp_mode dynamic --idx 0 --instance 0
```

- If you think this grasp is good grasp, press blank and poses can be saved, try to collect less than 30 grasps, and click **x** in isaacgym in the top right to close. The grasp pose could be saved in dir ***Grasp_Pose/***.

- After collection, unit axis for grasps in ***/Tink_Grasp_Transfer/Dataset/Grasps/*** in order to learn sdf function of each category.
```bash
python trans_unit.py 
```

- Other dexterous hand collection demo (Optional)
```bash
python shadow_dataset_human_shadow_add_issacgym_system_pytorch3d_mesh_new_dataset_multi_dexterous.py --idx 0 --instance 0 --cam_1 6 --cam_2 4
```

- Visualization 
```bash
# Put .pkl in to Annotation/visual_dict/new/
python show_data_mesh.py
```




## Grasp Transfer for Dataset Extension
<div align=center>
<img src="pic/method-transfer.png" width="740px">
</div>

### Dependencies
- [Tink](https://github.com/oakink/Tink) , this part is modified from Tink(OakInk)
- git clone https://github.com/oakink/DeepSDF_OakInk
follow the instruction and install all requirements:
The code is in C++ and has the following requirements: (using the same conda env annotate)

- [CLI11][1]
- [Pangolin][2]
- [nanoflann][3]
- [Eigen3.3.9][4]

[1]: https://github.com/CLIUtils/CLI11
[2]: https://github.com/stevenlovegrove/Pangolin
[3]: https://github.com/jlblancoc/nanoflann
[4]: https://eigen.tuxfamily.org


### Common Packages

```bash
pip install termcolor
pip install plyfile

### prepare mesh-to-sdf env
git clone https://github.com/marian42/mesh_to_sdf
cd mesh_to_sdf
pip install -e.


pip install scikit-image==0.16.2

put download packages in Transfer/third-party/
cd CLI11 # cd Pangolin/nanofl...
mkdir build
cd build
cmake ..
make -j8 
```

### Process steps
- The same process using Tink.
```bash
cd Tink_Grasp_Transfer/
python generate_sdf.py --idx 0
python train_deep_sdf.py --idx 0
python reconstruct_train.py --idx 0 --mesh_include
python tink/gen_interpolate.py --all --idx 0
python tink/cal_contact_info_shadow.py --idx 0 --tag trans
python tink/info_transform.py --idx 0 --all
python tink/pose_refine.py --idx 0 --all #--vis
```

- Or directly bash: 
```bash
sh transfer.sh
```

- Only save the success grasp and unit axis of dataset:
```bash
cd ../../../IsaacGym/python/collect_grasp/
# save the success grasp
sh run_clean.sh
# unit axis of dataset
python trans_unit_dataset_func.py

```
- You can change the grasp in to folder to make them small size
```bash
cd DexFuncGraspNet/Grasps_Dataset
python data_process_m.py
```
- Till now, the grasp dataset in folder: ***Annotation/Tink_Grasp_Transfer/Dataset/Grasps***, each grasps used for training in /0_unit_025_mug/sift/unit_mug_s009/new, which object quat are all [1 0 0 0], at same axis.


## DFG Dataset 

<div align=center>
<img src="pic/DexFuncGrasp.png" width="640px">
</div>

- We collect objects from online dataset such as OakInk, and collect grasps through steps above. we name it DFG dataset.
- Download source meshes and grasp labels for 12 categories from [DFG-Dataset](https://drive.google.com/file/d/1QVZh5OMwcZJtQJOob_Q5kzr-hBODYChy/view?usp=drive_link) dataset.
- Arrange the files as follows:
```
|-- DexFuncGraspNet
    |-- Grasps_Dataset
        |-- train
            |-- 0_unit_025_mug ##labeled objects
                |--unit_mug_s009.npy ##transferred objects
                |--unit_mug_s010.npy
                |--unit_mug_s011.npy
                | ...
                | ...
            |-- 0_unit_mug_s001
            |-- 1_unit_bowl_s101
            |-- 1_unit_bowl_s102
            |-- 4_unit_bottle12
            |-- 4_unit_bottle13
            | ...
        |-- test ###for testing

```

## DexFuncGraspNet
<div align=center>
<img src="pic/network.png" width="740px">
</div>

- As we propose this dataset, we provide the baseline method based on CVAE as shown in figure above:

### VRCNet

- First, we train the off-the-shelf [VRCNET](https://github.com/paul007pl/VRCNet) using our DFG dataset.


<div align=center>
<img src="pic/1.png" alt="input" class="img-responsive" width="35.1%" />
<img src="pic/2.png" alt="input" class="img-responsive" width="21.1%" />          
<img src="pic/3.png" alt="input" class="img-responsive" width="23.1%" />
</div>
          
- We use Pytorch3D to generate different view of partial point cloud.

    ```bash
    cd data_preprocess/
    sh process.sh
    ```
- OR Partial-Complete dataset from our DFG dataset can be download here : [VRC-Dataset](https://drive.google.com/file/d/1FJ5-4uw-7dPvpWVau7ChYrn93y0m92C6/view?usp=drive_link)
    ```bash
    |-- VRCNET-DFG
        |-- data
            |-- complete_pc
            |-- render_pc_for_completion
    ```

- Train VRCNET

    - [Pretrained VRCNET Model](https://drive.google.com/file/d/1benvNLM9HmLDlu4Y-AF3LAEn-T0LGRYk/view?usp=drive_link) for simulation is provided.
    
    ```bash
    |-- VRCNET-DFG
        |--log
            |--vrcnet_cd_debug
                |--best_cd_p_network.pth
    ```

    ```bash
    conda activate vrcnet
    ### change cfgs/vrcnet.yaml --load_model
    cd ../
    python train.py --config cfgs/vrcnet.yaml
    ### change cfgs/vrcnet.yaml --load_model
    python test.py --config cfgs/vrcnet.yaml # for test 
    ```

### CVAE

- Second, we train the CVAE grasp generation moudle.
- Dependencies
    - pytorch 1.7.1
    - [Pointnet2_Pytorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
    - open3d 0.9.0
- Common Packages

    ```bash
    conda create -n dexfuncgrasp python==3.7
    conda activate dexfuncgrasp
    
    pip install torch==1.7.1

    git clone https://github.com/erikwijmans/Pointnet2_PyTorch
    cd Pointnet2_PyTorch/pointnet2_ops_lib/
    pip install -e.

    pip install trimesh tqdm open3d==0.9.0 pyyaml easydict pyquaternion scipy matplotlib
    ```
- [Pretrained DexFuncGraspNet Model](https://drive.google.com/file/d/1ytPVvy6u9KeFsI8zaoT3SaTTsKUHUi_j/view?usp=drive_link) for simulation is provided.
    ```bash
    |-- DexFuncGraspNet
        |-- checkpoints
            |-- vae_lr_0002_bs_64_scale_1_npoints_128_radius_02_latent_size_2
                |-- latest_net.pth
    ```

    ```bash
    cd DexFuncGraspNet/
    # train cvae grasp generation net
    python train_vae.py --num_grasps_per_object 64 ###64 means batch_size(default:64)
    # generate grasps on test set
    python test.py # grasps generated in folder DexFuncGraspNet/test_result_sim [we use results from VRCNET]
    ```
### Optimize

- Third, the Refinement is using Pytorch Adam Optimizer.
    ```bash
    python refine_after_completion.py # grasps optimized in folder DexFuncGraspNet/test_result_sim_refine
    ```

### Visualize
- Put .pkl in to Annotation/visual_dict/new/

    ```bash
    cd ../Annotation/visual_dict/new/
    python show_data_mesh.py
    ```

## Simulation Experiment (BASELINE)
<div align=center>
<img src="pic/simulation_result2.png" width="740px">
</div>

- run the grasp verify in IsaacGym and get the success rate
```bash
cd ../../../IsaacGym/python/collect_grasp/
# simulation verify 
sh run_clean_test_sim.sh
# calcutate final success rate
python success_caulcate.py
```

- Our baseline results can be download here : [Baseline-Results](https://drive.google.com/file/d/1fWPR87JSEWzQoTuI-zCkjdaCT95jIQvA/view?usp=drive_link). And run bash above.

    |          Category        |    Successrate    |  Train/Test  |
    | :----------------------: | :---------------: | :-----------:| 
    |      Bowl                |   69.48           |    43/5      |  
    |      Lightbulb           |   74.23           |   44/7       |   
    |      Bottle              |   68.62           |   51/8       |    
    |      Flashlight          |   91.03           |   44/6       |    
    |      Screwdriver         |   75.60           |   32/7       |     
    |      Spraybottle         |   58.07           |   20/3       |    
    |      Stapler             |   85.77           |   28/6       |    
    |      Wineglass           |   55.70           |   35/5       |    
    |      Mug                 |   54.62           |   57/7       |    
    |      Drill               |   55.00           |   10/2       |     
    |      Camera              |   63.33           |   87/7       |     
    |      Teapot              |   57.14           |   41/7       |     
    |      Total               | **68.50**         |   492/67     | 

Total success rate is the average each success rate. not successgrasp/total grasp.
## Acknowledgments

This repo is based on [TeachNet](https://github.com/TAMS-Group/TeachNet_Teleoperation), [handpose3d](https://github.com/TemugeB/handpose3d), [OakInk](https://github.com/oakink/Tink), [DeepSDF](https://github.com/oakink/DeepSDF_OakInk),  [TransGrasp](https://github.com/yanjh97/TransGrasp), [6dofgraspnet](https://github.com/jsll/pytorch_6dof-graspnet), [VRCNET](https://github.com/paul007pl/VRCNet). Many thanks for their excellent works.

And our previous works about functional grasp generation as follows: [Toward-Human-Like-Grasp](https://github.com/zhutq-github/Toward-Human-Like-Grasp), [FGTrans](https://github.com/wurina-github/FGTrans), [Functionalgrasp](https://github.com/hjlllll/Functionalgrasp).
## Citation

```BibTeX
@inproceedings{
}
```
## License

Our code is released under [MIT License](./LICENSE).
