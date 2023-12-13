import numpy as np
import trimesh
import trimesh.transformations as tra
import torch
class ShadowGripper():
    """An object representing a Shadow gripper."""

    def __init__(self , filename, q=None, num_contact_points_per_finger=10, root_folder=''):
        """
        Create a ShadowHand gripper object.

        Keyword Arguments:
            q {list of int} -- configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
        """
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q
        self.root_folder = root_folder

        #object

        # self.obj = trimesh.load(filename)

        # self.joints = reading_xml()
        #lf5-lf1,rf4-rf1-->th5-th1

    def get_meshes(self, data):
        root_folder = self.root_folder
        joint = data[7:].clone().detach()#.numpy()#.detach()
        joint[9] = -joint[9]
        joint[13] = -joint[13]
        forearm = root_folder + 'gripper_models/shadowhand/forearm_electric.stl'
        wrist = root_folder + 'gripper_models/shadowhand/wrist.stl'
        palm = root_folder + 'gripper_models/shadowhand/palm.stl'
        lfknuckle = root_folder + 'gripper_models/shadowhand/knuckle.stl'
        LFJ4 = root_folder + 'gripper_models/shadowhand/lfmetacarpal.stl'
        LFJ3 = root_folder + 'gripper_models/shadowhand/F3.stl'
        LFJ2 = root_folder + 'gripper_models/shadowhand/F2.stl'
        LFJ1 = root_folder + 'gripper_models/shadowhand/F1.stl'
        rfknuckle = root_folder + 'gripper_models/shadowhand/knuckle.stl'
        RFJ3 = root_folder + 'gripper_models/shadowhand/F3.stl'
        RFJ2 = root_folder + 'gripper_models/shadowhand/F2.stl'
        RFJ1 = root_folder + 'gripper_models/shadowhand/F1.stl'
        mfknuckle = root_folder + 'gripper_models/shadowhand/knuckle.stl'
        MFJ3 = root_folder + 'gripper_models/shadowhand/F3.stl'
        MFJ2 = root_folder + 'gripper_models/shadowhand/F2.stl'
        MFJ1 = root_folder + 'gripper_models/shadowhand/F1.stl'
        ffknuckle = root_folder + 'gripper_models/shadowhand/knuckle.stl'
        FFJ3 = root_folder + 'gripper_models/shadowhand/F3.stl'
        FFJ2 = root_folder + 'gripper_models/shadowhand/F2.stl'
        FFJ1 = root_folder + 'gripper_models/shadowhand/F1.stl'
        TH3 = root_folder + 'gripper_models/shadowhand/TH3_z.stl'
        TH2 = root_folder + 'gripper_models/shadowhand/TH2_z.stl'
        TH1 = root_folder + 'gripper_models/shadowhand/TH1_z.stl'


        self.forearm = trimesh.load(forearm)
        self.wrist = trimesh.load(wrist)
        self.palm = trimesh.load(palm)
        self.LFJ4 = trimesh.load(LFJ4)
        self.lfknuckle = trimesh.load(lfknuckle)
        self.LFJ3 = trimesh.load(LFJ3)
        self.LFJ2 = trimesh.load(LFJ2)
        self.LFJ1 = trimesh.load(LFJ1)
        self.rfknuckle = trimesh.load(rfknuckle)
        self.RFJ3 = trimesh.load(RFJ3)
        self.RFJ2 = trimesh.load(RFJ2)
        self.RFJ1 = trimesh.load(RFJ1)
        self.mfknuckle = trimesh.load(mfknuckle)
        self.MFJ3 = trimesh.load(MFJ3)
        self.MFJ2 = trimesh.load(MFJ2)
        self.MFJ1 = trimesh.load(MFJ1)
        self.ffknuckle = trimesh.load(ffknuckle)
        self.FFJ3 = trimesh.load(FFJ3)
        self.FFJ2 = trimesh.load(FFJ2)
        self.FFJ1 = trimesh.load(FFJ1)
        self.TH3 = trimesh.load(TH3)
        self.TH2 = trimesh.load(TH2)
        self.TH1 = trimesh.load(TH1)
        self.palm.vertices *= 0.001
        self.wrist.vertices *= 0.001
        self.LFJ4.vertices *= 0.001
        self.lfknuckle.vertices *= 0.001
        self.LFJ3.vertices *= 0.001
        self.LFJ2.vertices *= 0.001
        self.LFJ1.vertices *= 0.001
        self.rfknuckle.vertices *= 0.001
        self.RFJ3.vertices *= 0.001
        self.RFJ2.vertices *= 0.001
        self.RFJ1.vertices *= 0.001
        self.mfknuckle.vertices *= 0.001
        self.MFJ3.vertices *= 0.001
        self.MFJ2.vertices *= 0.001
        self.MFJ1.vertices *= 0.001
        self.ffknuckle.vertices *= 0.001
        self.FFJ3.vertices *= 0.001
        self.FFJ2.vertices *= 0.001
        self.FFJ1.vertices *= 0.001
        self.TH3.vertices *= 0.001
        self.TH2.vertices *= 0.001
        self.TH1.vertices *= 0.001
        # self.obj.vertices *= 0.001

        # transform fingers relative to the base
        self.forearm.apply_translation([0 ,-0.01, 0])
        # palm_t = np.array([0,0,0])

        # palm_t = np.array(data[4:7].cpu())
        palm_t_ = data[4:7].detach().numpy()
        # datas[0] = data[1]
        # datas[1] = data[2]
        # datas[2] = data[3]
        # datas[3] = data[0]
        # self.palm.apply_transform(tra.quaternion_matrix(data[:4].cpu()))
        # palm_r = tra.quaternion_matrix(data[:4].cpu())

        zero = data[:4].clone().detach().numpy()#.detach().numpy()
        zeros = tra.quaternion_matrix(zero) #  qw, qx, qy, qz,
        # zeros = np.array([0,1,0,0])#.cuda()
        # zeros = tra.quaternion_matrix(zeros)
        palm_r_ = tra.euler_matrix(-0.5 * np.pi, 0 , 0)
        palm_r = np.dot(zeros,palm_r_)
        palm_r__ = tra.euler_matrix(0,  0.5 * np.pi, 0)
        palm_r___ = np.dot(palm_r,palm_r__)
        # palm_r____ = tra.euler_matrix(0.5*np.pi, 0, 0)
        # palm_r_____ = np.dot(palm_r___, palm_r____)
        # palm_r = np.dot(palm_r,palm_r___)
        palm_r = palm_r___
        self.palm.apply_transform(palm_r)
        # palm_t = np.array([0,0,0])
        self.palm.apply_translation(palm_t_)
        palm_t = torch.from_numpy(palm_t_).double()

        #LF
        lfj4_t = np.array([-0.017-0.016 ,0 ,0.044-0.023])#why?
        lfj4_r = tra.euler_matrix(0.571 * joint[0] ,0 ,0.821 * joint[0] )
        lfj4_r = np.dot(palm_r, lfj4_r)
        lfj4_t = np.dot(palm_r[:3,:3], lfj4_t)
        self.LFJ4.apply_transform(lfj4_r)
        self.LFJ4.apply_translation(palm_t + torch.from_numpy(lfj4_t))
        lfj3_t = np.array([-0.017+0.016 ,0 ,0.044+0.023])
        lfj3_t = np.dot(lfj4_r[:3,:3],lfj3_t)
        lfj3_r1 = tra.euler_matrix(0, joint[1] , 0)
        lfj3_r2 = tra.euler_matrix(joint[2] , 0, 0)
        lfj3_r = np.dot(lfj3_r2, lfj3_r1)
        lfj3_r = np.dot(lfj4_r,lfj3_r)
        self.LFJ3.apply_transform(lfj3_r)
        self.LFJ3.apply_translation(palm_t + torch.from_numpy(lfj4_t) + torch.from_numpy(lfj3_t))
        lfknuckle_t = lfj3_t.copy()
        self.lfknuckle.apply_transform(lfj3_r)
        self.lfknuckle.apply_translation(palm_t + torch.from_numpy(lfj4_t) + torch.from_numpy(lfknuckle_t))
        lfj2_t = np.array([0, 0, 0.045])
        lfj2_r = tra.euler_matrix(joint[3] , 0, 0)
        lfj2_r_ = np.dot(lfj3_r, lfj2_r)
        self.LFJ2.apply_transform(lfj2_r_)
        lfj2_t_ = np.dot(lfj3_r[:3, :3], lfj2_t)
        self.LFJ2.apply_translation(palm_t + torch.from_numpy(lfj4_t) + torch.from_numpy(lfj3_t) + torch.from_numpy(lfj2_t_))
        lfj1_t = np.array([0, 0, 0.025])
        lfj1_r = tra.euler_matrix(joint[4] , 0, 0)
        lfj1_r_ = np.dot(lfj2_r_, lfj1_r)
        self.LFJ1.apply_transform(lfj1_r_)
        lfj1_r_ = np.dot(lfj2_r_[:3, :3], lfj1_t)
        self.LFJ1.apply_translation(palm_t + torch.from_numpy(lfj4_t) + torch.from_numpy(lfj3_t) + torch.from_numpy(lfj2_t_) + torch.from_numpy(lfj1_r_))

        # RF
        rfj3_t = np.array([-0.011, 0, 0.095])
        rfj3_r1 = tra.euler_matrix(0, joint[5] , 0)
        rfj3_r2 = tra.euler_matrix(joint[6] , 0, 0)
        rfj3_r = np.dot(rfj3_r2, rfj3_r1)
        rfj3_r = np.dot(palm_r, rfj3_r)
        rfj3_t = np.dot(palm_r[:3, :3], rfj3_t)
        self.RFJ3.apply_transform(rfj3_r)
        self.RFJ3.apply_translation(palm_t + torch.from_numpy(rfj3_t))
        rfknuckle_t = rfj3_t.copy()
        self.rfknuckle.apply_transform(rfj3_r)
        self.rfknuckle.apply_translation(palm_t + torch.from_numpy(rfknuckle_t))
        rfj2_t = np.array([0, 0, 0.045])
        rfj2_r = tra.euler_matrix(joint[7] , 0, 0)
        rfj2_r_ = np.dot(rfj3_r, rfj2_r)
        self.RFJ2.apply_transform(rfj2_r_)
        rfj2_t_ = np.dot(rfj3_r[:3, :3], rfj2_t)
        self.RFJ2.apply_translation(palm_t + torch.from_numpy(rfj3_t) + torch.from_numpy(rfj2_t_))
        rfj1_t = np.array([0, 0, 0.025])
        rfj1_r = tra.euler_matrix(joint[8] , 0, 0)
        rfj1_r_ = np.dot(rfj2_r_, rfj1_r)
        self.RFJ1.apply_transform(rfj1_r_)
        rfj1_r_ = np.dot(rfj2_r_[:3, :3], rfj1_t)
        self.RFJ1.apply_translation(palm_t + torch.from_numpy(rfj3_t) + torch.from_numpy(rfj2_t_) + torch.from_numpy(rfj1_r_))

        # MF
        mfj3_t = np.array([0.011, 0, 0.095])
        mfj3_r1 = tra.euler_matrix(0, joint[9] , 0)
        mfj3_r2 = tra.euler_matrix(joint[10] , 0, 0)
        mfj3_r = np.dot(mfj3_r2, mfj3_r1)
        mfj3_r = np.dot(palm_r, mfj3_r)
        mfj3_t = np.dot(palm_r[:3, :3], mfj3_t)
        self.MFJ3.apply_transform(mfj3_r)
        self.MFJ3.apply_translation(palm_t + torch.from_numpy(mfj3_t))
        mfknuckle_t = mfj3_t.copy()
        self.mfknuckle.apply_transform(mfj3_r)
        self.mfknuckle.apply_translation(palm_t + torch.from_numpy(mfknuckle_t))
        mfj2_t = np.array([0, 0, 0.045])
        mfj2_r = tra.euler_matrix(joint[11] , 0, 0)
        mfj2_r_ = np.dot(mfj3_r, mfj2_r)
        self.MFJ2.apply_transform(mfj2_r_)
        mfj2_t_ = np.dot(mfj3_r[:3, :3], mfj2_t)
        self.MFJ2.apply_translation(palm_t + torch.from_numpy(mfj3_t) + torch.from_numpy(mfj2_t_))
        mfj1_t = np.array([0, 0, 0.025])
        mfj1_r = tra.euler_matrix(joint[12] , 0, 0)
        mfj1_r_ = np.dot(mfj2_r_, mfj1_r)
        self.MFJ1.apply_transform(mfj1_r_)
        mfj1_r_ = np.dot(mfj2_r_[:3, :3], mfj1_t)
        self.MFJ1.apply_translation(palm_t + torch.from_numpy(mfj3_t) + torch.from_numpy(mfj2_t_) + torch.from_numpy(mfj1_r_))

        #FF
        ffj3_t = np.array([0.033 ,0 ,0.095])
        ffj3_r1 = tra.euler_matrix(0, joint[13] , 0)
        ffj3_r2 = tra.euler_matrix(joint[14] , 0, 0)
        ffj3_r = np.dot(ffj3_r2,ffj3_r1)
        ffj3_r = np.dot(palm_r, ffj3_r)
        ffj3_t = np.dot(palm_r[:3, :3], ffj3_t)
        self.FFJ3.apply_transform(ffj3_r)
        self.FFJ3.apply_translation(palm_t + torch.from_numpy(ffj3_t))
        ffknuckle_t = ffj3_t.copy()
        self.ffknuckle.apply_transform(ffj3_r)
        self.ffknuckle.apply_translation(palm_t + torch.from_numpy(ffknuckle_t))
        ffj2_t = np.array([0,0,0.045])
        ffj2_r = tra.euler_matrix(joint[15] , 0, 0)
        ffj2_r_= np.dot(ffj3_r,ffj2_r)
        self.FFJ2.apply_transform(ffj2_r_)
        ffj2_t_ = np.dot(ffj3_r[:3,:3],ffj2_t)
        self.FFJ2.apply_translation(palm_t + torch.from_numpy(ffj3_t) + torch.from_numpy(ffj2_t_))
        ffj1_t = np.array([0,0,0.025])
        ffj1_r = tra.euler_matrix(joint[16] , 0, 0)
        ffj1_r_= np.dot(ffj2_r_,ffj1_r)
        self.FFJ1.apply_transform(ffj1_r_)
        ffj1_r_ = np.dot(ffj2_r_[:3,:3],ffj1_t)
        self.FFJ1.apply_translation(palm_t + torch.from_numpy(ffj3_t) + torch.from_numpy(ffj2_t_) + torch.from_numpy(ffj1_r_))

        #TH
        thj3_t = np.array([0.034 ,-0.009 ,0.029])
        thj3_r0 = tra.euler_matrix(0 ,0.785 ,0)
        thj3_r2 = tra.euler_matrix(0,0, -joint[17] )
        thj3_r1 = tra.euler_matrix(joint[18] , 0, 0)
        thj3_r = np.dot(thj3_r2, thj3_r1)
        thj3_r = np.dot(thj3_r0, thj3_r)
        thj3_r = np.dot(palm_r, thj3_r)
        thj3_t = np.dot(palm_r[:3, :3], thj3_t)
        self.TH3.apply_transform(thj3_r)
        self.TH3.apply_translation(palm_t + torch.from_numpy(thj3_t))
        thj2_t = np.array([0, 0, 0.038])
        thj2_r2 = tra.euler_matrix(joint[19] , 0, 0)
        thj2_r1 = tra.euler_matrix(0, -joint[20] , 0) #+ ---> -
        thj2_r = np.dot(thj2_r2, thj2_r1)
        thj2_r = np.dot(thj3_r,thj2_r)
        thj2_t = np.dot(thj3_r[:3,:3],thj2_t)
        self.TH2.apply_transform(thj2_r)
        self.TH2.apply_translation(palm_t + torch.from_numpy(thj3_t) + torch.from_numpy(thj2_t))
        thj1_t = np.array([0 ,0 ,0.032])
        thj1_r = tra.euler_matrix(0, -joint[21] , 0)#+ ---> -
        thj1_t = np.dot(thj2_r[:3,:3],thj1_t)
        thj1_r = np.dot(thj2_r,thj1_r)
        self.TH1.apply_transform(thj1_r)
        self.TH1.apply_translation(palm_t + torch.from_numpy(thj3_t) + torch.from_numpy(thj2_t) + torch.from_numpy(thj1_t))


        # self.fingers = trimesh.util.concatenate([self.finger_l, self.finger_r])
        self.hand = trimesh.util.concatenate([self.palm,
                                              self.LFJ4,self.lfknuckle,self.LFJ3,self.LFJ2,self.LFJ1,
                                              self.rfknuckle,self.RFJ3,self.RFJ2,self.RFJ1,
                                              self.mfknuckle,self.MFJ3,self.MFJ2,self.MFJ1,
                                              self.ffknuckle,self.FFJ3,self.FFJ2,self.FFJ1,
                                              self.TH3, self.TH2, self.TH1])
                                              # ,self.obj])

    # def get_meshes(self):
    #     """Get list of meshes that this gripper consists of.
    #
    #     Returns:
    #         list of trimesh -- visual meshes
    #     """

        return self.hand
        # return [self.palm, self.forearm, self.hand]
    # def get_obbs(self):
    #     """Get list of obstacle meshes.
    #
    #     Returns:
    #         list of trimesh -- bounding boxes used for collision checking
    #     """
    #     return [self.palm.bounding_box, self.forearm.bounding_box]

    # def get_closing_rays(self, transform):
    #     """Get an array of rays defining the contact locations and directions on the hand.
    #
    #     Arguments:
    #         transform {[nump.array]} -- a 4x4 homogeneous matrix
    #
    #     Returns:
    #         numpy.array -- transformed rays (origin and direction)
    #     """
    #     return transform[:3, :].dot(
    #         self.ray_origins.T).T, transform[:3, :3].dot(self.ray_directions.T).T
