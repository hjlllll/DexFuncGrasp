import torch
from . import networks
from os.path import join
import utils.utils as utils
from utils import FK_model
from datasets.utils import *


class GraspNetModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> sampling / evaluation)
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        if self.gpu_ids and self.gpu_ids[0] >= torch.cuda.device_count():
            self.gpu_ids[0] = torch.cuda.device_count() - 1
        self.device = torch.device('cuda:{}'.format(
            self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.loss = None
        self.pcs = None
        self.grasps = None
        # load/define networks
        self.net = networks.define_classifier(opt, self.gpu_ids, opt.arch,
                                              opt.init_type, opt.init_gain,
                                              self.device)

        self.criterion = networks.define_loss(opt)

        self.confidence_loss = None
        if self.opt.arch == "vae":
            self.kl_loss = None
            self.reconstruction_loss = None
            self.reconstruction_pc_loss = None

            self.angle_loss = None
            # self.loss_close = None
            # self.loss_away = None
        elif self.opt.arch == "gan":
            self.reconstruction_loss = None
        else:
            self.classification_loss = None

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch, self.is_train)

    def set_input(self, data):
        # input_pcs = torch.from_numpy(data['pc']).contiguous()
        input_pcs = data['pc'].contiguous().float()
        B, N, C = input_pcs.shape
        clip = 0.01
        sigma = 0.001
        # input_pcs += torch.clip(sigma * torch.randn(B,N,C), -1 * clip, clip)
        input_r = data['r'].contiguous().float()
        input_t = data['t'].contiguous().float() * 0.001
        input_a = data['angles'].squeeze(1).contiguous().float()
        # input_pcs = torch.from_numpy(data['pc']).contiguous().float()
        # input_r = torch.from_numpy(data['r']).contiguous().float()
        # input_t = torch.from_numpy(data['t']).contiguous().float() * 0.001 # 米
        # input_a = torch.from_numpy(data['angles']).squeeze(1).contiguous().float()
        input_grasps = torch.cat((input_r, input_t, input_a),dim=1)
        if self.opt.arch == "evaluator":
            targets = data['labels'].float()
            targets_hand_pc = data['hand_pc'].float()
            input_grasps = data['hand_pc'].float()
        else:
            # targets = torch.from_numpy(data['hand_keypoints']).float()
            # targets_hand_pc =  torch.from_numpy(data['hand_pc']).float()
            targets = data['hand_keypoints'].float()
            targets_hand_pc = data['hand_pc'].float()
        self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)
        self.grasps = input_grasps.to(self.device).requires_grad_(
            self.is_train)
        self.targets = targets.to(self.device)
        self.targets_hand_pc = targets_hand_pc.to(self.device)
        self.angles = input_a.to(self.device)
        self.r = input_r.to(self.device)
        self.t = input_t.to(self.device)

    def generate_grasps(self, pcs, z=None):
        with torch.no_grad():
            return self.net.module.generate_grasps(pcs, z=z)

    def evaluate_grasps(self, pcs, gripper_pcs):
        success, _ = self.net.module(pcs, gripper_pcs)
        return torch.sigmoid(success)

    def forward(self):
        return self.net(self.pcs, self.grasps, train=self.is_train)

    def backward(self, out, epoch):
        if self.opt.arch == 'vae':
            predicted_cp, confidence, mu, logvar = out
            # # 输入正向运动学层
            # 正则化
            outputs_r = predicted_cp[:, :4]
            outputs_t = predicted_cp[:, 4:7] * 1000.0
            outputs_a = predicted_cp[:, 7:] / 1.5708
            outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)
            predicted_key, transformed_pts = trans_xml_pkl_batch(outputs_a, outputs_t, outputs_r, device=self.device)
           
            predicted_key *= 0.001
            transformed_pts *= 0.001


            # points = self.pcs
            # id_list = torch.load('hand_point_idx.pth')
            # all = np.arange(0,2000)
            # hand_index = np.concatenate(
            #     (id_list['thumb'], id_list['firfinger'], id_list['midfinger'], id_list['ringfinger'], id_list['litfinger']),
            #     axis=0)
            # away_index = np.setdiff1d(all, hand_index)
            # away_points = transformed_pts[:, away_index, :]# b, n2, 3
            # ###对away points做 fps/random 采样
            # away_points = index_points(away_points, farthest_point_sample_tensor(away_points, 300))
            # close_points = torch.cat([transformed_pts[:, id_list['thumb'],:].unsqueeze(1), transformed_pts[:, id_list['firfinger'],:].unsqueeze(1)
            #                             , transformed_pts[:, id_list['midfinger'],:].unsqueeze(1), transformed_pts[:, id_list['ringfinger'],:].unsqueeze(1),
            #                         transformed_pts[:, id_list['litfinger'],:].unsqueeze(1)], dim=1)
            # # [b, 5, 22, 2500]
            # close_distance = ((close_points.unsqueeze(3).expand(-1, -1, -1, points.shape[1], -1) -
            #                 points.unsqueeze(1).expand(-1, close_points.shape[1], -1, -1).
            #                 unsqueeze(2).expand(-1, -1, close_points.shape[2], -1, -1))**2).sum(-1).sqrt()   
            # close_distances = torch.min(close_distance, -1)[0] # [b, 5, 22]

            # close_distances = torch.min(close_distances, -1)[0].cuda() 
            # # [b, 2500, 300]
            # away_distance = ((away_points.unsqueeze(1).repeat(1, points.shape[1], 1, 1) - points.unsqueeze(2).
            #                 repeat(1, 1, away_points.shape[1],1)) ** 2).sum(-1).sqrt()
            # away_distances = torch.log2(0.002/ (torch.min(away_distance, -2)[0] + 0.000001))  # [b, 300]
            # away_distances[away_distances <= 0] = 0
            # #loss_close = close_distances.sum() #* 10.0
            # close0 = close_distances[:,0].sum() / close_distances.shape[0]
            # close1 = close_distances[:,1].sum() / close_distances.shape[0]
            # close2 = close_distances[:,2].sum() / close_distances.shape[0]
            # close3 = close_distances[:,3].sum() / close_distances.shape[0]
            # close4 = close_distances[:,4].sum() / close_distances.shape[0]
            # self.loss_away = away_distances.sum()/away_distances.shape[0]  
            # self.loss_close = 5*close0 + 5*close1 + 5*close2 + 2.5 * close3 


            indexs = [200, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950]
            self.reconstruction_pc_loss, self.confidence_loss2 = self.criterion[1](
                transformed_pts[:, indexs, :],
                self.targets_hand_pc[:, indexs, :],
                confidence=confidence,
                confidence_weight=self.opt.confidence_weight,
                device=self.device)
            self.kl_loss = self.criterion[0](mu, logvar, device=self.device)

            loss_l1 = torch.nn.MSELoss() 
            index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
            self.angles[:, [3, 7, 11, 15]] += self.angles[:, [4, 8, 12, 16]] # 20230202 发现应该加上最后一个关节再索引去掉

            self.angle_loss = loss_l1(predicted_cp[:, 7:],  self.angles[:, index])
            self.r_loss,self.t_loss = loss_l1(outputs_r, self.r), loss_l1(outputs_t*0.001, self.t)

            
            ###FINAL 

            self.loss = 0.1 * self.kl_loss + 1 * self.reconstruction_pc_loss + 0.001 * self.confidence_loss2 \
                            + 10 * self.angle_loss + 1.5 * self.r_loss + 100 * self.t_loss
            

        elif self.opt.arch == 'gan':
            predicted_cp, confidence = out
            
        elif self.opt.arch == 'evaluator':
            grasp_classification, confidence = out
            self.classification_loss, self.confidence_loss = self.criterion(
                grasp_classification.squeeze(),
                self.targets,
                confidence,
                self.opt.confidence_weight,
                device=self.device)
            self.loss = self.classification_loss + 0.01* self.confidence_loss

        self.loss.backward()

    def optimize_parameters(self, epoch):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out, epoch)
        self.optimizer.step()


##################

    def load_network(self, which_epoch, train=True):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        checkpoint = torch.load(load_path, map_location=self.device)
        if hasattr(checkpoint['model_state_dict'], '_metadata'):
            del checkpoint['model_state_dict']._metadata
        net.load_state_dict(checkpoint['model_state_dict'])
        if train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.opt.epoch_count = checkpoint["epoch"]
        else:
            net.eval()

    def save_network(self, net_name, epoch_num):
        """save model to disk"""
        save_filename = '%s_net.pth' % (net_name)
        save_path = join(self.save_dir, save_filename)
        torch.save(
            {
                'epoch': epoch_num + 1,
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self, mode):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            prediction, confidence = out
            if mode == 'test':
                if self.opt.arch == "vae":


                    # # 输入正向运动学层
                    # 正则化
                    predicted_cp = prediction

                    outputs_r = predicted_cp[:, :4]
                    outputs_t = predicted_cp[:, 4:7] * 1000.0
                    outputs_a = predicted_cp[:, 7:] / 1.5708
                    outputs_r = outputs_r / (outputs_r.pow(2).sum(-1).sqrt()).reshape(-1, 1)
                    predicted_key, transformed_pts = trans_xml_pkl_batch(outputs_a, outputs_t, outputs_r,
                                                                        device=self.device)
                    predicted_key *= 0.001
                    transformed_pts *= 0.001

                    indexs = [200, 450, 550, 650, 750, 850, 950, 1050, 1150, 1250, 1350, 1450, 1550, 1650, 1750, 1850, 1950]
                    reconstruction_pc_loss, _ = self.criterion[1](
                        transformed_pts[:, indexs, :],
                        self.targets_hand_pc[:, indexs, :],
                        confidence=confidence,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device)

                    return reconstruction_pc_loss, predicted_cp, transformed_pts

                elif self.opt.arch == "gan":
                    predicted_cp = utils.transform_control_points(
                        prediction, prediction.shape[0], device=self.device)
                    reconstruction_loss, _ = self.criterion(
                        predicted_cp,
                        self.targets,
                        confidence=confidence,
                        confidence_weight=self.opt.confidence_weight,
                        device=self.device)
                    return reconstruction_loss, predicted_cp
            else:
                index = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21]
                self.angles[:, [3, 7, 11, 15]] += self.angles[:, [4, 8, 12, 16]] #  
                self.recover_label = torch.cat((self.r, self.t, self.angles[:, index]), dim=1)  

                correct = (abs((prediction - self.recover_label)).sum(-1)<5).sum().item()
                correct_idx = (abs((prediction - self.recover_label)).sum(-1)<5)
                return correct, len(self.recover_label), correct_idx
