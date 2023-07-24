import torch
import torch.nn as nn
import collections
from torch.nn import functional as F
from flops_counter import get_model_complexity_info
#from .resample2d_package.resample2d import Resample2d
# from .GCN_utils.gcn2 import GCN
from .networks.DCNv2.dcn_v2 import DCN 

class conv_bn_relu(nn.Module):
    def __init__(self, inp_dim, out_dim, k, stride=1, with_bn=True):
        super(conv_bn_relu, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu



class Feat_sampler(nn.Module):
    def __init__(self,head_conv, hps_channel, moudling=False):
        super(Feat_sampler, self).__init__()

        #self.resample = Resample2d()
        
        self.gradient_mul = 1.0
        self.hps_channel = hps_channel
        if self.hps_channel == 34:
            heads = {"face":5,"shoulder":2,
                            "left_elbow_wrist":2,"right_elbow_wrist":2,
                            "hip":2,
                            "left_knee_ankle":2,
                            "right_knee_ankle":2}  # COCO

        elif self.hps_channel == 28:
            heads = {"shoulder":2,
                            "left_elbow_wrist":2,"right_elbow_wrist":2,
                            "hip":2,
                            "left_knee_ankle":2,
                            "right_knee_ankle":2,
                            "head":2} # the partitions on crowdpose
        else:
            assert 'unsupport'

        self.heads = collections.OrderedDict(heads)
        
        # predict the one-hop offset
        self.j = 2
        self.searcher = nn.Conv2d(head_conv, len(self.heads)*self.j , 3, padding=1, stride=1, bias=True)
          
        
        inp_dim =64
        # feature transformation for each part
        feat_trans = []
        feat_agg = []
        for m in range(len(self.heads)):
            feat_trans.append(conv_bn_relu(head_conv, inp_dim, 1 ,with_bn= True))
            feat_agg.append(DCN(inp_dim, inp_dim, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1))
        self.feat_trans = nn.ModuleList(feat_trans)
        self.feat_agg = nn.ModuleList(feat_agg)

        
        # predict the second-hop offset separately
        pred_kps_list=[]
        for head in self.heads.keys():
            pred_kps_list.append(nn.Conv2d(inp_dim, self.heads[head]*2, 1, padding=0, stride=1, bias=True))
        self.pred_kps_list = nn.ModuleList(pred_kps_list)
        
        
        self.ct_feat_trans = nn.Conv2d(head_conv, inp_dim, 1, padding=0, stride=1, bias=True)   
        self.squeeze_ct = conv_bn_relu(inp_dim * len(heads) + head_conv, inp_dim, 3 ,with_bn=False) 
        self.pred_ct_hm = nn.Conv2d(inp_dim, 1, kernel_size=1, stride=1, padding=0)
        
    
    def resample(self, inp, offset):
        # import pudb;pudb.set_trace()
        out_h, out_w = offset.shape[-2:]
        new_h = torch.linspace(0, out_h-1, out_h).view(-1, 1).repeat(1, out_w)
        new_w = torch.linspace(0, out_w-1, out_w).repeat(out_h, 1)
        base_grid = torch.cat((new_w.unsqueeze(2), new_h.unsqueeze(2)), dim = 2).unsqueeze(0) # xy
        grid = offset.permute(0, 2, 3, 1) + base_grid.to(offset.device)  # unnormal
        grid[:,:,:,0] = grid[:,:,:,0]/(out_w-1) * 2 - 1
        grid[:,:,:,1] = grid[:,:,:,1]/(out_h-1) * 2 - 1  # normal to [-1, 1]

        out = F.grid_sample(inp, grid=grid, mode='bilinear', padding_mode='border')

        return out
    
    
    def feat_sampler(self, kps_feat, ct_feat, offset1):
        
        ct_hm_feat = [ct_feat]
        ct_feat = self.ct_feat_trans(ct_feat)
        off = collections.OrderedDict()
        for i,head in enumerate(list(self.heads.keys())):
            adapt_point_location = offset1[:,self.j*i:self.j*i+2,:,:].contiguous()
            kps_onehop_feat = self.resample(self.feat_agg[i](self.feat_trans[i](kps_feat)), adapt_point_location)
            ct_onehop_feat = self.resample(ct_feat, adapt_point_location)
            
            offset2 = self.pred_kps_list[i](kps_onehop_feat)
     
            off[head] = offset2 + offset1[:, self.j*i:self.j*i+2, :, :].repeat(1,self.heads[head],1,1)
            
            ct_hm_feat.append(ct_onehop_feat)

        ct_hm_feat =torch.cat(ct_hm_feat,dim=1)
        return off ,ct_hm_feat

    
   

    def post_process(self,res_dict):
        if self.hps_channel == 34:
            final_result = [res_dict["face"],res_dict["shoulder"],
                            res_dict["left_elbow_wrist"][:,:2,:,:],res_dict["right_elbow_wrist"][:,:2,:,:],
                            res_dict["left_elbow_wrist"][:,2:,:,:],res_dict["right_elbow_wrist"][:,2:,:,:],
                            res_dict["hip"],
                            res_dict["left_knee_ankle"][:,:2,:,:],res_dict["right_knee_ankle"][:,:2,:,:],
                            res_dict["left_knee_ankle"][:,2:,:,:],res_dict["right_knee_ankle"][:,2:,:,:]]

        elif self.hps_channel == 28:
            final_result = [res_dict["shoulder"],
                            res_dict["left_elbow_wrist"][:,:2,:,:],res_dict["right_elbow_wrist"][:,:2,:,:],
                            res_dict["left_elbow_wrist"][:,2:,:,:],res_dict["right_elbow_wrist"][:,2:,:,:],
                            res_dict["hip"],
                            res_dict["left_knee_ankle"][:,:2,:,:],res_dict["right_knee_ankle"][:,:2,:,:],
                            res_dict["left_knee_ankle"][:,2:,:,:],res_dict["right_knee_ankle"][:,2:,:,:],
                            res_dict["head"]] # the partitions on crowdpose
        
        final_result = torch.cat(final_result, dim=1)
        return final_result



    def forward(self, kps_feat, ct_feat):
        
        offset1 = self.searcher(kps_feat) # one-hop offset
        offset1_grad_mul = (1 - self.gradient_mul) * offset1.detach(
        ) + self.gradient_mul * offset1

        kps,ct_hm_feat = self.feat_sampler(kps_feat, ct_feat, offset1_grad_mul)
        kps = self.post_process(kps)
        
        ct_hm_feat = self.squeeze_ct(ct_hm_feat)
        ct = self.pred_ct_hm(ct_hm_feat)
    
        
        return kps, ct, offset1

    



        
if __name__ == "__main__":
    model = Feat_sampler(64)
    flops, params = get_model_complexity_info(model.cpu(), (128, 128), as_strings=False, print_per_layer_stat=True, channel=64)
    print('Flops:  %.3f' % (flops / 1e9))
    print('Params: %.2fM' % (params / 1e6))




