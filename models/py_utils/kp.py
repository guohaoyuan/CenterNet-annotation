import pdb
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr

from .kp_utils import _tranpose_and_gather_feat, _decode
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss
from .kp_utils import make_tl_layer, make_br_layer, make_kp_layer, make_ct_layer
from .kp_utils import make_pool_layer, make_unpool_layer
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]                                           # 残差块数量, [2, 2, 2, 2, 2, 4]    
        next_mod = modules[1]

        curr_dim = dims[0]                                              # 通道数cat, [256, 256, 384, 384, 384, 512]
        next_dim = dims[1]

        self.up1  = make_up_layer(                                      # residual block，先改变通道数再级联
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )                                                                   
        self.max1 = make_pool_layer(curr_dim)                           # 创建池化块，其实是kernel=2,s=2的最大值池化
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )                                                               # residual block，先改变通道数再级联
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )                                                               # 递归,residual block，先改变通道数再级联
        self.low3 = make_hg_layer_revr(                                 # residual block，先级联再改变通道数
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)                         # 上采样

        self.merge = make_merge_layer(curr_dim)                         # merge, 即x+y

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class kp(nn.Module):                                                    # CenterNet Model; 关于nstack(整个模型其实在实现中堆叠了nstack次),具体地说nstack被设置为2; https://blog.csdn.net/Chunfengyanyulove/article/details/94646724
    # 整体上来说，nstack=1表示就是CenterNet-52; nstack=2表示的就是CenterNet-104，其中两个52连接出利用中继监督
    # 另外，搭建网络过程是利用递归
    def __init__(
        self, db, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer, make_ct_layer=make_ct_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_tag_layer=make_kp_layer, make_regr_layer=make_kp_layer,
        make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(kp, self).__init__()

        self.nstack             = nstack
        self._decode            = _decode                               # decode就是网络输出了heatmap,embedding,offset后如何进行点匹配以及最终选择哪些点对作为结果的函数
        self._db                = db
        self.K                  = self._db.configs["top_k"]
        self.ae_threshold       = self._db.configs["ae_threshold"]
        self.kernel             = self._db.configs["nms_kernel"]
        self.input_size         = self._db.configs["input_size"][0]
        self.output_size        = self._db.configs["output_sizes"][0][0]

        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre                                       # self.pre定义的是网络的头部，网络先接了一个kernel size 7x7的convolution块以及一个residual块结构。主要作用先降低图片的分辨率

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])                                                              # CenterNet的主干结构是hourglasses，这里是就是其主干结构，make_xx_layer都是定义在kp_utils.py文件中的，知道其实hourglasses主干结构就可以了。
        # **并且注意到了吗，这里的定义都使用了for循环 for _ in range(nstack),其实作者所有的结构都定义了两个，两个结构通过前面提到的中继监督连接到一起。**

        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])                                                                          # hourglasses输出后，接一个convolution块

        # 然后定义的是接的三个分支，分别去输出top left 以及 bottom right, center的分支
        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])                                                                          # none
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])                                                                          # none

        self.ct_cnvs = nn.ModuleList([
            make_ct_layer(cnv_dim) for _ in range(nstack)
        ])                                                                          # none
        # 这里整三个none的原因是这样的，因为后面有左上角，右下角，中心点，为了结构好看，我们就这样创建了空块。其实这一块除了美观，没有什么作用

        ## keypoint heatmaps
        # keypoint heatmaps，用于输出tl以及br, ct的热图，这里是8 * 256 *256的
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])                                                                          # conv2d + bn + relu + conv2d
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])                                                                          # conv2d + bn + relu + conv2d

        self.ct_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])                                                                          # conv2d + bn + relu + conv2d

        ## tags
        # 用于输出 embeddings值  1 * 256 * 256的
        self.tl_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])                                                                          # conv2d + bn + relu + conv2d
        self.br_tags  = nn.ModuleList([
            make_tag_layer(cnv_dim, curr_dim, 1) for _ in range(nstack)
        ])                                                                          # conv2d + bn + relu + conv2d

        for tl_heat, br_heat, ct_heat in zip(self.tl_heats, self.br_heats, self.ct_heats):
            tl_heat[-1].bias.data.fill_(-2.19)                                      # 添加偏差，最后一层
            br_heat[-1].bias.data.fill_(-2.19)
            ct_heat[-1].bias.data.fill_(-2.19)

        # 下面这三个其实是中继结构，即将输出再接入下一个输入，后面的train以及test函数中会用到。
        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])                                                                          # residual块，只在104时存在

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])                                                                          # conv2d + bn                   
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])                                                                          # conv2d + bn          

        # 这里定义的是输出的回归坐标 ： 2 * 256 * 256; conv2d + bn + relu + conv2d
        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])                                                                          # conv2d + bn + relu + conv2d
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])                                                                          # conv2d + bn + relu + conv2d
        self.ct_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])                                                                          # conv2d + bn + relu + conv2d

        self.relu = nn.ReLU(inplace=True)                                           # relu

        # 这个是在52层和104层时候，都会输出heatmap和回归结果，相当于两个深度上的预测相结合
        # 这也是为什么，在_train()中的outs需要相加 

    def _train(self, *xs):
        image      = xs[0]                          # 原始图片
        tl_inds    = xs[1]
        br_inds    = xs[2]
        ct_inds    = xs[3]

        inter      = self.pre(image)                # convolution块 + residual块
        outs       = []

        layers = zip(
            self.kps,      self.cnvs,
            self.tl_cnvs,  self.br_cnvs, 
            self.ct_cnvs,  self.tl_heats, 
            self.br_heats, self.ct_heats,
            self.tl_tags,  self.br_tags,
            self.tl_regrs, self.br_regrs,
            self.ct_regrs
        )                                           # 这里拆包，原因在于灵活处理CenterNet-52和CenterNet-104两种情况
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_,  br_cnv_  = layer[2:4]
            ct_cnv_,  tl_heat_ = layer[4:6]
            br_heat_, ct_heat_ = layer[6:8]
            tl_tag_,  br_tag_  = layer[8:10]
            tl_regr_,  br_regr_ = layer[10:12]
            ct_regr_         = layer[12]

            kp  = kp_(inter)                        # 一个Hourglass
            cnv = cnv_(kp)                          # 一个convolution块

            tl_cnv = tl_cnv_(cnv)                   # none
            br_cnv = br_cnv_(cnv)                   # none
            ct_cnv = ct_cnv_(cnv)                   # none

            tl_heat, br_heat, ct_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), ct_heat_(ct_cnv)                       # conv2d + bn + relu + conv2d
            tl_tag, br_tag        = tl_tag_(tl_cnv),  br_tag_(br_cnv)                                              # conv2d + bn + relu + conv2d；此处没有center的embedding？因为center不需要配对，只是选择出分数高的点，用于排除掉incorrent的框           
            tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)                       # conv2d + bn + relu + conv2d;可以看出回归用的是卷积操作；Faster中的回归用的是FC

            tl_tag  = _tranpose_and_gather_feat(tl_tag, tl_inds)
            br_tag  = _tranpose_and_gather_feat(br_tag, br_inds)
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)

            outs += [tl_heat, br_heat, ct_heat, tl_tag, br_tag, tl_regr, br_regr, ct_regr]

            # 这里比较重要，这里就是中继结构的核心，还记得前面提到的inter吗？这里就是先将inter进行了self.inters_操作，
            # 然后将前面的输出cnv(哪里输出的上面找)，过一下self.cnvs_结构，然后对其进行求和，之后过了relu以及self.inters结构，
            # 最后作为输入进入到nstack==1的结构，在来一遍，其实self.inters_与self.cnvs_的结构是一样的，都是卷积层
            # https://blog.csdn.net/Chunfengyanyulove/article/details/94646724
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)

        return outs                                                                                                 # 前8个一组，后8个一组分别计算损失

    def _test(self, *xs, **kwargs):
        image = xs[0]

        inter = self.pre(image)

        outs          = []

        layers = zip(
            self.kps,      self.cnvs,
            self.tl_cnvs,  self.br_cnvs,
            self.ct_cnvs,  self.tl_heats,
            self.br_heats, self.ct_heats,
            self.tl_tags,  self.br_tags,
            self.tl_regrs, self.br_regrs,
            self.ct_regrs
        )
        for ind, layer in enumerate(layers):
            kp_, cnv_          = layer[0:2]
            tl_cnv_,  br_cnv_  = layer[2:4]
            ct_cnv_,  tl_heat_ = layer[4:6]
            br_heat_, ct_heat_ = layer[6:8]
            tl_tag_,  br_tag_  = layer[8:10]
            tl_regr_,  br_regr_ = layer[10:12]
            ct_regr_         = layer[12]

            kp  = kp_(inter)
            cnv = cnv_(kp)

            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                ct_cnv = ct_cnv_(cnv)

                tl_heat, br_heat, ct_heat = tl_heat_(tl_cnv), br_heat_(br_cnv), ct_heat_(ct_cnv)
                tl_tag, br_tag        = tl_tag_(tl_cnv),  br_tag_(br_cnv)
                tl_regr, br_regr, ct_regr = tl_regr_(tl_cnv), br_regr_(br_cnv), ct_regr_(ct_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr,
                         ct_heat, ct_regr]
            
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
                
        return self._decode(*outs[-8:], **kwargs)                                                   # 可以看到, 只用了第二次的输出作为预测，即后八个

    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.focal_loss  = focal_loss
        self.ae_loss     = _ae_loss
        self.regr_loss   = _regr_loss

    def forward(self, outs, targets):
        stride = 8

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        ct_heats = outs[2::stride]
        tl_tags  = outs[3::stride]
        br_tags  = outs[4::stride]
        tl_regrs = outs[5::stride]
        br_regrs = outs[6::stride]
        ct_regrs = outs[7::stride]

        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_ct_heat = targets[2]
        gt_mask    = targets[3]
        gt_tl_regr = targets[4]
        gt_br_regr = targets[5]
        gt_ct_regr = targets[6]
        
        # focal loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]
        ct_heats = [_sigmoid(c) for c in ct_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)
        focal_loss += self.focal_loss(ct_heats, gt_ct_heat)

        # tag loss
        pull_loss = 0
        push_loss = 0

        for tl_tag, br_tag in zip(tl_tags, br_tags):
            pull, push = self.ae_loss(tl_tag, br_tag, gt_mask)
            pull_loss += pull
            push_loss += push
        pull_loss = self.pull_weight * pull_loss
        push_loss = self.push_weight * push_loss

        regr_loss = 0
        for tl_regr, br_regr, ct_regr in zip(tl_regrs, br_regrs, ct_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
            regr_loss += self.regr_loss(ct_regr, gt_ct_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss

        loss = (focal_loss + pull_loss + push_loss + regr_loss) / len(tl_heats)
        return loss.unsqueeze(0), (focal_loss / len(tl_heats)).unsqueeze(0), (pull_loss / len(tl_heats)).unsqueeze(0), (push_loss / len(tl_heats)).unsqueeze(0), (regr_loss / len(tl_heats)).unsqueeze(0)
