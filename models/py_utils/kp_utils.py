import pdb
import torch
import torch.nn as nn

from .utils import convolution, residual

class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2

def make_merge_layer(dim):
    return MergeUp()

def make_tl_layer(dim):
    return None

def make_br_layer(dim):
    return None

def make_ct_layer(dim):
    return None

def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)

def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)

def make_kp_layer(cnv_dim, curr_dim, out_dim):                          # conv2d + bn + relu + conv2d
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_inter_layer(dim):                                              # 创建中继结构，实质是个残差块
    return residual(3, dim, dim)

def make_cnv_layer(inp_dim, out_dim):                                   # 创建conv
    return convolution(3, inp_dim, out_dim)

def _gather_feat(feat, ind, mask=None):                                 # 根据 ind 和 mask 提取 feat                     
    dim  = feat.size(2)                                                 # cat
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)       # 把保留的top_k的坐标作为ind传进去，提取到每一行的正确预测结果。
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _nms(heat, kernel=1):                                               # 在heatmap上利用maxpool来等效实现nms
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _tranpose_and_gather_feat(feat, ind):                               # 与_gather_feat()函数结合，根据 ind 提取 feat
    feat = feat.permute(0, 2, 3, 1).contiguous()                        # feat: [batch_size, height, width, cat]
    feat = feat.view(feat.size(0), -1, feat.size(3))                    # feat: [batch_size, height*width, cat]
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=20):                                                # 在heatmap上选取 topK corners
    batch, cat, height, width = scores.size()                           # cat 就是 channel

    # 在batch的每张图上挑选出topK corners，topk_scores指示某点分数，topk_inds指示某点序号(index)
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)      # 根据scores大小选出前k个角

    topk_clses = (topk_inds / (height * width)).int()                   # corners的类别,因为feature map的通道数等于类别数。

    topk_inds = topk_inds % (height * width)                            # 映射到feature map的通道上
    topk_ys   = (topk_inds / width).int().float()                       # y坐标
    topk_xs   = (topk_inds % width).int().float()                       # x坐标
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _decode(
    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr, ct_heat, ct_regr, 
    K=100, kernel=1, ae_threshold=1, num_dets=1000
):
    # tl_heat, br_heat: [bs, num_classes, height, width] tl点和br点的特征图(sigmoid后就是评分图), 这两个特征图的通道就表示类别, 通道数就是类别数
    # tl_tag,  br_tag:  [bs, 1, height, width] tl点和br点的embedding(每个点对应1维的值)
    # tl_regr, br_regr: [bs, 2, height, width] tl点和br点的offset(每个点对应2维的值, 分别表示x和y的offset)

    # 原文链接：https://blog.csdn.net/ying86615791/article/details/89531974

    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)                                    # 对每个heatmap利用sigmiod映射0-1
    br_heat = torch.sigmoid(br_heat)
    ct_heat = torch.sigmoid(ct_heat)

    # perform nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)                              # 对其进行nms操作，其实就是maxpooling,保留max部分，kernel_size = 3 x 3
    br_heat = _nms(br_heat, kernel=kernel)
    ct_heat = _nms(ct_heat, kernel=kernel)

    ## 在top left以及bottom right,找到最大的前K个点，并记录下他们的得分，位置，类别，坐标等信息，下面返回的结果分别代表的是：
    ## 类别得分，位置索引，类别，y坐标，x坐标
    tl_scores, tl_inds, tl_clses, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_clses, br_ys, br_xs = _topk(br_heat, K=K)
    ct_scores, ct_inds, ct_clses, ct_ys, ct_xs = _topk(ct_heat, K=K)

    #下面是将坐标扩充， 为后面拿到所有的坐标组合做准备。这里扩充完之后变成了下面的样子 左边是横向的扩充，右边是纵向的扩充 
    # #[1,1,1     [ 1,2,3, 
    # # 2,2,2       1,2,3, 
    # # 3,3,3]       1,2,3 ] 
    # # 这样就可以组合出所有的枚举坐标了。也就是下面干的事情

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)                 # 横向的扩充
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)                 # 纵向的扩充 
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)
    ct_ys = ct_ys.view(batch, 1, K).expand(batch, K, K)
    ct_xs = ct_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:                     # 微调框的位置offset
        ###############通过点的索引, 在regr中gather到topk点的offset(偏置), 并加到坐标中
        # 得到topk tl像素点的offsets, 每个像素点的offset是2维, 表示x和y的offset
        tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)           # 将tl_regr和tl_inds对应起来 # [bs, K, 2]
        tl_regr = tl_regr.view(batch, K, 1, 2)
        # 得到topk br像素点的offsets
        br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
        br_regr = br_regr.view(batch, 1, K, 2)
        ct_regr = _tranpose_and_gather_feat(ct_regr, ct_inds)
        ct_regr = ct_regr.view(batch, 1, K, 2)

        # 更新坐标，将热图求的坐标跟offset做求和操作。
        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]
        ct_xs = ct_xs + ct_regr[..., 0]
        ct_ys = ct_ys + ct_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    ## 这里首先不考类别，暴力的求出左上角点和右下角点的所有的组合框，即每个左上角点都与右下角点组合
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)

    ### 拿出所有的左上角点和右下角点的embedding的值，用于后面验证距离，只有距离相近，才能被判断为是同一个类别
    ###############通过expand为[bs, K, K]大小后, 将topk的左上角tl点 和 右下角br点 两两组合, 得到所有可能的box(一个样本一共有k*k个)
    tl_tag = _tranpose_and_gather_feat(tl_tag, tl_inds)                 # # 得到topk tl像素点的embedding [bs, K, K, 4]
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = _tranpose_and_gather_feat(br_tag, br_inds)                 # 得到topk br像素点的embedding
    br_tag = br_tag.view(batch, 1, K)
    ### 计算左上角点以及右下角点的距离的绝对值。
    dists  = torch.abs(tl_tag - br_tag)                                 # [bs, K, K]

    ###############计算box的置信度
    # 将topk的tl点的置信度 和 br点的置信度相加取平均, 作为所有可能box的置信度
    # 此时score包含对k*k个box的置信度
    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, 1, K).expand(batch, K, K)
    scores    = (tl_scores + br_scores) / 2                             # [bs, K, K]

    # reject boxes based on classes
    # 由于前面是枚举了所有可能的组合情况，所以肯定会有很多错误的匹配情况，这里开始，根据一系列条件，干掉错误的匹配情况。
    tl_clses = tl_clses.view(batch, K, 1).expand(batch, K, K)
    br_clses = br_clses.view(batch, 1, K).expand(batch, K, K)
    # topk点中, 找出bl点和br点处于不同通道的 点的 索引
    # 每个通道表示一个类别!!!如果一对tl点和br点不在同个通道, 表示他们不是属于同类物体, 就干掉这样的点
    # 这样就默认tl点和br点的topk的通道索引必须一样!!!
    cls_inds = (tl_clses != br_clses)

    # reject boxes based on distances 将距离大于阈值的干掉，这里是0.5
    dist_inds = (dists > ae_threshold)

    # reject boxes based on widths and heights 左上角不在右下角上方的干掉
    width_inds  = (br_xs < tl_xs)
    height_inds = (br_ys < tl_ys)

    ##将上面提到的全部干掉，上面是获取对应索引，此处根据索引赋值为-1，彻底干掉
    scores[cls_inds]    = -1
    scores[dist_inds]   = -1
    scores[width_inds]  = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)
    ### 拿到过滤后的topk的得分，以及topk的index
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)                                         # [bs, num_dets, 1]

    ##下面分别利用index过滤，拿到topkscore对应的坐标以及类别等
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)                                 # 前num_dets个box,将bboxes与inds对应起来；# 最终保留的框的坐标
    
    #width = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)
    #height = (bboxes[:,:,2] - bboxes[:,:,0]).unsqueeze(2)
    
    clses  = tl_clses.contiguous().view(batch, -1, 1)                   # 所有可能box(k*k个)的通道索引(类别索引)????
    clses  = _gather_feat(clses, inds).float()                          # 前num_dets个box的通道索引(类别索引)， 将clses与inds对应起来;;# 最终保留的框的类别

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()                   # 将tl_scores与inds对应起来;# 最终保留的框的tl_scores
    br_scores = br_scores.contiguous().view(batch, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()                   # 最终保留的框的br_scores

    ct_xs = ct_xs[:,0,:]
    ct_ys = ct_ys[:,0,:]
    
    center = torch.cat([ct_xs.unsqueeze(2), ct_ys.unsqueeze(2), ct_clses.float().unsqueeze(2), ct_scores.unsqueeze(2)], dim=2)
    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections, center

def _neg_loss(preds, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _sigmoid(x):
    x = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return x

def _ae_loss(tag0, tag1, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag0 = tag0.squeeze()
    tag1 = tag1.squeeze()

    tag_mean = (tag0 + tag1) / 2

    tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
    tag0 = tag0[mask].sum()
    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    tag1 = tag1[mask].sum()
    pull = tag0 + tag1

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _regr_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss
