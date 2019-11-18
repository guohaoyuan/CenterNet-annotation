import os
import pdb
import torch
import importlib
import torch.nn as nn

from config import system_configs
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)                          # 此处设计随机数种子，保证结果是确定的

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss_kp  = self.loss(preds, ys, **kwargs)
        return loss_kp

# for model backward compatibility 为了模型向后兼容
# previously model was wrapped by DataParallel module 以前的模型由DataParallel模块包装
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

class NetworkFactory(object):                                           # https://blog.csdn.net/ying86615791/article/details/89531974
    def __init__(self, db):
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        print("module_file: {}".format(module_file))                    # module_file: models.CenterNet-52 
        nnet_module = importlib.import_module(module_file)              # 导入models.CenterNet-52; NetworkFactory又通过importlib.import_module(module_file)导入了self.model和self.loss
        # module_file使用的system_configs.snapshot_name来自train.py中的configs["system"]["snapshot_name"] = args.cfg_file
        # NetworkFactory中的self.model和self.loss, 这二者来自CenterNet-52.py中的class model(kp), 这个model继承自kp.py中的class kp(nn.Module), 这个loss也是来自kp.py中的class AELoss(nn.Module)
        # 所以model主要框架都在这个class kp(nn.Module)里.
        # 只传入1张图片的list时, 模型执行_test函数. 所以在测试的时候(看test/coco.py中的def kp_decode函数), 输入被封装为[images](只有images这个元素)

        self.model   = DummyModule(nnet_module.model(db))
        self.loss    = nnet_module.loss
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes).cuda()    
        #  此处的DataParallel是作者自己写的一个在模块级别实现数据并行的类，该容器通过 将batch size个输入按照chunk_size分配给指定GPU 来并行化数据。

        total_params = 0
        for params in self.model.parameters():                          # 计算参数量
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))              # 打印总的参数量

        if system_configs.opt_algo == "adam":                           # 参数更新策略:adam
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )                                                           # # 只有requires_grad=True的参数需要optimize
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):                                                      # 模型传到GPU 
        self.model.cuda()

    def train_mode(self):                                               # 模型设置为训练模式
        self.network.train()

    def eval_mode(self):
        self.network.eval()                                             # 模型设置为推理模式

    def train(self, xs, ys, **kwargs):
        xs = [x for x in xs]
        ys = [y for y in ys]

        self.optimizer.zero_grad()
        loss_kp = self.network(xs, ys)
        loss        = loss_kp[0]
        focal_loss  = loss_kp[1]
        pull_loss   = loss_kp[2]
        push_loss   = loss_kp[3]
        regr_loss   = loss_kp[4]
        loss        = loss.mean()
        focal_loss  = focal_loss.mean()
        pull_loss   = pull_loss.mean()
        push_loss   = push_loss.mean()
        regr_loss   = regr_loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss, focal_loss, pull_loss, push_loss, regr_loss

    def validate(self, xs, ys, **kwargs):                               # 验证时不需要反向传播
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]

            loss_kp = self.network(xs, ys)
            loss       = loss_kp[0]
            focal_loss = loss_kp[1]
            pull_loss  = loss_kp[2]
            push_loss  = loss_kp[3]
            regr_loss  = loss_kp[4]
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):                                               # 学习率设置
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):                 # 加载预训练模型
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def load_params(self, iteration):                                   # 模型加载参数
        cache_file = system_configs.snapshot_file.format(iteration)
        print("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def save_params(self, iteration):                                   # 保存模型参数
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f) 
