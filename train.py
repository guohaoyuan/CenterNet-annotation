#!/usr/bin/env python
import os

import json
import torch
import numpy as np
import queue
import pprint
import random
import argparse
import importlib
import threading
import traceback

from tqdm import tqdm
from utils import stdout_to_tqdm
from config import system_configs
from nnet.py_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from db.datasets import datasets    # 这里是赋值为COCO数据集

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True   # 这两句并用，为模型寻找最优的初始化算法

def parse_args():   # 这里就是解析器对象
    parser = argparse.ArgumentParser(description="Train CenterNet") # 描述
    parser.add_argument("cfg_file", help="config file", type=str)   # cfg文件的路径
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)                        # 迭代开始的位置
    parser.add_argument("--threads", dest="threads", default=4, type=int)   # 线程个数

    #args = parser.parse_args()
    args, unparsed = parser.parse_known_args()
    return args

def prefetch_data(db, queue, sample_data, data_aug):    # 预取数据
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())                         # os.getpid()获得当前进程的id，设置随机数种子
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug) # 加载数据，进行数据增广
            queue.put(data)                             # 往队列中添加数据
        except Exception as e:
            traceback.print_exc()                       # 打印异常
            raise e

def pin_memory(data_queue, pinned_data_queue, sema):  
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return
    # python中的非阻塞使用互斥锁，锁定方法acquire
    # blocking参数：如果blocking为True，则当前线程会堵塞，直到获取到这个锁为止(blocking默认为True)；如果blocking为False，则当前线程不会堵塞。
    # 当一个线程调用锁的acquire()方法获得锁时，锁就进入“locked”状态。每次只有一个线程可以获得锁。如果此时另一个线程试图获得这个锁，该线程就会变为“blocked”状态，称为“阻塞”，直到拥有锁的线程调用锁的release()方法释放锁之后，锁进入“unlocked”状态。
    # 锁的好处：确保了某段关键代码只能由一个线程从头到尾完整地执行;锁的坏处：阻止了多线程并发执行，包含锁的某段代码实际上只能以单线程模式执行，效率大大下降，另外由于可以存在多个锁，不同的线程持有不同的锁，并试图获取对方持有的锁时，可能会造成死锁。
    
def init_parallel_jobs(dbs, queue, fn, data_aug):
    # 创建进程：Process([group [, target [, name [, args [, kwargs]]]]])，target表示调用对象，args表示调用对象的位置参数元组。kwargs表示调用对象的字典。name为别名。group实质上不使用。
    tasks = [Process(target=prefetch_data, args=(db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()         # Process以start()启动某个进程。
    return tasks

def train(training_dbs, validation_db, start_iter=0):
    # 从json文件读取参数
    learning_rate    = system_configs.learning_rate         # 学习率
    max_iteration    = system_configs.max_iter              # 最大迭代次数
    pretrained_model = system_configs.pretrain              # 预训练模型
    snapshot         = system_configs.snapshot              # 训练snapshot次就保存一次模型
    val_iter         = system_configs.val_iter              # 每隔几步验证一次
    display          = system_configs.display               # 每训练display次就显示一次loss
    decay_rate       = system_configs.decay_rate            # 衰减
    stepsize         = system_configs.stepsize              # 步长 ？？？

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue   = Queue(system_configs.prefetch_size)  # prefetch_size:预取数据量。Queue()表示多核运算中，队列的长度最大为prefetch_size
    validation_queue = Queue(5)                             # 表示队列长度最大为5

    # queues storing pinned data for training,队列存储固定数据以进行训练
    pinned_training_queue   = queue.Queue(system_configs.prefetch_size)
    pinned_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data)          # 
    sample_data = importlib.import_module(data_file).sample_data    # 将data_file导入，

    # allocating resources for parallel reading, 为并行读取分配资源
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data, True)      
    # 它调用init_parallel_jobs函数创建多进程，期间调用prefetch_data预取数据，期间调用sample_data函数。这些操作中涉及到了数据增强、各种groundtruth的生成等，
    if val_iter:     # 验证阶段数据增强=False
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data, False)

    training_pin_semaphore   = threading.Semaphore()    # 信号量，可以用于控制线程数并发数
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()    # 锁住线程
    validation_pin_semaphore.acquire()  

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)   # 参数元组 
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)       # 创建锁页线程
    training_pin_thread.daemon = True   
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True         # 子线程，在start之前使用，默认为False， true表示，无需等待子线程结束，主线程结束就结束
    validation_pin_thread.start()               # 子线程开始执行

    print("building model...")
    nnet = NetworkFactory(training_dbs[0])      # 搭建网络对象

    if pretrained_model is not None:    # 判断是否用预训练模型
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)   # 加载预训练模型的参数

    if start_iter:                                                      # 假如start_iter为0,就算了，否则执行   
        learning_rate /= (decay_rate ** (start_iter // stepsize))       # 根据start_iter来计算学习率

        nnet.load_params(start_iter)                                    # 根据命令行输入的参数，加载对应模型的参数，说白了就是从指定位置开始训练
        nnet.set_lr(learning_rate)                                      # 上面计算了学习率，此处是将学习率加载到模型中
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)                                      # 从头迭代时候直接将cfg文件中的学习率加载进去

    print("training start...")
    nnet.cuda()                                                         # 网络转化成GPU
    nnet.train_mode()                                                   # 训练模式
    with stdout_to_tqdm() as save_stdout:                               # Tqdm 是 Python 进度条库，可以在 Python 长循环中添加一个进度提示信息用法：tqdm(iterator)
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
            training = pinned_training_queue.get(block=True)
            training_loss, focal_loss, pull_loss, push_loss, regr_loss = nnet.train(**training)
            #training_loss, focal_loss, pull_loss, push_loss, regr_loss, cls_loss = nnet.train(**training)

            if display and iteration % display == 0:                    # 每迭代display次打印一次training loss
                print("training loss at iteration {}: {}".format(iteration, training_loss.item()))
                print("focal loss at iteration {}:    {}".format(iteration, focal_loss.item()))
                print("pull loss at iteration {}:     {}".format(iteration, pull_loss.item())) 
                print("push loss at iteration {}:     {}".format(iteration, push_loss.item()))
                print("regr loss at iteration {}:     {}".format(iteration, regr_loss.item()))
                #print("cls loss at iteration {}:      {}\n".format(iteration, cls_loss.item()))

            del training_loss, focal_loss, pull_loss, push_loss, regr_loss#, cls_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:   # 每迭代val_iter次打印一次validation loss
                nnet.eval_mode()                                        # 验证模式 
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(**validation)
                print("validation loss at iteration {}: {}".format(iteration, validation_loss.item()))
                nnet.train_mode()                                       # 验证结束后转换为训练模式

            if iteration % snapshot == 0:                               # 每过snapshot次就保存模型
                nnet.save_params(iteration)

            if iteration % stepsize == 0:                               # 每过stepsize就衰减一次学习率
                learning_rate /= decay_rate                             
                nnet.set_lr(learning_rate)                              # 将衰减后的学习率加载至模型

    # sending signal to kill the thread
    training_pin_semaphore.release()                                    # 发信号，杀线程
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:                                # 终止数据获取过程
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()

if __name__ == "__main__":
    args = parse_args() # 创建解析器对象

    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json") # 得到cfg.json文件的路径，config/CenterNet-xxx.json
    with open(cfg_file, "r") as f:  # 读取json文件
        configs = json.load(f)  
            
    configs["system"]["snapshot_name"] = args.cfg_file  # 添加snapshot_name
    system_configs.update_config(configs["system"]) # 添加一项后,更新json

    train_split = system_configs.train_split    # "trainval"
    val_split   = system_configs.val_split      # "minival"

    print("loading all datasets...")
    dataset = system_configs.dataset            # None, 据说默认为COCO
    # threads = max(torch.cuda.device_count() * 2, 4)
    threads = args.threads                      # 获得线程默认4
    print("using {} threads".format(threads))   # 
    training_dbs  = [datasets[dataset](configs["db"], train_split) for _ in range(threads)]     # 打印4次“start prefetching data...”，涉及构建训练集
    validation_db = datasets[dataset](configs["db"], val_split)                                 # 打印一次”start prefetching data...“，涉及构建验证集：图像读取，标志文件读取，图像预处理等

    print("system config...")
    pprint.pprint(system_configs.full)          # 格式化输出

    print("db config...")
    pprint.pprint(training_dbs[0].configs)

    print("len of db: {}".format(len(training_dbs[0].db_inds)))
    train(training_dbs, validation_db, args.start_iter)
