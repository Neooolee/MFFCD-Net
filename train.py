import os
import datetime
import torch
import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler as GradScaler
from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    Cuda                = True
    num_classes         = 2
    backbone            = "resnet50"
    input_shape         = [256, 256]
    train_epoch         = 100
    train_batch_size    = 4
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'cos'
    save_period         = 20
    save_dir            = 'logs'
    eval_period         = 100
    data_path           = 'train'
    cls_weights         = np.ones([num_classes], np.float32)
    num_workers         = 0
    ngpus_per_node      = torch.cuda.device_count()


    model           = Unet(num_classes=num_classes, pretrained=False, backbone=backbone).train()
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    model_train     = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()


    with open(os.path.join(r"train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(r"val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        

    show_config(
        num_classes = num_classes, backbone = backbone, model_path = '', input_shape = input_shape, \
        Init_Epoch = 0, Freeze_Epoch = 0, UnFreeze_Epoch = train_epoch, Freeze_batch_size = 1, Unfreeze_batch_size = train_batch_size, Freeze_Train = False, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    if True:
        UnFreeze_flag = False
        batch_size = train_batch_size
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, train_epoch)
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        


        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, data_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, data_path)
        

        train_sampler   = None
        val_sampler     = None
        shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)
        
        eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, data_path, log_dir, Cuda, \
                                            eval_flag=True, period=eval_period)

        
        for epoch in range(0, train_epoch):
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, train_epoch, Cuda, True, True, cls_weights, num_classes, 0, GradScaler(), save_period, save_dir, 0)



        loss_history.writer.close()
