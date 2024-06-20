import torch
import numpy as np
from monai.losses import DiceLoss, DiceCELoss, FocalLoss, DiceFocalLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pt2onnx(model, output_file, input_size, opset=12):
    import torch.onnx

    model.eval()
    model.cpu()

    dummy_input = torch.randn(input_size)

    if len(input_size) == 5:
        dynamic_axes = {'modelInput': {0: 'batch_size', 2: 'x', 3: 'y', 4: 'z'},
                        'modelOutput': {0: 'batch_size', 2: 'x', 3: 'y', 4: 'z'}}
    else:
        dynamic_axes = {'modelInput': {0: 'batch_size', 2: 'x', 3: 'y'},
                        'modelOutput': {0: 'batch_size', 2: 'x', 3: 'y'}}

    torch.onnx.export(model, dummy_input, output_file, export_params=True, 
                      opset_version=opset, do_constant_folding=True, 
                      input_names=['modelInput'], output_names=['modelOutput'], 
                      dynamic_axes=dynamic_axes)

    return 1

def save_model(NET, model_ff, input_size=np.array((1, 1, 128, 128, 88))):
    ori_state = NET.training
    original_device = next(NET.parameters()).device

    NET.eval()
    NET.cpu()
    torch.save(NET, model_ff)
    pt2onnx(NET, model_ff.replace('.pt', '.onnx'), input_size=input_size)    

    NET.to(original_device)    
    if ori_state:
        NET.train()

def get_loss(logits, mask, loss_type='ce'):    

    if loss_type == 'L1':
        loss = torch.nn.L1Loss()(logits, mask[None, ...].float())
        return loss
        
    mask_d = mask.long()
    if loss_type == 'ce':
        loss = torch.nn.CrossEntropyLoss()(logits, mask_d)
    elif loss_type == 'dicesoftmax':
        loss = DiceLoss(softmax=True, to_onehot_y=True, include_background=True)(logits, mask_d[None, ...])
    elif loss_type == 'dicefocalsoftmax':
        loss = DiceFocalLoss(softmax=True, to_onehot_y=True, include_background=True)(logits, mask_d[None, ...])
    elif loss_type == 'cesigmoid':
        loss = torch.nn.CrossEntropyLoss()(torch.sigmoid(logits), mask_d)
    elif loss_type == 'dicecesigmoid':
        loss = DiceCELoss(sigmoid=True, to_onehot_y=True, include_background=True)(logits, mask_d[None, ...])
    elif loss_type == 'dicesigmoid':
        loss = DiceLoss(sigmoid=True, to_onehot_y=True, include_background=True)(logits, mask_d[None, ...])
    elif loss_type == 'focalsigmoid':
        loss = FocalLoss(to_onehot_y=True, include_background=True)(logits, mask_d[None, ...])

    return loss

class LinearLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, final_lr=1e-6, last_epoch=-1):
        self.total_steps = total_steps
        self.final_lr = final_lr
        super(LinearLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr - (base_lr - self.final_lr) * self.last_epoch / self.total_steps for base_lr in self.base_lrs]

def get_scheduler(lr_rule, optimizer):
    print(f'using {lr_rule} Scheduler')
    
    if 'cosine' in lr_rule:
        tmax = 8
        if '@' in lr_rule:
            tmax = int(lr_rule.split('@')[1])
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax, eta_min=3e-6)
    elif 'onecycle' in lr_rule:
        steps = int(lr_rule.split('@')[1])
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, 5e-4, total_steps=steps)
    elif 'linear' in lr_rule:
        parts = lr_rule.split('@')
        total_steps = int(parts[1])
        final_lr = 1e-6
        if len(parts) > 2:
            final_lr = float(parts[2])
        return LinearLRScheduler(optimizer, total_steps=total_steps, final_lr=final_lr)
    else: #constant
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)
    

def get_dice(mask11, mask22, labels=None):
    mask1 = mask11.astype(int).flatten()
    mask2 = mask22.astype(int).flatten()
    if labels is None:
        #num_labels = max(mask1.max(), mask2.max()) + 1
        labels = np.unique(np.concatenate([mask1, mask2]))
        
    #print(num_labels)
    
    dice_scores = []
    
    for label in labels:
        mask1_label = (mask1 == label).astype(np.uint8)
        mask2_label = (mask2 == label).astype(np.uint8)
        
        intersection = np.sum(mask1_label * mask2_label)
        volume1 = np.sum(mask1_label)
        volume2 = np.sum(mask2_label)
        
        if volume1 + volume2 == 0:
            dsc = 1.0
        else:
            dsc = (2.0 * intersection) / (volume1 + volume2)
            
        dice_scores.append(dsc)
    
    return np.array(dice_scores)
