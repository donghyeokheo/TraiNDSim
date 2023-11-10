import torch

# MNIST binarization
def binarization(pic):
    threshold = 0.5
    pic[pic >= threshold] = 1
    pic[pic < threshold] = 0
    return pic

class Binarization():
    def __call__(self, pic):
        return binarization(pic)

# loss function
def SSE(logits, label):
    target = torch.zeros_like(logits)
    target[torch.arange(target.size(0)).long(), label] = 1
    out =  0.5*((logits-target)**2).sum()
    return out

def MSE(logits, label):
    return SSE(logits, label) / label.size(0)


def str_to_int(param):
    if (param == 'none') or (param == 'None'):
        param = None
    else:
        param = int(param)
    
    return param

def str_to_list(param, datatype):
    if (param == 'none') or (param == 'None'):
        param = None
    else:
        param = list(map(datatype, param.split(',')))
        
    return param

def time_hms(ss):
    h = int(ss // 3600)
    ss = ss - h*3600
    
    m = int(ss // 60)
    ss = ss - m*60
    
    s = int(ss)

    time = ""
    if h > 0:
        time += f"{h}h "
    if m > 0:
        time += f"{m}m "
    if s > 0:
        time += f"{s}s"

    return time

