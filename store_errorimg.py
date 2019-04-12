import torch
import numpy as np
import os
import uuid
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

unloader = torchvision.transforms.ToPILImage()

def save_image(tensor, cloth, action, path):
    dir = path
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    pth = os.path.join(dir,str(cloth),str(action))
    if not os.path.exists(pth):
        os.makedirs(pth)
    uuid_str = uuid.uuid4().hex()
    image.save(os.path.join(pth, 'error_imgs_' + str(uuid_str) + '.jpg'))
    
def store_errorimgs(input, output1, output2, target, acc1, acc2):
    with torch.no_grad():
        #output = torch.Tensor(8,5)
        #target = torch.ones(8,2,dtype=torch.long)
        #target = target[:,0]
        target1 = target[:,0]
        target2 = target[:,1]
        batch_size = target.size(0)
        _, pred1 = output1.topk(1, 1, True, True)
        pred1 = pred1.t()
        correct1 = pred1.eq(target1.view(1, -1).expand_as(pred1))
        correct1 = correct1.view(-1).numpy()  #classification result eg [0 1 0 1 0 0 0]
        res1 = pred1[0,:].numpy() #prediction result eg [0 3 1 2 0 1 4]
        _, pred2 = output2.topk(1, 1, True, True)
        pred2 = pred2.t()
        correct2 = pred2.eq(target2.view(1, -1).expand_as(pred2))
        correct2 = correct2.view(-1).numpy()  #classification result eg [0 1 0 1 0 0 0]
        res2 = pred2[0,:].numpy() #prediction result eg [0 3 1 2 0 1 4]
        correct = correct1 + correct2
        for idx in range(batch_size):
            if correct[idx] is not 2:
                save_image(input[idx,:,:,:], res1[idx], res2[idx], path)
            #print(correct,res)
        #return correct,res