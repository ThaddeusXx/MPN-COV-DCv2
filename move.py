import json
import os
import numpy as np
import shutil

txt_file = '/workspace/mnt/group/video/linshaokang/fast-MPN-COV/food/meta/train.txt'
f = open(txt_file,'r')
lines = f.readlines()
for line in lines:
    line=line.strip('\n')
    cl = line.split("/")[0]
    if not os.path.exists(("/workspace/mnt/group/video/linshaokang/fast-MPN-COV/food/train/"+str(cl)+"/")):
        os.makedirs(("/workspace/mnt/group/video/linshaokang/fast-MPN-COV/food/train/"+str(cl)+"/"))
    shutil.move(("/workspace/mnt/group/video/linshaokang/fast-MPN-COV/food/images/" + line+  ".jpg"), ("/workspace/mnt/group/video/linshaokang/fast-MPN-COV/food/train/"+str(cl)+"/"))