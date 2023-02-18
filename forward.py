import torch.nn.functional as F
import torch
#Forward Model with Seidel Aberrations
#Load the PSF arrays 

class svBlur(object):
    def __init__(self,psfs,windows,step_size,device):
        self.psfs = torch.from_numpy(psfs).unsqueeze(1) 
        self.step_size =  step_size
        self.windows = torch.from_numpy(windows)
        self.device = device

    def __call__(self,imgs):
        imgs = F.pad(imgs,(self.step_size,self.step_size,self.step_size,self.step_size),"constant",0)
        #imgs = imgs.expand(imgs.shape[0],self.psfs.size(0),imgs.shape[2],imgs.shape[3])
        #patched_imgs = imgs * self.windows.expand(imgs.shape).to(self.device)
        output = torch.sum(F.conv2d(imgs.expand(imgs.shape[0],self.psfs.size(0),imgs.shape[2],imgs.shape[3]) 
            * self.windows.expand(imgs.shape[0],self.psfs.size(0),-1,-1).to(self.device),self.psfs.to(self.device),
            groups = self.psfs.size(0)),dim=1,keepdim=True)
        return output


class svBlur_tx(object):
    def __init__(self,psfs,windows,step_size,device):
        self.psfs = torch.from_numpy(psfs).unsqueeze(1) 
        self.step_size =  step_size
        self.windows = torch.from_numpy(windows)
        self.device = device

    def __call__(self,imgs):
        imgs = F.pad(imgs,(self.step_size,self.step_size,self.step_size,self.step_size),"constant",0)
        #patched_imgs = imgs.expand(-1,self.psfs.size(0),-1,-1).to(self.device) * self.windows.expand(imgs.size(0),self.psfs.size(0),-1,-1).to(self.device)
        output = torch.sum(F.conv2d(imgs.expand(-1,self.psfs.size(0),-1,-1).to(self.device) * self.windows.expand(imgs.size(0),self.psfs.size(0),-1,-1).to(self.device),self.psfs.to(self.device),groups = self.psfs.size(0)),dim=1,keepdim=True)
        return output

class svBlur_color(object):
    def __init__(self,psfs,windows,step_size,device):
        self.psfs = torch.from_numpy(psfs).unsqueeze(1) 
        self.step_size =  step_size
        self.windows = torch.from_numpy(windows)
        self.device = device

    def __call__(self,imgs):
        imgs = F.pad(imgs,(self.step_size,self.step_size,self.step_size,self.step_size),"constant",0)
        imgs_r,imgs_g,imgs_b = torch.split(imgs,1,dim = 1)
        print(imgs_r.shape,imgs_g.shape,imgs_b.shape)
        #patched_imgs = imgs.expand(-1,self.psfs.size(0),-1,-1).to(self.device) * self.windows.expand(imgs.size(0),self.psfs.size(0),-1,-1).to(self.device)
        out_r = torch.sum(F.conv2d(imgs_r.expand(-1,self.psfs.size(0),-1,-1).to(self.device) * self.windows.expand(imgs.size(0),self.psfs.size(0),-1,-1).to(self.device)
                                    ,self.psfs.to(self.device),groups = self.psfs.size(0)),dim=1,keepdim=True)
        out_g = torch.sum(F.conv2d(imgs_g.expand(-1,self.psfs.size(0),-1,-1).to(self.device) * self.windows.expand(imgs.size(0),self.psfs.size(0),-1,-1).to(self.device)
                                    ,self.psfs.to(self.device),groups = self.psfs.size(0)),dim=1,keepdim=True)
        out_b = torch.sum(F.conv2d(imgs_b.expand(-1,self.psfs.size(0),-1,-1).to(self.device) * self.windows.expand(imgs.size(0),self.psfs.size(0),-1,-1).to(self.device)
                                    ,self.psfs.to(self.device),groups = self.psfs.size(0)),dim=1,keepdim=True)
        output = torch.stack((out_b,out_g,out_r),dim =1)
        return output