import os
import subprocess
from tap import Tap
import glob

class InferenceArgumentParser(Tap):
    layer_path:str
    model_path:str
    out_path:str
    quality: int = 1
    start_idx:int=0
    stop_idx:int=21
    workers: int = 4
    batch_size: int = 64
    size:int=64
    crop:tuple[int,int,int,int]=None
    reverse:int=0
    compile:int=1
    device:str='cuda'
    gpus:int=1
args = InferenceArgumentParser().parse_args()

# Generate a string "0,1,2,...,args.gpus-1"
gpu_ids = ",".join(str(i) for i in range(args.gpus))

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")

import torch.nn as nn
from torch.nn import DataParallel
import torch.nn.functional as F
from timesformer_pytorch import TimeSformer
import torch
import random
import gc
from pytorch_lightning import LightningModule
import scipy.stats as st
from torch.utils.data import DataLoader
import numpy as np
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import PIL.Image
from joblib import Parallel, delayed
import multiprocessing

PIL.Image.MAX_IMAGE_PIXELS = 933120000
print(f"Using {torch.cuda.device_count()} GPUs")

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = f'./'
    
    exp_name = 'pretraining_all'
    # ============== model cfg =============
    in_chans = 21 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 64
    tile_size = 64
    stride = tile_size // 3

    train_batch_size = 256 # 32
    valid_batch_size = 256
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    epochs = 50 # 30
    
    src_sd = 2

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-4 / warmup_factor
    min_lr = 1e-6
    num_workers = 16 * args.gpus
    seed = 42
    # ============== augmentation =============
    # valid_aug_list = [
    #     A.Resize(size, src_sd*size),
    #     A.Normalize(
    #         mean= [0] * in_chans,
    #         std= [1] * in_chans
    #     ),
    #     ToTensorV2(transpose_mask=True),
    # ]
def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False

def cfg_init(cfg, mode='val'):
    set_seed(cfg.seed)
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_img(path):
    image = cv2.imread(path, 0)
    if not args.crop is None:
        image = image[args.crop[1]:args.crop[1]+args.crop[3], args.crop[0]:args.crop[0]+args.crop[2]]
    pad0 = (256 - image.shape[0] % 256)
    pad1 = (256 - image.shape[1] % 256)
    image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
    image=np.clip(image,0,200)
    return image

def read_image_mask(start_idx,end_idx):
    images = []
    idxs = range(start_idx, end_idx)
    paths = []
    for i in idxs:
        print("loading", f"{args.layer_path}/{i:02}")
        path = f"{args.layer_path}/{i:02}"
        if os.path.exists(f"{path}.tif"):
            paths.append(f"{path}.tif")
        else:
            paths.append(f"{path}.jpg")
        assert(os.path.exists(paths[-1]))

    images = Parallel(n_jobs=-1)(delayed(read_img)(path) for path in paths)

    if args.reverse:
        images.reverse()

    print("done loading")
    
    images = np.stack(images, axis=2)

    return images

def get_img_splits(s,e):
    images = []
    xyxys = []

    image = read_image_mask(s,e)

    SD = CFG.src_sd

    x1_list = list(range(0, image.shape[1]*SD-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]*SD-CFG.tile_size+1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            tile = image[y1//SD:y2//SD, x1//SD:x2//SD]
            if np.any(tile!=0):
                images.append(tile)
                xyxys.append([x1, y1, x2, y2])
    test_dataset = CustomDatasetTest(images,np.stack(xyxys), CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(
            mean= [0] * CFG.in_chans,
            std= [1] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=False, drop_last=False,
                              )
    return test_loader, np.stack(xyxys),(image.shape[0]*SD//16,image.shape[1]*SD//16)

def get_transforms(data, cfg):
    if data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    return aug

class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        return image,xy
    
class RegressionPLModel(LightningModule):
    def __init__(self,pred_shape,size=64,enc='',with_norm=False):
        super(RegressionPLModel, self).__init__()
        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        self.backbone=TimeSformer(
                dim = 512,
                image_size = 64,
                patch_size = 16,
                num_frames = 30,
                num_classes = 16,
                channels=1,
                depth = 8,
                heads = 6,
                dim_head =  64,
                attn_dropout = 0.1,
                ff_dropout = 0.1
            )
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        x = self.backbone(torch.permute(x, (0, 2, 1,3,4)))
        x=x.view(-1,1,4,4)        
        return x
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/Arcface_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=16,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/MSE_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

def predict_fn(test_loader, model, device, test_xyxys, pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    mask_count_kernel = np.ones((CFG.size//16, CFG.size//16))
    model.eval()

    for step, (images, xys) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        batch_size = images.size(0)
        
        
        with torch.inference_mode():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)
        y_preds = torch.sigmoid(y_preds)  # Keep predictions on GPU

        y_preds = y_preds.squeeze(1)
    
        # Move results to CPU as a NumPy array
        y_preds_multiplied_cpu = y_preds.cpu().numpy()  # Shape: (batch_size, 64, 64)

        # Update mask_pred and mask_count in a batch manner
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1//16:y2//16, x1//16:x2//16] += y_preds_multiplied_cpu[i]
            mask_count[y1//16:y2//16, x1//16:x2//16] += mask_count_kernel

    mask_pred /= np.clip(mask_count, a_min=1, a_max=None)
    return mask_pred
import gc

if __name__ == "__main__":
    # Loading the model
    model = RegressionPLModel.load_from_checkpoint(args.model_path, strict=False)
    model.to(device)
    
    if args.compile:
        model = torch.compile(model)
        
    CFG.stride = CFG.tile_size // int(2**args.quality)
    
    print("processing quality ",args.quality, " => stride ", CFG.stride)
    
    model.eval()

    img_split = get_img_splits(args.start_idx,args.stop_idx)
    test_loader,test_xyxz,test_shape = img_split
    
    mask_pred = predict_fn(test_loader, model, device, test_xyxz, test_shape)
    
    mask_pred = np.clip(np.nan_to_num(mask_pred),a_min=0,a_max=1)
    mask_pred /= mask_pred.max()

    image_cv = (mask_pred * 255).astype(np.uint8)
    cv2.imwrite(args.out_path, image_cv)
