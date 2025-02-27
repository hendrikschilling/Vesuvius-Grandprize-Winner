# This repository has been archived. Further developement will resume at the monorepo of the Vesuvius Organization Team over at https://github.com/ScrollPrize/villa/tree/main/ink-detection    
Thank you to all contributors! 
# Vesuvius Grandprize Winning Solution
![Vesuvius Challenge GP Solution](pictures/logo.png)

The Repository contains the First Place Vesuvius Grand Prize solution. 
This repository is part of the First Place Grand Prize Submission to the Vesuvius Challenge 2023 from Youssef Nader, Luke Farritor and Julian Schilliger.

<!-- <img align="center" width="60" height="60" src="pictures/ThaumatoAnakalyptor.png">  -->
## Automatic Segmentation <img align="center" width="60" height="60" src="pictures/ThaumatoAnakalyptor.png"> 
Check out the automatic segmentation pipeline ThaumatoAnakalyptor of our winning Grand Prize submission by Julian Schilliger. 
[ThaumatoAnakalyptor](https://github.com/schillij95/ThaumatoAnakalyptor/tree/main) performs in full 3D and is also capable of segmenting in very mushy and twisted scroll regions.

# Ink Detection Overview<img align="center" width="60" height="60" src="pictures/logo.png"> :
Our final canonical model was a timesformer small architecture with divided space-time attention. 
The dataset underwent expansion and cleaning rounds to increase accuracy of the labels and become as accurate as possible, approximately 15 rounds were performed between the first letters and final solution. 
Our solution also consisted of 2 other architectures, Resnet3D-101 with pretrained weights, I3D with non-local block and maxpooling. 

Our implementation uses `torch`, `torch-lightning`,the [`timesformer-pytorch`](https://github.com/lucidrains/TimeSformer-pytorch) and [`3D-ResNets-PyTorch`](https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py). 


# 🚀 Get Started

EASY: build the docker image: 

```bash
docker build -t youssef_gp .
docker run --gpus all --shm-size=150g -it -v </your-path-to-train-scrolls>:/workspace/train_scrolls youssef_gp
```

Then to train:

```bash
python train_timesformer_og.py
```

Or to run inference with the already trained model:

```bash
python inference_timesformer.py --model_path timesformer_weights.ckpt --segment_path train_scrolls --segment_id 20231005123336
```

Important note: to install the ink labels and training data inside the docker image, run:

```bash
#to download the segments from the server
./download.sh
#propagates the inklabels into the respective segment folders for training
python prepare.py
```
You can find the weights of the canonical timesformer uploaded [here](https://drive.google.com/drive/folders/1rn3GMOvtJRMBHOxVhWFVSY6IVI6xUnYp?usp=sharing)
# Example Inference

To run inference of timesformer:

```bash
python inference_timesformer.py --segment_id 20231210121321 20231221180251 --segment_path $(pwd)/train_scrolls --model_path timesformer_weights.ckpt
```

The optional parameter ```--out_path``` can be used to specify the output path of the predictions.

# Alternative faster inference

```bash
python fast_inference_timesformer.py --layer_path <dir> --model_path <model>.ckpt --out_path ink_outer.jpg --compile 1 --quality 0
```

Faster inference variant that can be more than 10x faster or higher quality, but always renders to the native network output resolution, skipping the original upscaling.
Flags:
--compile, uses torch.compile for ~60% speedup
--quality 0-3, default 1, adjusts the stride for more oversampling. 2 should be better than GP baseline, 1 a bit faster and still good
--crop x y w h -
--reverse - reverse the layer order, try if ink detection doesn't seem to work at all
--median - use median instead of averaging to integrate strided predictions
--sr - 1-2 increase resolution by factor (use 2 in combination with quality 2+)
--focus - crop off borders of raw network output (1-3) without sr, 1-7 with sr, needs high quality values
--src-sd , default 2 scaledown of the source layer images, will be scaled up on the fly for lower memory use
