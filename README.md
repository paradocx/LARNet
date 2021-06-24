# LARNet: Lie Algebra ResNet 
Code for our paper：*LARNet:Lie Algebra Residual Network for Profile Face Recognition*  (ICML2021 accepted).  Here is the [arxiv version](https://arxiv.org/abs/2103.08147).

## Directory Structure
```
LARNet
├─ data
│    ├─ end2end                                        // The directory of training datasets for end2end
│    │                                                 //     e.g. MS1MV2
│    └─ plugin                                         // The directory of requirments for plugin
│           ├─ clean_feature.bin                       //     pretrained face model result, e.g. ArcFace,CosFace, and ours. Here we give a sample (ArcFace-MS1MV2, \#500)
│           ├─ pose_estimation.txt                     //     pose prior labels
│           └─ residual_sample_list.txt                //     The list of *clean_feature*
├─ src
│    ├─ end2end                                        // The code for end2end                      
│    │    ├─ End2end_train.py                          //     The training code for end2end
│    │    ├─ ResNet.py                                 //     The original architecture with our subnets' design
│    │    └─ otheroperation.py                         //     The code for loading different training datasets and caffe crop strategy
│    │
│    └─ plugin                                         // The code for plugin
│              ├─ Subnet_ablation.py                   //     The lightweight ablation experiments to show subnets' effectiveness        
│              ├─ Subnet_def.py                        //     The design of our subnets
│              └─ Subnet_train.py                      //     he training code for plugin based on *clean_feature*
│
└─test protocol                                        // The testing datasets                     
     ├─ IJBA
     ├─ CFP
```

## Requirments
- Python 3 with opencv
- pytorch
- ...

## Preparations
### Datasets
Training:
- **MS1MV2**: Our main training dataset is [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) .
- **CASIA-WebFace**: In some comparative experiments, we use [CASIA-WebFace](https://arxiv.org/abs/1411.7923). You can downlaod form [Google Drive](https://drive.google.com/open?id=1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz) or [Baidu Cloud](https://pan.baidu.com/s/1hQCOD4Kr66MOW0_PE8bL0w) with the key:y3wj.

Testing:
- **IJBA**:  We use the aligned images provided from  [Google Drive](https://drive.google.com/file/d/11p1eVSpyHZQUG0uBGyRoFnOXXTuZ501c/view?usp=sharing)　or  [Baidu Cloud](https://pan.baidu.com/s/1xLi6zDqwAeXEMV4aWi1k3g). 
- **CFP**: The original CFP dataset can be download from [Google Drive](https://drive.google.com/file/d/1B9QGThNd_-4Pg8O3si-EUYU9Px748p1C/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1U_CzmLsJ2OaX4rJeJ7r92g). And we offer a test protocol for further processing.
- ...

### Models
- **Pretrained model**. Because our plugin method does not need to change the structure and parameters of any pretrained model, it can directly use clean features of other methods as inputs, e.g. ArcFace, CosFace, and our results. Here we give a small training sample picked from  MS1MV2 randomly and its corresponding name list.
- **Pose labels**. We need to obtain the priori pose labels as inputs. Here we also provide a corresponding pose label file.

##Training
###plugin method
Make sure all files in *LARNet/data/plugin* are prepared well.
- Train the residual subnet:
```bash
cd LARNet/src/plugin
python Subnet_train.py
```
After this, a model will be saved in the *./plugin_subset.pth*
<br>
- ablation study for subnets:
Make sure *./plugin_subset.pth* exists.
```bash
cd LARNet/src/plugin
python Subnet_ablation.py
```
If you want to change the gating control function or the architecture of residual subnet, you can change the functions in the file *Subnet_def.py*:
```bash
def Gating_Control
```
or
```bash
class Res_Subnet
```

###end2end method
Make sure training datasets (e.g. MV1MV2) are prepared well.
- Train the entire subnet:
```bash
cd LARNet/src/end2end
python End2end_train.py
```
After this, a model will be saved in the *--model_dir/checkpoint.pth.tar*

##Testing



