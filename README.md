# LARNet
Code for the paper：&lt;LARNet:Lie Algebra Residual Network for Profile Face Recognition>(ICML2021)

···
>LARNet
>├─ data
>│    ├─ end2end                                        // The directory of training datasets for end2end
>│    │                                                 //     e.g. MS1MV2
>│    └─ plugin                                         // The directory of requirments for plugin
>│           ├─ clean_feature.bin                       //     pretrained face model result, e.g. ArcFace,CosFace, and ours. Here we give a sample (ArcFace-MS1MV2, \#500)
>│           ├─ pose_estimation.txt                     //     pose prior labels
>│           └─ residual_sample_list.txt                //     The list of *clean_feature*
>├─ src
>│    ├─ end2end                                        // The code for end2end                      
>│    │    ├─ End2end_train.py                          //     The training code for end2end
>│    │    ├─ ResNet.py                                 //     The original architecture with our subnets' design
>│    │    └─ otheroperation.py                         //     The code for loading different training datasets and caffe crop strategy
>│    │
>│    └─ plugin                                         // The code for plugin
>│              ├─ Subnet_ablation.py                   //     The lightweight ablation experiments to show subnets' effectiveness        
>│              ├─ Subnet_def.py                        //     The design of our subnets
>│              └─ Subnet_train.py                      //     he training code for plugin based on *clean_feature*
>│
>└─test protocol
>     ├─ IJBA
>     ├─ CFP
···
