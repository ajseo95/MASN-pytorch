Motion-Appearance Synergistic Networks for VideoQA (MASN)
========================================================================

Pytorch Implementation for the paper:

**[Attend What You Need: Motion-Appearance Synergistic Networks for Video Question Answering][1]** <br>
Ahjeong Seo, [Gi-Cheon Kang](https://gicheonkang.com), Joonhan Park, and [Byoung-Tak Zhang](https://bi.snu.ac.kr/~btzhang/) <br>
In ACL 2021

<!--![Overview of MASN](model_overview.jpg)-->
<img src="./model_overview.jpeg" width="90%" align="middle">

Requirements
--------
python 3.7, pytorch 1.2.0


Dataset
--------
- Download [TGIF-QA](https://github.com/YunseokJANG/tgif-qa) dataset and refer to the [paper](https://arxiv.org/abs/1704.04497) for details.
- Download [MSVD-QA and MSRVTT-QA](https://github.com/xudejing/video-question-answering).

Extract Features
--------
1. Appearance Features
- For local features, we used the Faster-RCNN pre-trained with Visual Genome. Please cite this [Link](https://github.com/peteanderson80/bottom-up-attention).
  * After you extracted object features by Faster-RCNN, you can convert them to hdf5 file with simple run: `python adaptive_detection_features_converter.py`
- For global features, we used ResNet152 provided by torchvision. Please cite this [Link](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).

2. Motion Features
- For local features, we use RoIAlign with bounding box features obtained from Faster-RCNN. Please cite this [Link](https://github.com/AceCoooool/RoIAlign-RoIPool-pytorch).
- For global features, we use I3D pre-trained on Kinetics. Please cite this [Link](https://github.com/Tushar-N/pytorch-resnet3d).


We uploaded our extracted features:
1) TGIF-QA
 * [`res152_avgpool.hdf5`][2]: appearance global features (3GB).
 * [`tgif_btup_f_obj10.hdf5`][3]: appearance local features (30GB).
 * [`tgif_i3d_hw7_perclip_avgpool.hdf5`][4]: motion global features (3GB).
 * [`tgif_i3d_roialign_hw7_perclip_avgpool.hdf5`][5]: motion local features (59GB).

2) MSRVTT-QA
 * [`msrvtt_res152_avgpool.hdf5`][10]: appearance global features (1.7GB).
 * [`msrvtt_btup_f_obj10.hdf5`][11]: appearance local features (17GB).
 * [`msrvtt_i3d_avgpool_perclip.hdf5`][12]: motion global features (1.7GB).
 * [`msrvtt_i3d_roialign_perclip_obj10.hdf5`][13]: motion local features (34GB).

3) MSVD-QA
 * [`msvd_res152_avgpool.hdf5`][14]: appearance global features (220MB).
 * [`msvd_btup_f_obj10.hdf5`][15]: appearance local features (2.2GB).
 * [`msvd_i3d_avgpool_perclip.hdf5`][16]: motion global features (220MB).
 * [`msvd_i3d_roialign_perclip_obj10.hdf5`][17]: motion local features (4.2GB).


Training
--------
Simple run
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --task Count --batch_size 32
```

For MSRVTT-QA, run
```sh
CUDA_VISIBLE_DEVICES=0 python main_msrvtt.py --task MS-QA --batch_size 32
```

For MSVD-QA, run
```sh
CUDA_VISIBLE_DEVICES=0 python main_msvd.py --task MS-QA --batch_size 32
```

### Saving model checkpoints  
By default, our model save model checkpoints at every epoch. You can change the path for saving models by `--save_path` options.
Each checkpoint's name is '[TASK]_[PERFORMANCE].pth' in default.


Evaluation & Results
--------
```sh
CUDA_VISIBLE_DEVICES=0 python main.py --test --checkpoint [NAME] --task Count --batch_size 32
```

Performance on TGIF-QA dataset:

  Model  |  Count   |  Action   |  Trans.  | FrameQA  |
 ------- | ------ | ------ | ------ | ------ |
MASN | 3.75 | 84.4 | 87.4 |  59.5|

You can download our pre-trained model by this link : [`Count`][6], [`Action`][7], [`Trans.`][8], [`FrameQA`][9]

Performance on MSRVTT-QA and MSVD-QA dataset:
Model  |  MSRVTT-QA   |  MSVD-QA   |
 ------- | ------ | ------ |
MASN | 35.2 | 38.0 |


Citation
--------
If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:
```text
@inproceedings{seo-etal-2021-attend,
    title = "Attend What You Need: Motion-Appearance Synergistic Networks for Video Question Answering",
    author = "Seo, Ahjeong  and
      Kang, Gi-Cheon  and
      Park, Joonhan  and
      Zhang, Byoung-Tak",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.481",
    doi = "10.18653/v1/2021.acl-long.481",
    pages = "6167--6177",
    abstract = "Video Question Answering is a task which requires an AI agent to answer questions grounded in video. This task entails three key challenges: (1) understand the intention of various questions, (2) capturing various elements of the input video (e.g., object, action, causality), and (3) cross-modal grounding between language and vision information. We propose Motion-Appearance Synergistic Networks (MASN), which embed two cross-modal features grounded on motion and appearance information and selectively utilize them depending on the question{'}s intentions. MASN consists of a motion module, an appearance module, and a motion-appearance fusion module. The motion module computes the action-oriented cross-modal joint representations, while the appearance module focuses on the appearance aspect of the input video. Finally, the motion-appearance fusion module takes each output of the motion module and the appearance module as input, and performs question-guided fusion. As a result, MASN achieves new state-of-the-art performance on the TGIF-QA and MSVD-QA datasets. We also conduct qualitative analysis by visualizing the inference results of MASN.",
}

```


License
--------
MIT License

Acknowledgements
--------
 This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (2015-0-00310-SW.StarLab/25%, 2017-0-01772-VTT/25%, 2018-0-00622-RMI/25%, 2019-0-01371-BabyMind/25%) grant funded by the Korean government.
 
 
 [1]: https://aclanthology.org/2021.acl-long.481/
 [2]: https://drive.google.com/file/d/1tWY3gU4XohzhZjV5Wia5L8XqfaV10127/view?usp=sharing
 [3]: https://drive.google.com/file/d/1rxLL6eqi3d9FXKq7e4Wx7jiisu7_gzJa/view?usp=sharing
 [4]: https://drive.google.com/file/d/1ejP_V3CuJFB_jaUYf-OM9up5bsnnETP3/view?usp=sharing
 [5]: https://drive.google.com/file/d/1JbHWs0yTExL7Lc_abCvaXX49IsazUVvw/view?usp=sharing
 [6]: https://drive.google.com/file/d/1Z3r20wd2Mxco47WWggmNKazonfYnUDy1/view?usp=sharing
 [7]: https://drive.google.com/file/d/1USUA5D9bN5Ar9rClfdhOUHiTYdX1di1P/view?usp=sharing
 [8]: https://drive.google.com/file/d/1jZLDt14ZRmfHEqc8Yat7beQA6n-N6-h7/view?usp=sharing
 [9]: https://drive.google.com/file/d/1bXGlOKWrqUlEOer2cRNIJ2654_H_2UeR/view?usp=sharing
 [10]: https://drive.google.com/file/d/16UswbSjfhHBBUih-cGCZgvurNLq-gOKx/view?usp=sharing
 [11]: https://drive.google.com/file/d/1KdsLDW3oE-xNtrzsoYKZv9N_hauOR1Of/view?usp=sharing
 [12]: https://drive.google.com/file/d/1mX0oxSQXDS2h2Fxz091q6NdKuHihr0Fj/view?usp=sharing
 [13]: https://drive.google.com/file/d/1wQERtue5TY3zEZJwX0u2t19ARhU4mhtY/view?usp=sharing
 [14]: https://drive.google.com/file/d/1XtQNShBMbW3jNwuZPMYP9p-5QbpgtHF6/view?usp=sharing
 [15]: https://drive.google.com/file/d/1efxWKIGxvmEV5nR9iJosTMOvpMzHwjlG/view?usp=sharing
 [16]: https://drive.google.com/file/d/143miiDN3m9-QqptxtA6U6BJfSP8XOcoW/view?usp=sharing
 [17]: https://drive.google.com/file/d/14DUT3_yazEFYqZRjzWgrZ6K3lYu0XfHm/view?usp=sharing
