# Improving Multi-Label Classification with Graph Neural Networks

>   Multi-label image classification is a wide-spread problematic that aims at predicting which objects are present in an image. 
Later improvements correlate with the improvements in single-label image classification, 
but such methods omit to capture the relationships between objects. 
It appears that some objects are more likely to be found together, 
therefore graphs of relationships between labels can be leveraged to improve the usual classification pipeline. 
Therefore, we review three graph neural networks based methods that use this knowledge to improve the performances of the models.

## Repository Organisation

- `src/`
- `notebooks/` : Notebooks to train the models
  - `baseline.ipynb` : Baseline models 
  - `ggnn.ipynb` : Trains a Graph Gated Neural Network (GGNN) adapted to the multi-label classification task. Inspired from [3]
  - `ssgrl.ipynb` : Trains a model following the Semantic Specific Graph Representation Learning   (SSGRL) [1] framework
  - `mlgcn.ipynb` : Trains a Multi Label Graph Convolution Network (GCN) [2]

## Data

Data expects to be put in an `input/` directory at the root. Three subfolders are needed:

- `VOC2007/` : Contains the Pascal VOC 2007 train, val and test data which can be obtained [here]( http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
- `glove/` : Contains the GloVe embeddings, which can be downloaded [here](https://nlp.stanford.edu/projects/glove/)
- `visual_genome/` : Contains the relationships of the Visual Genome dataset, downloadable [here](https://visualgenome.org/api/v0/api_home.html)



## Main References

- [1] Tianshui Chen, Muxin Xu, Xiaolu Hui, Hefeng Wu, and Liang Lin. Learning semantic-specific graphrepresentation for multi-label image recognition. ICCV 2019. [Link](http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.pdf).

- [2] Zhao-Min Chen, Xiu-Shen Wei, Peng Wang, and Yanwen Guo. Multi-label image recognition with graph convolutional networks. CVPR 2019. [Link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.pdf).

- [3] Kenneth Marino, Ruslan Salakhutdinov, and Abhinav Gupta. The more you know: Using knowledge graphs for image classification. CVPR 2017. [Link](https://arxiv.org/pdf/1612.04844.pdf).






##### *Project for the MVA classes Graphs in Machine Learning and Object Recognition & Computer Vision. 2019-2020*
