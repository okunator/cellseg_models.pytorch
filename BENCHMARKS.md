# Benchmarks

**This is a fair and controlled comparison between the different cell/nuclei-segmentation models that are implemented in this library.**

### Background info

- **<span style="color:green">Cell/Nuclei-segmentation</span>** performance is benchmarked against the [**Pannuke**](https://arxiv.org/abs/2003.10778) and [**Lizard**](http://arxiv.org/abs/2108) datasets.
- **<span style="color:orange">Panoptic-segmentation performance</span>** performance is benchmarked against **HGSOC** and **CIN2** datasets.

#### Segmentation Performance Metrics

- Panoptic Quality (PQ)
  - The bPQ (cell type unaware), mPQ (cell type aware) and cell type specific PQs for all the models are reported
- Mean Interception over Union (mIoU)
  - mIoU is also reported for the semantic-segmentation results of the panoptic-segmentation models.

#### Latency Metrics for Multipart Models

Remember that these models are multipart. Each of the models are composed of an encoder-decoder neural-network and a post-processing pipeline. Thus, for all of the models, we report:

- Number of parms in the encoder-decoder architecture
- Encoder-decoder FLOPS
- Encoder-decoder latency (img/s)
- Post-processing latencies (img/s)
- Total latency (img/s).

Note that the post-processing pipelines are often composed of several parts. For nuclei/cell-segmentation, the post-processing pipeline is composed of a nuclei instance separation part and a cell type majority voting part. The latency for these are benchmarked separately. For panoptic segmentation, also the semantic-segmentation post-processing part is benchmarked separately. **The reported latency metrics are an average over the validation split.**

#### Devices

The model latencies depend on the hardware. I'll benhmark the latencies on my laptop and on a HPC server.

- Laptop specs:
  - a worn-out NVIDIA GeForce RTX 2080 Mobile (8Gb VRAM)
  - Intel i7-9750H 6 x 6 cores @ 2.60GHz (32 GiB RAM)
- HPC specs:
  - Nvidia V100 (32 GB VRAM)
  - Xeon Gold 6230 2 x 20 cores @ 2,1 GHz (384 GiB RAM)

#### About the Datasets

**Pannuke** is the only dataset that contains fixed sized (256x256) patches so the benchmarking is straight-forward and not affected by the hyperparameters of the post-processing pipelines. However, the **Lizard**, **HGSOC**, and **CIN2** datasets contain differing sized images. This means, firstly, that the patching strategy of the training data-split will have an effect on the model performance, and secondly, that the inference requires a sliding-window approach. The segmentation performance is typically quite sensitive to the sliding-window hyperparameters, especially, to the `patch size` and `stride`. Thus, with these datasets, I'm going to also report the training data patching strategy and we also grid-search the best sliding-window hyperparameters.

#### Data Splits

**Pannuke** and **Lizard** datasets are divided in three splits. For these datasets, we report the mean of the 3-fold cross-validation. The **CIN2** and **HGSOC** datasets contain only a training splits and relatively small validation splits, thus, for those datasets we report the metrics on the validation split.

#### Regularization methods

The models are regularized during training via multiple regularization techniques to tackle distrubution shifts. Specific techniques (among augmentations) that are used in this benchmark are:

- [Spectral decoupling](https://arxiv.org/abs/2011.09468)
- [Label Smoothing](https://arxiv.org/abs/1512.00567)
- [Spatially Varying Label Smoothing](https://arxiv.org/abs/2104.05788)

#### Pre-trained backbone encoders

All the models are trained/fine-tuned with an IMAGENET pre-trained backbone encoder that is naturally reported.

#### Training Hyperparams

All the training hyperparameters are naturally reported.

#### Other Notes

Note that even if these benchmarks are not SOTA or differ from the original manuscripts, the reason for that are likely not the model-architecture or the post-processing method (since these are the same here) but rather the model weight initialization, loss-functions, training hyperparameters, regularization techniques, and other training tricks that affect the model performance.

## Baseline models

### <span style="color:green">Cell/Nuclei-segmentation</span>

#### Results Pannuke

##### Training Set-up

| Param                  | Value                                     |
| ---------------------- | ----------------------------------------- |
| Optimizer              | [AdamP](https://arxiv.org/abs/2006.08217) |
| Auxilliary Branch Loss | MSE-SSIM                                  |
| Type Branch Loss       | Focal-DICE                                |
| Encoder LR             | 0.00005                                   |
| Decoder LR             | 0.0005                                    |
| Scheduler              | Reduce on plateau                         |
| Batch Size             | 10                                        |
| Training Epochs        | 50                                        |
| Augmentations          | Blur, Hue Saturation                      |

#### Results Lizard

##### Training Set-up

Same as above.

##### Patching Set-up

##### Sliding-window Inference Hyperparams
