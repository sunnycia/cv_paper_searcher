|Index|Title|sentence|
|---|---|---|
|1|Song_Fusing_Subcategory_Probabilities_2015_CVPR_paper|...Two types of texture descriptors are applied to represent the images: the IFV [33] and Caffe [22] descriptors....|
|||...Since our focus is on the classification model rather than feature design, we choose to follow [8] and adopt IFV and Caffe as our feature descriptors....|
|||...With our configurations of IFV and Caffe descriptors, we obtained similar results to [8] when using SVM as the classifier....|
|||...Specifically, the subcategorization model achieves about 7% improvement for the DTD dataset, but only 2% and 0.7% improvement  if the Caffe descriptor is used alone,  for KTH-TIPS2 and FMD....|
|||...At the same time, Caffe with SVM provides much better accuracies than IFV with SVM for KTH-TIPS2 and FMD, but lower accuracies for DTD....|
|||...These observations suggest that when the Caffe descriptor is more discriminative for a certain dataset, there is less scope to explore intra-class variation and inter-class ambiguity with subcategories....|
|||...We thus also performed another set of experiments by generating subcategories and computing subcategory probabilities and fusion weights based on IFV only, while Caffe is only incorporated when traini...|
|||...It is also worth to note that for KTH-TIPS2 and FMD, we actually obtained higher accuracies using Caffe with SVM compared to DeCAF with SVM [8], and lower accuracies using IFV with SVM compared to tha...|
|||...We suggest that this could be due to the improved framework of Caffe and the smaller numbers of Gaussian modes we used to reduce the feature dimension....|
|||...The combined effect of IFV and Caffe with SVM is nevertheless similar to the results of [8]....|
|||...For DTD, we used almost identical configurations for IFV as [8], and the performance using IFV or Caffe with SVM is very similar to those reported in [8]....|
||11 instances in total. (in cvpr2015)|
|2|Yang_Multi-Scale_Recognition_With_ICCV_2015_paper|...We evaluate multi-scale DAG-structured variants of existing CNN architectures (e.g., Caffe [15], Deep19 [30]) on a variety of scene recognition benchmarks including SUN397 [39], MIT67 [26], Scene15 [9]....|
|||...We carry out an analysis on existing CNN architectures, namely Caffe and Deep19....|
|||...Results are shown for two query images and two Caffe layers in Fig....|
|||...The detailed parameter options for both Caffe model are described later in Sec....|
|||...Experimental Results  We explore DAG-structured variants of two popular deep models, Caffe [15] and Deep19 [30]....|
|||...We focus on the Caffe model, as it is faster and easier for diagnostic analysis....|
|||...These are the baseline Caffe results presented in the previous subsections....|
|||...Fine-tuning models on both Chain and DAG model for Caffe backbone....|
||8 instances in total. (in iccv2015)|
|3|Cohen_Deep_SimNets_CVPR_2016_paper|...(a) Single layer ConvNet compared against single layer SimNet on CIFAR-10 (b) CIFAR-10 cross-validation accuracies of single-layer networks as a function of the number of floating-point operations req...|
|||...In terms of implementation, we have integrated SimNets into Caffe toolbox ([21]), with the aim of making our code publicly available in the near future....|
|||...The SimNet to which we compared Caffe ConvNet is a two layer network that follows the general structure outlined in fig....|
|||...1(d), with l2 similarity and architectural choices taken to maximize the alignment with Caffe ConvNet: 5x5  4In this paper, we consider FLOPs to be a measure of computational complexity....|
|||...Two layer SimNet vs. Caffe ConvNet on CIFAR-10, SVHN and CIFAR-100  comparison of test accuracies, number of floating-point operations required to classify an image, and number of learned parameters....|
|||...Training hyper-parameters for the SimNet were configured via cross-validation, whereas for Caffe ConvNet we used the values that come built-in to Caffe....|
|||...As can be seen, the SimNet is roughly twice as efficient as Caffe ConvNet, yet achieves significantly higher accuracies on the more challenging benchmarks (CIFAR-10 and CIFAR-100)....|
||7 instances in total. (in cvpr2016)|
