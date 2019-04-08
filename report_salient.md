|Index|Title|sentence|
|---|---|---|
|1|Instance-Level Salient Object Segmentation|...Instance-Level Salient Object Segmentation  Guanbin Li1  2  ,  Yuan Xie1  Liang Lin1  Yizhou Yu2   1Sun Yat-sen University  2The University of Hong Kong  Abstract  Image saliency detection has recentl...|
|||...However, none of the existing methods is able to identify object instances in the detected salient regions....|
|||...In this paper, we present a salient instance segmentation method that produces a saliency mask with distinct object instance labels for an input image....|
|||...Our method consists of three steps, estimating saliency map, detecting salient object contours and identifying salient object instances....|
|||...For the first two steps, we propose a multiscale saliency refinement network, which generates high-quality salient region masks and salient object contours....|
|||...Once integrated with multiscale combinatorial grouping and a MAP-based subset optimization framework, our method can generate very promising salient object instance segmentation results....|
|||...To promote further research and evaluation of salient instance segmentation, we also construct a new database of 1000 images and their pixelwise salient instance annotations....|
|||...r proposed method is capable of achieving state-of-the-art performance on all public benchmarks for salient region detection as well as on our new dataset for salient instance segmentation....|
|||...Recently the accuracy of salient object detection has been improved rapidly [29, 30, 33, 45] due to the deployment of deep convolutional neural networks....|
|||...Nevertheless, most of previous methods are only designed to detect pixels that belong to any salient object, i.e....|
|||...a dense saliency map, but are unaware of individual instances of salient objects....|
|||...Input  Salient region  Filtered salient  object proposals  Salient instance   segmentation  Figure 1....|
|||...An example of instance-level salient object segmentation....|
|||...Middle left: detected salient region....|
|||...Middle right: filtered salient object proposals....|
|||...Different colors indicate different object instances in the detected salient region....|
|||...In this paper, we tackle a more challenging task, instance-level salient object segmentation (or salient instance segmentation for short), which aims to identify individual object instances in the det...|
|||...The next generation of salient object detection methods need to perform more detailed parsing within detected salient regions to achieve this goal, which is crucial for practical applications, includi...|
|||...We suggest to decompose the salient instance segmentation task into the following three sub-tasks....|
|||...In this sub-task, a pixel-level saliency mask is predicted, indicating salient regions in the input image....|
|||...2) Detecting salient object contours....|
|||...In this sub-task, we perform contour detection for individual salient object instances....|
|||...Such contour detection is expected to suppress spurious boundaries other than object contours and guide the generation of salient object proposals....|
|||...In this sub-task, salient object proposals are generated, and a small subset of salient object proposals are selected to best cover the salient regions....|
|||...Finally, a CRF based refinement method is applied to improve the spatial coherence of salient object instances....|
|||...As their output is derived from receptive fields with a uniform size, they may not perform well  12386  on images with salient objects at multiple different scales....|
|||...lution of the original input image, making it infeasible to accurately detect the contours of small salient object instances....|
|||...Given the aforementioned sub-tasks of salient instance segmentation, we propose a deep multiscale saliency refinement network, which can generate very accurate results for both salient region detectio...|
|||...Such information integration is paramount for both salient region detection [6] and contour detection [5]....|
|||...Given the detected contours of salient object instances, we apply multiscale combinatorial grouping (MCG) [3] to generate a number of salient object proposals....|
|||...We further filter out noisy or overlapping proposals and produce a compact set of segmented salient object instances....|
|||...Finally, a fully connected CRF model is employed to improve spatial coherence and contour localization in the initial salient instance segmentation....|
|||...In summary, this paper has the following contributions:   We develop a fully convolutional multiscale refinement network, called MSRNet, for salient region detection....|
|||... information for saliency inference but also attentionally determine the pixel-level weight of each salient map by looking at different scaled versions of the same image....|
|||... MSRNet generalizes well to salient object contour detection, making it possible to separate distinct object instances in detected salient regions....|
|||...When integrated with object proposal generation and screening techniques, our method can generate high-quality segmented salient object instances....|
|||... A new challenging dataset is created for further research and evaluation of salient instance segmentation....|
|||...We have generated benchmark results for salient contour detection as well as salient instance segmentation using MSRNet....|
|||...In this section, we discuss the most relevant work on salient region detection, object proposal generation and instance-aware semantic segmentation....|
|||...Recently, deep CNNs have pushed the research on salient region detection into a new phase....|
|||...Though they have been widely used as a foregoing step for object detection, they are not tailored for salient object localization....|
|||...[16] proposed to generate a ranked list of salient object proposals, the overall quality of their result needs much improvement....|
|||...In this paper, we generate salient object proposals on the basis of salient object contour detection results....|
|||...Our overall framework for instance-level salient object segmentation....|
|||...Inspired by this problem, we propose salient instance segmentation, which simultaneously detects salient regions and identifies object instances inside them....|
|||...Because salient object detection is not associated with a predefined set of semantic categories, it is a challenging problem closely related to generic object detection and segmentation....|
|||...2, our method for salient instance segmentation consists of four cascaded components, including salient region detection, salient object contour detection, salient instance generation and salient inst...|
|||...Specifically, we propose a deep multiscale refinement network and apply it to both salient region detection and salient object contour detection....|
|||...Next, we generate a fixed number of salient object proposals on the basis of the results of salient object contour detection and apply a subset optimization method for further screening these object proposals....|
|||...Finally, the results from the previous three steps are integrated in a CRF model to generate the final salient instance segmentation....|
|||...Multiscale Refinement Network  We formulate both salient region detection and salient object contour detection as a binary pixel labeling problem....|
|||...Since salient objects could have different scales, we propose a multiscale refinement network (MSRNet) for both salient region detec tion and salient object contour detection....|
|||...3.1.1 Refined VGG Network  Salient region detection and salient object contour detection are closely related and both of them require low-level cues as well as high-level semantic information....|
|||...o deep models based on the same multiscale refinement network architecture to perform two subtasks, salient region detection and salient object contour detection....|
|||...As the number of training images for salient contour detection is much smaller, in practice, we first train a network for salient region detection....|
|||...A duplicate of this trained network is further fine-tuned for salient contour detection....|
|||...As the number of contour and non-contour pixels are extremely imbalanced in each training batch for salient object contour detection, the penalty for misclassifying contour pixels is 10 times the pena...|
|||...When training MSRNet for salient region detection, we initialize the bottom-up backbone network with a VGG16 network pretrained on ImageNet and the top-down refinement stream with random values....|
|||...Salient Instance Proposal  We choose the multiscale combinatorial grouping (MCG) algorithm [3] to generate salient object proposals from the detected salient object contours....|
|||...Specifically, given an input image, we first generate four salient object contour maps (three from scaled versions of the input and one from the fused map)....|
|||...To ensure a high recall rate of salient object instances, we generate 800 salient object proposals for any given image....|
|||...We discard those proposals with fewer than 80% salient pixels to guarantee that any remaining proposal mostly resides inside a detected salient region....|
|||...Given the set of initially screened salient object proposals, we further apply a MAPbased subset optimization method proposed in [51] to produce a compact set of object proposals....|
|||...The number of remaining object proposals in the compact set forms the final number of predicted salient object instances in the image....|
|||...We call each remaining salient object proposal a detected salient instance....|
|||...Refinement of Salient Instance Segmentation  As salient object proposals and salient regions are obtained independently, there exist discrepancies between the union of all detected salient instances a...|
|||...In this section, we propose a fully connected CRF model to refine the initial salient instance segmentation result....|
|||...Suppose the number of salient instances is K. We treat the background as the K + 1st class, and cast salient instance segmentation as a multi-class labeling problem....|
|||...If a salient pixel is covered by a single detected salient instance, the probability of the pixel having the label associated with that salient instance is 1....|
|||...If a salient pixel is not covered by any detected salient instance, the probability of the pixel having any label is 1 K ....|
|||...Note that salient object proposals may have overlaps and some object proposals may occupy non-salient pixels....|
|||...If a salient pixel is covered by k overlapping salient instances, the probability of the pixel having a label associated with one of the k salient instances is 1 k ....|
|||...If a background pixel is covered by k overlapping salient instances, the probability of the pixel having a label associated with 1 one of the k salient instances is k+1 , and the probability of the pi...|
|||...Given this initial salient instance probability map, we employ a fully connected CRF model [26] for refinement....|
|||...ons (p) and pixel intensities (I), and encourages nearby pixels with similar colors to take similar salient instance labels, while the second kernel only considers spatial proximity when enforcing smo...|
|||...A New Dataset for Salient Object Instances  As salient instance segmentation is a completely new problem, no suitable datasets exist....|
|||...In order to promote the study of this problem, we have built a new dataset with pixelwise salient instance labels....|
|||...To reduce the ambiguity in salient region detection results, these images were mostly selected from existing datasets for salient region detection, including ECSSD [48], DUT-OMRON [49], HKU-IS [29], a...|
|||...Two-thirds of the chosen images contain multiple occluded salient object instances while the remaining one-third consists of images with no salient regions, a single salient object instance or multipl...|
|||...To reduce label inconsistency, we asked three human annotators to label detected salient regions with different instance IDs in all selected images using a custom designed interactive segmentation tool....|
|||...We only kept the images where salient regions were divided into an identical number of salient object instances by all the three annotators....|
|||...At the end, our new salient instance dataset contains 1,000 images with high-quality pixelwise salient instance labeling as well as salient object contour labeling....|
|||...We combine the training sets of both the MSRA-B dataset (2500 images) [34] and the HKUIS dataset (2500 images) [29] as our training set (5000 images) for salient region detection....|
|||...As discussed in Section 3.1.3, this trained model is used as the initial model  for salient contour detection, and is further fine-tuned on the training set of our new dataset for salient instances an...|
|||...We fine-tune MSRNet on the augmented dataset for 10K iterations and keep the model with the lowest validation error as our final model for salient object contour detection....|
|||...It takes around 50 hours to train our multiscale refinement network for salient region detection and another 20 hours for salient object contour detection....|
|||...In our experiments, it takes 0.6 seconds to perform either salient region detection or salient object contour detection on a testing image with 400x300 pixels....|
|||...It takes 20 seconds to generate a salient instance segmentation with MCG being the bottleneck which needs 18 seconds to generate salient object proposals for a single image....|
|||...Evaluation on Salient Region Detection  To evaluate the performance of our MSRNet on salient region detection, we conduct testing on six benchmark datasets: MSRA-B [34], PASCAL-S [31], DUTOMRON[49], H...|
|||...It is a more meaningful measure in evaluating the applicability of a saliency model in salient instance segmentation....|
|||...1 Comparison with the State of the Art  We compare the proposed MSRNet with other 8 state-ofthe-art salient region detection methods, including GC [10], DRFI [24], LEGS [44], MC [52], MDF [29], DCL+ [...|
|||...Comparison of precision-recall curves among 9 salient region detection methods on 3 datasets....|
|||...ents are complementary to each other, which makes MSRNet not only capable of detecting more precise salient regions (with higher resolution) but also discovering salient objects at multiple scales....|
|||...ation on Salient Instance Segmentation  To evaluate the effectiveness of our proposed framework for salient instance segmentation as well as to promote further research on this new problem, we adopt t...|
|||...tional contour detection [2, 47] to evaluate the performance of salient object contour detection, and adopt three standard measures: fixed contour threshold (ODS), per-image best threshold (OIS), and ...|
|||...We define performance measures for salient instance segmentation by drawing inspirations from the evaluation of instance-aware semantic segmentation....|
|||...Benchmark results from our proposed method in both salient object contour detection and salient instance segmentation are given in Table 2....|
|||...Our method can handle challenging cases where multiple salient object instances are spatially connected to each other....|
|||...Examples of salient instance segmentation results by our MSRNet based framework....|
|||...The most important component of our framework is a multiscale saliency refinement network, which generates highquality salient region masks and salient object contours....|
|||...To promote further research and evaluation of salient instance segmentation, we have also constructed a new database with pixelwise salient instance annotations....|
|||...Quantitative benchmark results of salient object contour detection and salient instance segmentation on our new dataset....|
|||...Conclusions  In this paper, we have introduced salient instance segmentation, a new problem related to salient object detection,  This work was supported by Hong Kong Innovation and Technology Fund (I...|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Deep contrast learning for salient object  detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Unconstrained salient object detection via proposal subset optimization....|
||111 instances in total. (in cvpr2017)|
|2|Ruochen_Fan_Associating_Inter-Image_Salient_ECCV_2018_paper|...Associating Inter-Image Salient Instances for Weakly  Supervised Semantic Segmentation  Ruochen Fan1[0000000319910146], Qibin Hou2[0000000283888708], Ming-Ming Cheng2[0000000155508758] Gang Yu3[000000...|
|||...In this paper, we use an instance-level salient object detector to automatically generate salient instances (candidate objects) for training images....|
|||...Using similarity features extracted from each salient instance in the whole training set, we build a similarity graph, then use a graph partitioning algorithm to separate it into multiple subgraphs, e...|
|||...Our graph-partitioning-based clustering algorithm allows us to consider the relationships between all salient instances in the training set as well as the information within them....|
|||...Such methods merely require supervisions of one or more of the  2  Ruochen Fan, Qibin Hou and Ming-Ming Cheng  (a) input images  (b) salient instances  (c) proxy GT  (d) output results  Fig....|
|||...1: Input images (a) are fed into a salient instance detection method (e.g., S4Net [12]) giving instances shown in colour in (b)....|
|||...Our system automatically generates proxy ground-truth data (c) by assigning correct tags to salient instances and rejecting noisy instances....|
|||...aliency extractors, such as MSRNet [24] and S4Net [12], are now not only able to predict gray-level salient objects but also instance-level masks....|
|||...Inspired by the advantages of such instance-level salient object detectors, in this paper, we propose to carry out the instance distinguishing task in the early saliency detection stage, with the help...|
|||...In order to make use of the salient instance masks with their bounding boxes, two main obstacles need to be overcome....|
|||...Firstly, an image may be labeled with multiple keywords, so determining a correct keyword (tag) for each class-agnostic salient instance is essential....|
|||...Secondly, not all salient instances generated by the salient instance  Associating Inter-Image Salient Instances  3  detector are semantically meaningful; incorporating such noisy instances would deg...|
|||...Therefore, recognizing and excluding such noisy salient instances is important in our approach....|
|||...In this paper, we take into consideration both the intrinsic properties of a salient instance and the semantic relationships between all salient instances in the whole training set....|
|||...Here we use the term intrinsic properties of a salient instance to refer to the appearance information within its (single) region of interest....|
|||...In fact, it is possible to predict a correct tag for a salient instance using only its intrinsic properties: see [19, 22, 42]....|
|||...nformation within each region of interest, there are also strong semantic relationships between all salient instances: salient instances in the same category typically share similar semantic features....|
|||...More specifically, our proposed framework contains an attention module to predict the probability of a salient instance belonging to a certain category, based on its intrinsic properties....|
|||...ic relationships, we use a semantic feature extractor which can predict a semantic feature for each salient instance; salient instances sharing similar semantic information have close semantic feature...|
|||...Based on the semantic features, a similarity graph is built, in which the vertices represent salient instances and the edge weights record the semantic similarity between a pair of salient instances....|
|||...In summary, the main contributions of this paper are:   the first use of salient instances in a weakly supervised segmentation framework, significantly simplifying object discrimination, and performin...|
|||... a weakly supervised segmentation framework exploiting not only the information inside salient instances but also the relationships between all objects in the whole dataset....|
|||...In this paper, differently from all the aforementioned methods, we propose a weakly supervised segmentation framework using salient instances....|
|||...ag-assignment problem is modeled as graph partitioning, in which both the relationships between all salient instances in the whole dataset, as well as the information within them are taken into consid...|
|||...However, with the development of deep  Associating Inter-Image Salient Instances  5  Fig....|
|||...Instances are extracted from the input images by a salient instance detector (e.g., S4Net [12])....|
|||...Semantic features are obtained from the salient instances and used to build a similarity graph....|
|||...Graph partitioning is used to determine the final tags of the salient instances....|
|||...Given training images labelled only with keywords, we use an instance-level saliency segmentation network, S4Net [12], to extract salient instances from every image....|
|||...Each salient instance has a bounding box and a mask indicating a visually noticeable foreground object in an image....|
|||...These salient instances are classagnostic, so the extractor S4Net does not need to be trained for our training set....|
|||...Although salient instances contain ground-truth masks for training a segmentation mask, there are two major limitations in the use of such salient instances to train a segmentation network....|
|||...Determining the correct keyword associated with each salient instance is necessary....|
|||...We refer to such salient instances as noisy instances....|
|||...Both limitations can be removed by solving a tag-assignment problem, in which we associate salient instances with correct tags based on image keywords, and tag others as noisy instances....|
|||...Our pipeline takes into consideration both the intrinsic characteristics of a single region, and the relationships between all salient instances....|
|||...inspired by class activation mapping (CAM) [48], we use an attention module to identify the tags of salient instances di Semantic Features...Attention ModelSalient Instance DetectorInput ImageResultsS...|
|||...One weakness of existing weakly supervised segmentation work is that it treats the training set image by image, ignoring the relationships between salient instances across the entire training set....|
|||...However, salient instances belonging to the same category share similar contextual information which is of use in tag-assignment....|
|||...Our architecture extracts semantic features for each salient instance; regions with similar semantic information have similar semantic features....|
|||...The tag-assignment problem now becomes one of graph partitioning, making use not only of the intrinsic properties of a single salient instance, but the global relationships between all salient instances....|
|||...Assuming that a salient instance has a bounding box (x0, y0, x1, y1) in image I, the  probability of this salient instance belonging to the i-th category pi is:  pi =   1  (x1  x0)(y1  y0)  x1  y1  Xx...|
|||...3.2 Semantic Feature Extractor  The attention module introduced above assigns tags to salient instances from their intrinsic properties, but fails to take relationships between all salient instances i...|
|||...To discover such relationships, we use a semantic feature extractor to produce  Associating Inter-Image Salient Instances  7  feature vectors for each input region of interest, such that regions of i...|
|||... y +   (f  ct   y),  (4)  4 Tag-Assignment Algorithm  In order to assign a correct keyword to every salient instance with or identify it as a noisy instance, we use a tag-assignment algorithm, exploit...|
|||...In detail, assume that n salient instances have been produced from the training set by S4Net, and n semantic features extracted for each salient instance, denoted as fj , j = 1, ....|
|||...3.1 described, we predict the probability of every salient instance j belonging to category i, written as pij , i = 0, ....|
|||..., n, where category 0 means the salient instance is a noisy one....|
|||...Let the image keywords for a salient instance j be the set Kj ....|
|||...The purpose of the tag-assignment algorithm is to predict the final tags of the salient instances xij, i = 0, ....|
|||...We associate semantic similarity with the edges of a weighted undirected similarity graph having a vertex for each salient instance, and an edge for each pair of salient instances which are strongly similar....|
|||...Edge weights give the similarity of a salient instance pair....|
|||...As salient instances in the same category have similar semantic content and semantic features, a graph partitioning algorithm should ensure the vertices inside a subset are strongly related while the ...|
|||...As xi is a binary vector with length n, this formula simply sums the weights of  Associating Inter-Image Salient Instances  9  edges between all vertices in subgraph i....|
|||...To further explain this formulation, consider a salient instance, such as the vertex bounded by dotted square in Figure 3(b), which belongs to category ia....|
|||...Sharing similar semantic content, the vertex representing this salient instance has strong similarity with the vertices in subset ia....|
|||...The objective of the optimization problem reaches a maximum if and only if this vertex is partitioned into subset ia, meaning that the salient instance is assigned a correct tag....|
|||...4.3 The Graph Partitioning with Attention and Noisy Vertices  The tag assignment problem in Section 4.2 identifies keywords for salient instances using semantic relationships between the salient instances....|
|||...However, the intrinsic properties of a salient instance are also important in tag assignment....|
|||...As explained in Section 3.1, the attention module predicts the probability pij that a salient instance j belongs to category i....|
|||...10  Ruochen Fan, Qibin Hou and Ming-Ming Cheng  As the salient instances are obtained by the class-agnostic S4Net, some salient instances may fall outside the categories of the training set....|
|||...However, unlike in the original version, center loss is calculated by cosine distance instead of Euclidean distance for consistency  Associating Inter-Image Salient Instances  11  Table 1: Ablation s...|
|||... 34, 582  (a) Ablation results Random refers to keywords of an image being assigned randomly to the salient instances....|
|||...The results of the whole pipeline with or without noisy salient instance filtering are also given....|
|||...(b) Influence of r The retention ratio r determines the proportion of salient instances labeled as valid during graph partitioning....|
|||...The results on the three datasets, especially for the simple ImageNet set which contains more noisy salient instances, show that the noise filtering mechanism further improves segmentation performance....|
|||...The balancing ratio  balances information within salient instances to global object relationship information across the whole dataset....|
|||...If  is set to 0, graph partitioning depends solely on the global relationship information; as  increases, the influence of the intrinsic properties of the salient instances also increases....|
|||...When  = 30, 1.3% performance gain is obtained as intrinsic properties of the salient instances are also taken into consideration during graph partitioning....|
|||...etention ratio r The other key hyper-parameter, the retention ratio r, determines the proportion of salient instances to be regarded as valid in graph partitioning, as a proportion (1  r) of the insta...|
|||...Eliminating a proper number of salient instances having low confidence improves the quality of the proxy-ground-truth and benefits the final segmentation results, but too small a retention ratio leads...|
|||...It is  Associating Inter-Image Salient Instances  13  Table 3: Pixel-level segmentation results on the PASCAL VOC 2012 val and test sets compared to those from existing state-of-the-art approaches....|
|||...Because our academic version CPLEX restricts the maximum number of variables to be optimized, we use batches of 400 salient instances in implementation....|
|||...akly supervised segmentation framework, focusing on generating accurate proxy-ground-truth based on salient instances extracted from the training images and tags assigned to them....|
|||...In this paper, we introduce salient instances to weakly supervised segmentation, significantly simplifying the object discrimination operation in existing work and enabling our framework to conduct in...|
|||...In order to improve the accuracy of tagassignment, both the information from individual salient instances, and from the relationships between all objects in the whole dataset are taken into consideration....|
|||...1617 (2014) 3, 9, 14  Associating Inter-Image Salient Instances  15  4....|
|||...Cheng, M., Mitra, N.J., Huang, X., Torr, P.H., Hu, S.: Global contrast based salient region  detection....|
|||...Hou, Q., Cheng, M.M., Hu, X., Borji, A., Tu, Z., Torr, P.: Deeply supervised salient object  detection with short connections....|
|||...Jiang, H., Wang, J., Yuan, Z., Wu, Y., Zheng, N., Li, S.: Salient object detection: A discriminative regional feature integration approach....|
|||...Li, G., Xie, Y., Lin, L., Yu, Y.: Instance-level salient object segmentation....|
|||...In: CVPR (2017)  1, 4  Associating Inter-Image Salient Instances  17  47....|
||82 instances in total. (in eccv2018)|
|3|cvpr18-Revisiting Salient Object Detection  Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects|...Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects  Revisiting Salient Object Detection:  Md Amirul Islam*  Mahmoud Kalash*  Neil D. B. Bruce  University of Manitoba  Universi...|
|||...Specifically, there is not universal agreement about what constitutes a salient object when multiple observers are queried....|
|||...This implies that some objects are more likely to be judged salient than others, and implies a relative rank exists on salient objects....|
|||...We also show that the problem of salient object subitizing can be addressed with the same network, and our approach exceeds performance of any prior work across all metrics considered (both traditiona...|
|||...Introduction  The majority of work in salient object detection considers either a single salient object [37, 38, 7, 8, 31, 32, 9, 19, 17, 24, 39, 18] or multiple salient objects [13, 27, 36], but does...|
|||...There is a paucity of data that includes salient objects that are hand-segmented by multiple observers....|
|||...We present a solution in the form of a deep neural network to detect salient objects, consider the relative ranking of salience of these objects, and predict the total number of salient objects....|
|||...Left to right: input image, detected salient regions, rank order of salient objects, confidence score for salient object count (subitizing)....|
|||...Colors indicate the rank order of different salient object instances....|
|||...This includes detection of all salient regions in an image, and accounting for inter-observer variability by assigning confidence to different salient regions....|
|||...Success is measured against other algorithms based on the rank order of salient objects relative to ground truth orderings in addition to traditional metrics....|
|||...Recent efforts also consider the problem of salient object subitizing....|
|||...As a whole, our work generalizes the problem of salient object detection, we present a new model that provides predictions of salient objects according to the traditional form of this problem, multipl...|
|||...Salient Object Detection:  Convolutional Neural Networks (CNNs) have raised the bar in performance for many problems in computer vision including salient object detection....|
|||...Some CNN based methods exploit superpixel and object region proposals to achieve accurate salient object detection [9, 19, 17, 22, 39, 18]....|
|||...Other methods [24, 31, 38] use an end-to-end encoderdecoder architecture that produces an initial coarse saliency map and then refines it stage-by-stage to provide better localization of salient objects....|
|||...a new upsampling method to reduce artifacts of deconvolution which results in a better boundary for salient object detection....|
|||...Salient Object Subitizing:  Recent work [35, 7] has also addressed the problem of subitizing salient objects in images....|
|||...This task involves counting the number of salient objects, regardless of their importance or semantic category....|
|||...The first salient object subitizing network proposed in [35] applies a feed-forward CNN to treat the problem as a classification task....|
|||...Proposed Network Architecture  We propose a new end-to-end framework for solving the problem of detecting multiple salient objects and ranking the objects according to their degree of salience....|
|||...semantic segmentation, salient object detection) require pixel-precise information to produce accurate predictions....|
|||...The architecture based on iterative refinement of a stacked representation is capable of effectively detecting multiple salient objects....|
|||...Each of the channels in the SCM learns a soft weight for each spatial location of the nested relative salience stack in order to label pixels based on confidence that they belong to a salient object....|
|||...er missing spatial details in each stage of refinement and also preserve the relative rank order of salient objects....|
|||...  3.2.1 Rank-Aware Refinement Unit  Most existing works [24, 32, 37, 8] that have shown success for salient object detection typically share a common structure of stage-wise decoding to recover per-pi...|
|||...Following [12], we integrate gate units in our rank-aware refinement unit that control the information passed forward to filter out the ambiguity relating to figure-ground and salient objects....|
|||...odel forces the channel dimension to be the same as the number of participants involved in labeling salient objects....|
|||...Note that, we only forward the NRSS to the next stage, allowing the network to learn contrast between different levels of confidence for salient objects....|
|||...Stacked Representation of Ground(cid:173)truth  The ground-truth for salient object detection or segmentation contains a set of numbers defining the degree of saliency for each pixel....|
|||...N ) where each map Gi includes a binary indication that at least i observers judged an object to be salient (represented at a per-pixel level)....|
|||...N is the number of different participants involved in labeling the salient objects....|
|||...The stacked groundtruth saliency maps G provides better separation for multiple salient objects (see Eq....|
|||...Similar to our multiple salient object detection network, the subitizing network is also based on ResNet-101 [6] except we remove the last block....|
|||...end a fully connected layer at the end to generate confidence scores for each of 0, 1, 2, 3, and 4+ salient objects existing in the input image followed by another fully connected layer leads to gener...|
|||...As a classifier, the subitizing network reduces two cross entropy losses l1 sub(cf , n) between the number of salient objects n in ground-truth, and the total predicted objects....|
|||...A New Dataset for Salient Object Subitizing: Since salient object subitizing is not a widely addressed problem, a limited number of datasets [35] were created....|
|||...It is an evident from the table that, there is a considerable number of images with more than two salient objects but only few images with more than 7....|
|||...To reduce imbalance  sub(c, n) and lf  7145  # Salient Object  1  #Images  Distribution (%)  300 0.35  2  227 0.27  3  136 0.16  4  72 0.08  5  43 0.05  6  28 0.03  7  8+  Total  18 0.02  26 0.03  85...|
|||...Count and percentage of images corresponding to different numbers of salient objects in the Pascal-S dataset....|
|||...In this dataset, salient object labels are based on an experiment using 12 participants to label salient objects....|
|||...Virtually all existing approaches for salient object segmentation or detection threshold the ground-truth saliency map to obtain a binary saliency map....|
|||...This is one of the most highly used salient object segmentation datasets, but is unique in having multiple explicitly tagged salient regions provided by a reasonable sample size of observers....|
|||...Since a key objective of this work is to rank salient objects in an image, we use the original ground-truth maps (each pixel having a value corresponding to the number of observers that deemed it to b...|
|||...Evaluation Metrics: For the multiple salient object detection task, we use four different standard metrics to measure performance including precision-recall (PR) curves, F-measure (maximal along the c...|
|||...In ordered to evaluate the rank order of salient objects, we introduce the Salient Object Ranking (SOR) metric which is defined as the Spearmans Rank-Order Correlation between the ground truth rank or...|
|||...Performance Comparison with State(cid:173)of(cid:173)the(cid:173)art  The problem of evaluating salient detection models is challenging in itself which has contributed to differences among benchmarks ...|
|||...Quantitative Evaluation: Table 2 shows the performance score of all the variants of our model, and other recent methods on salient object detection....|
|||...Right: Precision-Recall curves for salient object detection corresponding to a variety of algorithms....|
|||...We can see that our method can predict salient regions accurately and produces output closer to ground-truth maps in various challenging cases e.g., instances touching the image boundary (1st & 2nd ro...|
|||...The nested relative salience stack at each stage provides distinct representations to differentiate between multiple salient objects and allows for reasoning about their relative salience to take place....|
|||...Predicted salient object regions for the Pascal-S dataset....|
|||...Each row shows outputs corresponding to different algorithms designed for the salient object detection/segmentation task....|
|||...4.2.1 Application: Ranking by Detection  As salient instance ranking is a completely new problem, there is not existing benchmark....|
|||...In order to promote this direction of studying this problem, we are interested in finding the ranking of salient objects from the predicted saliency map....|
|||...Rank order of a salient instance is obtained by averaging the degree of saliency within that instance mask....|
|||...It is worth noting that no prior methods report results for salient instance ranking....|
|||...The proposed method significantly outperforms other approaches in ranking multiple salient objects and our analysis shows that learning salient object detection implicitly learns rank to some extent, ...|
|||...5 shows a qualitative comparison of the state-of-the-art approaches designed for salient object detection....|
|||...Qualitative depiction of rank order of salient objects....|
|||...4.2.2 Application: Salient Object Subitizing  As mentioned prior, salient object detection, ranking, and subitizing are interrelated....|
|||...It is therefore natural to consider whether salient region prediction and ranking provide guidance to subitize....|
|||...These are most common for ties in the ground truth, and for scenes with many salient objects....|
|||...Conclusion  In this paper, we have presented a neural framework for detecting, ranking, and subitizing multiple salient objects that introduces a stack refinement mechanism to achieve better performance....|
|||...Deeply supervised salient object detection with short connections....|
|||...Adaptive location for multiple salient objects detection....|
|||...Deep contrast learning for salient object  detection....|
|||...Contextual hypergraph modeling for salient object detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Non-local deep features for salient object detection....|
|||...Towards the success rate of one: Real-time unconstrained salient object detection....|
|||...A stagewise refinement model for detecting salient objects in images....|
|||...What is and what is not a salient object?...|
|||...learning salient object detector by ensembling linear exemplar regressors....|
|||...Unconstrained salient object detection via proposal subset optimization....|
|||...Amulet: Aggregating multi-level convolutional features for salient object detection....|
||78 instances in total. (in cvpr2018)|
|4|Deng-Ping_Fan_Salient_Objects_in_ECCV_2018_paper|...We provide a comprehensive evaluation of salient object detection (SOD) models....|
|||...Our analysis identifies a serious design bias of existing SOD datasets which assumes that each image contains at least one clearly outstanding salient object in low clutter....|
|||...Beyond object category annotations, each salient image is accompanied by attributes that reflect common challenges in real-world scenes....|
|||...Keywords: Salient object detection  Saliency benchmark  Dataset  Attribute  1  Introduction  This paper considers the task of salient object detection (SOD)....|
|||...Specifically, most datasets assume that an image contains at least one salient object, and thus discard images that do not contain salient  2  Fan et al....|
|||...Sample images from our new dataset including non-salient object images (first row) and salient object images (rows 2 to 4)....|
|||...For salient object images, instancelevel ground truth map (different color), object attributes (Attr) and category labels are provided....|
|||...Our main contribution is the collection of a new high quality SOD dataset, named the SOC, Salient Objects in Clutter....|
|||...It differs from existing datasets in three aspects: 1) salient objects have category  Bringing Salient Object Detection to the Foreground  3  annotation which can be used for new research such as wea...|
|||...datasets designed for SOD tasks, especially in the aspects including annotation type, the number of salient objects per image, number of images, and image quality....|
|||...Previous SOD datasets only annotate the image by drawing (b) pixel-accurate silhouettes of salient objects....|
|||...Different from (d) MS COCO object segmentation dataset [27] (Objects are not necessarily being salient), our work focuses on (c) segmenting salient object instances....|
|||...2.1 Datasets  Early datasets are either limited in the number of images or in their coarse annotation of salient objects....|
|||...For example, the salient objects in datasets MSRAA [29] and MSRA-B [29] are roughly annotated in the form of bounding boxes....|
|||...ASD [1] and MSRA10K [11] mostly contain only one salient object in each image, while the SED2 [2] dataset contains two objects in a single image  4  Fan et al....|
|||...2 (b)) with more than one salient object in images....|
|||...2 (c)) salient objects annotation....|
|||...The ILSO [22] dataset contains instance-level salient objects annotation but has boundaries roughly labeled as shown in Fig....|
|||...To sum up, as discussed above, existing datasets mostly focus on images with clear salient objects in simple backgrounds....|
|||...limitations of existing datasets, a more realistic dataset which contains realistic scenes with non-salient objects, textures in the wild, and salient objects with attributes, is needed for future inv...|
|||...Single-task models have the single goal of detecting the salient objects in images....|
|||...Recently, a deep architecture [17] with  Bringing Salient Object Detection to the Foreground  5  Table 1....|
|||...MSR [22] was designed for both salient region detection and salient object contour detection,  6  Fan et al....|
|||...Almost all of the existing SOD datasets make the assumption that an image contains at least one salient object and discard the images that do not contain salient objects....|
|||...In a realistic setting, images do not always contain salient objects....|
|||...For example, some amorphous background images such as sky, grass and texture contain no salient objects at all [6]....|
|||...The non-salient objects or background stuff may occupy the entire scene, and hence heavily constrain possible locations for a salient object....|
|||...[41] proposed a state-of-the-art SOD model by judging what is or what is not a salient object, indicating that the non-salient object is crucial for reasoning about the salient object....|
|||...This suggests that the non-salient objects deserve equal attention as the salient objects in SOD....|
|||...Thus, we define the non-salient objects as images without salient objects or images with stuff categories....|
|||...Bringing Salient Object Detection to the Foreground  7  Based on the characteristics of non-salient objects, we collected 783 texture images from the DTD [21] dataset....|
|||...To this end, we gathered 6,000 images from more than 80 categories, containing 3,000 images with salient objects and 3,000 images without salient objects....|
|||...4 (a) shows the number of salient objects for each category....|
|||...3) Global/Local Color Contrast of Salient Objects....|
|||...As described in [26], the term salient is related to the global/local contrast of the foreground and background....|
|||...It is essential to check whether the salient objects are easy to detect....|
|||...4) Locations of Salient Objects....|
|||...Previous benchmarks often adopt this incorrect way to analyze the location distribution of salient objects....|
|||...From these statistics, we can observe that salient objects in our dataset do not suffer from center bias....|
|||...5) Size of Salient Objects....|
|||...The size of an instance-level salient object is defined as the proportion of pixels in the image [26]....|
|||...4 (g), the size of salient objects in our SOC varies in a broader range, compared with the only existing instance-level ILSO [22] dataset....|
|||...6) High-Quality Salient Object Labeling....|
|||...Similar to famous SOD task oriented benchmark dataset Bringing Salient Object Detection to the Foreground  9  (a) ILSO  (b) SOC  (c) MSCOCO  (d) SOC  Fig....|
|||...The list of salient object image attributes and the corresponding description....|
|||...After the first stage, we have 3,000 salient object images annotated with bboxes....|
|||...In the second stage, we further manually label the accurate silhouettes of the salient objects according to the bboxes....|
|||...In the end, we keep 3,000 images with high-quality, instance-level labeled salient objects....|
|||...7) Salient Objects with Attributes....|
|||...Left: Attributes distribution over the salient object images in our SOC dataset....|
|||...ACBOCLHOMBOCOVSCSOACBOCLHOMBOCOVSCSOACBOCLHOMBOCOVSCSOBringing Salient Object Detection to the Foreground  11  4.1 Evaluation Metrics  In a supervised evaluation framework, given a predicted map M ge...|
|||...Multi-task: Different from models mentioned above, MSR [22] detects the instance-level salient objects using three closely related steps: estimating saliency maps, detecting salient object contours, a...|
|||...4.3 Attributes-based Evaluation  We assign the salient images with attributes as discussed in Sec....|
|||...Bringing Salient Object Detection to the Foreground  13  Table 4....|
|||...Attributes-based performance on our SOC salient objects sub-dataset....|
|||...5 Discussion and Conclusion  To our best knowledge, this work presents the currently largest scale performance evaluation of CNNs based salient object detection models....|
|||...will evolve and grow over time and will enable research possibilities in multiple directions, e.g., salient object subitizing [46], instance level salient object detection [22], weakly supervised base...|
|||...Bringing Salient Object Detection to the Foreground  15  References  1....|
|||...Achanta, R., Hemami, S., Estrada, F., Susstrunk, S.: Frequency-tuned salient re gion detection....|
|||...Borji, A., Cheng, M.M., Jiang, H., Li, J.: Salient object detection: A benchmark....|
|||...Borji, A., Sihite, D.N., Itti, L.: Salient object detection: a benchmark....|
|||...: Global contrast  based salient region detection....|
|||...Jiang, H., Cheng, M.M., Li, S.J., Borji, A., Wang, J.: Joint Salient Object Detection  and Existence Prediction....|
|||...Li, G., Xie, Y., Lin, L., Yu, Y.: Instance-Level Salient Object Segmentation....|
|||...Li, G., Yu, Y.: Deep Contrast Learning for Salient Object Detection....|
|||...Li, X., Zhao, L., Wei, L., Yang, M.H., Wu, F., Zhuang, Y., Ling, H., Wang, J.: DeepSaliency: Multi-task deep neural network model for salient object detection....|
|||...: The secrets of salient object  segmentation....|
|||...Liu, N., Han, J.: DHSNet: Deep Hierarchical Saliency Network for Salient Object  Detection....|
|||...Luo, Z., Mishra, A.K., Achkar, A., Eichel, J.A., Li, S., Jodoin, P.M.: Non-local  deep features for salient object detection....|
|||...Wang, J., Jiang, H., Yuan, Z., Cheng, M.M., Hu, X., Zheng, N.: Salient Object Detection: A Discriminative Regional Feature Integration Approach....|
|||...Wang, L., Lu, H., Wang, Y., Feng, M., Wang, D., Yin, B., Ruan, X.: Learning to detect salient objects with image-level supervision....|
|||...learning salient object detector by ensembling linear exemplar regressors....|
|||...IEEE (2017)  Bringing Salient Object Detection to the Foreground  17  42....|
|||...Zhang, J., Ma, S., Sameki, M., Sclaroff, S., Betke, M., Lin, Z., Shen, X., Price, B.,  Mech, R.: Salient object subitizing....|
|||...Zhang, J., Dai, Y., Porikli, F.: Deep Salient Object Detection by Integrating Multilevel Cues....|
|||...Zhang, P., Wang, D., Lu, H., Wang, H., Ruan, X.: Amulet: Aggregating multi-level convolutional features for salient object detection....|
||76 instances in total. (in eccv2018)|
|5|Li_The_Secrets_of_2014_CVPR_paper|...tat.ucla.edu  Abstract  In this paper we provide an extensive evaluation of fixation prediction and salient object segmentation algorithms as well as statistics of major datasets....|
|||...Our analysis identifies serious design flaws of existing salient object benchmarks, called the dataset design bias, by over emphasising the stereotypical concepts of saliency....|
|||...The dataset design bias does not only create the discomforting disconnection between fixations and salient object segmentation, but also misleads the algorithm designing....|
|||...Based on our analysis, we propose a new high quality dataset that offers both fixation and salient object segmentation ground-truth....|
|||...With fixations and salient object being presented simultaneously, we are able to bridge the gap between fixations and salient objects, and propose a novel method for salient object segmentation....|
|||...Finally, we report significant benchmark progress on 3 existing datasets of segmenting salient objects....|
|||...Most of the works in computer vision focus on one of the following two specific tasks of saliency: fixation prediction and salient object segmentation....|
|||...in an salient object segmentation dataset, image labelers annotate an image by drawing pixel-accurate silhouettes of objects that are believed to be salient....|
|||...Various datasets of fixation and salient object segmentation have provided objective ways to analyze algorithms....|
|||...In this paper, we explore the connection between fixation prediction and salient object segmentation by augmenting 850 existing images from PASCAL 2010 [12] dataset with eye fixations, and salient obj...|
|||...With fixations and salient object labels simultaneously presented in the same set of images, we report a series of interesting findings....|
|||...First, we show that salient object segmentation is a valid problem because of the high consistency among labelers....|
|||...Second, unlike fixation datasets, the most widely used salient object segmentation dataset is heavily biased....|
|||...As a result, all top performing algorithms for salient object segmentation have poor generalization power when they are tested on more realistic images....|
|||...Finally, we demonstrate that there exists a strong correlation between fixations and salient objects....|
|||...4 we propose a new model of salient object segmentation....|
|||...By combining existing fixation-based saliency models with segmentation techniques, our model bridges the gap between fixation prediction and salient object segmentation....|
|||...Despite its simplicity, this model significantly outperforms state-of 4321  the-arts salient object segmentation algorithms on all 3 salient object datasets....|
|||...Related Works  In this section, we briefly discuss existing models of fixation prediction and salient object segmentation....|
|||...We also discuss the relationship of salient object to generic object segmentation such as CPMC [7, 19]....|
|||...[21] proposed the MSRA-5000 dataset with bounding boxes on the salient objects....|
|||...Inspired by this new dataset, a line of papers has proposed [9, 23, 22] to tackle this new challenge of predicting full-resolution masks of salient objects....|
|||...A typical fixation ground-truth contains several fixation dots, while a salient object ground-truth usually have one or several positive regions composed of thousands of pixels....|
|||...In comparison, a salient object detector aims at enumerating a subset of objects that exceed certain saliency threshold....|
|||...first 200) segments do not always correspond to salient objects or their parts....|
|||...We then conduct the experiment of 12 subjects to label the salient objects....|
|||...Given an image, a subject is asked to select the salient objects by clicking on them....|
|||...Similar to our fixation experiment, the instruction of labeling salient objects is intentionally kept vague....|
|||...For salient object segmentation task, the test/ground-truth saliency maps are binary maps obtained by first averaging the individual segmentations from the test/ground-truth subset, and then threshold...|
|||...C) and salient object (Fig....|
|||...The labeling of salient objects is based on the full segmentation (Fig....|
|||...While Bruce dataset was originally designed for fixation prediction, it was recently augmented by [5] with 70 subjects under the instruction to label the single most salient object in the image....|
|||...ion algorithms: ITTI [17], AIM [6], GBVS [1], DVA [16], SUN [26], SIG [15], AWS [13]; and following salient object segmentation algorithms: FT [1], GC [9], SF [23], and PCAS [22]....|
|||...4323  A: Original imageB: Full segmentationC: Fixation ground-truthD: Salient object ground-truthcompletely empty....|
|||...Images with two or more equally salient objects are very likely to become empty after thresholding....|
|||...Inter-subject consistency of 2 salient object segmentation datasets and 5 fixation datasets....|
|||...Similar to our consistency analysis of salient object dataset, we evaluate the consistency of eye fixations among subjects (Tab....|
|||...s ill-defined, we observe highly consistent behaviors among human labelers in both eye-fixation and salient object segmentation tasks....|
|||...Sharply contrasted to the fixation benchmarks, the performance of all salient object segmentation algorithms drop significantly when migrating from the popular FT dataset....|
|||...This result is alarming, because the magnitude of the performance drop from FT to any dataset by any algorithm, can easily dwarf the 4year progress of salient object segmentation on the widely used FT dataset....|
|||...Image statistics of the salient object segmentation datasets....|
|||...Dataset design bias  The performance gap among datasets clearly suggests new challenges in salient object segmentation....|
|||... analyze the following image statistics in order to find the similarities and differences of todays salient object segmentation datasets:  Local color contrast: Segmentation or boundary detection is a...|
|||...Right: The F-Measure scores of salient object segmentation....|
|||...At a first glance, our observation that FT contains unnaturally strong object boundaries seem acceptable, especially for a dataset focusing on salient object analysis....|
|||...Fixations and F-measure  Previous methods of salient object segmentation have reported big margin of F-measures over all fixation algorithms....|
|||...he saliency maps generated by all fixation algorithms, and then benchmark these algorithms on all 3 salient object datasets....|
|||...We also notice that the F-measure of the ground-truth human fixation maps is rather low compared to the inter-subject consistency scores in the salient object labeling experiment....|
|||...Once equipped with appropriate underlying representation, the human fixation map, as well as their algorithm approximations, generate accurate results for salient object segmentation....|
|||...From Fixations to Salient Object Detection  Many of todays well-known salient object algorithms have the following two components: 1) a suitable representation for salient object segmentation, and 2) ...|
|||...In this section, we build a salient object segmentation model by combining existing techniques of segmentation and fixation based saliency....|
|||...This simple combination results in a novel salient object segmentation method that outperforms all previous methods by a large margin....|
|||...The representation of CPMC-like object proposal is easIf all salient ily adapted to salient object segmentation....|
|||...objects can be found from the pool of object candidates, we can reduce the problem of salient object detection to a much easier problem of salient segment ranking....|
|||...We generate the salient object segmentation by averaging the top-K segments at pixel level....|
|||...Note that no appearance feature is used in our method, because our goal is to demonstrate the connection between fixation and salient object segmentation....|
|||...Best results of our model compared to best results of existing salient object algorithms....|
|||...Our model achieves better Fmeasure than all major salient object segmentation algorithms on all three datasets, including the heavily biased FT dataset....|
|||...F-measures of our salient object segmentation method under different choices of K, in comparison to CPMC ranking function....|
|||...4, we compare our results with the state-of-theart salient object algorithms with K = 20 (the benchmark result on other datasets is provided in our website)....|
|||...Our method outperforms the state-of-the-art salient object segmentation algorithms by a large margin....|
|||...In this dataset, we achieved an improvement of 11.82% with CPMC+GBVS in comparison to the best performing salient object algorithm PCAS....|
|||...Conclusion  In this paper, we explore the connections between fixation prediction and salient object segmentation by providing a new dataset with both fixations and salient object annotations....|
|||...Our analysis suggests that the definition of a salient object is highly consistent among human subjects....|
|||...We also point out significant dataset design bias in major salient object benchmarks....|
|||...We argue that the problem of salient object segmentation should move beyond the textbook examples of visual saliency....|
|||...A possible new direction is to look into the strong correlation between fixations and salient objects....|
|||...This simple model outperforms state-of-the-art salient object segmentation algorithms on all major datasets....|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Image signature: Highlighting sparse salient regions....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||73 instances in total. (in cvpr2014)|
|6|He_Delving_Into_Salient_ICCV_2017_paper|...Delving into Salient Object Subitizing and Detection  Shengfeng He1  Jianbo Jiao2  Xiaodan Zhang2  Guoqiang Han1 1South China University of Technology, China 2City University of Hong Kong, Hong Kong  ...|
|||...Lau2  3University of Chinese Academy of Sciences, China  Abstract  Subitizing (i.e., instant judgement on the number) and detection of salient objects are human inborn abilities....|
|||...We propose a multi-task deep neural network with weight prediction for salient object detection, where the parameters of an adaptive weight layer are dynamically determined by an auxiliary subitizing network....|
|||...The numerical representation of salient objects is therefore embedded into the spatial representation....|
|||...existing multi-task architectures, and the auxiliary subitizing network provides strong guidance to salient object detection by reducing false positives and producing coherent saliency maps....|
|||...Moreover, the proposed method is an unconstrained method able to handle images with/without salient objects....|
|||...Finally, we show state-of-theart performance on different salient object datasets....|
|||...Traditional methods focus solely on detecting salient objects....|
|||...First,  (a) Input  (b) w/o subitizing (c) w/ subitizing  (d) GT  Figure 1: Our method augments salient object detection with subitizing....|
|||...Subitizing provides strong guidance to accurately detect salient objects from complex background....|
|||...most methods assume the presence of salient objects, and fail if this is not the case....|
|||...[36, 38] propose to predict the number of salient objects, they only use it to filter images without salient objects....|
|||...No interactions are considered between the number and the detection of salient objects....|
|||...In this paper, we aim to explore the interaction between numerical and spatial representations in the salient object detection task....|
|||...While this CNN is trained for salient object detection, the weights of its adaptive weight layer are dynamically determined by an auxiliary subitizing network, allowing the layer to encode subitizing ...|
|||... We design a deep network to detect salient objects with the guidance of subitizing, by introducing an adaptive weight layer....|
|||...This layer integrates two different tasks by adaptively assigning weights according to the predicted number of salient objects....|
|||...Specifically, the proposed method outperforms existing methods on an unconstrained salient object dataset....|
|||...We are beginning to see some salient object detection works in these two years....|
|||...Multi-task Sharing Networks  Given an image, we aim to detect salient objects with the aid of subitizing....|
|||...Here, we explore three multi-task architectures for salient object detection, as shown in Fig....|
|||...twork, by exhaustively tuning the number of shared layers, to find out how shared knowledge affects salient object subitizing and detection performances....|
|||...For the subitizing task, the last convolution layer connects with a fully connected layer and then outputs the number of salient objects....|
|||...While sharing the 3rd to 5th convolution layers produces the best performance for salient object detection, sharing the 4th to 5th convolution layers produces the best performance for subitizing....|
|||...Proposed Network Architecture  This section presents the proposed network architecture  and the overall method for salient object detection....|
|||...ew  The proposed network is a multi-task deep neural network, containing three main components: the salient object detection network, subitizing network, and an adap 1061  Figure 3: The architecture ...|
|||...The upper part is the salient object detection network, and the bottom part is the subitizing network....|
|||...An adaptive weight layer is added in the middle of the salient object detection network, where its weights are dynamically determined by the subitizing network, to encode numerical representation into...|
|||...The salient object detection network is constructed based on the convolution-deconvolution pipeline....|
|||...a rich feature representation, while the deconvolution stage serves as a shape generator to segment salient objects based on the extracted features....|
|||...The final output is a probability map that indicates how likely each pixel belongs to the salient objects....|
|||...Deep Neural Network with Weight Prediction  Given an input image I, the salient object detection network produces a saliency map m from a set of weights ....|
|||...The salient object detection is posed as a regression problem, and the saliency value of each pixel (x, y) in m can be described as:  used to detect salient objects for any input images....|
|||...; , a(n)),  (2)  where a(n) is the adaptive weights determined according to the predicted number of salient objects, n, of the input image....|
|||...In this way, detecting salient objects is not only dependent on the static weights , but also the adaptive weights a(n)....|
|||...These adaptive weights can be considered as numerical representation of the salient objects....|
|||...etwork is trained,  is fixed and  4.2.1 Predicting Weights with Subitizing  To encode the number of salient objects into the adaptive weights, we implement the numerical feature embedding in a convolu...|
|||...In this way, the introduced adaptive weight matrix parameterizes this convolution layer as a function of the predicted number of salient objects....|
|||...The output features fn of the subitizing network is directly used as the adaptive weight matrix for salient object detection....|
|||...fully connected layer followed by a classification layer to predict the existence and the number of salient objects....|
|||...Before integrating to the salient object detection network, the subitizing network is first pre-trained solely for the subitizing task on the SOS dataset [36]....|
|||...Salient Object Detection Network  As mentioned earlier, we design the salient object detection network based on the convolution-deconvolution pipeline, and these two stages are connected by the adapti...|
|||...Salient Object Subitizing Network  A human is only able to identify up to 4 salient objects at a glance, effortlessly and consistently....|
|||...To accurately learn the object boundaries, we segment all the salient objects from the bounding boxes using the available ground truth segmentations from MS COCO [20] and Pascal VOC 2007 [7]....|
|||...We evaluate the proposed method on the MSO dataset [36], which is the test set of the SOS dataset [36] for salient object detection....|
|||...Although MSRA10K [4] is the largest dataset for salient object detection, most of the deep learning based models are trained on this dataset or its subset....|
|||...MAE  MAE  Ours  MAP [38]  0.654 0.370  0.187 0.119  0.061 0.068  0.846 0.511  0.235 0.139  Table 3: Unconstrained salient object detection evaluation on the MSO dataset [36]....|
|||...The proposed method outperforms MAP [38] not only on images with salient objects, but also on background images....|
|||...We can see that integrating subitizing is effective in salient object detection, contributing to an overall F-measure improvement of about 10% and producing 20% less error....|
|||...The main reason is that the proposed adaptive weight layer dynamically enriches the representation space of the salient object detection network, parameters will be generated according to the input context....|
|||...Beside the full MSO dataset, we further report the performance on background images only and on images with salient objects, to better verify the performance of subitizing embedding....|
|||...The proposed method augmented subitizing strategy achieves the best results on the scenarios with different numbers of salient objects....|
|||...Compared to the other deep learning based methods, we embed subitizing knowledge in salient object detection and thus achieve better performance....|
|||...5, the unconstrained method MAP [38] may misdetect the non-existence of salient objects....|
|||...Application: Detection Augmenting Subitizing  As mentioned earlier salient object detection and subitizing are mutually involved in the human visual system....|
|||...Hence, we are interested to find out if salient object detection may provide effective guidance to subitizing....|
|||...While salient object detection is used for weight prediction, subitizing produces our final prediction....|
|||...Failure Cases  Although detecting salient objects with subitizing guidance achieves good performances, the subitizing prediction may not necessarily agree with the ground truth on the number of salien...|
|||...As the subitizing network produces different numbers of salient objects from those of the ground truth, the salient object network outputs different saliency maps....|
|||...Conclusion  In this paper, we have explored the interactions between numerical and spatial representations in salient object detection....|
|||...To address the multi-task problem from a different point of view, we propose a multi-task deep neural network to detect salient objects with the augmentation of subitizing using dynamic weight prediction....|
|||...Extensive experiments demonstrate that subitizing knowledge provides strong guidance to salient object detection, and the proposed method achieves state-of-the-art performance on four datasets....|
|||...The subitizing guidance may sometimes disagree with the ground truth saliency maps on the number of salient objects....|
|||...We can see that the proposed method outperforms MSO and a fine-tuned VGG network on the scenarios with different numbers of salient objects....|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Global IEEE TPAMI,  contrast based salient region detection....|
|||...Supercnn: A superpixelwise convolutional neural network for salient object detection....|
|||...Deep contrast learning for salient object  detection....|
|||...The secrets  of salient object segmentation....|
|||...Real-time salient object detection with a minimum spanning tree....|
|||...Minimum barrier salient object detection at 80 fps....|
|||...Unconstrained salient object detection via proposal subset optimization....|
||72 instances in total. (in iccv2017)|
|7|Zhang_Salient_Object_Subitizing_2015_CVPR_paper|...The phenomenon, known as Subitizing, inspires us to pursue the task of Salient Object Subitizing (SOS), i.e....|
|||...predicting the existence and the number of salient objects in a scene using holistic cues....|
|||...It attains 94% accuracy in detecting the existence of salient objects, and 42-82% accuracy (chance is 20%) in predicting the number of salient objects (1, 2, 3, and 4+), without resorting to any objec...|
|||...Finally, we demonstrate the usefulness of the proposed subitizing technique in two computer vision applications: salient object detection and object proposal....|
|||...Introduction  How quickly can you tell the number of salient objects in each image in Fig....|
|||...In this paper, we propose a subitizing-like approach to estimate the number (0, 1, 2, 3 and 4+) of salient objects in a scene, without resorting to any object localization process....|
|||...Solving this Salient Object Subitizing (SOS) problem can benefit many computer vision tasks and applications....|
|||...Knowing the existence and the number of salient objects without the expensive detection process (e.g., sliding window detection) can enable a machine vision system to select different processing pipel...|
|||...SOS can help a computer vision system suppress the object detection process, until the existence of salient objects is detected, and it can also provide cues for selecting among search strategies and ...|
|||...Differentiating between scenes with zero, a single and multiple salient objects can also facilitate applications like robot vision [45], egocentric video summarization [31], snap point prediction [57]...|
|||...The annotations from the AMT workers are further analyzed in a more controlled offline setting, which shows a high inter-subject consistency in subitizing salient objects....|
|||...Our ultimate goal is to develop a fast and accurate computational method to estimate the number of salient objects  1  Figure 2: Sample images of the proposed SOS dataset....|
|||...nd Convolutional Neural Network (CNN) classifier attains 94% accuracy in detecting the existence of salient objects, and 42-82% accuracy (chance is 20%) in predicting the number of salient objects (1,...|
|||...We formulate the Salient Object Subitizing (SOS) problem, which aims to predict the number of salient objects in a scene without resorting to any object localization process....|
|||...We demonstrate applications of the SOS technique in guiding salient object detection and object proposal generation, resulting in state-of-the-art performance....|
|||...In the task of salient object detection [35, 48], we demonstrate that SOS can help improve accuracy by identifying images that contain no salient object....|
|||...Salient object detection aims at localizing salient objects in a scene by a foreground mask [1, 13] or bounding boxes [35, 23, 21, 48]....|
|||...However, existing salient object detection methods assume the existence of salient objects in an image....|
|||...Thus, counting salient objects using existent salient object detection methods can be quite unreliable....|
|||...Detecting the existence of salient objects....|
|||...Some works address the problem of detecting the existence of salient objects in an image....|
|||...In [55], a global feature based on several saliency maps is used to determine the existence of salient objects in thumbnail images, assuming an image either contains a single salient object or none....|
|||...It is worth noting that the testing images in [55, 45] are substantially simplified compared to ours, and the methods of [55, 45] cannot provide information about the number of salient objects....|
|||... counting approaches, in that it targets at category-independent inference of the number of generic salient objects, whose appearance can dramatically vary from category to category, and from image to...|
|||...The number of salient objects is shown in the red rectangle on each image....|
|||...The SOS Dataset  We describe the collection of the Salient Object Subitizing dataset, and then provide the labeling consistency analysis of the annotation collected via Amazon Mechanical Turk....|
|||...Image Source  To collect a dataset of images with different numbers of salient objects, we gathered a set of images from three object detection datasets, COCO [34], ImageNet [44] and VOC07 [19], and a...|
|||...f images from the SUN dataset to 2000, because most images in this dataset do not contain obviously salient objects, and we do not want the images from this dataset to dominate the category for no sal...|
|||...These images are a bit ambiguous about what should be counted as an individual salient object....|
|||...A few images do not have a clear notion about what should be counted as an individual salient object, and labels on those images tend to be divergent....|
|||...The ImageNet dataset contains more images with three salient objects than the other datasets....|
|||...However, the flexible viewing time allowed the AMT workers to look closely at these images, which may have had an influence over their attention and their answers to the number of salient objects....|
|||...Moreover, will shortening the viewing time change the common answers to the number of salient objects?...|
|||...was exposed to the subject for only 500 ms. After that, the subject was asked to tell the number of salient objects by choosing an answer from 0, 1, 2, 3, and 4+....|
|||... subitizing test, indicating that changing the viewing time may slightly affect the apprehension of salient objects....|
|||...epancy may be attributed to the fact that objects at the image center tend to be thought of as more salient than other ones given a short viewing time (see images in the top row of Fig....|
|||...Salient Object Subitizing  Since it remains an open problem to robustly detect salient objects, we propose a Salient Object Subitizing method for estimating the number of salient objects without resor...|
|||...Global Image Features  Although salient objects can have dramatically different appearance in color, texture and shape, we expect that global geometric information can be used to differentiate images ...|
|||...We use the state-of-the-art salient detection method of [58], and binarize its saliency maps using Otsus method [39]....|
|||...Since this counting-based method cannot determine the existence of salient objects in images, we only report its performance in predicting the number of salient objects....|
|||...The fine-tuned CNN attains over 90% AP scores in predicting images with no salient object and with a single salient object....|
|||...However, SalPyr is not as effective as HOG and GIST in predicting the existence of salient objects in an image....|
|||...Counting based on pixel connectivity is only reliable in idealistic cases, where salient objects are separated and well detected....|
|||...Many images in the SOS dataset have cluttered backgrounds and overlapping foreground objects, making the prediction of the number salient objects a non-trivial task....|
|||...For the remaining categories, there is still a considerable gap between human and machine performance, especially for categories with more than one salient object (compare Fig....|
|||...Application I: Salient Object Detection  In this section, we describe a simple application of the SOS technique for improving the accuracy of salient object detection....|
|||...Salient object detection aims to automatically localize salient objects in an image....|
|||...However, most salient object detection methods assume the presence of salient objects in an image; as a consequence, these methods can output unexpected results for images that contain no salient object [55]....|
|||...This suggests that we can exploit our CNN-based SOS method to identify images that contain no salient objects, as a precomputation for salient object detection methods....|
|||...For a given image, if our SOS method predicts that the image contains zero salient objects, then we do not apply salient object detection methods on that image....|
|||...We have found that this simple scheme can significantly improve efficiency and reduce false alarms for salient object detection....|
|||...Experiment on the MSO Dataset  Existing salient object detection benchmark datasets lack images that contain zero salient objects....|
|||...Moverover, these datasets usually have a majority of images where these is only a single salient object....|
|||...The MSO dataset has more balanced proportions of images with zero salient objects, one salient object, and multiple salient objects....|
|||...We believe that this dataset provides a more realistic setting to evaluate salient object detection methods....|
|||...We annotated a bounding box for each individual salient object in an image....|
|||...As shown in Table 4, more than 50% images in our MSO dataset contain either zero salient objects or more than one salient objects....|
|||...We test two state-of-the-art algorithms on our MSO dataset: Edge Boxes (EB) [59], which is an object proposal method, and LBI [48], which is a salient object detection method....|
|||...This improvement is attributed to the high accuracy of our CNN-based subitizing classifier in identifying images with no salient object (see Fig....|
|||...For example, for images predicted as containing a single salient object, we can prioritize bounding boxes covering all the salient regions....|
|||...mance of our subitizing classifier generalizes to a different dataset for detecting the presence of salient objects in images, we evaluate it on the web thumbnail image test set proposed in [55]....|
|||...50% of these images contain a single salient object, and the rest contain no salient object....|
|||...ects, and selectively reduce the number of retrieved proposals according to the predicted number of salient objects....|
|||...Because in the VOC07 dataset, objects are annotated regardless of whether they are salient or not, images predicted as containing no salient object can often have many background objects annotated....|
|||...identified as containing 1, 2 or 3 salient objects, and N proposals otherwise....|
|||...oduced the Salient Object Subitizing problem, which aims to predict the existence and the number of salient objects in an image using global image features, without resorting to any localization proce...|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...The secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
||70 instances in total. (in cvpr2015)|
|8|Xin_Li_Contour_Knowledge_Transfer_ECCV_2018_paper|...Contour Knowledge Transfer for Salient Object  Detection  Xin Li1, Fan Yang1, Hong Cheng1, Wei Liu1, Dinggang Shen2  1 University of Electronic Science and Technology of China, Chengdu 611731, China 2...|
|||...In recent years, deep Convolutional Neural Networks (CNNs) have broken all records in salient object detection....|
|||...Our goal is to overcome this limitation by automatically converting an existing deep contour detection model into a salient object detection model without using any manual salient object masks....|
|||...e two tasks, we further propose a contour-to-saliency transferring method to automatically generate salient object masks which can be used to train the saliency branch from outputs of the contour bran...|
|||...Different from these fully supervised methods, our method requires no groundtruth salient object mask for training deep CNNs....|
|||...Over the past decades, the techniques of salient object detection have evolved dramatically....|
|||...Traditional methods [3, 4, 20] only use low-level features and cues for identifying salient regions in an image, leading to their inability to summarize high-level semantic knowledge....|
|||...Recently, fully-supervised approaches [8, 9, 21, 24] based on deep Convolutional Neural Networks (CNNs) have greatly improved the performance of salient object detection....|
|||...The success of these methods depends mostly on a huge number of training images containing manually annotated salient objects....|
|||...Unfortunately, in salient object detection, annotations are provided in the form of pixel-wise masks....|
|||...To eliminate the need for time-consuming image annotation, we propose to facilitate feature learning in salient object detection by borrowing knowledge from an existing contour detection model....|
|||...Although salient object detection and contour extraction seem inherently different, they are actually related to each other....|
|||...For example, salient regions are often surrounded by contours....|
|||...Our goal is to convert a trained contour detection model (CEDN) [35] into a saliency detection model without using any manually labeled salient object masks....|
|||...After that, the trained branch in turn transfers the learned saliency knowledge,  OursInputAmuletGTUCFDSSContour Knowledge Transfer for Salient Object Detection  3  in the form of saliency-aware cont...|
|||...Although the generated salient object masks and saliency-aware contour labels may contain errors in the beginning, they gradually become more reliable after several iterations....|
|||...cy-to-Contour procedure), becoming a powerful saliency detection model, where one branch focuses on salient object contour detection and the other branch predicts saliency score of each pixel....|
|||...Despite not using manually annotated salient object labels for training, our proposed method is capable of generating a reliable saliency map for each input (See Fig....|
|||... this paper makes the following three major contributions:   We present a new idea and solution for salient object detection by automatically converting a well-trained contour detection model into a s...|
|||... We introduce a simple yet effective contour-to-saliency transferring method to bridge the gap between contours and salient object regions....|
|||...In recent years, fully-supervised CNNs have demonstrated highly accurate performance in salient object detection tasks....|
|||...[19] proposed the use of a multi-task fully-convolutional neural network for salient object detection....|
|||...[30] proposed a recurrent FCN to encode saliency prior knowledge for salient object detection....|
|||...introduce short connections into the Holistically-nested Edge Detector (HED) network architecture [31] so as to solve scale-space problems in salient object detection....|
|||...Notable previous attempts at detecting salient object(s), while using no saliency mask for training, are Weakly Supervised Saliency (WSS) [29] and Supervision by Fusion (SBF) [37] methods....|
|||...Furthermore, the contour knowledge is successfully transferred for salient region detection....|
|||...To the best of our knowledge, the idea of transferring contour knowledge for salient object detection has not been investigated before....|
|||...3 Approach  3.1 Overview  This paper tackles the problem of borrowing contour knowledge for salient object detection without the need of labeled data....|
|||...Given an existing contour detection network (CEDN) [35], our objective is to convert this already well-trained model  Contour Knowledge Transfer for Salient Object Detection  5  Fig....|
|||...ocedures above enables the saliency branch to progressively derive semantically strong features for salient object detection, and the contour branch learns to identify only the contours of salient reg...|
|||...The two-branch C2S-Net roots in the CEDN [35] for salient object detection....|
|||...h-level feature representations from an input image, the contour decoder identifies contours of the salient region, and the saliency decoder estimates the saliency score of each pixel....|
|||...Because salient object detection is a more difficult task than contour detection, we add another convolutional layer in each saliency decoder group....|
|||...enc)Contour Knowledge Transfer for Salient Object Detection  7  where Lsal(Ii) is the ground-truth salient object mask of the i-th image, and esal(Lsal(Ii), C(Fi; s)) is the per-pixel loss of S(Fi; s...|
|||...The detected contours provide important cues for salient object detection....|
|||...As observed by many previous works [6, 7], salient objects are usually well-surrounded by contours or edges....|
|||...Therefore, we can leverage this important cue to bridge the gap between object contours and salient object regions....|
|||...ps in a large collection of unlabeled images, our goal is to utilize them to generate corresponding salient object masks, so as to simulate strong human supervision over saliency branch training....|
|||...ective function to pick out only a very few masks B from C that are most likely to cover the entire salient regions to form the salient object mask Lsal for each image....|
|||...Si denotes the score reflecting the likelihood of region mask bi to be a salient region mask....|
|||...According to [6, 7], a region that is better surrounded by contours is more likely to be a salient region....|
|||...We initialize parameters of both  ImageIteration 1Iteration 2Iteration 3ImageIteration 1Iteration 2Iteration 3Contour Knowledge Transfer for Salient Object Detection  9  fenc and fcont by parameter v...|
|||...After that, we use the proposed contour-to-saliency transfer method to produce salient object masks Lsal as training samples for updating the saliency decoder parameters s....|
|||...These generated results are then utilized to produce salient object masks on unlabeled set N using Eq....|
|||...4, the estimated salient object masks and contour maps become more and more reliable, and then provide useful information for network training....|
|||...Compared with only sharing the  Contour Knowledge Transfer for Salient Object Detection  11  Table 1....|
|||...Automatically generating a reliable salient object mask for each image, based on generated proposal candidate masks C (about 500 proposals), is a challenging task....|
|||...Specifically, for AVG-P, we first simply take an average of all proposals (generated from detected contours) to form a saliency map for each image, and then use SalCut [3] to produce its salient object mask....|
|||...6 and only the proposal with the highest score is picked out to serve as salient object mask for each image....|
|||...3.3 to produce salient object masks for all images....|
|||...In order to obtain a fair comparison with existing weakly supervised and unsupervised deep models, we first use the same training  Contour Knowledge Transfer for Salient Object Detection  13  Table 2....|
|||...It also should be noted that our method requires no manual salient object label for training the network while other top-ranked deep models are trained with pixel-wise annotations....|
|||...It can be seen that our method can consistently and accurately highlight the salient objects in different challenging cases....|
|||...To bridge the gap between contours and salient object regions, we propose a novel transferring method that can automatically generate a saliency mask for each image from its contour map....|
|||...DHSDCLDSSUCFAmuletWSSOursGTInputSBFContour Knowledge Transfer for Salient Object Detection  15  References  1....|
|||...Deng, Q., Luo, Y.: Edge-based method for detecting salient objects....|
|||...Du, S., Chen, S.: Salient object detection via random forest....|
|||...Hu, P., Shuai, B., Liu, J., Wang, G.: Deep level sets for salient object detection....|
|||...Li, G., Yu, Y.: Deep contrast learning for salient object detection....|
|||...Li, X., Zhao, L., Wei, L., Yang, M.H., Wu, F., Zhuang, Y., Ling, H., Wang, J.: Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...Li, X., Yang, F., Chen, L., Cai, H.: Saliency transfer: An example-based method  for salient object detection....|
|||...Li, X., Yang, F., Cheng, H., Chen, J., Guo, Y., Chen, L.: Multi-scale cascade  network for salient object detection....|
|||...: The secrets of salient object  segmentation....|
|||...Liu, N., Han, J.: Dhsnet: Deep hierarchical saliency network for salient object  detection....|
|||...Luo, Z., Mishra, A., Achkar, A., Eichel, J., Li, S., Jodoin, P.M.: Non-local deep  features for salient object detection....|
|||...Wang, L., Lu, H., Wang, Y., Feng, M., Wang, D., Yin, B., Ruan, X.: Learning to  detect salient objects with image-level supervision....|
|||...Zhang, P., Wang, D., Lu, H., Wang, H., Ruan, X.: Amulet: Aggregating multi-level  convolutional features for salient object detection....|
||67 instances in total. (in eccv2018)|
|9|Zhang_Unconstrained_Salient_Object_CVPR_2016_paper|...en2 Brian Price2 Radom r M ech2  1Boston University  2Adobe Research  Abstract  We aim at detecting salient objects in unconstrained images....|
|||...In unconstrained images, the number of salient objects (if any) varies from image to image, and is not given....|
|||...We present a salient object detection system that directly outputs a compact set of detection windows, if any, for an input image....|
|||...Our system leverages a Convolutional-NeuralNetwork model to generate location proposals of salient objects....|
|||...% relative improvement in Average Precision compared with the state-of-the-art on three challenging salient object datasets....|
|||...Introduction  In this paper, we aim at detecting generic salient objects in unconstrained images, which may contain multiple salient objects or no salient object....|
|||...Solving this problem entails generating a compact set of detection windows that matches the number and the locations of salient objects....|
|||...(Existence) Is there any salient object in the image?...|
|||...(Localization) Where is each salient object, if any?...|
|||...Furthermore, individuating each salient object (or reporting that no salient object is present) can critically alleviate the ambiguity in the weakly supervised or unsupervised learning scenario [10, 2...|
|||...In this paper, we will use the term salient region detection when referring to these methods, so as to distinguish from the salient object detection task solved by our approach, which includes individ...|
|||...Some methods generate a ranked list of bounding box candidates for salient objects [21, 43, 52], but they lack an effective way to fully answer the questions of Existence and Localization....|
|||...Other salient object detection methods simplify the detection task by assuming the existence of one and only one salient object [48, 45, 32]....|
|||...In contrast to previous works, we present a salient object detection system that directly outputs a compact set of detections windows for an unconstrained image....|
|||...Our system leverages the high expressiveness of a Convolutional Neural Network (CNN) model to generate a set of scored salient object proposals for an image....|
|||...A key difference between salient object detection and object class detection is that saliency greatly depends on the surrounding context....|
|||...Therefore, the salient object proposal scores estimated on local image regions can be inconsistent with the ones estimated on the global scale....|
|||...To summarize, the main contributions of this work are:   A salient object detection system that outputs compact  detection windows for unconstrained images,   A novel MAP-based subset optimization for...|
|||...Salient region detection aims at generating a dense foreground mask (saliency map) that separates salient objects from the background of an image [1, 11, 41, 50, 25]....|
|||...Some methods allow extraction of multiple salient objects [33, 28]....|
|||...Most of these methods critically rely on  the assumption that there is only one salient object in an image....|
|||...Predicting the existence of salient objects....|
|||...In [49, 40], a binary classifier is trained to detect the existence of salient objects before object localIn [53], a salient object subitizing model is proization....|
|||...posed to suppress the detections on background images that contain no salient object....|
|||...While they can lead to substantial speed-ups over sliding window approaches for object detection, these proposal methods are not optimized for localizing salient objects....|
|||...Some methods [43, 21] generate a ranked list of proposals for salient objects in an image, and can yield accurate localization using only the top few proposals....|
|||...A Salient Object Detection Framework  Our salient object detection framework comprises two steps....|
|||...We first present the subset optimization formulation, as it is independent of the implementation of our proposal generation model, and can be useful beyond the scope of salient object detection....|
|||...On the other hand, salient objects can also overlap each other to varying extents....|
|||...At the same time, we favor a compact set of detections that explains the observations, as salient objects are distinctive and rare in nature [16]....|
|||...es the question of Existence, as the number of detections tends to be zero if no strong evidence of salient objects is found (Eq....|
|||...Salient Object Proposal Generation by CNN  We present a CNN-based approach to generate scored i=1 for salient objects....|
|||...As our CNN model takes the whole image as input, it is able to capture context information for localizing salient objects....|
|||...We find this uniformly sampling inefficient for salient object detection, and sometimes it even worsens the performance in our task (see Sec....|
|||...To train our CNN model, we use about 5500 images from the training split of the Salient Object Subitizing (SOS) dataset [53]....|
|||...The SOS dataset comprises unconstrained images with varying numbers of salient objects....|
|||...In particular, the SOS dataset has over 1000 background/cluttered images that contain no salient objects, as judged by human annotators....|
|||...As the SOS dataset only has annotations about the number of salient objects in an image, we manually annotated object bounding boxes according to the number of salient objects given for each image....|
|||...Following [21, 43], we use the PASCAL evaluation protocol [18] to evaluate salient object detection performance....|
|||...Traditional salient region detection methods [1, 11, 41, 50, 25] cannot be fairly evaluated in our task, as they only generate saliency maps without individuating each object....|
|||...Therefore, we mainly compare our method with two state-of-the-art methods, SC [21] and LBI [43], both of which output detection windows for salient objects....|
|||...We also evaluate a recent CNN-based object proposal model, MultiBox (MBox) [17, 46], which is closely related to our salient object proposal method....|
|||...20 object classes in PASCAL VOC [18]), regardless whether they are salient or not....|
|||...We evaluate our method mainly on three benchmark salient object datasets: MSO [53], DUT-O [51] and MSRA [29]....|
|||...MSO contains many background images with no salient object and multiple salient objects....|
|||...MSRA comprises 5000 images, each containing one dominant salient object....|
|||...However, it has been used for evaluating salient object detection in [21, 43]....|
|||...As in [21, 43], we use all the annotated bounding boxes in VOC07 as class-agnostic annotations of salient objects....|
|||...On VOC07, our method is slightly worse than MBox [46], but VOC07 is not a salient object dataset....|
|||...Reporting the nonexistence of salient objects is an important task by itself [53, 49]....|
|||...Thus, we further evaluate how our method and the competing methods handle background/cluttered images that do not contain any salient object....|
|||...As shown in Table 2, the proposal score generated by SC and LBI is a poor indicator of the existence of salient objects, since their scores are not calibrated across images....|
|||...e also include the results on five subsets of images for more detailed analysis: 1) 886 images with salient objects, 2) 611 images with a single salient object, 3) 275 images with multiple salient obj...|
|||...However, on images with a large object, uniform region sampling worsens the performance, as it may introduce window proposals that are only locally salient, and it tends to cut the salient object....|
|||...Even on constrained images with a single salient object, our subset optimization formulation still provides 12% relative improvement over the best baseline (shown in red in Table 3)....|
|||...This shows the robustness of our formulation in handling images with varying numbers of salient objects....|
|||...Conclusion  We presented a salient object detection system for unconstrained images, where each image may contain any number of salient objects or no salient object....|
|||...Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...The  secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Learning optimal seeds for diffusion-based salient object detection....|
|||...A unified approach to salient object  detection via low rank matrix recovery....|
|||...FASA: Fast, Accurate, and Size Aware Salient Object Detection....|
|||...Minimum barrier salient object detection at 80 fps....|
||66 instances in total. (in cvpr2016)|
|10|What Is and What Is Not a Salient Object_ Learning Salient Object Detector by Ensembling Linear Exemplar Regressors|...Learning Salient Object Detector by Ensembling Linear Exemplar Regressors  What is and What is not a Salient Object?...|
|||...tute for Multidisciplinary Science, Beihang University  Abstract  Finding what is and what is not a salient object can be helpful in developing better features and models in salient object detection (...|
|||...As a result, we propose a novel salient object detector by ensembling linear exemplar regressors....|
|||... the extracted object proposals, and a linear exemplar regressor is trained to encode how to detect salient proposals in a specific image....|
|||...In SOD, a key step is to distinguish salient and non-salient objects using the visual attributes....|
|||...However, in complex scenarios it is often unclear which attributes inherently make an object pop-out and how to separate salient and non-salient objects sharing some attributes (see Fig....|
|||...As a result, an investigation on what is and what is not a salient object is necessary before developing SOD models....|
|||...The main attributes of salient objects in the three images are location (1st row), shape (2nd row) and color (3rd row), while attributes shared with non-salient objects are shape (1st row), color (2nd...|
|||...In [10], salient objects were considered to be unique and have compact spatial distribution, or be distinctive with respect to both their local and global surroundings [13]....|
|||...Based on these findings, heuristic features can be designed to identify whether a region [8, 24, 16, 44], a superpixel [19, 14] or a pixel [37, 34, 42] is salient or not....|
|||...Typically, these models can achieve impressive performance when salient and non-salient objects are remarkably different....|
|||...However, in complex scenarios that salient and non-salient objects may share some visual attributes, making them difficult to be separated (see Fig....|
|||... fine-tuned, Moreover, it is still unclear what visual attributes contribute the most in separating salient and non-salient objects due to the black box characteristic of  43214142  Figure 2....|
|||...As a result, finding what is not a salient object is as important as knowing what is a salient object, and the answer can be helpful for designing better features and developing better SOD models....|
|||...ed in the construction process so as to find a more precise definition on what is and what is not a salient object....|
|||...Moreover, from the 10, 000 images included in the dataset, we find that objects can become salient in diversified ways that may change remarkably in different scenes, which may imply that a SOD model ...|
|||...Inspired by these two findings on what are salient and non-salient objects, we propose a simple yet effective approach for image-based SOD by ensembling plenty of linear exemplar regressors....|
|||...ndness propagation process so as to derive a foregroundness map that is capable  to roughly pop-out salient objects and suppress non-salient ones that have many similar candidates....|
|||...y its shape, attention and foregroundness descriptor, and such descriptor are then delivered into a salient object detector formed by ensembling various linear exemplar regressors so as to detect the ...|
|||...ge by using the same proposal descriptor, while each regressor encodes a specific way of separating salient objects from non-salient ones....|
|||...ce for training and testing SOD models; 2) We conduct an investigation on what is and what is not a salient object in constructing the dataset, based on which an effective salient object detector is p...|
|||...(a) Scene complexity: simple (top) to complex (bottom), (b) Number of salient objects: single (top) to multiple (bottom), (c) Object size: small (top) to large (bottom), (d) Object position: center (t...|
|||...nd What is Not a Salient Object  To obtain a comprehensive explanation on what is and what is not a salient object, a feasible solution is to investigate the whole process in constructing a new SOD da...|
|||...From these observations, we can infer the key attributes of salient and non-salient objects as well as the subjective bias that may inherently exist in image-based SOD datasets....|
|||...he second stage, these two engineers are further asked to manually label the accurate boundaries of salient objects in 10, 000 images tagged with Yes. Note that we have 10 volunteers involved in the w...|
|||...3, images in XPIE cover a variety of simple and complex scenes with different numbers, sizes and positions of salient objects....|
|||...Sometimes the most salient region is considered to be not an object due to its semantic attributes (e.g., the rock and road in Fig....|
|||...From these three reasons, we can also derive a definition of salient object that can be combined with previous definitions [5]....|
|||...That is, a salient object should have a limited similar distractors, relatively clear and simple shape and high objectness....|
|||...Moreover, we find that salient objects can popout for having specific visual attributes in different scenarios, which implies that a good SOD model should encode all probable ways that salient objects...|
|||...In other words, the inter-object similarities between objects are useful cues in separating salient and non-salient objects....|
|||...Inspired by this fact, we propose to estimate a foregroundness map to depict where salient objects may reside by using such inter-object similarity....|
|||...In this manner, large salient objects can pop-out as a whole (see Fig....|
|||...With these feature vectors, a linear exemplar regressor (v) can be trained to separate salient and non-salient objects on a specific image by minimizing  min    1 2  kwk2  2 + C + XOO+  O + C  XOO  O,  s.t....|
|||... O  O+, wTvO + b  1  O, O  0,  O  O, wTvO + b  O  1, O  0,  (7)  where C + and C  are empirically set to 1/ O+  and 1/ O  to balance the influence of probable salient objects and distractors....|
|||...Ensembling for Salient Object Detection  Given all  linear exemplar regressors, a proposal O in a testing image gains  I  saliency scores, denoted as {I(vO) I  I}....|
|||...Learning a Salient Object Detector by En sembling Linear Exemplar Regressors  4.1....|
|||...Training Linear Exemplar Regressors  Given the foregroundness map, we can train a salient object detector by ensembling linear exemplar regressors....|
|||...It covers many complex scenes with different numbers, sizes and positions of salient objects....|
|||...First, the investigation about what is and what is not a salient object provides useful cues in designing effective features for separating salient and non-salient objects....|
|||...Actually, the proposal-based framework is similar to the way that human perceives salient objects....|
|||...Third, various ways of separating salient and non-salient objects in diversified scenes are isomorphically represented with the exemplar-based linear regressors....|
|||...The enhancement-based fusion strategy for combining exemplar scores makes the learned salient detector emphasize more on the most relevant linear exemplar  43264147  Figure 6....|
|||...In this manner, regressors with the capability of separating salient and nonsalient objects in this scene can be emphasized, making the ensembled model scene adaptive....|
|||... to enhance the contrast of saliency maps, followed by a morphological operation to obtain a smooth salient object (see Fig....|
|||...Actually, saliency is a relative concept and with the training data from only one image we can simply infer how to separate specific salient objects from specific non-salient ones....|
|||...On the contrary, the extended bag of negatives can tell us more about how to separate specific salient objects from massive non-salient ones....|
|||...Conclusions  Knowing what is and what is not a salient object is important for designing better features and developing better models for image-based SOD....|
|||...By investigating the visual attributes of salient and non-salient objects, we find that non-salient objects often have many similar candidates, complex shape and low objectness, while salient objects ...|
|||...eserving foregroundness propagation, which can be used to extract effective features for separating salient and non-salient objects....|
|||...Moreover, we train an effective salient object detector by ensembling plenty of linear exemplar regressors....|
|||...Frequency-tuned salient region detection....|
|||...What is a salient object?...|
|||...a dataset and a baseline  model for salient object detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Deep contrast learning for salient object  detection....|
|||...The secrets  of salient object segmentation....|
|||...DHSNet: Deep hierarchical saliency net work for salient object detection....|
|||...Learning to detect a salient object....|
|||...Minimum barrier salient object detection at 80 fps....|
|||...Harf: Hierarchy-associated rich  features for salient object detection....|
||62 instances in total. (in cvpr2017)|
|11|Zou_HARF_Hierarchy-Associated_Rich_ICCV_2015_paper|...rsite Paris-Est, Ecole des Ponts ParisTech  nikos.komodakis@enpc.fr  Abstract  The state-of-the-art salient object detection models are able to perform well for relatively simple scenes, yet for more ...|
|||... such an issue, this paper proposes a novel hierarchy-associated feature construction framework for salient object detection, which is based on integrating elementary features from multi-level regions...|
|||...This leads to a rich feature representation, which is able to represent the context of the whole object/background and is much more discriminative as well as robust for salient object detection....|
|||...Extensive experiments on the most widely used and challenging benchmark datasets demonstrate that the proposed approach substantially outperforms the state-of-theart on salient object detection....|
|||...The focus of the present paper is exactly on this task (usually also referred to as salient object detection in the computer vision literature), which is essentially equivalent to solving a binary for...|
|||...Thus the state-of-the-art salient object detection models typically compute visual features based on regions for saliency evaluation, where it is important to note that the robustness and richness of ...|
|||...However, one problem from this is that the features extracted from small regions might be not sufficiently discriminative for detecting salient objects in complex scenes....|
|||...A suggestion might be to adjust the segmentation parameters so that an object is composed of very few regions to facilitate the salient object detection task....|
|||...Most previous models suffer from their limited robustness to highlight the whole salient object in complex scenes....|
|||...accurate salient object detection....|
|||...As a result, even the state-of-the-art saliency models still have significant difficulties in completely highlighting salient objects in complex scenes (see Figure 1)....|
|||... extract features that are both more robust and also richer based on the over-segmented regions for salient object detection?....|
|||...To that end,  this paper proposes a novel hierarchyassociated feature construction framework for salient object detection, which integrates various elementary features from both the target region and ...|
|||...Our hypothesis is that features computed in this manner are able to represent the context of the entire object/background and are much more discriminative as well as robust for salient object detection....|
|||...ely used benchmark datasets demonstrate that the HARF representation achieves higher performance of salient object detection and the proposed approach substantially outperforms the state-of-the-art mo...|
|||...We propose a novel hierarchy-associated feature construction framework for salient object detection that allows much more robust and discriminative features, which utilize information from multiple im...|
|||...Concerning the salient object detection problem, which is the focus of this paper, in [36] it was defined as a binary segmentation problem for application to object recognition....|
|||...Since then, plenty of saliency models have been proposed for detecting salient objects in images based on various theories and principles, such as information theory[37], graph theory [15, 23, 41], st...|
|||...oreover, a variety of effective measures and priors are explored to achieve a higher performance of salient object detection, e.g., local and global contrast measures [1, 9, 16, 28, 30, 33, 40], cente...|
|||...Apart from detecting salient objects in a single image, salient object detection also has been extended to identifying common salient objects shared in multiple images and video sequences....|
|||...For a comprehensive survey on salient object detection models, the interested reader is referred to [3]....|
|||...Some recent models have exploited hierarchical architectures for salient object detection....|
|||...le in [28] the most confident regions in a saliency tree are selected to improve the performance of salient object detection....|
|||...In contrast, we focus on extracting discriminative and robust features through hierarchical representation for detecting salient objects in complex scenes....|
|||...Thus the extracted features only from the small regions would be not sufficiently discriminative for salient object detection....|
|||...Notice that, the whole image at the root-node is not included for HARF computation because it certainly contains both salient object and background regions....|
|||...However, salient object detection focuses on highlighting salient objects from background, and does not need to assign semantic labels to any objects....|
|||...ression  With the HARF representation for each leaf region of the binary segmentation tree, we cast salient object detection as a regression problem that predicts the saliency of a region....|
|||...MSRA-B includes 5000 images, most of which contain a single salient object from a variety of scenes....|
|||...Many images in PASCAL-1500 contain multiple salient objects with various scales, locations and or highly clustered backgrounds, which are challenging in salient object detection....|
|||...With the HARF1 or HARF2 representation over the CNN and traditional features, the performance of salient object detection increases even further, yielding a substantial improvement....|
|||...y-connected layers of CNN (which carry more semantic information than the lower ones) contribute to salient object detection or not, we generate saliency results by using only the lower CNN layers (la...|
|||...All the salient object detection results of the compared models are generated using the authors implementations with their default parameters....|
|||...el are always higher than others at the top-right corners, which suggests that our model highlights salient objects in a more complete manner thanks to the HARF representation....|
|||...examples we make the following observations:  Heterogeneous appearances: For some images containing salient objects with heterogeneous appearances (e.g., the first row), previous models tend to highli...|
|||...Low contrast: For some images showing low contrast  to salient objects (e.g., the 2nd row), the proposed model can suppress irrelevant background regions and can highlight salient objects that have mo...|
|||...Multiple salient objects: The proposed model generates better looking saliency maps for those images containing multiple salient objects (e.g., the last two rows)....|
|||...The HARF structure is able to represent the global context of the whole salient object or background, which allows much more robust and discriminative features....|
|||...mark datasets demonstrated that the proposed approach substantially outperforms the state-ofthe-art salient object detection models both quantitatively (over 31% MAE decrease) and qualitatively....|
|||...Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Global contrast based salient region detection....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Submodular salient region detec tion....|
|||...Contextual hypergraph modeling for salient object detection....|
|||...Learning to detect a salient object....|
|||...Learning optimal seeds for diffusion-based salient object detection....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object  detection via low rank matrix recovery....|
|||...Modeling attention to salient proto objects....|
||52 instances in total. (in iccv2015)|
|12|Kruthiventi_Saliency_Unified_A_CVPR_2016_paper|...itute of Science, Bangalore, India  Abstract  Human eye fixations often correlate with locations of salient objects in the scene....|
|||...Our network shows a significant improvement over the current state-of-the-art for both eye fixation prediction and salient object segmentation across a number of challenging datasets....|
|||...Illustrative images (a) with their corresponding eye fixation predictions (b), groundtruth (c) and salient object segmentation predictions (d), groundtruth (e)....|
|||...The second task of salient object segmentation requires the generation of a pixel-accurate binary map indicating the presence of striking objects in the image....|
|||...An example image with eye fixation and salient object segmentation maps generated by our model along with the ground-truth maps are shown in Fig....|
|||...Recent studies have shown that the two tasks of eye fixation prediction and salient object segmentation are correlated [8, 9]....|
|||...Human eye fixations are often found to be guided by the locations of salient objects in the scene....|
|||...[9] who used a simple eye fixation based model for segmenting salient objects in an image and achieved stateof-the-art results....|
|||...5781  In this work, we propose a deep convolutional architecture for simultaneously predicting the human eye fixations and segmenting salient objects in an image....|
|||...These works defined immediate attention-grabbing objects in a scene as salient objects and formulated the problem of salient object segmentation to predict a pixel-accurate binary mask of the salient ...|
|||...[17] proposed a frequency domain approach for segmenting salient objects using low-level features of color and luminance....|
|||...[20] segmented salient objects using fixation points as identification markers on objects and found an optimal contour around the fixation points....|
|||...[9] proposed a salient object segmentation model which used the saliency maps from existing fixation prediction algorithms to classify image regions marked by an object proposal algorithm as a salient object....|
|||...Recently, in the realm of salient object segmentation, Zhao et al....|
|||...Architecture overview of the proposed network for simultaneously predicting human eye fixations and segmenting salient objects....|
|||...Network Architecture  We propose a fully convolutional deep network with a branched architecture for simultaneously predicting eye fixations and segmenting salient objects....|
|||...Efficient detection of these salient regions in an image would require the model to capture the global context of the image before assigning scores to its individual regions....|
|||...Salient Object Segmentation  Salient object segmentation consists of two sub-tasks: detecting the salient objects in the image and determining the spatial extent of the object by identifying its boundaries....|
|||...While detection of a salient object requires the image regions to be characterized using contextually rich semantic features, the task of finding the objects spatial extent requires lower level semant...|
|||...We extract features from the max-pool layers of CONV-2, CONV-4, CONV-5 and CONV-6 blocks for the task of salient object segmentation....|
|||...Architectural details of the proposed deep convolutional network for segmenting salient objects and predicting eye fixations  salient object segmentation [8]....|
|||...Following the multi-scale approach described earlier for salient object segmentation, we tap the max-pool layer of the CONV-6 block using an inception module with layers of receptive fields: 1  1, 3  ...|
|||...In the case of salient object segmentation, the groundtruth maps are binary and have sharp edges at the object boundaries....|
|||...Since the networks salient object predictions are coarse in resolution, we use the fully connected Conditional Random Field (CRF) formulation of Phillip et al....|
|||...The unary costs for a node to take the labels salient and background are defined using the networks object saliency map prediction....|
|||...The additive inverse of this object saliency map is taken to be the unary cost for the salient label....|
|||...Each of these images is provided with a pixel-accurate ground truth binary mask indicating the salient object and is used for training the network for segmentation task....|
|||...2, are shared between both the tasks of salient object segmentation and eye fixation prediction and are trained for both the tasks simultaneously using all the images in the batch....|
|||...PASCAL-S [9], DUT-OMRON [40], iCoSeg [41] and ECSSD [42] datasets consisting of 850, 5168, 643 and 1000 images respectively are used for evaluating the model on the task of salient object segmentation....|
|||...Qualitative results of our approach along with other state-of-the-art methods for salient object segmentation....|
|||...We used Mean Absolute Error (MAE) and Weighted F-Measure to evaluate the performance of our network for salient object segmentation....|
|||...5.2.1 Salient Object Segmentation  Mean Absolute Error (MAE) : MAE is computed as the mean of pixel-wise absolute difference between the continuous object saliency map and the binary ground-truth ....|
|||...We use the mean and maximal F w  values to evaluate the salient object segmentation capabilities of a model....|
|||...Results  5.3.1 Salient Object Segmentation  The quantitative results obtained by the proposed method on PASCAL-S, DUT-OMRON, iCoSeg and ECSSD datasets for salient object segmentation are shown in Table 2....|
|||...qualitative results for salient object segmentation are shown in Fig....|
|||...Our network also captures multiple salient objects (third row) and weighs their relative importance in the scene appropriately....|
|||...neously training the network, we train relevant parts of the network independently for the tasks of salient object segmentation and eye fixation prediction....|
|||...Quantitative results of our approach on salient object segmentation compared against other state-of-the art methods on PASCALS, DUT-OMRON, iCoSeg and ECSSD datasets....|
|||...Conclusion  In this work, we have proposed a novel deep convolutional architecture capable of simultaneously predicting human eye fixations and segmenting the salient objects in an image....|
|||...Also, our network has a branched architecture to efficiently capture both the low-level and highlevel semantics necessary for salient object segmentation....|
|||...Quantitative Results on DUT-OMRON dataset when the networks are trained simultaneously versus independently for the tasks of eye fixation prediction and salient object segmentation....|
|||...We evaluate our method on four datasets of eye fixation prediction and salient object segmentation and show that it outperforms the existing state-of-the-art approaches....|
|||...[8] A. Borji, What is a salient object?...|
|||...a dataset and a baseline model for salient object detection, Image Processing, IEEE Transactions on, vol....|
|||...[9] Y. Li, X. Hou, C. Koch, J. M. Rehg, and A. L. Yuille, The  secrets of salient object segmentation, in CVPR, 2014....|
|||...[10] F. Perazzi, P. Kr ahenb uhl, Y. Pritch, and A. Hornung, Saliency filters: Contrast based filtering for salient region detection, in CVPR, 2012....|
|||...Shum, Learning to detect a salient object, Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol....|
|||...[17] R. Achanta, S. Hemami, F. Estrada, and S. Susstrunk, Frequency-tuned salient region detection, in CVPR, 2009....|
|||...[19] S. S. R and R. V. Babu, Salient object detection via object ness measure, in ICIP, 2015....|
|||...[45] A. Borji, D. N. Sihite, and L. Itti, Salient object detection:  A benchmark, in ECCV, 2012....|
||50 instances in total. (in cvpr2016)|
|13|Deeply Supervised Salient Object Detection With Short Connections|...Deeply Supervised Salient Object Detection with Short Connections  Qibin Hou1 Ming-Ming Cheng1  Xiaowei Hu1 Ali Borji2  Zhuowen Tu3  Philip Torr4  1CCCE, Nankai University  2CRCV, UCF  3UCSD 4Universi...|
|||...Our method produces stateof-the-art results on 5 widely tested salient object detection benchmarks, with advantages in terms of efficiency (0.08 seconds per image), effectiveness, and simplicity over ...|
|||...Introduction  The goal in salient object detection is to identify the most visually distinctive objects or regions in an image....|
|||...This motivates recent research efforts of using Fully Convolutional Neural Networks (FCNs) for salient object detection [29, 46, 52, 13, 36]....|
|||...This motivated us to develop a new method for salient object detection by introducing short connections to the skip-layer structure within the HED [49] architecture....|
|||...es can be transformed to shallower side-output layers and thus can help them better locate the most salient region; shallower sideoutput layers can learn rich low-level features that can help refine t...|
|||...By combining features from different levels, the resulting architecture provides rich multi-scale feature maps at each layer, a property that is essentially need to do salient object detection....|
|||...Experimental results show that our method produces state-of-the-art results on 5 widely tested salient object detection benchmarks, with advantages in terms of efficiency (0.08 seconds per image), eff...|
|||...The majority of salient object detection methods are based on hand-crafted local features [22, 25, 50], global features [7, 42, 37, 24], or both (e.g., [3])....|
|||...Here, we focus on discussing recent salient object detection methods based on deep learning architectures....|
|||...d all the previous state-of-the-art records in nearly every sub-field of computer vision, including salient object detection....|
|||...presented a multi-context deep learning framework for salient object detection....|
|||...However, experimental results show that such a successful architecture is not suitable for salient object detection....|
|||...he standard HED architecture [49] as well as its extended version, a special case of this work, for salient object detection and gradually move on to our proposed architecture....|
|||...In this part, we extend the HED architecture for salient object detection....|
|||...During our experiments, we observe that deeper layers can better locate the most salient regions, so based on the architecture of HED we connect another side output to the last pooling layer in VGGNet [45]....|
|||...Besides, since salient object detection is a more difficult task than edge detection, we add two other convolutional layers with different filter channels and spatial sizes in each side output, which ...|
|||...A result comparison between the original HED and enhanced HED for salient object detection can be found in Table 4....|
|||...In addition, the deep side outputs can indeed locate the salient objects/regions, some detailed information is still lost....|
|||...pproach is based on the observation that deeper side outputs are capable of finding the location of salient regions but at the expense of the loss of details, while shallower ones focus on low-level f...|
|||...The main focus of saliency locating stage is on looking for the most salient regions for a given image....|
|||...s that with the help of deeper side information, lower side outputs can both accurately predict the salient objects/regions and refine the results from deeper side outputs, resulting in dense and accu...|
|||...Though our DCNN model can precisely find the salient objects/regions in an image, the saliency maps obtained are quite smooth and some useful boundary information is lost....|
|||...The pairwise potential is defined as  two classes in our case, we use the inferred posterior probability of each pixel being salient as the final saliency map directly....|
|||...Most images in this dataset have only one salient object....|
|||...most widely used datasets in salient object detection literature....|
|||...Most of images in this dataset have low contrast with more than one salient object....|
|||...It contains 300 images, most of which possess multiple salient objects....|
|||...(17)  As stated in [1], this metric favors methods that successfully detect salient pixels but fail to detect non-salient regions over methods that successfully detect non-salient pixels but make mist...|
|||...Albeit the fusion prediction map gets denser, some non-salient pixels are wrongly predicted as salient ones even though the CRF is used thereafter....|
|||...It can be easily seen that our proposed method not only highlights the right salient region but also produces coherent boundaries....|
|||...It is also worth mentioning that thanks to the short connections, our approach gives salient regions more confidence, yielding higher contrast between salient objects and the background....|
|||...ch as HKUIS [29], PASCALS [34], and SOD [39, 40], which contain a large number images with multiple salient objects....|
|||...This indicates that our method is capable of detecting and segmenting the most salient object, while other methods often fail at one of these stages....|
|||...Conclusion  In this paper, we developed a deeply supervised network for salient object detection....|
|||...With these short connections, the activation of each side-output layer gains the capability of both highlighting the entire salient object and accurately locating its boundary....|
|||...Our approach significantly advances the stateof-the-art and is capable of capturing salient regions in both simple and difficult cases, which further verifies the merit of the proposed architecture....|
|||...Global contrast based salient region detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...Deep contrast learning for salient object  detection....|
|||...Contextual hypergraph modeling for salient object detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Learning to detect a salient object....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||47 instances in total. (in cvpr2017)|
|14|Zhang_Supervision_by_Fusion_ICCV_2017_paper|...networks (DNNs), deep (convolutional) models have been built in recent years to address the task of salient object detection....|
|||...To address this problem, this paper makes the earliest effort to train a deep salient object detector without using any human annotation....|
|||...o generate the learning curriculum and pseudo ground-truth for supervising the training of the deep salient object detector....|
|||...Introduction  With the goal of discovering the object regions that can attract human visual attention in images, salient object detection has been gaining intensive research interest in recent years....|
|||...Previous Works  The salient object detection approaches proposed in early ages mainly explored image saliency by evaluating the  Corresponding author....|
|||...DNNs), researchers have investigated several deep (convolutional) models for addressing the task of salient object detection [17, 32, 46, 25, 4, 16]....|
|||...[46] proposed a multi-context deep learning framework for salient object detection, which jointly modeled global context and local context in a unified framework....|
|||...Motivation and Contributions  Studies in this field have demonstrated that the DNNbased salient object detectors are highly effective and can achieve top results on modern benchmark datasets (see Fig....|
|||...However, all current DNN-based salient object detectors require the large-scale manual supervision in the form of pixel-level human annotation....|
|||...(A) shows some examples of the salient object detection results generated by the existing approaches, where the first row are the original images, the second row are the saliency maps obtained by the ...|
|||...e benchmark datasets, where the blue histograms indicate the average performance of the traditional salient object detectors [31, 27, 47, 35, 18, 45] and the red histograms indicate the average perfor...|
|||...The performance gap between the traditional salient object detectors and the deep salient object detectors is mainly caused by (i) the powerful deep learning technique and (ii) the large-scale manuall...|
|||...e the earliest effort to explore: Is pixel-level human annotation indispensable for building strong salient object detector? and moreover, Can deep salient object detectors be trained entirely without...|
|||...Training deep salient object detector without using any human annotation is very challenging....|
|||...Thus, in this paper, we propose to take advantage of the existing unsupervised salient object detector to provide the needed pseudo supervision....|
|||...s direction, one naive strategy is to adopt the saliency maps generated by an existing unsupervised salient object detector to provide the initial pseudo ground-truth, and then train the DNN-based sal...|
|||...The underlying reasons are two folds: Firstly, only using one unsupervised salient object detector is  1In this paper, unsupervised learning refers to learning without using  human annotation....|
|||...Basically, such difficulty is inferred based on the inconsistency of the fused weak salient object detectors....|
|||...3) to train deep salient object detector without using any human annotation....|
|||...The proposed unsupervised learning framework for training deep salient object detector under the supervision by fusion....|
|||...Afterwards, the dynamic learning curriculum and pseudo ground-truth maps are generated to provide supervision for training the deep salient object detector (the yellow blocks in Fig....|
|||...The Proposed Approach  Given N training images {In}, n  [1, N ], we use three unsupervised salient object detectors [29, 45, 44]3 to generate the initial weak saliency maps {WSMm n }, m  [1, M ], M = ...|
|||...reliability (in inter-image fusion) are then replaced by  3Here we choose to use three unsupervised salient object detectors by considering the tradeoff of the effectiveness and efficiency in fusing d...|
|||...4050  the saliency maps generated by the learnt deep salient object detector, which forms the new weak saliency maps for guiding the learning in the next iteration....|
|||...{WSMm m=1, the goal of intra-image fusion is to infer the superpixel-level reliability of each weak salient object detector to integrate the weak saliency maps with considering the different difficult...|
|||...(1)  Afterwards, for inferring the superpixel-level reliabilities of the weak salient object detectors {am n } and the difficulties of various superpixel regions {bn,i}, we adopt the GLAD fusion model...|
|||...In this model, bn,i = 0 means the superpixel region is very ambiguous and hence even the best weak salient object detector has a 50% chance of predicting it correctly, while bn,i =  means the superpix...|
|||...For am n , a very large value that closes to + means the weak salient object detector always predicts correctly, while a very small value that closes to  means the salient object detector always predi...|
|||...(4)  Then, the binary label of the m-th weak salient object detector on the n-th training images {Lm  n } is obtained by:  Lm  n = (cid:26)1, m  n  T others  0,  , T =  1  M N  N  M  Xn=1  Xm=1  m n ,...|
|||...Afterwards, we adopt the GLAD fusion model to infer the imagelevel reliabilities of the weak salient object detectors {m} and the difficulties of various training images {n}....|
|||...Training Deep Salient Object Detector under  Supervision by Fusion  2.3.1 The Network Architecture  We build our deep salient object detector based on the DHSNet [25] due to its effectiveness and efficiency....|
|||...The architecture of the network for the adopted deep salient object detector....|
|||...For training such deep salient object detector without using any human annotation, we introduce three channels of supervisory signals, including the superpixel-level fusion maps {SFMn}, the image-leve...|
|||...Experimental Settings  We implemented comprehensive experiments by using five widely used salient object benchmark datasets, which are the MSRA10K [5], ECSSD [37], SOD [26], DUTO [38], and PASCAL-S [2...|
|||...More encouragingly, when compared with a number of other stateof-the-art supervised deep salient object detectors, our un 8Based on our statistics, manual annotation needs around 31 seconds per-image,...|
|||...4) The performance drop on the relatively more challenging salient object detection datasets, e.g., DUT-O, tends to be more significant than it on the relatively less challenging ones, e.g., ECSSD, wh...|
|||...This paper has proposed a novel unsupervised learning framework to train the DNN-based salient object detector....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...Contextual hypergraph modeling for salient object detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The  secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency net work for salient object detection....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Minimum barrier salient object detection at 80 fps....|
||46 instances in total. (in iccv2017)|
|15|Chen_Look_Perceive_and_ICCV_2017_paper|...ng University  Abstract  Recently, CNN-based models have achieved remarkable success in image-based salient object detection (SOD)....|
|||...nto an inception-segmentation module and jointly fine-tuning them on images with manually annotated salient objects, the proposed networks show impressive performance in segmenting salient objects....|
|||...For imagebased SOD, there are two major tasks that need to be addressed, including popping-out salient objects as a whole and suppressing all probable distractors....|
|||...Considering that salient objects may be sometimes embedded in cluttered background and share some visual attributes with certain distractors, SOD remains a challenging task especially in such complex scenes....|
|||...[7] extracted a 26, 675d descriptor for each superpixel and fed it into several cascaded fullyconnected (FC) layers so as to identify whether a superpixel is salient or not....|
|||...chitecture of the proposed networks is mainly inspired by the work of [23], which demonstrates that salient objects can be annotated (and detected) by the human-being (and the classic random forest mo...|
|||...These two streams are then merged into an inception-segmentation module that can detect salient visual content through an inception-like block followed by convolution and deconvolution layers....|
|||...In this manner, complex salient objects can be detected as a whole, while distractors can be well suppressed (see Fig....|
|||...is paper include: 1) We propose novel two-stream fixation-semantic CNNs that can effectively detect salient objects in images; 2) we conduct a comprehensive analysis of state-of-the-art deep SOD model...|
|||...In many scenarios, such deep models have achieved state-ofthe-art performance in salient object detection....|
|||...Salient regions are detected  by measuring the reconstruction residuals that reflect the distinctness between background and salient regions....|
|||...e adopted to progressively refine the details in saliency maps so as to highlight the boundaries of salient objects....|
|||...bined with the global descriptor extracted by the first stream to determine whether a superpixel is salient via FC layers....|
|||...A salient map can be thus generated by By repeatedly processing every superpixel....|
|||...After that, another deep networks with only FC layers are adopted to predict the saliency of each candidate object from the global perspective so that salient objects can be detected as a whole....|
|||...In this work, we propose to address this issue by simulating the ways in which ground-truth annotations of salient objects are generated in eye-tracking experiments....|
|||...3, 340 8, 000 2, 500 2, 500 10, 000  25, 000 9, 500 10, 565 2, 500   10  000 from MSRA10K [28] with salient object masks and 15  ,  ,  000  from SALICON [16] with fixation density maps....|
|||...The bottom half is the two-stream module for extracting fixation and semantic cues, while the top half is the inception-segmentation module for feature fusion and salient prediction....|
|||...rature, including:  1) DUT-OMRON [44] contains 5, 168 complex images with pixel-wise annotations of salient objects....|
|||...2) PASCAL-S [23] contains 850 natural images that are pre-segmented into objects/regions and free-viewed by 8 subjects in eye-tracking tests for salient object annotation....|
|||...Many images contain multiple disconnected salient objects or salient objects that touch image boundaries....|
|||...GT indicates ground-truth mask of salient objects....|
|||...This indicates that the fixation stream can facilitate the detection of salient objects....|
|||...In this case, the Fmax score drops sharply (see Table 3), implying that the fixation stream can provide useful cues to detect salient objects....|
|||...In this case, salient objects are simply detected and segmented via CONV and DECONV layers....|
|||...However, such successful cases do not mean that deep models already capture all the essential characteristics of salient objects in all scenes....|
|||...  To validate this point, we test the six deep models over the psychological patterns, in which salient objects are very simple and can be easily detected by the human-being and the fixation predictio...|
|||...e fact that existing deep models rely heavily on the features learned from natural images, in which salient objects often have obvious semantic meanings....|
|||...Conclusion  This paper proposes two-stream fixation-semantic CNNs for image-based salient object detection....|
|||...These two streams are then fused into the inception-segmentation module in which salient objects can be efficiently and accurately segmented....|
|||...simulate the saccade shift processes of the human-being in eye-tracking experiments so as to detect salient objects beyond natural images....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Learning to detect a salient object....|
|||...Background prior-based salient object detection via deep reconstruction residual....|
|||...Deeply supervised salient object detection with short connection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...Saliency unified: A deep architecture for simultaneous eye fixation prediction and salient object segmentation....|
|||...Deep contrast learning for salient object  detection....|
|||...Contextual hypergraph modeling for salient object detection....|
|||...The secrets of salient object  segmentation....|
|||...DHSNet: Deep hierarchical saliency  network for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object  detection via low rank matrix recovery....|
|||...Minimum barrier salient object detection at 80 fps....|
||46 instances in total. (in iccv2017)|
|16|Kim_Salient_Region_Detection_2014_CVPR_paper|...nmo@ee.kaist.ac.kr  Abstract  In this paper, we introduce a novel technique to automatically detect salient regions of an image via highdimensional color transform....|
|||...Our main idea is to represent a saliency map of an image as a linear combination of high-dimensional color space where salient regions and backgrounds can be distinctively separated....|
|||...This is based on an observation that salient regions often have distinctive colors compared to the background in human perception, but human perception is often complicated and highly nonlinear....|
|||...or to a feature vector in a high-dimensional color space, we show that we can linearly separate the salient regions from the background by finding an optimal linear combination of color coefficients i...|
|||...Its goal is to detect salient regions, in terms of saliency map, from an image where the detected regions would draw the attentions of humans at the first sight of an image....|
|||...As demonstrated in many previous works, salient region detection is useful in many applications including segmentation [20], object recognition [22], image retargetting [32], photo re-arrangement [25]...|
|||...The development of salient region detection is often inspired by the human visual perception concepts....|
|||...One important concept is how distinct to a certain extent [9] the salient region is compared to other parts of an image....|
|||...Since color is a very important visual cue to humans, many salient region detection techniques are built  (a) Inputs Figure 1....|
|||...Examples of our salient region detection....|
|||...(b) Saliency maps  (c) Salient regions  upon distinctive color detection from an image....|
|||...Starting from a few initial color examples of detected salient regions and backgrounds, our technique estimates an optimal linear combination of color values in the high-dimensional color transform sp...|
|||...Assumptions Since our technique uses only color information to separate salient regions from the background, our technique shares a limitation when identically-colored objects are present in both the ...|
|||...Nevertheless, we show that many salient regions can simply be detected using only color information via our highdimensional color transform space, and we achieve high detection accuracy and better per...|
|||...Related Works  Representative works in salient region detection are reviewed in this section....|
|||...We refer readers to [30] for a more extensive comparative study of state-of-the-art salient region detection techniques....|
|||...[14] performed salient object segmentation with multi-scale superpixel-based saliency and closed boundary prior....|
|||...[26] used the uniqueness and distribution of the CIELab color to find the salient region....|
|||...[28] used an unsupervised approach to sample patches of an image which are salient by using patch features....|
|||...efinite foreground and the definite background regions will be used as initial color samples of the salient regions and background....|
|||...Initial Salient Regions Detection  Superpixel Saliency Features As demonstrated in recent works [15, 26, 27, 34], features from superpixels are effective and efficient for salient object detection....|
|||...Trimap Construction The initial saliency map usually does not detect salient objects accurately and may contain many ambiguous regions....|
|||...This trimap construction step is to identify very salient pixels from the initial saliency map that definitely belong to salient regions and backgrounds, and use our high-dimensional color transform m...|
|||...catch the salient pixels more accurately, instead of using a single global threshold to obtain our trimap from the initial saliency map, we propose using a multi-scale analysis with adaptive thresholding....|
|||...Therefore, it is able to locally capture very salient regions even though the local region might not be the most salient globally within the whole image....|
|||...Our goal is to find a linear combination of color coefficients in the high dimensional color transform space such that colors of salient regions and colors of backgrounds can be distinctively separated....|
|||...The different magnitudes in the color gradients can also handle cases when salient regions and backgrounds have different amount of defocus and different color contrast....|
|||...The result shows that the performance is undesirable when only RGB is used, and using various nonlinear RGB transformed color spaces and gamma corrections helps to catch the salient regions more accurately....|
|||...mples in our trimap to estimate an optimal linear combination of color coefficients to separate the salient region color and the background color....|
|||...color samples and to further separate the color distance between the salient region and the background....|
|||...s of our algorithm against previous algorithms on three representative benchmark datasets: the MSRA salient object dataset [19], the Extended Complex Scene Saliency Dataset (ECCSD) [33], and the Inter...|
|||...This dataset contains comparatively obvious salient objects on the simple background and is considered as a less challenging dataset in saliency detection....|
|||...So, this dataset can test the generalization ability of the salient region detection methods....|
|||...This dataset is quite interesting because it contains many people which are relatively hard to detect as salient objects....|
|||...Some visual examples of salient object detection on the MSRA dataset are presented in Figure 9 which demonstrate effectiveness of the proposed method....|
|||...Conclusions  We have presented a high-dimensional color transformbased salient region detection, which estimates foreground regions by using the linear combination of various color space....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Center-surround divergence of feaIn ICCV, pages  ture statistics for salient object detection....|
|||...Learning to  detect a salient object....|
|||...Unsupervised salient object segmentation based on kernel IEEE Transdensity estimation and two-phase graph cut....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
||44 instances in total. (in cvpr2014)|
|17|Hongmei_Song_Pseudo_Pyramid_Deeper_ECCV_2018_paper|...Pyramid Dilated Deeper ConvLSTM for  Video Salient Object Detection  Hongmei Song1, Wenguan Wang1[0000000208029567], Sanyuan Zhao1, Jianbing Shen1,2, and Kin-Man Lam3  1 Beijing Lab of Intelligent Inf...|
|||...This paper proposes a fast video salient object detection model, based on a novel recurrent network architecture, named Pyramid Dilated Bidirectional ConvLSTM (PDB-ConvLSTM)....|
|||...iency detection can also be divided into two categories, i.e., eye fixation prediction [41, 39] and salient object detection [49, 47]....|
|||...With unsupervised video object segmentation as an example application task, we further show that the proposed video saliency model, equipped with a CRF segmentation module,  Pyramid Dilated Deeper Co...|
|||...2 Related Work  Image/Video Salient Object Detection....|
|||...Recently, with the popularity of deep neural network, various deep learning based image salient object detection models were proposed, e.g., multi-stream network with embedded superpixels [22, 25], re...|
|||...Conventional video salient object detection methods [8, 9, 28, 42] extract spatial and temporal features separately and then integrate them together to generate a spatiotemporal saliency map....|
|||...[44] introduced FCN to video salient object detection by using adjacent pairs of frames as input, which substantially improves the precision and achieves a speed of 2 fps....|
|||...Those models have similar goal with video salient object detection, aside from they seeking to get a binary fore-/background mask for each video frame....|
|||...3 Our Approach  This section elaborates on the details of the proposed video salient object detection model, which consists of two key components....|
|||...Architecture overview of the proposed video salient object detection model, which consists of two components, e.g., a spatial saliency learning module based on Pyramid Dilated Convolution (PDC) ( 3.1)...|
|||...PDB-ConvLSTM takes the spatial features learnt from the PDC module as inputs, and outputs improved spatiotemporal saliency representations for final video salient object prediction ( 3.2)....|
|||...More specially, let F  RW HM denote the input 3D feature tensor, a set k=1 and different k=1 (strides are set as 1) are adopted for generating a set of  of K dilated convolution layers with kernels {C...|
|||...A region will be reasonably salient if only we see it from a proper distance and see its proper spatial context....|
|||...With above definitions, ConvLSTM can be  Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection  7  formulated as follows:  i  Xt + WH f  Xt + WH o  Xt + WH  it = (WX ft = (WX ot = (WX ct...|
|||...In order to extract more powerful spatiotemporal information and let the network adapt to salient targets at different scales, we further extend DB-ConvLSTM with a PDC-like structure....|
|||...Let G  {0, 1}473473  Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection  9  and S  [0, 1]473473 denote the groundtruth saliency map and predicted saliency respectively, the overall lo...|
|||...LM AE is based on MAE metric, which is widely used in salient object detection....|
|||...One is for examining the performance of the proposed model for the main purpose, video salient object detection ( 4.1)....|
|||...or evaluating the effectiveness of the proposed model on unsupervised video object segmentation, as salient object detection has been shown as an essential preprocessing step for unsupervised segmenta...|
|||...For video salient object detection, we evaluate the performance on three public datasets, i.e., Densely Annotated VIdeo Segmentation (DAVIS) [31], Freiburg-Berkeley Motion Segmentation (FBMS) [2] and ...|
|||...ViSal is the first dataset specially designed for video salient object detection and includes 17 challenging video clips....|
|||...Salient Object Detection  We compared our model with 18 famous saliency methods, including 11 image salient object detection models: Amulet [51], SRM [36], UCF [52], DSS [16], MSR [23], NLDF [29], DCL...|
|||...Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection  11  Table 1....|
|||... 0.040 0.877 0.041 0.648 0.172 0.726 0.099 0.734 0.096 0.671 0.132 0.126 0.731 0.022 0.917  7 video salient object detection approaches: SGSP [27], GAFL [43], SAGE [42], STUW [8], SP [28], FCNS [44], ...|
|||...Note that FCNS and FGRNE are deep learning based video salient object detection models....|
|||...As seen, our model consistently produces accurate salient object estimations with various challenging scenes....|
|||...4.2 Performance on Unsupervised Video Object Segmentation  Video salient object detection model produces a sequence of probability maps that highlight the most visually important object(s)....|
|||...As demonstrated in [42], such salient object estimation could offer meaningful cue for unsupervised video primary object segmentation, which seeks to a binary foreground/background  12  H. Song, W. W...|
|||...Thus video salient object detection can be used as a pre-processing step for unsupervised video segmentation....|
|||...This is mainly because the proposed PDB Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection  13  Table 3....|
|||...5 Conclusions  This paper proposed a deep video salient object detection model which consists of two essential components: PDC module and PDB-ConvLSTM module....|
|||...Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection  15  References  1....|
|||...: Global contrast  based salient region detection....|
|||...Li, G., Xie, Y., Lin, L., Yu, Y.: Instance-level salient object segmentation....|
|||...Li, G., Xie, Y., Wei, T., Wang, K., Lin, L.: Flow guided recurrent neural encoder  for video salient object detection....|
|||...Li, G., Yu, Y.: Deep contrast learning for salient object detection....|
|||...Liu, N., Han, J.: Dhsnet: Deep hierarchical saliency network for salient object  detection....|
|||...Luo, Z., Mishra, A.K., Achkar, A., Eichel, J.A., Li, S., Jodoin, P.M.: Non-local deep features for salient object detection....|
|||...Wang, T., Borji, A., Zhang, L., Zhang, P., Lu, H.: A stagewise refinement model  for detecting salient objects in images....|
|||...Wang, W., Shen, J., Dong, X., Borji, A.: Salient object detection driven by fixation  prediction....|
|||...Wang, W., Shen, J., Shao, L.: Video salient object detection via fully convolutional  networks....|
|||...IEEE TIP 25(11), 50255034 (2016)  Pyramid Dilated Deeper ConvLSTM for Video Salient Object Detection  17  46....|
|||...Zhang, P., Wang, D., Lu, H., Wang, H., Ruan, X.: Amulet: Aggregating multi-level convolutional features for salient object detection....|
||44 instances in total. (in eccv2018)|
|18|Yang_Saliency_Detection_via_2013_CVPR_paper|...t or the entire image, whereas a few methods focus on segmenting out background regions and thereby salient objects....|
|||...Instead of considering the contrast between the salient objects and their surrounding regions, we consider both foreground and background cues in a different way....|
|||...Saliency detection is carried out in a two-stage scheme to extract background regions and foreground salient objects efficiently....|
|||...We note that saliency models have been developed for eye fixation prediction [6, 14, 15, 17, 19, 25, 33] and salient object detection [1, 2, 7, 9, 23, 24, 32]....|
|||...The latter is to accurately detect where the salient object should be, which is useful for many high-level vision tasks....|
|||...In this paper, we focus on the bottom-up salient object detection tasks....|
|||...[32] analyze multiple cues in a unified energy minimization framework and use a graph-based saliency model [14] to detect salient objects....|
|||...develop a hierarchical graph model and utilize concavity context to compute weights between nodes, from which the graph is bi-partitioned for salient object detection....|
|||...The main observation is that the distance between a pair of background regions is shorter than that of a region from the salient object and a region from the background....|
|||...In the second stage, we apply binary segmentation on the resulted saliency map from the first stage, and take the labelled foreground nodes as salient queries....|
|||...In contrast, the semi-supervised method [12] requires both background and salient seeds, and generates a binary segmentation....|
|||...Furthermore, it is difficult to determine the number and locations of salient seeds as they are generated by random walks, especially for the scenes with different salient objects....|
|||...As our model incorporates local grouping cues extracted from the entire image, the proposed algorithm generates well-defined boundaries of salient objects and uniformly highlights the whole salient regions....|
|||...when salient queries are given, and using 1  f   3....|
|||...Examples in which the salient objects appear at the image boundary....|
|||...Second, it reduces the effects of imprecise queries, i.e., the ground-truth salient nodes are inadvertently selected as background queries....|
|||...Due to the imprecise labelling results, the pixels with the salient objects have low saliency values....|
|||...The example in which imprecise salient queries are selected in the second stage....|
|||...By integration of four saliency maps, some salient parts of object can be identified (although the whole object is not uniformly highlighted), which provides sufficient cues for the second stage detec...|
|||...While most regions of the salient objects are highlighted in the first stage, some background nodes may not be adequately suppressed (See Figure 4 and Figure 5)....|
|||...Ranking with Foreground Queries  The saliency map of the first stage is binary segmented (i.e., salient foreground and background) using an adaptive threshold, which facilitates selecting the nodes of...|
|||...We expect that the selected queries cover the salient object regions as much as possible (i.e., with high recall)....|
|||...Once the salient queries are given, an indicator vector y is formed to compute the ranking vector f  using Eq....|
|||...Despite some imprecise labelling, salient objects can be well detected by the proposed algorithm as shown in Figure 6....|
|||...The salient object regions are usually relatively compact (in terms of spatial distribution) and homogeneous in appearance (in terms of feature distribution), while background regions are the opposite....|
|||...In other words, the intra-object relevance (i.e., two nodes of the salient objects) is statistically much larger than that of object-background and intra-background relevance, which can be inferred fr...|
|||...Therefore, the sum of the relevance values of object nodes to the ground-truth salient queries is considerably larger than that of background nodes to all the queries....|
|||...Similarly, in spite of the saliency maps after the first stage of Figure 5 are not precise, salient object can be well detected by the saliency maps after the foreground queries in the second stage....|
|||...The main steps of the proposed salient object detection algorithm are summarized in Algorithm 1....|
|||...4: Bi-segment Sbq to form salient foreground queries and an indicator vector y. Compute the saliency map Sf q by Eq....|
|||...The first one is the MSRA dataset [23] which contains 5,000 images with the ground truth of salient region marked by bounding boxes....|
|||...The second one is the MSRA-1000 dataset, a subset of the MSRA dataset, which contains 1,000 images provided by [2] with accurate human-labelled masks for salient objects....|
|||...The precision value corresponds to the ratio of salient pixels correctly assigned to all the pixels of extracted regions, while the recall value is defined as the percentage of detected salient pixels...|
|||...We note that the proposed algorithm uniformly highlights the salient regions and preserves finer object boundaries than the other methods....|
|||...Conclusion  We propose a bottom-up method to detect salient regions in images through manifold ranking on a graph, which incorporates local grouping cues and boundary priors....|
|||...Global contrast based salient region detection....|
|||...Random walks on graphs  for salient object detection in images....|
|||...Automatic salient object segmentation based on contex and shape prior....|
|||...Center-surround divergence of feature  statistics for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Automatic salient object  extraction with contextual cue....|
|||...Geodesic saliency using  tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
||44 instances in total. (in cvpr2013)|
|19|Yeong_Jun_Koh_Sequential_Clique_Optimization_ECCV_2018_paper|...Therefore, we develop the sequential clique optimization (SCO) technique to efficiently determine the cliques corresponding to salient object tracks....|
|||...Experimental results show that the proposed algorithm significantly outperforms the state-of-the-art video object segmentation and video salient object detection algorithms on recent benchmark datasets....|
|||...Keywords: Video object segmentation, primary object segmentation, salient object detection, sequential clique optimization  1  Introduction  Video object segmentation (VOS) [14] is the task to segment...|
|||...In this regard, VOS is closely related to video salient object detection (SOD) [710], in which the objective is to detect salient objects in a video....|
|||...Note that the salient objects mean that they appear frequently in the video and have dominant color and motion features....|
|||...However, it is challenging to delineate salient objects in videos without any user annotations....|
|||...Then, we perform instance matching, by selecting one object instance from each frame, in order to construct the most salient object track....|
|||...By repeating the SCO process, we can extract multiple salient object tracks....|
|||...Finally, we convert these salient object tracks into VOS results in unsupervised and semi-supervised settings....|
|||...[15] propose an end-to-end learning framework, which combines appearance and motion information to provide pixel-wise segmentation results for salient objects....|
|||...Some algorithms [4042] adopt a priori knowledge, such as the boundary prior that boundary regions tend to belong to the background and thus be less salient than center regions....|
|||...Also, an instance-level segmentation algorithm for salient objects was proposed in [14], which uses both saliency maps and object proposals....|
|||...[9] combine color and motion saliency maps based on the salient foreground model and the non-salient background model....|
|||...Some algorithms [8, 10, 48] exploit spatial and temporal features jointly to detect spatiotemporally salient regions....|
|||...Third, we extract salient object tracks by finding cliques in the graph and convert the tracks into VOS results....|
|||...Lee, and C.-S. Kim  Input frames  Instance generation  Complete (cid:1863)-partite graph   Finding salient object tracks  Segmentation results  Fig....|
|||...For each proposal, we use the maximum of the category-wise scores, since the purpose of the proposed algorithm is to segment out salient objects in videos regardless of their categories....|
|||...3.2 Finding Salient Object Tracks  Problem: The set of all object instances, V = O1 O2   OT , includes non-salient objects, as well as salient ones....|
|||...From V, we extract as many salient objects as possible,  Sequential clique optimization for video object segmentation  5  (a)  (b)  (c)  Fig....|
|||...Illustration of finding salient object tracks over four frames in Boxing-fisheye: (a) complete 4-partite graph, (b) 1st salient object track 1, and (c) 2nd salient object track 2....|
|||...while excluding non-salient ones, assuming that a salient object should have dominant features in each frame and appear frequently through the sequence....|
|||...To this end, we construct the most salient object track, by selecting an object instance in each frame, which corresponds to one identical salient object....|
|||...Then, after removing all instances in the track from V, we repeat the process to extract the next salient track, and so on....|
|||...To extract the most salient object, we perform the instance matching by selecting one object instance (one node) from each frame (each node subset) Ot....|
|||...Also, object instances in a clique, representing a salient object track, should have  high saliency scores....|
|||...However, as the iteration goes on, the clique converges to a salient object track, in which the nodes represent an identical object and thus exhibit high similarity weights in general....|
|||...Let 1 denote the most salient object track, obtained by this SCO process....|
|||..., M }, until no node remains in G. In general, if p < q, p is more salient than q....|
|||...As a result, we obtain the set of the refined salient object tracks {  1,  2, ....|
|||...Given a refined salient object track   = { t}T we determine whether to discard ot, t t, we compare the weight w(o,  count the number of object instances o,   t=1, at frame t from  ....|
|||...Therefore, Proposed-F additionally picks another salient object track  p, only when  1 and  p are spatially adjacent in most frames in a video....|
|||...61, 62] and the conventional SOD algorithms in [8, 10, 24, 48, 6365], which also extract primary or salient objects from each frame at the pixel level....|
|||...Note that SCO yields multiple salient object tracks, which are used to produce VOS results....|
|||...4.2 Comparison with Salient Object Detection Techniques  To assess SOD results, we adopt three performance metrics: precision-recall (PR) curves, F-measure, and mean absolute error (MAE)....|
|||...Then, we chose a salient instance from each frame to construct the salient object track....|
|||...By applying SCO repeatedly, we obtained multiple salient object tracks....|
|||...Wang, W., Shen, J., Shao, L.: Video salient object detection via fully convolutional networks....|
|||...Li, G., Xie, Y., Lin, L., Yu, Y.: Instance-level salient object segmentation....|
|||...Perazzi, F., Kr ahenb uhl, P., Pritch, Y., Hornung, A.: Saliency filters: Contrast based filtering  for salient region detection....|
|||...Li, G., Yu, Y.: Deep contrast learning for salient object detection....|
|||...Hu, P., Shuai, B., Liu, J., Wang, G.: Deep level sets for salient object detection....|
||41 instances in total. (in eccv2018)|
|20|Jiang_Submodular_Salient_Region_2013_CVPR_paper|...rsity of Maryland, College Park, MD, 20742  {zhuolin, lsd}@umiacs.umd.edu  Abstract  The problem of salient region detection is formulated as the well-studied facility location problem from operations...|
|||...High-level priors are combined with low-level features to detect salient regions....|
|||...ive function, which maximizes the total similarities (i.e., total profits) between the hypothesized salient region centers (i.e., facility locations) and their region elements (i.e., clients), and pen...|
|||...Most saliency models [2, 20, 4, 6, 8] are based on a contrast prior between salient objects and backgrounds....|
|||...However, given the ground truth salient regions in Figure 1(b), even for the first simple example, these approaches either fail to separate the object from the background, as in Figures 1(c) and 1(e),...|
|||...(a) Input images; (b) Ground truth salient regions; (c)(e): Saliency maps using [2, 6, 4] with contrast priors; (f) Saliency map using [26] with a low-rank prior....|
|||...[26, 28] represent an image as a low-rank matrix plus sparse noise, where the background is modeled by the low-rank matrix and the salient regions are indicated by the sparse noise (i.e., low-rank prior)....|
|||...We present a submodular objective function for efficiently creating saliency maps from natural images; these maps can then be used to detect multiple salient regions within a single image....|
|||...Our objective function consists of two terms: a similarity term (between the selected centers of salient regions and image elements (superpixels) assigned to that center), and the facility costs for t...|
|||...Hence it favors the extraction of high-quality potential salient regions....|
|||...The second term penalizes the number of selected potential salient region centers, so it avoids oversegmentation of salient regions....|
|||...It reduces the redundancy among selected salient region centers because the small gain obtained by splitting a region through the introduction of an extrane ous region center is offset by the facility cost....|
|||...This high level prior is integrated with low level feature information into a unified objective function to identify salient regions....|
|||...uper-pixels, which are less likely to cross object boundaries and lead to more accurately segmented salient regions....|
|||...Unlike approaches that identify only one salient region in an image [7], our approach identifies multiple salient regions simultaneously without any strong assumptions about the statistics of the back...|
|||...The main contributions of our paper are:  Salient region selection is modeled as the facility location problem, which is solved by maximizing a submodular objective function....|
|||...This provides a new perspective using submodularity for salient region detection, and it achieves state-of-art performance on two public saliency detection benchmarks....|
|||... We present an efficient greedy algorithm by using the  We naturally integrate high-level priors with low-level saliency into a unified framework for salient region detection....|
|||...Related Work  Existing salient region detection approaches can be roughly divided into two categories: bottom-up and topdown approaches....|
|||...Top-down approaches make use of high level knowledge about interesting objects to identify salient regions [29, 5, 14]....|
|||...e approaches integrate multiple saliency maps generated from different features or priors to detect salient regions....|
|||...This provides a set of potential salient regions....|
|||...Finally, the saliencies of the potential salient regions and their constituent superpixels are computed from color and spatial location information....|
|||...Extraction of Potential Salient Regions  We model the problem of identifying high quality potential salient regions as selecting a subset, A, of J as the final region centers....|
|||...K is the maximum number of salient regions that the algorithm might identify, and is a parameter specified by the user....|
|||...An example of detecting salient regions with different components....|
|||... A  = 9 locations marked as red are iteratively selected; (g) Potential salient region extraction....|
|||... A  = 5 locations marked as red are selected; (j) Potential salient region extraction with prior....|
|||...Five regions generated; (k) Ground truth salient region; (l) Saliency map without prior; (m) Saliency map with prior; (n) Salient region mask based on the saliency map in (m)....|
|||...3.3.3 Potential Salient Region Extraction Given a set of selected facility locations A, let the current i = maxjA cij, and the famaximal profit from vi be cur i = arg maxjA cij....|
|||...Hence, we i cluster the image elements that share the same facility location as the most profitable to obtain potential salient regions {ri}  i=1... A ....|
|||...V (ri) =  A region which has a wider spatial distribution is typically less salient than regions which have small spatial spread [20, 8]....|
|||...Examples of extracting salient regions....|
|||...ages; (b) Ground truth salient regions; (c) High-level prior map; (d) Saliency map without high level prior; (e) Saliency map with high level prior; (f) Salient Region extraction based on (e) by simpl...|
|||..., S  Algorithm 1 Submodular Salient Region Detection 1: Input: I, G = (V, E), cij , K and ....|
|||...Conclusion  We presented a greedy-based salient region detection approach by maximizing a submodular function, which can be viewed as the facility location problem....|
|||...The saliency maps in (j) are used to segment the salient regions by simple thresholding....|
|||...The saliency maps in (j) are used to segment the salient regions by simple thresholding....|
|||...Design and perceptual validation of performance measures for salient object segmentation, 2010....|
|||...A unified approach to salient object detection via low rank  matrix recovery, 2012....|
||40 instances in total. (in cvpr2013)|
|21|Shuhan_Chen_Reverse_Attention_for_ECCV_2018_paper|...Reverse Attention for Salient Object Detection  Shuhan Chen[0000000200945157], Xiuli Tan, Ben Wang, and Xuelong Hu  School of Information Engineering,  Yangzhou University, China  {c.shuhan, t.xiuli02...|
|||...Benefit from the quick development of deep learning techniques, salient object detection has achieved remarkable progresses recently....|
|||...To this end, this paper presents an accurate yet compact deep network for efficient salient object detection....|
|||...By erasing the current predicted salient regions from side-output features, the network can eventually explore the missing object parts and details which results in high resolution and accuracy....|
|||...Keywords: Salient Object Detection  Reverse Attention  Side-output Residual Learning  1  Introduction  Salient object detection, also known as saliency detection, aims to localize and segment the most...|
|||...Recently, with the quick development of deep convolutional neural networks (CNNs), salient object detection has achieved significant improvements over conventional hand-crafted feature based approaches....|
|||...it is inevitable to lose resolution and difficult to refine, making it infeasible to locate salient objects accurately, especially for the object boundaries and small objects....|
|||...Nevertheless, the existing archaic fusions are still incompetent for saliency detection under complex real-world scenarios, especially when dealing with multiple salient objects with diverse scales....|
|||...To this end, we present an accurate yet compact deep salient object detection network which achieved comparable performance with state-of-the-art methods, thus enables for real-time applications....|
|||...In generally, more convolutional channels with large kernel size leads to better performance in salient object detection  Reverse Attention for Salient Object Detection  3  Img&GT  s-out 1  s-out 2  ...|
|||...In a different way, we introduce residual learning [25] into the architecture of HED [5], and regard salient object detection as a super-resolution reconstruction problem [26]....|
|||...However, the performance is not satisfactory enough if we directly apply it for salient object detection due to its challenging....|
|||...In summary, the contributions of this paper can be concluded as: (1) We introduce residual learning into the architecture of HED for salient object detection....|
|||...[28] applied recurrent unit into FCNs to iteratively refine each salient region....|
|||...Although it is natural to apply it for salient object detection, the performance is not satisfactory enough....|
|||...(cid:44)(cid:81)(cid:83)(cid:88)(cid:87)(cid:3)(cid:76)(cid:80)(cid:68)(cid:74)(cid:72)  (cid:54)(cid:76)(cid:71)(cid:72)(cid:16)(cid:82)(cid:88)(cid:87)(cid:83)(cid:88)(cid:87)(cid:3)  (cid:85)(cid:7...|
|||...Since most of the existing saliency detection networks are fine-tuned from image classification networks which are only responsive to small and sparse discriminative object parts,  Reverse Attention ...|
|||...Reverse Attention for Salient Object Detection  9  3.5 Difference to Other Networks  Though shares the same name, the proposed network significantly differs from reverse attention network [33], which ...|
|||...Here, we perform a  Reverse Attention for Salient Object Detection  11  Table 3....|
|||...6, including complex scenes, low contrast between salient object and background,  12  Shuhan Chen et al....|
|||...multiple (small) salient objects with diverse characteristics (e.g., size, color)....|
|||...ng all the cases into account, it can be observed clearly that our approach not only highlights the salient regions correctly with less false detection but also produces sharp boundaries and coherent ...|
|||...Visual comparisons with the existing methods in some challenging cases: complex scenes, low contrast, and multiple (small) salient objects....|
|||...Reverse Attention for Salient Object Detection  13  Precision recall curve  Precision recall curve  Precision recall curve  i  i  n o s c e r P  1  0.9  0.8  0.7  0.6  0.5  0.4  0.3  0.2  0.1  0  0  D...|
|||... NLDF 0.048  UCF 0.168  Amulet Ours 0.080 0.022  5 Conclusions  As a low-level pre-processing step, salient object detection has great applicability in various high-level tasks yet remains not being w...|
|||...Reverse Attention for Salient Object Detection  15  References  1....|
|||...Li, X., Zhao, L., Wei, L., Yang, M.H., Wu, F., Zhuang, Y., Ling, H., Wang, J.: Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...Li, G., Yu, Y.: Deep contrast learning for salient object detection....|
|||...Luo, Z., Mishra, A., Achkar, A., Eichel, J., Li, S., Jodoin, P.M.: Non-local deep  features for salient object detection....|
|||...(2017) 53005309 Instance-level salient object segmentation....|
|||...Zhang, P., Wang, D., Lu, H., Wang, H., Ruan, X.: Amulet: Aggregating multi-level  convolutional features for salient object detection....|
|||...Xiao, H., Feng, J., Wei, Y., Zhang, M.: Deep salient object detection with dense  connections and distraction diagnosis....|
|||...Hu, P., Shuai, B., Liu, J., Wang, G.: Deep level sets for salient object detection....|
|||...: Learning to detect a salient object....|
|||...: The secrets of salient object  segmentation....|
|||...Jiang, H., Wang, J., Yuan, Z., Wu, Y., Zheng, N., Li, S.: Salient object detection: A discriminative regional feature integration approach....|
|||...Borji, A., Cheng, M.M., Jiang, H., Li, J.: Salient object detection: A benchmark....|
|||...Liu, N., Han, J.: Dhsnet: Deep hierarchical saliency network for salient object  detection....|
|||...Kim, J., Pavlovic, V.: A shape-based approach for salient object detection using  deep learning....|
|||...(2016) 455470  Reverse Attention for Salient Object Detection  17  46....|
||40 instances in total. (in eccv2018)|
|22|cvpr18-Flow Guided Recurrent Neural Encoder for Video Salient Object Detection|...Flow Guided Recurrent Neural Encoder for Video Salient Object Detection  Guanbin Li1  Yuan Xie1  Tianhao Wei2  Keze Wang1  Liang Lin1,3   1Sun Yat-sen University  2Zhejiang University  3SenseTime Grou...|
|||...In this paper, we present flow guided recurrent neural encoder (FGRNE), an accurate and end-to-end learning framework for video salient object detection....|
|||...It can be considered as a universal framework to extend any FCN based static saliency detector to video salient object detection....|
|||...Although image based salient object detection has been extensively studied during the past decade, video based salient object detection is much less explored  The first two authors contribute equally ...|
|||...Nevertheless, directly applying these methods to video salient object detection is non-trivial and challenging....|
|||...1, state-ofthe-art still-image salient object detectors (e.g....|
|||...DSS [10]) deteriorates drastically from the inability to maintain the visual continuity and temporal correlation of salient objects between consecutive frames....|
|||...Recently, with the thriving application of deep CNN in salient object detection of static images, there are also attempts to extend CNN to video salient object detection [36, 16]....|
|||...resentation, which can be exploited to extend any FCN based still-image  saliency detector to video salient object detection....|
|||...n temporal domain and is complementary to feature warping towards an improved performance for video salient object detection....|
|||...Still(cid:173)Image Salient Object Detection  Image salient object detection has been extensively studied for decades....|
|||...In recent years, the profound deep CNN has pushed the research on salient object detection into a new phase and has become the dominant research direction in this field....|
|||...In contrast to these still-image based salient object detection methods, we focus on video salient object detection, which incorporates both temporal and motion information to improve the feature map ...|
|||...It can be considered as a universal framework to extend any FCN based models to video salient object detection, and can easily benefit from the improvement of stillimage salient object detectors....|
|||...Video Salient Object Detection  Compared with saliency detection in still images, detecting video salient objects is much more challenging due to the high complexity in effective spatio-temporal model...|
|||...Moreover, this rough strategy of spatio-temporal modeling lacks explicit compensation for objects motion, making it hard to detect the salient objects with strenuous movement....|
|||...rent Neural Encoder  Given the a video frame sequence Ii, i = 1, 2, ..., N , the objective of video salient object detection is to output the saliency maps of all frames, Si, i = 1, 2, ..., N ....|
|||...State-ofthe-art salient object detectors for static image are mostly based on FCN structure [20, 23, 18, 10]....|
|||...4.1.2 Evaluation Criteria  Similar to image-based salient object detection, we adopt precision-recall curves (PR), maximum F-measure and mean absolute error (MAE) as the evaluation metrics....|
|||...FGRNE is compatible with any FCN based stillimage salient object detectors....|
|||...In this paper, we choose the state-of-the-art deeply supervised salient object detection (DSS) [10] method with public trained model as a baseline and take the updated DSS with FGRNE embedded as our f...|
|||...The first six are the latest stateof-the-art salient object detection methods for static images while the last three are video-based saliency models....|
|||...As shown in the figures, our method (FGRNE) significantly outperforms all state-of-theart static and dynamic salient object detection algorithms on both DAVIS and FBMS....|
|||...The evident performance gain towards that of Sa also reveals the importance of motion modeling for video salient object detection....|
|||...patiotemporal coherence of the feature representation, which greatly boost the performance of video salient object detection....|
|||...sed host network model, we apply to incorporate our FGRNE in two other recently published FCN based salient object detection methods, including DCL [20]  ject Segmentation Methods  The problem setting...|
|||...Conclusion  In this paper, we have presented an accurate and endto-end framework for video salient object detection....|
|||...an be considered as a universal framework to extend any FCN based static saliency detector to video salient object detection, and can easily benefit from the future improvement of image based salient ...|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Deeply supervised salient object detection with short connections....|
|||...Video salient object detecarXiv preprint  tion using spatiotemporal deep features....|
|||...Deep contrast learning for salient object  detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Video salient object detection via fully convolutional networks....|
|||...Instance-level salient object segmentation....|
|||...Minimum barrier salient object detection at 80 fps....|
||39 instances in total. (in cvpr2018)|
|23|cvpr18-Salient Object Detection Driven by Fixation Prediction|...We build a novel neural network called Attentive Saliency Network (ASNet)1 that learns to detect salient objects from fixation maps....|
|||...Our work offers a deeper insight into the mechanisms of attention and narrows the gap between salient object detection and fixation prediction....|
|||...Although promising results have been achieved, they occasionally fail to detect the most salient object in  Corresponding author: Jianbing Shen....|
|||...Additionally, for current computational saliency models, their connection with how humans explicitly choose salient objects or watch natural scenes are less clear (as discussed in [3, 6])....|
|||...These studies confirmed a strong correlation between fixations and salient objects....|
|||...It is then used for salient object detection in a top-down manner....|
|||...Our contributions are manifold:   We aim to infer salient objects (captured in lower network layers) from the fixation map (encoded in higher layers) within a unified neural network....|
|||...This goes one step beyond previous deep learning based saliency models and offers a deep insight into the confluence between fixation prediction and salient object detection....|
|||...Related Work  In this section, we first briefly review the fixation prediction ( 2.1) and salient object detection literature ( 2.2)....|
|||...The Relationship between FP and SOD  Although SOD has been extensively studied in computer vision research, only few studies (e.g., [39, 4, 34]) have explored how humans explicitly choose salient objects....|
|||...Instead of performing FP or SOD separately, we exploit the correlation between fixations and salient objects via tightly coupling these two tasks in a unified deep learning architecture....|
|||...Our Approach  Given an input image, the goal is to produce a pixel-wise saliency map to highlight salient object regions....|
|||...The whole network is simultaneously trained to predict fixation locations and to detect salient objects in an end-to-end way (3.3)....|
|||...Detecting Object Saliency with Fixation Prior  The fixation map P gives a coarse but informative prior regarding visually salient regions....|
|||...A number of previous studies for pixel-labeling tasks such as semantic segmentation [44], and salient object detection [35, 48], have shown that neural networks are capable of producing fine-gained  1...|
|||...The fixation map is learned from the upper layers and is used by the ASNet to locate the salient objects....|
|||...The network is trained for detecting and successively refining the salient object via aggregating information from high-level fixation map and the spatially rich information from low-level network features....|
|||...In each time step, the convLSTM is trained for inferring the salient object with the knowledge of fixation information, and sequentially optimizes the features with the updated memory cell and hidden ...|
|||...Given the ground-truth salient object annotation S (here S  {0, 1}1414 for conv5-3 layer), the overall loss function is defined as:  LSal(S, Q) = LC(S, Q)+1LP (S, Q)+2LR(S, Q)  +3LF (S, Q)+4LM AE(S, Q...|
|||...og qx(cid:1),  (8)  where N is the total number of pixels and sk  S, qk  Q.  refers to the ratio of salient pixels in ground truth S. Weighted cross-entropy loss handles the imbalance between number o...|
|||...These two datasets have annotations for fixations and salient objects, respectively....|
|||...Conclusions  We proposed a deep learning network, ASNet, towards a better interpretable and efficient SOD model, which leverages fixation prediction for detecting salient objects....|
|||...Such prior was further utilized for teaching the network where the salient object is and the detailed object saliency was rendered step by step by considering finer and finer features in a top-down manner....|
|||...Frequency-tuned salient region detection....|
|||...What is a salient object?...|
|||...A dataset and a baseline model for salient object detection....|
|||...Look, perceive and segment: Finding the salient objects in images via two-stream fixation-semantic cnns....|
|||...Global contrast based salient region detection....|
|||...Deeply supervised salient object detection with short connections....|
|||...Saliency unified: A deep architecture for simultaneous eye fixation prediction and salient object segmentation....|
|||...Deep contrast learning for salient object  detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...DHSNet: Deep hierarchical saliency network for salient object detection....|
|||...Learning to detect a salient object....|
|||...Non-local deep features for salient object detection....|
|||...Everyone knows what is interesting: Salient locations which should be fixated....|
|||...A stagewise refinement model for detecting salient objects in images....|
|||...Video salient object detection via fully convolutional networks....|
||39 instances in total. (in cvpr2018)|
|24|Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper|...an, 710072, P. R. China   {liunian228, junweihan2010}@gmail.com   Abstract            Traditional 1 salient  object  detection  models  often  use  hand-crafted  features  to  formulate  contrast  and...|
|||...In this  work,  we  propose  a  novel  end-to-end  deep  hierarchical  saliency network (DHSNet) based on convolutional neural  networks for detecting salient objects....|
|||...In  recent  researchers  have  developed  many  computational  models  for  salient  object  detection  and  applied  them  to  benefit  many  other  applications,  such  as  image  summarization  [1]...|
|||...For images in (a), we show the salient object detection results of a  global  contrast  based  method  in  (b),  a  background  prior  based  method in (c), the results of the GV-CNN in (d), the final...|
|||...On the  other hand, they are often difficult to detect salient objects  with  large  sizes  and  complex  textures,  especially  when  image  backgrounds  are  also  cluttered  or  have  similar  appe...|
|||...However, it often fails when salient  objects touch image boundaries or have similar appearance      (c)   (see  column   in  Figure  1)....|
|||...with  backgrounds  Compactness  prior  [12]  advocates  that  salient  object  regions  are  compact  and  perceptually  homogeneous  elements....|
|||...Although these priors can further provide informative  information  for  salient  object  detection,  they  are  usually  explored  empirically  and  modelled  by  hand-designed  formulations....|
|||... are  usually  very  time-consuming,  becoming the bottleneck of the computational efficiency of  a salient object detection algorithm....|
|||...w to efficiently  preserve object details become the most intrinsic problems  for further promoting salient object detection methods....|
|||...s the whole images as the inputs and outputs  saliency  maps  directly,  hierarchically  detecting  salient  objects from the global view to local contexts, from coarse  scale to fine scales (see Figu...|
|||...Consequently,  the  GV-CNN  can  obtain optimal global salient object detection results, being  robust  to  complex  foreground  objects  and  cluttered  backgrounds,  even  if  they  are  very  simil...|
|||...The  contributions  of  this  paper  can  be  summarized  as   follows:    (1)  We  propose  a  novel  end-to-end  saliency  detection  model, i.e., the DHSNet, to detect salient objects....|
|||...  with  other  11  state-of-the-art  approaches  demonstrate the great superiority of DHSNet on the salient  object  detection  problem,  especially  on  complex  datasets....|
|||...ncy  detection,  which  includes  two  branches,  i.e.,  eye  fixation  prediction  [32,  33]  and  salient  object detection [34-36]....|
|||...For  salient  object  detection,  Wang  et  al....|
|||...rks hard to learn enough global structures, thus their  results  are  often  distracted  by  local  salient  patterns  in  cluttered backgrounds and are not able to highlight salient  objects  uniform...|
|||...DHSNet for Salient Object Detection   As  shown  in  Figure  2,  DHSNet  is  composed  of  the  GV-CNN  and  the  HRCNN....|
|||...The  GV-CNN  first  coarsely  detects  salient  objects  in  a  global  perspective,  then  the  HRCNN hierarchically and progressively refines the details  of  the  saliency  map  step  by  step....|
|||...e  ground  truth  saliency  mask,  the  fully  connected  layer  learns  to  detect  and  localize  salient  objects  of  the  input  image from the feature maps ahead by integrating various  saliency...|
|||...Datasets   We  conducted  evaluations  on  four  widely  used  salient  object  benchmark  datasets....|
|||...Most  images contain only one salient object and the backgrounds  are  usually  clear....|
|||...DUT-OMRON  [10]  includes  5,168  images  with  one  or  more  salient  objects  and  relatively  complex  backgrounds....|
|||...As we can see,  DHSNet  not  only  detects  and  localizes  salient  objects  accurately,  but  also  preserves  object  details  subtly....|
|||...ground objects (row 3 and 8),  cluttered backgrounds and complex foregrounds (row 4, 5  and 7), and salient objects touching image boundaries (row  1,  2  and  6)....|
|||...We  can  also  see  that  LEGS,  MDF,  and  MCDL  often  are  distracted by local salient patterns in cluttered backgrounds   684  and  are  not  able  to  highlight  salient  objects  uniformly....|
|||...ion,  recall  and  F-measure will drop, which demonstrates the superiority of  adopting RCLs in the salient object detection problem....|
|||...The results in the last row of Table 3 demonstrate the  superiority  of DHSNet  over  traditional  encoder-decoder  networks  on  the  salient  object  detection  task....|
|||...Conclusions   In  this  paper,  we  proposed  DHSNet  as  a  novel  end-to-end salient object detection  model....|
|||...Center-surround divergence of  feature statistics for salient object detection....|
|||...Global contrast based salient region detection....|
|||...[9]   Frequency-tuned salient region detection....|
|||...Background Prior-Based Salient Object Detection via Deep  Reconstruction  Residual....|
|||...Saliency  filters:  Contrast  based  filtering  for  salient  region  detection....|
|||...Fusing  generic  objectness  and  visual  saliency  for  salient  object  detection....|
|||...The   secrets of salient object segmentation....|
|||...Learning  to  detect  a  salient  object....|
||37 instances in total. (in cvpr2016)|
|25|Zhang_Amulet_Aggregating_Multi-Level_ICCV_2017_paper|...Amulet: Aggregating Multi-level Convolutional Features  for Salient Object Detection  Pingping Zhang Dong Wang Huchuan Lu Hongyu Wang Xiang Ruan  Dalian University of Technology, China  Tiwaki Co.Ltd ...|
|||...However, how to better aggregate multi-level convolutional feature maps for salient object detection is underexplored....|
|||...In this work, we present Amulet, a generic aggregating multi-level convolutional feature framework for salient object detection....|
|||...By aggregating multi-level convolutional features in this efficient and flexible manner, the proposed saliency model provides accurate salient object labeling....|
|||...Thus, these low-level feature based methods are very far away from distinguishing salient objects from the clutter background and can not generate satisfied predictions....|
|||...ently find the optimal multi-level feature aggregation strategy, and 3) how to efficiently preserve salient objects boundaries should become the most intrinsic problems in salient object detection....|
|||... trained on the MSRA10K dataset [5]) achieves new state-of-the-art performance on other large-scale salient object detection datasets, including the recent DUTS [42], DUT-OMRON [47], ECSSD [46], HKU-I...|
|||...Related Work  In this section, we briefly review existing representative models for salient object detection....|
|||...Salient object detection  Over the past decades, lots of salient object detection methods have been developed....|
|||...The majority of salient object detection methods are based on low-level hand-crafted features, e.g., image contrast [10, 19], color [26, 2], texture [46, 47]....|
|||...A lot of research efforts have been made to develop various deep architectures for useful features that characterize salient objects or regions....|
|||...[41] first propose two deep neural networks to integrate local pixel estimation and global proposal search for salient object detection....|
|||...[20] propose to encode low-level distance map and high-level sematic features of deep CNNs for salient object detection....|
|||...In contrary to the above methods only used specific-level features, we observe that features from all levels are potential saliency cues and helpful for salient object detection....|
|||...Although this method can detect salient objects from different levels, the inner connection of different-level predictions is missing....|
|||...Bidirectional information aggregating learning  j , j = 1, ..., T } and Yn = {yn  Given the salient object detection training dataset S = {(Xn, Yn)}N n=1 with N training pairs, where Xn = {xn j , j = ...|
|||...Most of the images in this dataset contain only one salient object....|
|||...Images of this dataset have one or more salient objects and relatively complex background....|
|||...Images of this dataset are well chosen to include multiple disconnected salient objects or objects touching the image boundary....|
|||...The SED1 has 100 images each containing only one salient object, while the SED2 has 100 images each containing two salient objects....|
|||...Pixel-wise annotation of salient objects was generated by [19]....|
|||...When testing, the proposed salient object detection algorithm runs at about 16 fps with 256  256 resolution....|
|||...Evaluation Metrics: We utilize three main metrics to evaluate the performance of different salient object detection algorithms, including the precision-recall (PR) curves, F-measure and mean absolute ...|
|||...The above overlapping-based evaluations usually give higher score to methods which assign high saliency score to salient pixel correctly....|
|||...However, the evaluation on nonsalient regions can be unfair especially for the methods which successfully detect non-salient regions, but miss the detection of salient regions....|
|||...unds (the first two rows), objects near the image boundary (the 3-4 rows) and multiple disconnected salient objects (the 5-6 rows)....|
|||...Whats more, with our BPR component, our saliency maps provide more accurate boundaries of salient objects (the 1, 3, 4, 6 rows)....|
|||...This indicates that our proposed BPR is capable of detecting and localizing the boundary of most salient objects, while other methods often fail at this fact....|
|||...Conclusion  In this paper, we propose a generic aggregating multilevel convolutional feature framework for salient object detection....|
|||...What is a salient object?...|
|||...a dataset and a baseline model for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Deep contrast learning for salient object  detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Learning to detect salient objects with image-level supervision....|
||37 instances in total. (in iccv2017)|
|26|Li_Contextual_Hypergraph_Modeling_2013_ICCV_paper|...2013 IEEE International Conference on Computer Vision 2013 IEEE International Conference on Computer Vision  Contextual Hypergraph Modeling for Salient Object Detection  Xi Li, Yao Li, Chunhua Shen, A...|
|||...As a result, the problem of salient object detection becomes one of finding salient vertices and hyperedges in the hypergraph....|
|||...an alternative approach based on centerversus-surround contextual contrast analysis, which performs salient object detection by optimizing a cost-sensitive support vector machine (SVM) objective funct...|
|||...Experimental results on four challenging datasets demonstrate the effectiveness of the proposed approaches against the stateof-the-art approaches to salient object detection....|
|||...Recently, a large body of work concentrates on salient object detection [417], whose goal is to discover the most salient and attention-grabbing object in an image....|
|||...Because it is difficult to define saliency analytically, the performance of salient object detection is evaluated on datasets containing human-labeled bounding boxes or foreground masks....|
|||...09/ICCV.2013.413  3321 3328  Image  Hypergraph saliency Figure 1: Illustration of our approaches to salient object detection....|
|||...Global salient object detection approaches [4,5,7,11,12] estimate the saliency of a particular image region by measuring its uniqueness in the entire image....|
|||...We then show that the use of a hypergraph captures more comprehensive contextual information, and therefore enhances the accuracy of salient object detection....|
|||...Here, we propose two approaches to salient object detection based on hypergraph modeling and imbalanced maxmargin learning....|
|||...that of detecting salient vertices and hyperedges in a hypergraph at multiple scales....|
|||...Example results of our approaches to salient object detection are shown in Fig....|
|||...ethod based on imbalanced maxmargin learning, which is capable of effectively discovering the local salient image regions that significantly differ from their surrounding image regions....|
|||...(cid:5)  eE  (4)  (5)  Image  Hypergraph saliency Standard graph saliency Figure 4: Illustration of salient object detection using two different types of graphs (i.e., hypergraph and standard pairwise...|
|||...Clearly, our hypergraph saliency measure is able to accurately capture the intrinsic structural properties of the salient object....|
|||...Hypergraph modeling for saliency detection To more effectively find salient object regions, we propose a hypergraph modeling based saliency detection method that forms contexts of superpixels to captu...|
|||...redges (i.e., superpixel cliques) to effectively capture the intrinsic structural properties of the salient object, as shown in Fig....|
|||...In addition, a salient hyperedge should have the following two properties: 1) it should be enclosed by strong image edges; and 2) its intersection with the image boundaries ought to be small [5, 13]....|
|||...Each image in SED-100 contains only one salient object....|
|||...Each image in the aforementioned datasets contains a human-labelled foreground mask used as ground truth for salient object detection....|
|||...7 shows their quantitative results of salient object detection in the aspect of PR curves....|
|||...both the internal consistency and strong boundary properties of salient objects....|
|||...These approaches are implemented using their either publicly available source code  3326 3333  Image  GT  Ours  GS SP [5]  LR [12]  SF [11]  CB [13]  SVO [15]  RC [7]  HC [7]  RA [16]  FT [4]  CA [14...|
|||...10 shows several salient object detection examples of all the thirteen approaches....|
|||...11 gives three intuitive examples of salient object segmentation (i.e., binarization using the adaptive threshold [4]) based on the proposed approach....|
|||...12 shows some image retar 3327 3334  Figure 11: Examples of salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Segmenting salient objects from  images and videos....|
|||...Automatic salient object extraction with contextual cue....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...This indicates that our approach is capable of effectively preserving the intrinsic structural information on salient objects during image retargeting....|
|||...Conclusion  In this work, we have proposed two salient object detection approaches based on hypergraph modeling and centerversus-surround max-margin learning....|
|||...hat captures the intrinsic contextual saliency information on image pixels/superpixels by detecting salient vertices and hyperedges in a hypergraph....|
|||...Furthermore, we have developed a local salient object detection approach based on centerversus-surround max-margin learning, which solves an imbalanced cost-sensitive SVM optimization problem....|
|||...Compared with the twelve state-of-the-art approaches, we have empirically shown that the fusion of our approaches is able to achieve more accurate and robust results of salient object detection....|
||37 instances in total. (in iccv2013)|
|27|Deep Level Sets for Salient Object Detection|...Deep Level Sets for Salient Object Detection  Ping Hu  Bing Shuai  Jun Liu  Gang Wang  phu005@ntu.edu.sg  bshuai001@ntu.edu.sg  jliu029@ntu.edu.sg  wanggang@ntu.edu.sg  School of Electrical and Electr...|
|||...Later, it was extended to object-level saliency detection [54, 55, 12, 64, 10, 34, 60] which targets at computing saliency maps to highlight the regions of salient objects ac (a)  (b)  (c)  (d)  (e)  Figure 1....|
|||...Because of the unawareness of image content, purely low-level cues are difficult to detect salient objects in complex scenes....|
|||...For this kind of models, it is critical to effectively learn the semantic relationship between salient objects and background from data....|
|||...It is difficult for a network to learn saliency at boundaries of salient regions....|
|||...In this paper, to relive these limitations, we propose an end-to-end deep Level Set network for salient object detection....|
|||...When applying it to salient object detection, which is also a binary segmentation task, our target is to generate a level set function with an interface that accurately separates salient objects from ...|
|||...Instead of directly learning a binary label for each pixel independently, our network is trained to learn the level sets for salient objects....|
|||...e and area can be implicitly represented in the energy function, so the network can be aware of the salient object as a whole instead of learning saliency for every pixel independently....|
|||...In summary, this work has the following three contributions:   We use Level Set formulation to help deep networks learn information about salient objects more easily and naturally....|
|||...The trained network can detect salient objects precisely and output salient maps that are compact and uniform....|
|||...2301  The most widely used one is contrast prior, which believes that salient regions present high contrast over background in certain context [14, 12, 34, 21, 40, 17]....|
|||...Zhang et al [59] and Jiang et al [22] detect salient objects from the perspective of objects uniqueness and surroundness....|
|||...Another useful assumption called center bias prior assumes that salient objects tend to be located at the center of images [54, 64, 59, 56, 53]....|
|||...These methods tend to fail at complex situations because they dont aware the image content or cant effectively learn the interaction between salient objects and background....|
|||...Methods proposed in [33, 30, 24] merge low-level, mid-level and high-level features learned by the VGG16 net [44] together to hierarchically detect salient objects....|
|||...Instead of predicting saliency for every single pixel, superpixel and object region proposal are also combined with deep network [46, 26, 29, 62, 31, 50] to achieve accurate segmentation of salient object....|
|||...Deep Level Sets for Salient Object Detection  3.1.1 Formulation of Level Sets  When applying level set methods [38, 61] for binary segmentation in 2D space , the interface C   is defined as the bounda...|
|||...With these in mind, we combine the level set method with deep networks to detect salient objects....|
|||...el sets to have longer segmentation interfaces, so that it can express more details about shapes of salient objects....|
|||...The last two terms with  > 0 force the salient map to be uniform both inside and outside salient regions....|
|||...The SED2 [2] is composed of 100 images with two salient objects....|
|||...Our model is able to produce saliency maps that highlight salient regions accurately and uniformly....|
|||...0; =0.75; =0; =0;  In this paper, an end-to-end deep level set network have been proposed to detect salient objects....|
|||...Experiments on benchmark datasets demonstrate that the proposed deep level set network can detect salient objects effectively and efficiently....|
|||...7(l)-(n), using Level Sets helps the network to detect more compact salient regions and detect more details of shape....|
|||...Frequency-tuned salient region detection....|
|||...Saliency unified: A deep architecture for simultaneous eye fixation prediction and salient object segmentation....|
|||...Deep contrast learning for salient object  [9] M.-M. Cheng, N. J. Mitra, X. Huang, and S.-M. Hu....|
|||...Global contrast based salient region detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The  secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency net work for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object  detection via low rank matrix recovery....|
|||...Minimum barrier salient object detection at 80 fps....|
||37 instances in total. (in cvpr2017)|
|28|Zhang_Minimum_Barrier_Salient_ICCV_2015_paper|...Minimum Barrier Salient Object Detection at 80 FPS  Jianming Zhang1  Stan Sclaroff1  Zhe Lin2 Xiaohui Shen2 Brian Price2 Radom r M ech2  1Boston University  2Adobe Research  Input  SO  AMC  HS  SIA  H...|
|||...Abstract  We propose a highly efficient, yet powerful, salient object detection method based on the Minimum Barrier Distance (MBD) Transform....|
|||...Powered by this fast MBD transform algorithm, the proposed salient object detection method runs at 80 FPS, and significantly outperforms previous methods with similar speed on four large benchmark dat...|
|||...Introduction  The goal of salient object detection is to compute a saliency map that highlights the salient objects and suppresses the background in a scene....|
|||...Due to the emerging applications on mobile de vices and large scale datasets, a desirable salient object detection method should not only output high quality saliency maps, but should also be highly c...|
|||...In this paper, we address both the quality and speed requirements for salient object detection....|
|||...The Image Boundary Connectivity Cue, which assumes that background regions are usually connected to the image borders, is shown to be effective for salient object detection [39, 33, 36, 35]....|
|||...The proposed salient object detection method runs at about 80 FPS using a single thread, and achieves comparable or better performance than the leading methods on four benchmark datasets....|
|||...We propose a fast salient object detection algorithm based on the MBD transform, which achieves state-ofthe-art performance at a substantially reduced computational cost....|
|||...While saliency detection methods are often optimized for eye fixation prediction, salient object detection aims at uniformly highlighting the salient regions with well defined boundaries....|
|||...Therefore, many salient object detection methods combine the contrast/uniqueness cue with other higher level priors [5, 24, 34, 26, 6], e.g....|
|||...The image boundary prior has been used for salient object detection, assuming that most image boundary regions are background....|
|||...A few attempts have been made to speed up salient object detection....|
|||...However, the past papers [30, 8] did not propose a raster-scanning algorithm to make the MBD transform practical for fast salient object detection....|
|||...ith a new theoretic error bound result, which we believe should be useful beyond the application of salient object detection, e.g....|
|||...In the application of salient object detection, there is no noticeable difference between FastMBD and the exact MBD transform in performance....|
|||...Minimum Barrier Salient Object Detection   D(x)  dI (x)  < I....|
|||...seeds are connected by a path in S.  In this section, we describe an implementation of a system for salient object detection that is based on FastMBD....|
|||...Lastly, several efficient post-processing operations are introduced to finalize the salient map computation....|
|||...MBD Transform for Salient Object Detection  Similar to [33], to capture the image boundary connectivity cue, we set the pixels along the image boundary as the seeds, and compute the MBD transform for ...|
|||...This appearance-based cue is more robust when the salient regions touch the image boundary, and it is complementary to the geometric cue captured by the MBD map B....|
|||...These operations do not add much computational burden, but can effectively enhance the performance for salient object segmentation....|
|||...To make the smoothing level scale with the size of the salient regions,  is adaptively determined by   = s,  (9)  where  is a predefined constant, and s is the mean pixel value on the map B. Secondly,...|
|||...Limitations  Input  MB  MB+  A key limitation of the image boundary connectivity cue is that it cannot handle salient objects that touch the image boundary....|
|||...Our method MB fails to highlight the salient regions that are connected to the image boundary, because it basically only depends on the image boundary connectivity cue....|
|||...9, MB+ cannot fully highlight the salient region, either....|
|||...Conclusion  Figure 9: Some failure cases where the salient objects touch the image boundary....|
|||...Based on FastMBD, we proposed a fast salient object detection method that runs at about 80 FPS....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Global contrast based salient region detection....|
|||...Learning to detect a salient object....|
|||...Learning optimal seeds for diffusion-based salient object detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object  detection via low rank matrix recovery....|
|||...The secrets  of salient object segmentation....|
||37 instances in total. (in iccv2015)|
|29|cvpr18-Learning to Promote Saliency Detectors|...feng@gmail.com, aliborji@gmail.com  Abstract  Salient  Background  The categories and appearance of salient objects vary from image to image, therefore, saliency detection is an image-specific task....|
|||...Introduction  Detecting salient objects or regions of an image, i.e....|
|||...ods usually utilize low-level features and heuristic priors which are not robust enough to discover salient objects in complex scenes, neither are capable of capturing semantic objects....|
|||...The small binary mask in each image indicates the salient object of this image....|
|||...They can learn high-level semantic features from training samples, thus are more effective in locating semantically salient regions, yielding more accurate results in complex scenes....|
|||...For example, signs and persons are salient objects in the first column of Figure 1 , while they belong to the background in the second column....|
|||...Second, categories and appearance of salient objects vary from image to image, while small training data is not enough to capture the diversity....|
|||...For example, the six salient objects shown in Figure 1 come from six different categories and differ wildly in their appearance....|
|||...Consequently, it might be hard to learn a unified detector to handle all varieties of salient objects....|
|||...1644  Considering the large diversity of salient objects, we avoid training a deep neural network (DNN) that directly maps images into labels....|
|||...As a nonparametric model, the NN classifier can adapt well to new data and handle the diversity of salient objects....|
|||...Compared with directly learning to detect diverse salient objects, this would be easier for the network to learn on limited data....|
|||...During training, the DNN is provided with the true salient and background regions, of which the label of a few randomly selected pixels are flipped, to produce anchors....|
|||...Top-down (TD) saliency aims at finding salient regions specified by a task, and is usually formulated as a supervised learning problem....|
|||...Second, thanks to deep learning, our method is capable of capturing semantically salient regions and does well on complex  1645  (c)  (d)  64 128256 512  512  64  64  64 64 64 64  (a)  (b)  Region  e...|
|||...During training, the salient and background pixels for producing anchors are selected using a randomly flipped ground truth ((d) and (e) in the figure), see Sec.3.1....|
|||...Each image consists a salient and a background region, i.e....|
|||...The salient or background region Cmk is also mapped into vectors in D-dimensional metric space by a DNN  with parameter :  mk = (Cmk; ),  (2)  in which mk is the mapping of the salient or background r...|
|||...Iterative testing scheme  In the testing phase, since the ground-truth is unknown, it is not possible to obtain precise salient and background regions to produce anchors as in the training time....|
|||...Given the anchors, we use the nearest neighbor classifier as in Eqn.3 to compute the probability of each pixel belonging to salient regions, i.e....|
|||... 1  t + 1  Z (t) m ,  (5)  where Y (t+1) is the prior saliency map which will be used for selecting salient and background regions in the next it m  eration....|
|||...en partially separate them, and thus can provide information regarding categories and appearance of salient objects in the image....|
|||...1 Compute the embedding vector n of each pixel xn  of X. for t  {1, ..., T } do  2  3  4  5  Select the approximate salient C1 and background region C2 according to Y (t)....|
|||...The precision of a binary map is defined as the ratio of the number of salient pixels it correctly labels, to all salient pixels in this binary map....|
|||...The recall value is the ratio of the number of correctly labeled salient pixels to all salient pixels in the ground-truth map:  precision =   T S  DS    DS   ,  recall =   T S  DS    T S   ,  (6)  in ...|
|||...It can be seen that the saliency maps produced by our methods highlight salient regions that are missed by the baselines....|
|||...Conclusion  Acknowledgment  In this paper, we propose a novel learning method to promote existing salient object detection methods....|
|||...Frequency-tuned salient region detection....|
|||...Deeply supervised salient object detection with short connections....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Non-local deep features for salient object detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Learning to detect salient objects with image-level supervision....|
|||...A stagewise refinement model for detecting salient objects in im 1652  ages....|
||36 instances in total. (in cvpr2018)|
|30|Jiang_Generic_Promotion_of_ICCV_2015_paper|...peng@sdu.edu.cn  Abstract  In this work, we propose a generic scheme to promote any diffusion-based salient object detection algorithm by original ways to re-synthesize the diffusion matrix and constr...|
|||... inverse normalized Laplacian matrix as the original diffusion matrix and promote the corresponding salient object detection algorithm, which leads to superior performance as experimentally demonstrat...|
|||...In the field of saliency detection, two branches have developed, which are visual saliency detection [4,9,10,1215, 19,29,34,39,41] and salient object detection [1,57,11,16 18,2025,27,32,33,35,37,38,40,42]....|
|||...While the former tries to predict where the human eye focuses on, the latter aims to detect the whole salient object in an image....|
|||...Over the past several years, contrast based methods [1, 6, 11, 32] significantly promote the benchmark of salient object detection....|
|||...However, these methods usually miss small local salient regions or bring some outliers such that the resultant saliency maps tend to be nonuniform....|
|||...To tackle these problems, diffusion-based methods [16, 24, 33, 38] use diffusion matrices to propagate saliency information of seeds to the whole salient object....|
|||...In this work, we aim at a generic scheme that promotes any diffusion-based salient object detection algorithm by constructing a good diffusion matrix and a good seed vector at the same time....|
|||...As demonstrated by comprehensive experiments and analysis, the promotion leads to superior performance in salient object detection....|
|||...Diffusion-Based Methods  A diffusion-based salient object detection method usually segments an image into N superpixels first by an algorithm such as SLIC [2]....|
|||...ually s is not complete and we need to propagate the partial saliency information in s to the whole salient region along the graph to obtain the final saliency map [24]....|
|||...Diffusion Map  Diffusion-based salient objection detection algorithms (e.g., [16, 24, 38]) usually use a positive semi-definite matrix, A, to define the diffusion matrix....|
|||...  Based on Eq.s 7 and 8, we make a novel interpretation of the working mechanism of diffusion-based salient object detection: the saliency of a node (called focus node) is determined by all the seed v...|
|||...4.  are less likely to be the salient regions we search for....|
|||...The main steps of the proposed salient object detection algorithm are summarized in Algorithm 1....|
|||...220  Algorithm 1 Promoted Diffusion-Based Salient Object Detection Input: An image on which to detect the salient object....|
|||...6.4, respectively, and compare different salient object detection algorithms, as detailed in Sec....|
|||...When comparing with other salient object detection methods in Sec....|
|||...Promotion of Visual Saliency  Visual saliency detection predicts human fixation locations in an image, which are often indicative of salient objects around....|
|||...Therefore, we use the detected visual saliency as the seed information, and conduct diffusion on it to detect the salient object region in an image....|
|||...In other words, we promote a visual saliency detection algorithm by diffusion for the task of salient object detection....|
|||...2, previous visual saliency detection methods which usually can not highlight the whole salient object all get significantly boosted after difrw ....|
|||...The promotion is so significant that some promoted methods even outperform some state-of-the-art salient objection detection methods, as observed by comparing Fig....|
|||...ltiply a factor GT (j) in Step 1 to ensure that the non-zero seed values are selected from only the salient region; we solve the nonnegative least-squares problem in step 3 of Alg....|
|||...2 to ensure nonnegative elements of s. The adapted OMP will stop when  kfresk2 is below a threshold, c, or the nonnegative seed val ues at the salient region are all selected, as shown in Step 5 of Alg....|
|||...sed ones including PCA [28], GMR [38], MC [16], DSR [21], BMS [40], HS [37], GC [7] and RBD [42] on salient object detection....|
|||...Conclusions  In this work, we make a novel analysis of the working mechanism of the diffusion-based salient object detection....|
|||...Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Image signature: Highlight ing sparse salient regions....|
|||...Learning to detect a salient object....|
|||...Learning optimal seeds for diffusion-based salient object detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object  detection via low rank matrix recovery....|
||36 instances in total. (in iccv2015)|
|31|Jiang_Salient_Object_Detection_2013_CVPR_paper|...There are various applications for salient object detection, including object detection and recognition [25, 46], image compression [21], image cropping [35], photo collage [17, 47], dominant color de...|
|||...This is a principle way in image classification [19], but rarely studied in salient object detection....|
|||...Related work  The following gives a review of salient object detection (segmentation) algorithms that are related to our approach....|
|||...A comprehensive survey of salient object detection can be found from [9]....|
|||...the salient object usually lies in the center of an image, is investigated in [23, 50]....|
|||...A low rank matrix recovery scheme is proposed for salient object detection [41]....|
|||...Besides, spectral analysis in the frequency domain is used to detect salient regions [1, 20]  Additionally, there are several works directly checking if an image window contains an object....|
|||...The recent learning approach [33] aims to predict eye fixation, while our approach is for salient object detection and moreover, we solve the problem by introducing and exploring multi-level regional ...|
|||...Regional contrast descriptor  A region is likely thought to be salient if it is different from its surrounding regions....|
|||...The appearance features attempt to describe the distribution of colors and textures in a region, which can characterize the common properties of the salient object and the background....|
|||...The geometric features include the size and position of a region that may be useful to describe the spatial distribution of the salient object and the background....|
|||...For instance, the salient object tends to be placed near the center of the image while the background usually scatters over the entire image....|
|||...Image regions with similar appearances might belong to the background in one image but belong to the salient object in some other ones....|
|||...It is not enough to merely use the property features to check if one region is in the background or the salient object....|
|||...A region is considered to be confident if the number of the pixels belonging to the salient object or the background exceeds 80% of the number of the pixels in the region, and its saliency score is se...|
|||...This data set [31] includes 5000 images, originally containing labeled rectangles from nine users drawing a bounding box around what they consider the most salient object....|
|||...We manually segmented the salient object (contour) within the user-drawn rectangle to obtain binary masks....|
|||...This data set [3] contains two subsets: SED1 that has 100 images containing only one salient object and SED2 that has 100 images containing two salient objects....|
|||...This data set is a collection of salient object boundaries based on the Berkeley segmentation data set....|
|||...Seven subjects are asked to choose the salient object(s) in 300 images....|
|||...We generate the pixel-wise annotation of the salient objects based on the boundary annotation....|
|||...In this paper, we use it to evaluate the performance of salient object detection....|
|||...Precision corresponds to the percentage of salient pixels correctly assigned, and recall is the fraction of detected salient pixels belonging to the salient object in the ground truth....|
|||...In the property descriptor, the geometric features are ranked higher as salient objects tend to lie in the center in most images....|
|||...As can be seen, our approach (DRFI) achieves the best performance on the MSRA-B and SED1 data sets in which each image contains one single salient object....|
|||...Intuitively, our approach has limited ability when discovering all the salient objects within one image (higher recall)....|
|||...For example, in the first two rows, other approaches may be distracted by the textures on the background while our method almost successfully highlights the whole salient object....|
|||...Conclusions  In this paper, we address the salient object detection problem using a discriminative regional feature integration approach....|
|||...Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object In CVPR, pages  detection via low rank matrix recovery....|
||36 instances in total. (in cvpr2013)|
|32|cvpr18-A Bi-Directional Message Passing Model for Salient Object Detection|...u.cn, lhchuan@dlut.edu.cn,  heyou f@126.com, wg134231@alibaba-inc.com  Abstract  Recent progress on salient object detection is beneficial from Fully Convolutional Neural Network (FCN)....|
|||...The saliency cues contained in multi-level convolutional features are complementary for detecting salient objects....|
|||...In this paper, we propose a novel bi-directional message passing model to integrate multilevel features for salient object detection....|
|||...Although numerous valuable models have been proposed, it is still difficult to locate salient object accurately especially in some complicated scenarios....|
|||...In this paper, we propose a novel bi-directional message passing model for salient object detection....|
|||...In summary, the MCFEM and GBMPM in our model work collaboratively to accurately detect the salient objects (see the third column in Fig....|
|||... feature extraction module to capture rich context information for multi-level features to localize salient objects with various scales....|
|||...The integrated features are complementary and robust for detecting salient objects in various scenes....|
|||...In salient object detection, a lot of deep learning models with various network architectures have been proposed....|
|||...[27] propose two convolutional neural networks to combine local superpixel estimation and global proposal search for salient object detection....|
|||...[18] build a twostage network for salient object detection....|
|||...Different from them, we propose a gated bi-directional message passing module to integrate multi-level features for accurately detecting salient objects....|
|||...Some attempts [37, 8, 14] have been conducted to exploit multi-level CNN features for salient object detection....|
|||...Overview of Network Architecture  In this paper, we propose a bi-directional message passing model to address salient object detection....|
|||...We first feed the input image into the VGG16 net to produce multi-level feature maps which capture different information about the salient objects....|
|||...Multi(cid:173)scale Context(cid:173)aware Feature Extraction  Visual context is quite important to assist salient object detection....|
|||...However, the salient objects have large variations in scale, shape and position....|
|||...Besides, the semantic information at deeper layers helping localize the salient objects and spatial details at shallower ones are both important for saliency detection....|
|||...The HKU-IS dataset proposed in [15] has 4447 images and most of the images include multiple disconnected salient objects....|
|||...The precision value is the ratio of ground truth salient pixels in the predicted salient region....|
|||...It can be seen that our method can accurately detect salient objects....|
|||...We can observe that the proposed MCFEM is effective in salient object detection, especially implemented with dilated convolutional layers, which outperforms the  1747  Input  GT  SRM  Ours Figure 4....|
|||...Conclusion  In this paper, we propose a novel bi-directional message passing model for salient object detection....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Deeply supervised salient object detection with short connections....|
|||...Submodular salient region detection....|
|||...Instance-level salient obIn Proceedings of IEEE Conference on  ject segmentation....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Non-local deep features for salient object detection....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Learning to detect salient objects with image-level supervision....|
|||...A stagewise refinement model for detecting salient objects in images....|
|||...Deep contrast learning for salient object detection....|
||35 instances in total. (in cvpr2018)|
|33|Feng_Local_Background_Enclosure_CVPR_2016_paper|...david.feng, nick.barnes, shaodi.you}@nicta.com.au  cdmccarthy@swin.edu.au  Abstract  Recent work in salient object detection has considered the incorporation of depth cues from RGB-D images....|
|||...Recently the field has expanded to include the detection of entire salient regions or objects [1, 3]....|
|||...We note that salient objects tend to be characterised by being locally in front of surrounding regions, and the distance between an object and the background is not as important as the fact that the b...|
|||...The first, which is proportional to saliency, is the angular density of background around a region, encoding the idea that a salient object is in front of most of its surroundings....|
|||...These approaches do not consider relative depth, and work best when the range of salient objects is closer than the background....|
|||...More recently, the effectiveness of global contrast for RGB salient object detection [7] has inspired similar approaches for RGB-D saliency....|
|||...difference between regions, some methods instead use signed depth difference, improving results for salient objects in front of background [8]....|
|||...[15] observe that while a salient object should be in front of its surrounds, patches on that object may be at a similar depth....|
|||...Depth contrast methods are unlikely to produce good results when a salient object has low depth contrast compared to the rest of the scene (see Figure 1)....|
|||...[23] explore orientation and background priors for detecting salient objects, and use PageRank and MRFs to optimize their saliency map....|
|||...Note that patches lying on salient objects tend to be enclosed by the local background set....|
|||...Given an RGB-D image with pixel grid I(x, y), we aim to segment the pixels into salient and non-salient pixels....|
|||...Note that the pop-out structure corresponding to salient objects is correctly identified....|
|||...Saliency Detection System  We construct a system for salient object detection using the proposed feature....|
|||...an important component of pre-attentive visual attention, with closer objects more likely to appear salient to the human visual system [16]....|
|||...Boundary refinement is a common postprocessing step employed in existing salient object detection systems (e.g....|
|||...Experiments  The performance of our saliency system is evaluated on two datasets for RGB-D salient object detection....|
|||...We then evaluate the contribution of prior application and Grabcut refinement on our salient object detection system on both datasets....|
|||...Finally, we compare our salient object detection system with three state-of-the-art RGB-D salient object detection systems: LMH [22], ACSD [15], and a recently proposed method that exploits global pri...|
|||...We incorporate this feature into a salient object detection system using depth prior, spatial prior, and Grabcut refinement....|
|||...other state-of-the-art RGB-D salient object detection systems....|
|||...This demonstrates that our feature is able to identify salient structure from depth more effectively than existing contrast-based methods....|
|||...Figure 8 shows the output of our salient detection system compared with state-of-the-art methods....|
|||...The angular statistics employed by our depth feature provide a more robust measure of salient structure....|
|||...Failure Cases Since our method measures pop-out structure, it does not produce good results when the salient object is surrounded in all directions by background with lower depth....|
|||...Comparison of output saliency maps produced by our salient object detection system against the output of GP [23], ACSD [15], and LMH [22]....|
|||...Depth incorporating with color improves salient object detection....|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Depth information fused salient object detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Global contrast based salient region detection....|
|||...Depth really matters: Improving visual salient region detection with depth....|
||32 instances in total. (in cvpr2016)|
|34|Kim_Automatic_Content-Aware_Projection_ICCV_2017_paper|...At first, the salient contents such as linear structures and salient regions in the image are preserved by optimizing the single Panini projection model....|
|||...Then, the multiple Panini projection models at salient regions are interpolated to suppress image distortion globally....|
|||...[2] proposed to minimize the distortion of contents such as salient lines and regions through optimization techniques....|
|||...However, it does not consider objects of interest in an image, and hard constraint on linear structures can cause large distortion on salient contents....|
|||...lines) and salient regions in images and videos into account for contents-preserving projection which are very important to increase the subjective quality of projected images....|
|||...The contents analysis step such as viewpoint selection and salient region extraction can be replaced with any other methods....|
|||...The methods mentioned above are very simple and fast, but have a common drawback  they can not preserve all lines and salient objects simultaneously....|
|||...The contents analysis steps such as viewpoint selection and salient region extraction can be replaced with any other methods....|
|||...As in [11], we consider two properties, conformality of salient objects and curvature of linear structures, to measure distortions....|
|||...We first extract line segments and salient objects to be preserved automatically....|
|||...If we have multiple salient objects in an image, we compute multiple optimal Pannini projection parameters for multiple salient objects, respectively....|
|||...lines) and salient regions in images and videos into account for contents-preserving projection which is very important to increase the subjective quality of projected images....|
|||...To extract salient objects, we compute scene saliency as the combination of appearance and motion saliency of the image as  Sscene  i  = wSappear  i  + (1  w)Smotion  i  ,  (1)  i  , Sappear  where Ss...|
|||..., and Smotion  i  i  To find salient objects, we define the appearance saliency as a probability of object existence in the image....|
|||...Thus, we extract local peaks as salient objects by applying non-maximum suppression to the scene saliency....|
|||...To estimate optimal parameters to preserve both linear structure and salient objects, we define two distortion measures as illustrated in Fig....|
|||...To consider shapes of salient objects, we adopt the dis tortion measure of [2] which is defined as  C(p) = (cid:18)cosp  up p  +  up  p(cid:19)2  +(cid:18)cosp  vp p    up  p(cid:19)2  ....|
|||...This optimization is globally applied to consider every linear structure and salient object in the image simultaneously....|
|||...Thus, for one salient point, we project an image around the salient point with a model of which viewpoint is centered at the salient point....|
|||...Then, shapes around salient points are preserved....|
|||...However, regions between salient points have strong distortions because projection models are different from each other....|
|||...To do this, we define anchor points as the salient points projected by the global model....|
|||...After the alignment of the local models, we interpolate local models to fill the regions between salient points smoothly....|
|||...To preserve shapes around salient points, weight wP (u, v) is defined in an exponential form decreasing according to dp(u, v)....|
|||...As the second measure, we consider the conformality, i.e., measuring the degree to which the appearance of an original spherical image is distorted around salient points....|
|||...To measure the conformality, we sample four points around a salient point....|
|||...4, the conformality measure is defined as  Conf ormality =  min (1, 2, 3, 4) max (1, 2, 3, 4)  ,  (12)  where n is a distance between the salient point and the projected sampled point....|
|||...If the four values from 1 to 4 are similar to each other, this value is close to 1, which means that the shape around the salient point is less distorted....|
|||...3.1, we used manually extracted salient points and line segments as input....|
|||...5 show the results of the projected images, and the red points and green lines represent salient points and line segments, respectively....|
|||...For the proposed method, automatically extracted line segments and salient points are also used as inputs....|
||31 instances in total. (in iccv2017)|
|35|Wang_Deep_Networks_for_2015_CVPR_paper|...The final saliency map is generated by a weighted sum of salient object regions....|
|||...In contrast, global methods [1, 24, 29] take the entire image into consideration to predict the salient regions which are characterized by holistic rarity and uniqueness, and thus help detect large ob...|
|||...image contents like edges and noise, global methods are less effective when the textured regions of salient objects are similar to the background (See Figure 1(d))....|
|||...In the global search stage, we search for the most salient object regions....|
|||...The final saliency map is generated by the sum of salient object regions weighted by their saliency values....|
|||...The methods that consider only local contexts tend to detect high frequency content and suppress the homogeneous regions inside salient objects....|
|||...As such, generic object detection are closely related to salient object segmentation....|
|||...[4] use a graphical model to exploit the relationship of objectness and saliency cues for salient object detection....|
|||...In this work, we propose a DNN-based saliency detection method combining both local saliency estimation and global salient object candidate search....|
|||...liency detection, where the DNN-L estimates local saliency of each pixel and the DNN-G searches for salient object regions based on global features to enforce label dependencies....|
|||...The network takes a RGB image patch of 51  51 pixels as an input, and exploits a softmax regression model as the output layer to generate the probabilities of the central pixel being salient and nonsalient....|
|||...it sufficiently overlaps with the ground truth salient region G:  B T G   0.7  min( B ,  G )....|
|||...its overlap with the ground truth salient region is less than a predefined threshold:  B T G  < 0.3  min( B ,  G )....|
|||...Figure 3(c) shows the output of the first layer, where locally salient pixels with different features are highlighted by different feature maps....|
|||...The accuracy score Ai measures the average local saliency value of the i-th object candidate, whereas the coverage score Ci measures the proportion of salient area covered by the i-th object candidate....|
|||...The red candidate region covering almost the entire local salient region has a high coverage score but a low accuracy score....|
|||...The green candidate region located inside the local salient region has a high accuracy score but a low coverage score....|
|||...Most of the images include only one salient object with high contrast to the background....|
|||...Many images in this data set have multiple salient objects of various sizes and locations....|
|||...The precision and recall of a saliency map are computed by segmenting a salient region with a threshold, and comparing the binary map with the ground truth....|
|||...Based on the overlap rate oi with the ground truth salient region, the i-th candidate region is classified as foreground (oi > 0.7) or background (oi < 0.2)....|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Global contrast based salient region detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
||31 instances in total. (in cvpr2015)|
|36|Predicting Salient Face in Multiple-Face Videos|...Predicting Salient Face in Multiple-face Videos  Yufan Liu, Songyang Zhang, Mai Xu, and Xuming He  Beihang University, Beijing, China  ShanghaiTech University, Shanghai, China  Abstract  Although the ...|
|||...Therefore, we propose in this paper a novel deep learning (DL) based method to predict salient face in multiple-face videos, which is capable of learning features and transition of salient faces acros...|
|||...In particular, we first learn a CNN for each frame to locate salient face....|
|||...Taking CNN features as input, we develop a multiple-stream long short-term memory (M-LSTM) network to predict the temporal transition of salient faces in video sequences....|
|||...Particularly, detecting salient objects, such as faces, plays an important role in video analytics, human-computer interface design and event understanding....|
|||...Existing literature on saliency prediction typically focuses on finding salient face in static images [21]....|
|||...This figure mainly demonstrates transition of salient faces and characters of long/short-term correlation between salient faces across frames....|
|||...More importantly, both longand shortterm correlation of salient faces across frames, which is critical in modeling attention transition across frames for multiple-face videos (see Figure 1), is not ta...|
|||...(3) We propose a DL-based method to predict the salient face with transition across frames, which integrates a CNN and an LSTM-based RNN model....|
|||... (RNN), we develop a multiplestream LSTM (M-LSTM) network for predicting the dynamic transitions of salient faces alongside video frames, taking the extracted CNN features as the input....|
|||...Among 65 videos in MUFVET-I, for instance, 24 videos have salient objects other than faces, among which 3 videos have new objects appearing in the scenes....|
|||...The small value of  also implies that other features need to be learned for predicting salient face....|
|||...Second, we design a CNN to learn the features related to salient face at each static video frame, which is discussed in Section 3.2....|
|||...Section 3.3 presents MLSTM that learns to predict salient face, by taking into consideration saliency-related features of CNN and the temporal transition of salient faces across video frames....|
|||...After the convolutional feature extraction in GoogleNet, we use two fully connected (FC) layers, with softmax activation function, to decide whether the face is salient or not....|
|||...The first FC layer has 128 units, whose outputs are used as the features for predicting the salient face....|
|||...The second FC layer has 2 units, indicating the salient or non-salient face....|
|||...For training CNN, we automatically label each detected face to be salient or non-salient, according to the fixations falling into the face region....|
|||...Hence, the faces that take up above 60% fixations are annotated as salient faces, and other faces are seen as non-salient ones....|
|||...M-LSTM for Salient Face Prediction  The CNN defined above mainly extract spatial information of each face at a single frame....|
|||...To model temporal dynamics of attention transition in videos, we now develop a novel M-LSTM to predict salient face in the video setting....|
|||...It is worth mentioning that the LSTM chunk is capable of learning long/short-term dependency of salient face transition as well as overcoming the problem of vanishing gradient....|
|||...After training M-LSTM, wn,t can be achieved for predicting salient face....|
|||...ce in [41], while the consideration of temporal transition enables our method to accurately predict salient face across frames....|
|||...The main reason is that the utilization of only static features in [21] may predict wrong salient face in a video....|
|||...From this figure, one may observe that our method is capable of finding the salient face....|
|||...For example, we can see from Figure 8 that the salient face is changed from the girl to the man and then back to the girl, which is extremely consistent with our prediction....|
|||...On the contrary, [41] finds all three faces as salient ones, and [21] misses the salient face of the speaking man....|
|||...Again, this figure verifies that our method is able to precisely locate salient face by considering temporal saliency transition in M-LSTM....|
|||...To predict the salient face in multiple-face videos, we proposed in this paper a DLbased method, in which both CNN and RNN are combined in a framework and then trained over MUFVET-II....|
|||...Specifically, CNN, fined-tuned on Google Net, was adopted in our DL-based method, for automatically learning the features relevant to locating the salient face....|
||31 instances in total. (in cvpr2017)|
|37|Jiang_Saliency_Detection_via_2013_ICCV_paper|...We jointly consider the appearance divergence and spatial distribution of salient objects and the background....|
|||...e absorbed time of transient node measures its global similarity with all absorbing nodes, and thus salient objects can be consistently separated from the background when the absorbed time is used as ...|
|||...All bottom-up saliency methods rely on some prior knowledge about salient objects and backgrounds, such as contrast, compactness, etc....|
|||...[35] cast saliency detection into a graph-based ranking problem, which performs label propagation on a sparsely connected graph to characterize the overall differences between salient object and background....|
|||...In addition, equilibrium distribution based saliency models only highlight the boundaries of salient object while object interior still has low saliency value....|
|||...As salient objects seldom occupy all four image boundaries [33, 5] and the background regions often have appearance connectivity with image boundaries, when we use the boundary nodes as absorbing node...|
|||...Inspired  1550-5499/13 $31.00  2013 IEEE 1550-5499/13 $31.00  2013 IEEE DOI 10.1109/ICCV.2013.209 DOI 10.1109/ICCV.2013.209  1665 1665  scenes with complex salient objects....|
|||...h transient node can reflect its overall similarity with the background, which helps to distinguish salient nodes from background nodes....|
|||...Furthermore, the main objectives in [9, 14, 31] are to predict human fixations on natural images as opposed to identifying salient regions that correspond to objects, as illustrated in this paper....|
|||...[11], which exploits the hitting time on the fully connected graph and the sparsely connected graph to find the most salient seed, based on which some background seeds are determined again....|
|||...te the problem of using the equilibrium distribution to measure saliency, the identification of the salient seed is difficult, especially for the  1666 1666  states and t transient states, renumber t...|
|||...Because the salient objects seldom occupy all image borders [33], we duplicate the boundary superpixels around the image borders as the virtual background absorbing nodes, as shown in Figure 2....|
|||...Consequently, the background regions near the image center possibly present comparative saliency with salient objects, thereby decreasing the contrast of objects and backgrounds in the resulted saliency maps....|
|||...However, if the kind of region belongs to salient object, its saliency will be also incorrectly suppressed....|
|||...This dataset is first used for salient object segmentation evaluation [23], where seven subjects are asked to label the foreground salient object masks....|
|||...The precision is defined as the ratio of salient pixels correctly assigned to all the pixels of extracted regions....|
|||...Examples in which the salient objects appear at the image boundaries....|
|||...It should be noted that the absorbing nodes may include object nodes when the salient objects touch the image boundaries, as shown in Figure 4....|
|||...We note that the proposed method more uniformly highlights the salient regions while adequately suppresses the backgrounds than the other methods....|
|||...This dataset contains the ground truth of salient region marked as bounding boxes by nine subjects....|
|||...We accumulate the nine ground truth, and then choose the pixels with consistency score higher than 0.5 as salient region and fit a bounding box in the salient region....|
|||...Frequency tuned salient region detection....|
|||...Fusing generic objectness  and visual saliency for salient object detection....|
|||...Global  contrast based salient region detection....|
|||...Random walks on graphs  for salient object detection in images....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Learning to  detect a salient object....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection  via low rank matrix recovery....|
|||...Automatic salient object  extraction with contextual cue....|
||31 instances in total. (in iccv2013)|
|38|Zhu_Saliency_Pattern_Detection_ICCV_2017_paper|...ling@temple.edu, {wujin, denghuiping, liujin}@wust.edu.cn  Abstract  In this paper we propose a new salient object detection method via structured label prediction....|
|||...ncy Object Detection is more popular in the computer vision community as it is designed for general salient object discovery from an image [33]....|
|||...Intuitively, a salient object should visually stands out from its surroundings [23]....|
|||...As the pixel-wise prediction ignores the relationship between pixels, inner regions of a proto-object may take very different salient values....|
|||...Bottom-up attention refers to detect salient objects from the perceptual data that only comes from images itself....|
|||...ery pixel is evaluated independently, nearby pixels with the same semantics may take very different salient values due to the variation  of local context....|
|||...It generates candidate proposals that may enclose the salient objects as shown in Fig....|
|||...l enclosed n = d2 pixels: pi  R, i = 1, 2, ..., n, where s = [s1, s2, ..., sn] and si refers to the salient value of pi....|
|||...Intuitively, a certain lj can be a representative for the saliency structure of many different image patches if considering their pixel-level salient values as binary....|
|||...We observe that salient regions roughly occupy 10% to 40% of pixels in an image on average....|
|||...It produces a binary map that indicates the rough locations of salient objects....|
|||...1, the salient values of all pixels in each sliding window of a proposal are the weighted combination of selected SSPs....|
|||...Saliency propagation  Taking some high confidential salient regions as seed, most saliency propagation methods require an adjacent similarities to distribute saliency mass to similar nearby regions al...|
|||...Following [4], we solve the following quadratic energy model to obtain the salient value yi for segment xi:  argmin  yi Xi  kii(yi  vi)2 +  1  2Xi,j  E(i, j)(yi  yj)2,  (9)  where vi  v is the average...|
|||...hree kinds of training samples are collected: 1) positive patches, at least 50% of pixels in it are salient; 2) weak positive patches, the percentage of salient pixels is in the range of (0, 50%); 3) ...|
|||...Visual comparison of salient maps  We first compare the results of all evaluated methods qualitatively....|
|||...Form top to bottom, those examples include images with: 1) clutter background, 2) salient object with monotonous color, 3) large salient objects, 4) small salient objects, 5) multiple salient objects,...|
|||...In other word, these measures care more about the performance of methods on detecting the salient regions than their ability of avoiding highlighting non-salient regions....|
|||...Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...Deep contrast learning for salient object detection....|
|||...Deepsaliency: Multi-task deep neural  5475  network model for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Learning to detect a salient object....|
|||...Learning Optimal Seeds for Diffusion-Based Salient Object Detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A multi-size superpixel approach for salient object detection based on multivariate normal distribution estimation....|
||31 instances in total. (in iccv2017)|
|39|cvpr18-Progressively Complementarity-Aware Fusion Network for RGB-D Salient Object Detection|...               Progressively Complementarity-aware Fusion Network    for RGB-D Salient Object Detection             Hao Chen                          Youfu Li*   City University of Hong Kong, Kowloon,...|
|||... How  incorporate  cross-modal  complementarity  sufficiently is the cornerstone question for RGB-D salient  object detection....|
|||...The experiments on public datasets  show the effectiveness of the proposed CA-Fuse module and  the RGB-D salient object detection network....|
|||...Introduction   The  aim  of  salient  object  detection  is  to  identify  the  object/objects attracting human beings most in a scene [2,  3]....|
|||...dy of deep convolutional neural  networks  (CNNs)  [11-17]  have  been  designed  for  RGB-induced  salient  object  detection  and  have  achieved  appealing  performance....|
|||...However,  when  the  salient  object  and  background  these  RGB-induced saliency detection models may be powerless  to discriminate the salient object from background....|
|||...architecture  of  the  proposed  progressively  complementarity-aware  fusion  network  for  RGB-D  salient  object  detection....|
|||...To  be  more  specific,  the  deeper  features  typically carry more global contextual information and are  more likely to locate the salient object correctly, while the  shallower  details....|
|||...the   In  summary,   the  proposed  RGB-D  salient  object   detection network enjoys several distinguished benefits:   1)  The  cross-modal  complementarity  can  be  explicitly   3052     Figure 3:...|
|||...It  is  able  to  capture  cross-level  complementary information sufficiently to locate the salient  object and meanwhile highlight its details in an end-to-end  manner (only 0.06s testing time for e...|
|||...Related work      Previous  RGB-D  salient  object  detection  models  [31-41]  fuse  RGB  and  depth  information  by  three  main  modes:  serializing  RGB  and  depth  as  undifferentiated  4-chann...|
|||...Thus, the final loss function  of the whole RGB-D salient object detection network is         L  final  =  6  +  m l CAR  m  =  2  d  (  K      Pw  k  k RD  k  =  2  ,  Y   ),  (4)     where wk is the...|
|||...Visually,  the  shallow  identifying  edge  information  and  the  deep  layers  are  able  to  learn  global  contexts  to  locate  the  salient  object....|
|||...3(b) module also fails to utilize the  cross-modal complement sufficiently  to  remove  confusing  background  and  refine  salient  details....|
|||...8 such as the appearance or depth of the salient object is  not  distinctive  from  the  background  (e.g.,  the  1st-3rd  rows  and the 4th-5th rows, respectively)....|
|||...In the 6th row, the depth of the salient   object is locally-connected with some background objects....|
|||...Also,  the  6th  row  includes  multiple  disconnected  salient  objects....|
|||...And  in  the  7th  row,  the  appearance  of  the  salient  object is intra-variant....|
|||...In  these  challenging  cases,  most  of  other  methods  are  unlikely  to  locate  the  salient  object  due  to  the  lack  of  high-level  contextual reasoning or robust multi-modal fusion strategy....|
|||...Although the CTMF method is able to obtain more correct  and uniform saliency maps  than  others,  the  fine  details  of  the salient objects are lost severely due to the deficiency of  cross-level fusion....|
|||...Conclusion       In  this  work,  we  propose  an  end-to-end  RGB-D  salient  object detection network, which is complementarity-aware  for  fusing  both  cross-modal  and  cross-level  features....|
|||...Global contrast based salient region detection....|
|||...DHSNet:  Deep  Hierarchical  Saliency   Network for Salient Object Detection....|
|||...Deeply  supervised  salient  object  detection  with  short  connections....|
|||...Background  prior-based  salient  object  detection  via  deep  reconstruction  residual....|
|||...Depth really  Matters:  Improving  Visual  Salient  Region  Detection  with  Depth....|
|||...Local  background enclosure for RGB-D salient object detection....|
|||...Depth-Aware Salient Object Detection and Segmentation via  Multiscale  Discriminative  Saliency  Fusion  and  Bootstrap  Learning....|
|||...Rgbd  salient  object  detection:  a  benchmark  and  algorithms....|
|||...Frequency-tuned salient region detection....|
||30 instances in total. (in cvpr2018)|
|40|Tong_Salient_Object_Detection_2015_CVPR_paper|...Second, a strong classifier based on samples directly from an input image is learned to detect salient pixels....|
|||...Although significant progress has been made, it remains a challenging task to develop effective and efficient algorithms for salient object detection....|
|||...ch areas: visual attention which is extensively studied in neuroscience and cognitive modeling, and salient object detection which is of great interest in computer vision....|
|||... method, where a set of training samples is collected, where positive samples are pertaining to the salient objects while negative samples are from the background in this image....|
|||...Bootstrap learning for salient object detection....|
|||...While it is able to identify salient pixels, the results contain a significant amount of false detections....|
|||...While the above-mentioned contrast-based methods are simple and effective, pixels within the salient objects are not always highlighted well....|
|||...[10] utilize a soft abstraction method to remove unnecessary image details and produce perceptually accurate salient regions....|
|||...[43] construct a salient object detection method based on boundary connectivity....|
|||...As these two categories bring forth different properties of efficient and effective salient detection algorithms, we propose a bootstrap learning approach which exploits the strength of both bottom-up...|
|||...Bootstrap Saliency Model  Figure 2 shows the main steps of the proposed salient object detection algorithm....|
|||...The dark channel of image patches is mainly generated by colored or dark objects and shadows, which usually appear in the salient regions as shown in Figure 3....|
|||...Multiscale Saliency Maps  The accuracy of the saliency map is sensitive to the number of superpixels as salient objects are likely to appear at different scales....|
|||...The SOD dataset [26] is composed of 300 images from the Berkeley segmentation dataset where each one is labeled with salient object boundaries, based on which the pixel level ground truth [34] is built....|
|||...Some of the images in the SOD dataset include more than one salient object....|
|||...It is challenging due to the fact that every image has two salient objects....|
|||...We note that these salient objects appear at different image locations although the center-bias is used in the proposed algorithm....|
|||...n of the LBP features (effective for texture classification), the proposed method is able to detect salient objects accurately despite similar appearance to the background regions as shown in the four...|
|||...Conclusion  In this paper, we propose a bootstrap learning model for salient object detection in which both weak and strong saliency models are constructed and integrated....|
|||...Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Center-surround divergence of  feature statistics for salient object detection....|
|||...The secrets  of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
||30 instances in total. (in cvpr2015)|
|41|Scharfenberger_Statistical_Textural_Distinctiveness_2013_CVPR_paper|...rio, Canada  Abstract  A novel statistical textural distinctiveness approach for robustly detecting salient regions in natural images is proposed....|
|||...To achieve saliency detection in an automatic manner, one must define what constitutes as a salient object based on some quantifiable visual attributes such as intensity, color, structure, texture, si...|
|||...In the context of saliency in natural images, one can then view salient objects of interest as objects that possess textural characteristics that are highly distinctive from a human observer perspecti...|
|||...aracteristics are:  1. the choice of appropriate texture representations for distinguishing between salient and non-salient regions,  2. the added computational complexity associated with textural cha...|
|||...roach is designed to take explicit advantage of the textural characteristics in the image to detect salient regions in an efficient yet characteristic manner....|
|||...All these approaches are designed to identify salient regions with high visual stimuli, but tend to blur saliency maps and to highlight local features such as small objects....|
|||...To better preserve the structure of salient regions, Hou et al....|
|||...However, these methods highlight boundaries of salient regions rather than their entire region....|
|||...However, extracting salient regions in images with textured background properly is challenging for these methods....|
|||...However, down sampling or reducing the dimensionality of the patches may lead to a loss of small salient regions....|
|||...[21] abstract input images into homogeneous elements, and determine salient regions by applying two contrast measures based on the spatial distribution and uniqueness of elements....|
|||...Matrix decomposition based on a previously trained background model is then performed to identify salient regions, which have to be refined using strong  978978978980980  Figure 2: Architecture for s...|
|||...However, none of these approaches explicitly consider rotational-invariant neighborhood-based texture representations (atoms) for salient region detection....|
|||...ng the relationships between each texture pair  980980980982982  Figure 4: Example image containing salient objects, and the learned texture model with pixels associated with the corresponding atoms (...|
|||...Statistical textural distinctiveness graphical  model construction  In natural images, salient regions of interest can be characterized as regions that are visually distinct from the rest of the scene...|
|||...In this work, we first consider a salient region of interest as regions that have highly unique and distinctive textural characteristics when compared to the rest of the scene (see Fig....|
|||...Here, we introduce the concept of statistical textural distinctiveness, where an area of interest is salient if it has low textural pattern commonality compared to the rest of the scene....|
|||... potential of our proposed statistical texture distinctiveness approach (TD) for robustly detecting salient regions, we evaluated our method based on the public EPFL database [1]....|
|||...In the second experiment, we compare the performance of our approach with other approaches, with segmenting salient objects using both adaptive thresholds and GrabCut [23]....|
|||...In the second experiment, we applied an image dependent threshold on the saliency maps to segment salient regions....|
|||...obtained showed that the distribution of saliency values follows a Gaussian mixture model, with non-salient values having larger probabilities than salient values....|
|||...In comparison to other approaches, the textural distinctiveness scheme can detect more salient regions with high precision....|
|||...c) Precision, recall and F-measure for cut-based (GrabCut [23]) segmentation of salient objects, initialized with saliency maps from all tested saliency approaches....|
|||...rimental results using a public natural image dataset demonstrated strong potential for identifying salient regions in images in an efficient manner, thus illustrating the usefulness of explicitly inc...|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...To increase the robustness of segmenting salient regions, Cheng et al....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
||30 instances in total. (in cvpr2013)|
|42|Lee_Deep_Saliency_With_CVPR_2016_paper|... advances in saliency detection have utilized deep learning to obtain high level features to detect salient regions in a scene....|
|||...ow level features such as color, texture and location information to investigate characteristics of salient regions including objectness, boundary convexity, spatial distribution, and global contrast....|
|||...Yet, it was still very hard to differentiate salient regions from their adjacent non-salient regions because their feature distances were not directly encoded....|
|||...l features can play complementary roles to assist high level features with the precise detection of salient regions....|
|||...Related Works  In this section, representative works in salient region detection are reviewed....|
|||...Recent trends in salient region detection utilize learningintroduced by Liu based approaches, which were first et al....|
|||...l Color TransChain(MC) form(HDCT) [15], and Hierarchical Saliency(HS) [29] are the top 6 models for salient region detection reported in the benchmark paper [5]....|
|||...d often generate high-dimensional features to increase discriminative power [15, 13] to distinguish salient regions from non-salient regions....|
|||...where 0 and 1 denote non-salient and salient region labels respectively, and z0 and z1 are the score of each label of training data....|
|||...el using only the high level feature map from the deep CNN detected the approximate location of the salient objects but was unable to capture detailed location because the high level feature maps had ...|
|||...On the other hand, the model with only the low level features failed to distinguish the salient object in the first row....|
|||...With both the low level feature distances and the high level feature map, the models could correctly capture salient objects and their exact boundaries....|
|||...Also, we found that the ELD-map often helps to find salient objects that are difficult to detect using only CNN as shown in the second row....|
|||....164 0.227  0.095 0.102 0.127 0.125 0.150 0.142 0.181 0.177 0.218  Table 4: The F-measure scores of salient region detection algorithms on five popular datasets....|
|||...Table 5: The Mean Absolute Error(MAE) of salient region detection algorithms on five popular datasets....|
|||...From each image, we use about 30 salient superpixels and 70 non-salient superpixels; around 0.9 million input data are generated....|
|||...The PR graph and f-measure score tend to be more informative than ROC curve because salient pixels are usually less than nonsalient [5]....|
|||...The overlapping-based evaluations give higher score to methods which assign high saliency score to salient pixel correctly....|
|||...However, the evaluation on non-salient regions can be unfair especially for the methods which successfully detect non-salient regions, but missed the detection of salient regions [5]....|
|||...From the top to the bottom, row 1-2 are the images with a low-contrast salient object, row 3-4 are with complicated background, row 5-6 are with multiple salient objects and row 7 is with a salient ob...|
|||...s difficult cases including low-contrast objects (row 1-3), complicate backgrounds (row 4-6), small salient objects (row 7-8), multiple salient objects(row 9-10) and touching boundary examples (row 11...|
|||...Our algorithm shows especially good performance on images with low-contrast salient objects and complicated backgrounds, and also works well on other difficult scenes....|
|||...The first and the second results contain correct salient objects but also highlight non-salient regions....|
|||...The third and fourth examples have the extremely difficult scenes with a small, low-contrast and boundary touching the salient object....|
|||...For these difficult scenes, MCDL [21] and MDF [16] also fail to find the salient objects precisely....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...The secrets of salient object segmentation....|
|||...of Computer  Learning to detect a salient object....|
||29 instances in total. (in cvpr2016)|
|43|Zeisl_Automatic_Registration_of_2013_ICCV_paper|...2013 IEEE International Conference on Computer Vision  Automatic Registration of RGB-D Scans via Salient Directions  Bernhard Zeisl ETH Zurich  zeislb@inf.ethz.ch  Kevin K oser  GEOMAR, Kiel kkoeser@g...|
|||...We utilize the principle of salient directions present in the geometry and propose to extract (several) directions from the distribution of surface normals or other cues such as observable symmetries....|
|||...Each salient direction is then exploited to render an orthographic view, and by this way removing the perspective effects that had been introduced by the particular scanner position....|
|||...The remainder of the paper is structured as follows: After a discussion of existing registration techniques in the next section, we show how to obtain viewpoint invariance from salient directions in Sec....|
|||...5 cover details of our approach for salient direction detection and pose estimation....|
|||...(middle, right:) Generated salient direction rectified (SDR) renderings along corresponding salient directions....|
|||...Viewpoint Invariance via Salient Directions Our novel approach to register widely separated scans builds upon image features rather than 3D geometry features, because image features are plenty, well l...|
|||...Instead we exploit the entire scene information by the concept of salient directions....|
|||...Let us now define what we mean by a salient direction....|
|||...A salient direction is a real-world direction in global coordinates d sal that can be observed locally as d sal i  in independent scans i and j:  , d sal  j  sal  d  = R  T i d  sal i = R  T j d  sal j  ....|
|||...(1)  Intuitively, imagine d sal is the north direction, that is repre Figure 3: Orthographic renderings along a salient direction....|
|||...A salient direction rectified (SDR) image, is an image which is obtained by rendering the scene along a salient direction d sal  i with orthographic projection matrix  (cid:4)  (cid:5)  with Pi d  sal...|
|||...Given a salient direction d sal with corresponding local directions d sal in scans i and j, then corresponding points in the two SDR-images relate to each other via a 2D similarity transformation....|
|||...Figure 4: Support regions for different detected salient directions (color coded), shown for 3 cube faces....|
|||...Detection of salient directions (per scan)....|
|||...Normalization of image data with respect to salient di rections (per direction per scan)  3....|
|||...Salient Direction Detection and Image Nor malization Given a salient world direction that can be identified in two different scans, we have shown that we can transform the image content in a way that ...|
|||...However, in this contribution we demonstrate the idea using salient directions derived from characteristics of geometric structures, that is peaks in the distribution of surface normals (cf....|
|||...(cid:2)(cid:3)a(cid:3)2 + (cid:3)b(cid:3)2  2aTb  (cid:3)a  b(cid:3)2 = 1  (cid:3)  2  dered from salient directions highly supported by structures near the scanner, and repeatability of salient dire...|
|||...Note that there exists a sign ambiguity for the symmetry plane normal, thus we use both possible normal directions as salient direction....|
|||...6) are set to 10 Repeatability of Salient Directions It is essential for successful registration that we extract at least one salient direction (up to small variation) in both viewpoints....|
|||...Thus, in these regions corresponding salient directions (defined as direc at maximum) can and should get suptions differing by 10 port....|
|||...We now determine repeatability scores by comparing the number of corresponding salient directions to the total number of detected salient directions....|
|||...Conclusion  In this work we have presented the novel concept of obtaining viewpoint invariance by means of an orthographic projection along detected salient directions in range data....|
|||...We have proven that resulting salient direction rectified (SDR) images for corresponding salient directions in different scans are identical up to a 2D similarity transformation in the general case or...|
|||...We have proposed to utilize modes in the distribution of surface normals for salient direction detection....|
|||...(Lower left parts) Repeatability scores for salient directions, i.e....|
|||...the ration of found and present salient directions in the scan overlap....|
||28 instances in total. (in iccv2013)|
|44|Cheng_Efficient_Salient_Region_2013_ICCV_paper|...oup, Oxford Brookes University  Shuai Zheng Vibhav Vineet Nigel Crook  Abstract  Detecting visually salient regions in images is one of the fundamental problems in computer vision....|
|||...We propose a novel method to decompose an image into large scale perceptually homogeneous elements for efficient salient region detection, using a soft image abstraction representation....|
|||...ssignment of comparable saliency values across similar regions, and producing perceptually accurate salient region detection....|
|||...We evaluate our salient region detection approach on the largest publicly available dataset with pixel accurate annotations....|
|||...Introduction  The automatic detection of salient object regions in images involves a soft decomposition of foreground and background image elements [7]....|
|||...ting human fixation points [6, 32] (another major research direction of visual attention modeling), salient region detection methods aim at uniformly highlighting entire salient object regions, thus b...|
|||...two emerging trends:  In terms of improving salient region detection, there are  Global cues: which enable the assignment of comparable saliency values across similar image regions and which are prefe...|
|||...3), which abstract unnecessary details, assign comparable saliency values across similar image regions, and produce perceptually accurate salient regions detection results (b)....|
|||...We extensively evaluate our salient object region detection method on the largest publicly available dataset with 1000 images containing pixel accurate salient region annotations [2]....|
|||...ls [5], as well as quantitative analysis of different methods in the two major research directions: salient object region detection [7] and human fixation prediction [6, 32]....|
|||...[52] find salient image regions using information theory....|
|||...While these algorithms are generally better at preserving global image structures and are able to highlight entire salient object regions, they suffer from high computational complexity....|
|||...[42] made the important observation that decomposing an image into perceptually homogeneous elements, which abstract unnecessary details, is important for high quality salient object detection....|
|||...erceptually homogeneous elements, resulting in the efficient evaluation of global cues and improved salient object region detection accuracy....|
|||...centage of salient pixels correctly assigned, while recall measures the percentage of salient pixel detected....|
|||...The GC saliency map is better at uniformly highlighting the entire salient object region but its precision recall values are worse....|
|||...However, the saliency map in (d) is closer to the ground truth (b) and better reflects the true salient region in the original image (a)....|
|||...Our method aims at the first problem: finding the most salient and attention-grabbing object in a scene....|
|||...Although this dataset only contains images with non-ambiguous salient objects, we argue that efficiently and effectively finding saliency object region for such images is already very important for ma...|
|||...Experimental results on the largest public available dataset show that our salient object region detection results are 25% better than the previous best results (compared against 18 alternate methods)...|
|||...In IEEE CVPR,  Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||27 instances in total. (in iccv2013)|
|45|cvpr18-Detect Globally, Refine Locally  A Novel Approach to Saliency Detection|...du.cn, aliborji@gmail.com  Abstract  Effective integration of contextual information is crucial for salient object detection....|
|||...To address this problem, we proposes a global Recurrent Localization Network (RLN) which exploits contextual information by the weighted response map in order to localize salient objects more accurately....|
|||...When it comes to the imagebased salient object detection, two major problems need to be tackled: how to highlight salient objects against the cluttered background and how to preserve the boundaries of...|
|||...However, in view of the fact that salient objects may share some similar visual attributes with the background distractors and sometimes multiple salient objects  (a)  (b)  (c)  (d)  Figure 1....|
|||...ome maps are too cluttered which can introduce misleading information when detecting and segmenting salient objects....|
|||...CWM serves to filter out the distractive and cluttered background and make salient objects stand out....|
|||...former recurrently focuses on the spatial distribution of various scenarios to help better localize salient objects and the latter helps refine the saliency map by the relations between each pixel and...|
|||...propose a multi-context deep learning structure for salient object detection....|
|||...Local features and global cues are incorporated for generating a weighted sum of salient object regions....|
|||...The multi-scale feature maps at each layer can assist to locate salient regions and recover detailed structures at the same time....|
|||...Intuitively, if pixel i is salient at position (x, y), the pixel in the response map related to it should be assigned a higher value....|
|||...However, some detailed structures along the boundaries of salient objects are still missing....|
|||...HKU-IS has 4,447 images which are selected by meeting at least one of the following three criteria: multiple salient objects with overlapping, objects touching the image boundary and low color contrast....|
|||...The examples are shown in various scenarios, including multiple salient objects (row 1-2), the small ob 43263132  *  Baseline  CWM  RM BRN  ECSSD  THUR15K  HKU-IS  DUTS  DUT-OMRON  F-measure MAE F-me...|
|||...ject (row 3), the object touching the image boundary (row 4) and salient objects sharing similar color appearance with the background (row 5-7)....|
|||...Conclusion  In this paper, we propose a novel Localization-toRefinement network for salient object detection from the global and local view....|
|||...The Recurrent Localization Network (RLN) can learn to better localize salient objects by  1  0.9  0.8  0.7  0.6  0.5  0.4  0.3  0.2  0.1  0        Precision  Recall  Fmeasure  Baseline  RM1  RM1*  RM1...|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Look, perceive and segment: Finding the salient objects in images via two-stream fixation-semantic cnns....|
|||...GlobIEEE TPAMI,  al contrast based salient region detection....|
|||...Deeply supervised salient object detection with short connections....|
|||...Deep contrast learning for salient object  detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
|||...Learning to detect salient objects with image-level supervision....|
|||...A stagewise refinement model for detecting salient objects in images....|
||27 instances in total. (in cvpr2018)|
|46|Zhao_Saliency_Detection_by_2015_CVPR_paper|...In this paper, we tackle this problem by proposing a multi-context deep learning framework for salient object detection....|
|||...Objects like the house cannot be classified as salient objects from the low-contrast background either based on low-level saliency cues or background priors, but they are semantically salient in high-...|
|||...An appropriate scope of context is also very important to help a salient object stand out from its context meanwhile keep those non-salient objects suppressed in background....|
|||...the red which are salient objects....|
|||...dashed boxes in Figure 2(a)) is adopted to determine the saliency, then all these object are highlighted as salient objects, as shown in Figure 2(c)....|
|||...2(d), if the flower and the leaf are considered together, then only the flower is classified as the salient object; if the car and the guard fence are considered as in the same picture, then only the ...|
|||...The last layer of the network structure has 2 neurons followed by a softmax function as output, indicating the probabilities of centered superpixel whether being in background or belonging to a salient object....|
|||...y is the prediction of saliency for the centered superpixel, where y = 1 for salient superpixel and y = 0 for background....|
|||...The label of each window is determined by thresholding the overlap ratio between the centered superpixel and corresponding groundtruth salient object mask....|
|||...The MSRA10k dataset is a subset of the MSRA Salient Object Dataset [36], which originally provides salient object annotation in terms of bounding boxes provided by 3-9 users....|
|||...Benchmark Datasets  ASD [1] includes 1, 000 images sampled from the MSRA Salient Object Database [36]....|
|||...SED1 [5] contains 100 images of a single salient object  annotated manually by three users....|
|||...SED2 [5] contains 100 images of two salient objects an notated manually by three users....|
|||...Our approach significantly outperforms all the state-of-the-art salient object segmentation algorithms....|
|||...It is obvious that our approach is able to highlight the salient object parts more coherently, and has a better prediction especially in complex scene with confusing background, such as the cases in t...|
|||...In CVPR,  Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Global contrast based salient region detection....|
|||...Image signature: Highlighting sparse salient regions....|
|||...Submodular salient region  detection....|
|||...The secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Comparing salient object detection  results without ground truth....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||26 instances in total. (in cvpr2015)|
|47|Bruce_A_Deeper_Look_CVPR_2016_paper|...Recent efforts have explored the relationship between human gaze and salient objects, and we also examine this point further in the context of FCNs....|
|||... several distinct problems are treated as problems of modeling saliency, including gaze prediction, salient object segmentation, and objectness measures [2] respectively....|
|||...While the principal focus of this work is on gaze prediction, we also consider  the problem of salient object segmentation and further explore the relationship between salient objects and human gaze patterns....|
|||... we present a model deemed Fully Convolutional Saliency (FUCOS), that is applied to either gaze, or salient object  e r o c S C U A     ) s e g a m  i   l l     a   s s o r c a d e g a r e v a (  0.74...|
|||...For some experiments involving salient object segmentation, we also consider an alternative model based on deconvolutional neural networks [26] for comparison....|
|||...Note that versions specific to salient object segmentation (and not gaze) are introduced later on in section 4....|
|||...The evaluation is largely focused on modeling gaze prediction, however we also consider salient object segmentation....|
|||...Given all of these considerations, benchmarking for both fixation data and salient object segmentation is based on the methods described by Li et al....|
|||...Model performance for salient object segmentation is based on Precision-Recall (PR) analysis....|
|||...Note for the pascal dataset, two algorithms trained for salient object prediction are also included....|
|||...This provides an effective means of comparing the predictions of most salient regions across different algorithms....|
|||...Predicted salient object regions from the Pascal dataset....|
|||...Output corresponds to algorithms intended for salient object segmentation (Left to Right): GC, SF, PCAS, MCG+GBVS, SALFCN, SAL-DC  learning solutions as a function of the nature of the different datasets....|
|||...Salient Object Segmentation  As discussed in the introduction, the notion of saliency has become fragmented, and encompasses both gaze prediction and salient object segmentation, among other tasks....|
|||...As such, we have also considered the efficacy of fully convolutional network models to predict salient object regions (as defined by ground truth object masks)....|
|||...its structure may allow more precise recovery of spatial information corresponding to boundaries of salient objects....|
|||...It is interesting to note the efficacy of these models relative to existing solutions for the problem of salient object segmentation....|
|||...lower recall values, perhaps reflecting a more precise adherence to spatial boundaries of segmented salient objects....|
|||...Precision-Recall curves for salient object detection corresponding to a variety of algorithms (See Legend)....|
|||...patterns directly in the context of this paper, in examining the performance of models intended for salient object segmentation in their ability to predict fixation patterns....|
|||...Discussion  In this paper, we present a deep learning model for saliency prediction, and demonstrate the effectiveness of this approach for both gaze prediction and salient object segmentation....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Image signature: Highlighting sparse salient regions....|
|||...The secrets of salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||26 instances in total. (in cvpr2016)|
|48|Li_Deep_Contrast_Learning_CVPR_2016_paper|...Deep Contrast Learning for Salient Object Detection  Guanbin Li  Yizhou Yu  Department of Computer Science, The University of Hong Kong  {gbli, yzyu}@cs.hku.hk  Abstract  Salient object detection has ...|
|||...Resulting saliency maps are typically blurry, especially near the boundary of salient objects....|
|||...Though early work primarily focused on predicting eye-fixations in images, research has shown that salient object detection, which emphasizes object-level integrity of saliency prediction results, is ...|
|||...Despite recent progress, salient object detection remains a challenging problem that calls for more accurate solutions....|
|||...For example, local contrast features may fail to detect homogenous regions inside salient objects while global contrast suffers from complex background....|
|||...This model can not only infer semantic properties of salient objects, but also capture visual contrast among multi-scale feature maps....|
|||...However, directly applying existing fully convolutional network architecture to salient object detection would not be most appropriate because a standard fully convolutional model is not particularly ...|
|||...1, the architecture of our deep contrast network for salient object detection consists of two complementary components, a fully convolutional stream  479    CONV1_1+RELU   CONV1_2+RELU  POOLING_1    ...|
|||...First, the network should be deep enough to produce multi-level features for detecting salient objects at different scales....|
|||...ion of all network parameters in MS-FCN and the fusion layer, i is a weight balancing the number of salient pixels and unsalient ones, and  I ,  I  and  I + denote the total number of pixels, unsalien...|
|||...3, the saliency maps generated from the proposed method without CRF are fairly coarse and the contours of salient objects may not be well preserved....|
|||...The proposed saliency refinement model can not only generate smoother results with pixelwise accuracy but also well preserve salient object contours....|
|||...Most of the images has a single salient object....|
|||...Dut-OMRON contains 5,168 challenging images, each of which has one or more salient objects and relatively complex backgrounds....|
|||...HKU-IS is another large dataset containing 4447 challenging images, most of which have either low contrast or multiple salient objects....|
|||...t network using the MSRA-B dataset, it only takes 1.5 seconds for the trained model (DCL) to detect salient objects in a testing image with 400x300 pixels on a PC with an NVIDIA Titan Black GPU and a ...|
|||...We can also see that our model without CRF (DCL) significantly outperforms all evaluated salient object detection algorithms across all the considered datasets....|
|||...Conclusions  In this paper, we have introduced an end-to-end deep contrast network for salient object detection....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Learning optimal seeds for diffusion-based salient object detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
||25 instances in total. (in cvpr2016)|
|49|cvpr18-Deep Unsupervised Saliency Detection  A Multiple Noisy Labeling Perspective|...Most of the images in MSRA-B dataset only have one salient object....|
|||...The SOD saliency dataset [14] contains 300 images, where many images contain multiple salient objects with low contrast....|
|||...The SED2 [1] dataset contains 100 images with each image contains two salient objects....|
|||...n  Recall  2P recision + Recall  ,  (9)  where 2 = 0.3, P recision corresponds to the percentage of salient pixels being correctly detected, Recall is the fraction of detected salient pixels in relati...|
|||...Small salient object detection is quite challenging which increase the difficulty of this dataset....|
|||...Background of the third image is very complex, and all the competing methods fail to detect salient object....|
|||...The fourth image is in very low-contrast, where most of the competing methods failed to capture the whole salient objects with the last penguin mis-detected, especially for those unsupervised saliency methods....|
|||...The salient objects in the last row are quite small, and the competing methods failed to  9035  Figure 4....|
|||...capture salient regions, while our method capture the whole salient region with high precision....|
|||...e, we plan to investigate the challenging scenarios of multiple saliency object detection and small salient object detection under our  r o r r  E e     t  l  u o s b A n a e M     0.2  0.15  0.1  0.0...|
|||...Global contrast based salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Deeply supervised salient object detection with short connections....|
|||...Deep contrast learning for salient object In Proc....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Non-local deep features for salient object detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
|||...A stagewise refinement model for detecting salient objects in images....|
|||...Supervision by fusion: Towards unsupervised learning of deep salient object detector....|
|||...Attention to the scale: Deep multi-scale salient object detection....|
|||...Deep salient object detection by integrating multi-level cues....|
|||...Integrated deep and shallow networks for salient object detection....|
|||...Harf: Hierarchy-associated rich features for salient object detection....|
||25 instances in total. (in cvpr2018)|
|50|Jia_Category-Independent_Object-Level_Saliency_2013_ICCV_paper|...Abstract  It is known that purely low-level saliency cues such as frequency does not lead to a good salient object detection result, requiring high-level knowledge to be adopted for successful discove...|
|||...ect candidates without the need of category information, and then enforce the consistency among the salient regions using a Gaussian MRF with the weights scaled by diverse density that emphasizes the ...|
|||...Other than cognitively understanding the way human perceive images and scenes, finding salient regions and objects in the images helps various tasks such as speeding up object detection [27, 23] and c...|
|||...his bears much importance in understanding human visual systems, we focus on the problem of finding salient objects, aiming to find consistent foreground objects, which is often of interest in many fu...|
|||...h as common object detectors [15, 28] and global image statistics [6] aid the selection of the most salient regions in  1550-5499/13 $31.00  2013 IEEE 1550-5499/13 $31.00  2013 IEEE DOI 10.1109/ICCV.2...|
|||...lly-connected Markov random field (MRF) that takes into consideration the overall agreement between salient regions over the whole image, with an explicit emphasis on nodes that are more likely to be ...|
|||... we will show that our method returns a more objectcentric saliency map that is consistent over the salient object, achieving state-of-the-art performance on the bench mark MSRA and Weizmann datasets...|
|||...To obtain a consistent salient object detection, an important structural choice is to use a fully-connected graph, rather than a locally connected graph as many previous approaches do [5], as locally ...|
|||...Object Detection  Our method starts with finding an informative prior that captures the potential salient regions from images....|
|||...It could be observed that although the saliency map is still coarse, it provides a reasonable initialization for the final saliency map as it correctly identifies the salient object location....|
|||...More importantly, such saliency prior is not biased towards specific low-level appearances such as highfrequency regions, which often misses the inside region of the salient objects....|
|||...Saliency Computation with Graph-based  Foreground Agreement The pixel level prior gives us a reasonably informative result on the salient regions of the images....|
|||...a pixel i is computed as  (cid:3)  (cid:7) Wijsj + (1  Wij)(1  sj)  (cid:8)  (3)  agreement between salient regions in the image, based on the similarities between pixel level features....|
|||...The idea is that if a pixel has a high salient prior, then pixels that appear similar in the image should also receive high salient scores even if it lies in a region with low contrast, thus ensuring ...|
|||...ents  We evaluated our method on the MSRA saliency dataset containing 1000 images together with the salient object annotated by human participants as the ground-truth saliency map, and compared the pe...|
|||...Conclusion  In this paper we proposed a novel image saliency algorithm that utilizes the object-level information to obtain better discovery of salient objects....|
|||...ect candidates without the need of category information, and then enforce the consistency among the salient regions using a Gaussian MRF with the weights scaled by diverse density, which emphasizes th...|
|||...Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Image signature: Highlighting sparse salient regions....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
||25 instances in total. (in iccv2013)|
|51|Li_Visual_Saliency_Based_2015_CVPR_paper|...This dataset has become less challenging over the years because images there typically include a single salient object located away from the image boundary....|
|||...f advanced saliency models, we have created a large dataset where an image likely contains multiple salient objects, which have a more general spatial distribution in the image....|
|||...While these empirical priors can improve saliency results for many images, they can fail when a salient object is off-center or significantly overlaps with the image boundary....|
|||...This paper proposes a simple but very effective neural network architecture to make deep CNN features applicable to saliency modeling and salient object detection....|
|||... any information around the considered image region, thus is not able to tell whether the region is salient or not with respect to its neighborhood as well as the rest of the image....|
|||...As a result, we obtain M refined saliency maps,  {A(1), A(2), ..., A(M )}, interpreting salient parts of the input image at various granularity....|
|||...ere chosen by following at least one of the following criteria:  1. there are multiple disconnected salient objects; 2. at least one of the salient objects touches the image  boundary;  3. the color c...|
|||...To reduce label inconsistency, we asked three people to annotate salient objects in all 7320 images individually using a custom designed interactive segmentation tool....|
|||...We define label consistency as the ratio between the number of pixels labeled as salient by all three people and the number of pixels labeled as salient by at least one of the people....|
|||...In summary, 50.34% images in HKU-IS have multiple disconnected salient objects while this number is only 6.24% for the MSRA dataset; 21% images in HKU-IS have salient objects touching the image bounda...|
|||...Most of the images contain only one salient object....|
|||...SED1 has 100 images each containing only one salient object while SED2 has 100 images each containing two salient objects....|
|||...This dataset is very challenging since many images contain multiple salient objects either with low contrast or overlapping with the image boundary....|
|||...Each image may contain one or multiple salient objects....|
|||...Our new dataset contains 4447 images with pixelwise annotation of salient objects....|
|||...on model for 15 image segmentation levels using the MSRA dataset, it only takes 8 seconds to detect salient objects in a testing image with 400x300 pixels on a PC with an NVIDIA GTX Titan Black GPU an...|
|||...an be seen, our method performs well in a variety of challenging cases, e.g., multiple disconnected salient objects (the first two rows), objects touching the image boundary (the sec ond row), clutter...|
|||...Thus it is more likely for them to miss salient pixels....|
|||...Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object  detection via low rank matrix recovery....|
||25 instances in total. (in cvpr2015)|
|52|Mauthner_Encoding_Based_Saliency_2015_CVPR_paper|...Recent research has emphasized the need for analyzing salient information in videos to minimize dataset bias or to supervise weakly labeled training of activity detectors....|
|||...In contrast to previous methods we do not rely on training information given by either eye-gaze or annotation data, but propose a fully unsupervised algorithm to find salient regions within videos....|
|||...We evaluate our approach on several datasets, including challenging scenarios with cluttered background and camera motion, as well as salient object detection in images....|
|||...Therefore, the region containing the horse is the eponymous region, and in general the horse should be at least part of the most salient region....|
|||...tain expressive saliency maps, above mentioned human preferences may even be misleading for general salient object detection tasks....|
|||...These considerations lead us to the goal of this work: finding eponymous and therefore salient video or image regions....|
|||...In contrast to using fixation maps as ground-truth, [16] proposed a large dataset with bounding-box annotations of salient objects....|
|||...By labeling 1000 images of this dataset, [1] refined the salient object detection task, see [3] for a review....|
|||...In contrast to salient object detection, video saliency or finding salient objects in videos is a rather unexplored field....|
|||...A Bayesian Saliency Formulation  Following the Gestalt principle for figure-ground segregation, we are searching for surrounded regions as they are more likely to be perceived as salient areas [20]....|
|||...To distinguish salient foreground pixels x  F from surrounding background pixels, we employ a histogram based Bayes classifier on the input image I....|
|||...Second, following our original motivation by Gestalt theory, the foreground map for scale i should contain highlighted areas for salient regions of size i or smaller....|
|||...Besides these locally computed foreground maps, global estimation of salient parts also offers valuable information....|
|||...Salient Object Detection  One of the most similar tasks to localizing activities in videos is salient object detection in still images....|
|||...A comparison with the state-of-the-art in salient object segmentation is shown in Figure 4....|
|||...Therefore, we evaluate the impact of the latter object-center prior by cropping images of the ASD dataset such that salient objects are located near the borders....|
|||...We compare our EBSG against the top performing BMS [32] using two cropping levels: First, salient objects touch the closest image border and second, intersect the closest border by 5 pixel....|
|||...(a)  (b)  (c)  (d)  Figure 4: (a) comparison of EBSG and EBSGR (b) to stateof-the-art in salient object detection on ASD dataset....|
|||...Implicitly enforcing figure-ground segregation on individual scales allows us to preserve salient regions of various sizes....|
|||...lent results on challenging video sequences with cluttered background and camera motion, as well as salient object detection in images....|
|||...Frequency-tuned Salient Region Detection....|
|||...Global contrast based salient region detection....|
|||...The secrets  of Salient Object Segmentation....|
|||...Learning to Detect A Salient Object....|
||24 instances in total. (in cvpr2015)|
|53|Tu_Real-Time_Salient_Object_CVPR_2016_paper|...ogy, University of Science and Technology of China  Abstract  In this paper, we present a real-time salient object detection system based on the minimum spanning tree....|
|||...Due to the fact that background regions are typically connected to the image boundaries, salient objects can be extracted by computing the distances to the boundaries....|
|||...We further introduce a boundary dissimilarity measure to compliment the shortage of distance transform for salient object detection....|
|||...Introduction  The goal of salient object detection is to identify the most important objects in a scene....|
|||...[26] have been shown to be effective in bottomup salient object detection....|
|||...[34] use the minimum barrier distance (MBD) [24, 6] for salient object detection....|
|||...The proposed salient object detection method achieves superior results to the state-of-the-art methods....|
|||...Distance Transform with a Minimum Span ning Tree  Previously, the geodesic distance (GD) and the barrier distance (BD) are used for salient object detection....|
|||...rties  f (u  v) = f (v  u)  f (u  v) >= 0  (3)  (4)  In [26, 35], the geodesic distance is used for salient object detection....|
|||...(6)  The barrier distance is shown to be more robust than geodesic distance for salient object detection [34]....|
|||...We describe our salient object detection system in this section....|
|||...The overall salient object detection system is summarized in Figure 3....|
|||...Measuring the Boundary Connectivity  We set all pixels along the image boundary as a set of seed nodes to exploit the background and connectivity priors for salient object detection....|
|||...transform is more favoured in salient object detection and we adopt the BD transform in our final experimental results....|
|||...n not produce satisfying results when the background has many cluttered and isolated regions or the salient object touches the image boundary....|
|||...We apply the proposed distance transform to measure boundary connectivity for salient object detection....|
|||...The proposed salient object detection is evaluated and compared with state-of-the-art algorithms and achieves comparable or better results in numerical evaluation....|
|||...Minimum barrier salient object detection at 80 fps....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...The  secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||24 instances in total. (in cvpr2016)|
|54|Wang_A_Stagewise_Refinement_ICCV_2017_paper|...s (CNNs) have been successfully applied to a wide variety of problems in computer vision, including salient object detection....|
|||...To detect and segment salient objects accurately, it is necessary to extract and combine high-level semantic features with lowlevel fine details simultaneously....|
|||...This is beneficial for the classification task which does not need spatial information, but presents challenges for densely segmenting salient objects....|
|||...The master net helps locate salient objects, while the refinement  43214019  Figure 1....|
|||...inement nets help renovate sharp and detailed boundaries in coarse saliency maps for highresolution salient object segmentation....|
|||...The most  widely used feature is contrast, which is based on the fundamental hypothesis that salient objects are usually in high contrast with the background....|
|||...To address the above-mentioned issues, fully convolutional networks (FCN) haven been trained end-to-end for densely segmenting salient objects....|
|||...Firstly, a fully connected layer after the conv5 3 layer in [33] results in losing spatial information of salient objects....|
|||...To further distinguish salient objects from the background, we employ a pyramid pooling module (PPM) for gathering global context information....|
|||...It can be seen that the saliency maps generated from the proposed method with PPM can preserve salient object boundaries and suppress background noise....|
|||...The notation lg  {0, 1} indicates the foreground or background label of the pixel at location (i, j) and Pr(li,j = lg ) represents its corresponding probability of being salient or not....|
|||...It can be seen that our method is capable of uniformly highlighting the inner part of salient objects as well as suppressing the background clutter....|
|||...This is because the pyramid pooling module can aggregate global context information which is important for distinguishing salient objects from the background in a global view....|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...GlobIEEE TPAMI,  al contrast based salient region detection....|
|||...Random walks on graphs for salient object detection in images....|
|||...Deep contrast learning for salient object  detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
|||...Learning to detect salient objects with image-level supervision....|
||24 instances in total. (in iccv2017)|
|55|Du_A_Graphical_Model_2015_CVPR_paper|...signatures, supervised latent Dirichlet allocation is used to learn the latent distributions of the salient regions over the visual vocabulary and hierarchical Dirichlet processes are implemented to i...|
|||...In our formulation, a signature corresponds to a document, a salient region corresponds to a topic, an observation corresponds to a word, and an authorship corresponds to a label....|
|||...A salient region is a distribution over the features in the visual vocabulary, which groups similar co-occurring observations....|
|||...Each author is modeled as a combination of all salient regions with different proportions....|
|||...For a query signature, classification is performed by computing the salient region proportions for the signature based on observations....|
|||...Further, instead of guessing the number of salient regions empirically, HDP is used to estimate the number needed for the given dataset....|
|||...Section 3 provides a detailed description of how to build the supervised topic models and how HDP is used to estimate the number of salient regions....|
|||... = [1, 2, ..., R], where each r is the distribution of the salient region r over the vocabulary, and  St is the mean of the salient regions of the tth signature....|
|||...N(cid:89)  q(, S , ) = q( )  q(Sn n)  (4)  n=1  Here  is a R-dimensional variational Dirichlet hyperparameter that governs the salient region distribution of each signature....|
|||...In the M-step, each of the salient regions r is estimated by counting how many times each observation is assigned to this salient region among all signatures....|
|||... and 2 are estimated by the relationships between salient regions and the labels for all training signatures....|
|||...HDP for salient region estimation  (cid:124) estest  In LDA and sLDA, the number of salient regions needs to be prefixed and it is always chosen empirically....|
|||...For the tth signature, draw salient region proportions  t from Dir()  2....|
|||...For the nth observation:  (a) Draw a salient region assignment St,n from  M ult(t)  (b) Draw an observation Ot,n from M ult(St,n )  processing new and massive data, it is not possible to easily choos...|
|||...We give a brief introduction as follows: Let t,n be the salient region for the nth observation in the tth signature, t,rs be the existing salient regions for signature t, Nt,r be the number of observa...|
|||...If a new salient region is needed for this observation, we draw one salient region t,r from G0 and increase Mt by one as follows:  t,r 1,1, ..., t,r1, , H   Mr0 M +   r0  +    M + 0  H  (9)  R(cid:88)...|
|||...If the first term of the right-hand side of Equation (9) is chosen, the new salient region for t,n is picked among the existing salient regions  with a probability proportional to the number of times ...|
|||...If the second term is chosen, a new salient region is introduced and the total number of salient regions is increased by one....|
|||...First we run HDP on the DS-I parital dataset to estimate the number of salient regions....|
|||...230 salient regions are estimated from HDP after 1000 iterations, and empirically K = 1500 is chosen....|
|||...Effects of parameters  The two parameters that have impact on the performance are the number of centers K in K-means algorithm and the  number of salient regions R in sLDA....|
|||...Number of salient regions: The choice of R depends on the size of vocabulary and the observations....|
|||...ild the observations and vocabulary, sLDA is used to model each author as proportions of the hidden salient regions, and HDP is used to indicate the proper number of salient regions needed for each da...|
||23 instances in total. (in cvpr2015)|
|56|Margolin_What_Makes_a_2013_CVPR_paper|...Introduction  The detection of the most salient region of an image has numerous applications, including object detection and recognition [13], image compression [10], video summarization [16], and pho...|
|||...As illustrated in Figure 1(d), this could lead to missing homogeneous regions of the salient object....|
|||...A pixel is deemed salient if the pattern of its surrounding patch cannot be explained well by other image patches....|
|||...In this figure the patch px (marked in red) should be considered as salient in image Im2 and non-salient in image Im1....|
|||...Hence, the patch px will be considered more salient in image Im2 than in image Im1....|
|||...In this image, the drawings on the wall are salient because they contain unique patterns, compared to the buildings facade....|
|||...Putting it all together  We seek regions that are salient in both color and pattern....|
|||...SED1 [3]: 100 images of a single salient object anno tated manually by three users....|
|||...SED2 [3]: 100 images of two salient objects annotated  manually by three users....|
|||...SOD [17]: 300 images taken from the Berkeley Segmentation Dataset for which seven users selected the boundaries of the salient objects....|
|||...According to [4], the Top-4 highest scoring salient object detection algorithms are: SVO [5], CR [6], CNTX [9], and CBS [11]....|
|||...This dataset is aimed at gaze-prediction, which differs from our task of salient object detection....|
|||...Frequencytuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Learning to detect a salient object....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...It can be seen that while SVO [5] detects the salient regions, parts of the background are erroneously detected as salient....|
|||...The CBS method [11] relies on shape priors and therefore often detects only parts of the salient objects (e.g., the flower) or convex background regions (e.g., the water of the harbor)....|
|||...Our method integrates color and pattern distinctness, and hence captures both the outline, as well as the inner pixels of the salient objects....|
|||...We do not make any assumptions on the shape of the salient regions, hence, we can handle convex as well as concave shapes....|
|||...In this paper we have shown that the statistics of patches in the image plays a central role in identifying the salient patches....|
||23 instances in total. (in cvpr2013)|
|57|Zhang_Saliency_Detection_A_2013_ICCV_paper|...Furthermore, BMS is also shown to be advantageous in salient object detection....|
|||...xation heat map; (c) and (d) are the saliency maps generated by BMS for eye fixation prediction and salient object detection respectively....|
|||...the mean attention map, is a full-resolution preliminary saliency map that can be further processed for a specific task such as eye fixation prediction or salient object detection [5]....|
|||...2 shows two types of saliency maps of BMS for eye fixation prediction and salient object detection....|
|||...We also show with both qualitative and quantitative results that the outputs of BMS are useful in salient object detection....|
|||...majority of the previous saliency models use centersurround filters or image statistics to identify salient patches that are complex (local complexity/contrast) or rare in their appearance (rarity/imp...|
|||...to a local gradient operator plus Gaussian blurring on natural images, and thus cannot detect large salient regions very well....|
|||...The salient region detection method of [16] also employs a feature channel thresholding step....|
|||...Boolean maps should be generated in such a way that more salient regions have higher chances to be separated from the surrounding background....|
|||...Six categories: 50/80/60 with large/medium/small salient regions; 15 with clustering background; 15 with repeating distractors; 15 with both large and small salient regions....|
|||...The input images are roughly arranged in ascending order of the size of their salient regions....|
|||...Moreover, they tend to favor the boundaries rather than the interior regions of large salient objects, like the car and the STOP sign in the last two examples, even with the help of multi-scale proces...|
|||...Images are roughly arranged in ascending order of the size of their salient regions....|
|||...Salient object detection aims at segmenting salient objects from the background....|
|||...Models for salient object detection have different emphasis compared with models for eye fixation prediction....|
|||...However, salient object detection requires object level segmentation, which means the corresponding saliency map should be highresolution with uniformly highlighted salient regions and 3Note that some...|
|||...The ImgSal dataset [27] used in the previous section also has ground-truth salient regions labeled by 19 subjects....|
|||...The labeled salient regions of this dataset are not very precise, and thus unsuitable for quantitative evaluation using the PR metric....|
|||...BMS is the only model that consistently achieves state-of-the-art performance on five benchmark eye tracking datasets, and it is also shown to be useful in salient object detection....|
|||...This may help to redeem the limitation that salient regions that touch the image borders cannot be well detected using the surroundedness cue alone....|
|||...Frequency-tuned salient region detection....|
|||...Global  contrast based salient region detection....|
|||...Image signature: Highlight ing sparse salient regions....|
||23 instances in total. (in iccv2013)|
|58|Frintrop_Traditional_Saliency_Reloaded_2015_CVPR_paper|...[21] is still competitive with current state-of-the-art methods for salient object segmentation if some important adaptions are made....|
|||...Furthermore, we integrate the saliency system into an object proposal generation framework to obtain segment-based saliency maps and boost the results for salient object segmentation....|
|||...In this paper, we show that the traditional structure of saliency models based on multi-scale Difference-ofGaussians is still competitive with current salient object detection methods....|
|||...Since the original model was designed to simulate eye movements, we need some adaptations to achieve high performance for salient object segmentation....|
|||...od: a pixel that differs strongly from its neighbors, e.g., at an object border, is considered more salient than one surrounded by similar pixels....|
|||...at the traditional, biologicallyinspired concept is still valid and obtains competitive results for salient object segmentation if adapted accordingly....|
|||...However, on benchmarks for salient object detection and segmentation these models usually perform less well and it has been claimed that other methods are required for such tasks.1  During the last de...|
|||...Since salient items are salient because of their difference to the surrounding, computing such a contrast is an essential step....|
|||...We show in this paper how a FIT-based approach can be implemented to achieve state-of-the-art performance for salient object segmentation and that the resulting system is thus suitable for many applications....|
|||...[21] and outline in detail which adaptions have been made to obtain state-of-the-art performance for salient object segmentation....|
|||...We implemented also an orientation channel based on Gabor filters, but it turned out that it is less useful for salient object segmentation....|
|||...This is especially important when simulating eye movements and prioritizing data processing, but less so when performing salient object segmentation....|
|||...Segmentation and saliency are combined by selecting for each salient blob the segments which overlap more than k percent with the blob (we use k = 30%)....|
|||...This set of segments per blob constitutes one object proposal; the number of proposals per frame is thus the same as the number of salient blobs, but the proposals have more precise boundaries than the blobs....|
|||...This measure is the most common comparison measure for salient object segmentation....|
|||...re integration theory is still valid, and that it can be used equally well as other methods to find salient objects in images....|
|||...In CVPR,  Frequency-tuned salient region detection....|
|||...Image signature: High lighting sparse salient regions....|
|||...Global contrast based salient region detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||23 instances in total. (in cvpr2015)|
|59|Liu_Adaptive_Partial_Differential_2014_CVPR_paper|...We assume that the saliency of image elements can be carried out from the relevances to the saliency seeds (i.e., the most representative salient elements)....|
|||...Location is another important prior for modeling salient regions....|
|||...Inspired by recent advances in machine learning, compressive sensing [34, 22] and operations research [15] are also utilized to detect salient image features....|
|||...The work in [34, 22] assumes that a natural image can always be decomposed into a distinctive salient foreground and a homogenous background....|
|||...[15] formulate saliency detection as a semi-supervised clustering problem and use the well-studied facility location model to extract cluster centers for salient regions....|
|||...[26] propose a supervised approach to learn to detect a salient region in an image....|
|||...The bottom row shows the ground truth (GT for short) salient region and saliency maps computed by some state-of-the-art saliency detection methods....|
|||...ixed formulation and boundary conditions to describe all types of saliency due to the complexity of salient regions in real world images....|
|||...Similarly, let v be a vector field on V and denote vp as the vector at p.  Input ImageSuperpixel  SetmentationCenter PriorColor PriorGuidance MapSaliency Score MapMasked Salient RegionBackground Prior...|
|||...That is, we assume that our attention is firstly attracted by the most representative salient image elements (this paper names them as saliency seeds) and then the visual attention will be propagated ...|
|||...(a) input image and GT salient region....|
|||...(a) input image and GT salient region....|
|||...We also present ground truth (GT) salient regions and the saliency maps for compared methods....|
|||...The center-surround contrast based methods, such as IT [13], GB [10] and CA [9], can only detect parts of boundaries of salient objects....|
|||...However, the geodesic distance to boundary strategy in that method tends to recognize background parts as salient regions when their colors are significantly different from the boundary....|
|||...In SM [15], regions inside a salient object which share a similar color with the background will be regarded as part of the background....|
|||...Fusing generic objectness  and visual saliency for salient object detection....|
|||...Global  contrast based salient region detection....|
|||...Submodular salient region detection....|
|||...Learning  to detect a salient object....|
|||...Design and perceptual validation of performance  measures for salient object segmentation....|
|||...Saliency filters: Contrast  based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank  matrix recovery....|
||23 instances in total. (in cvpr2014)|
|60|Fan_Structure-Measure_A_New_ICCV_2017_paper|...s crucial for gauging the progress of object segmentation algorithms, in particular in the field of salient object detection where the purpose is to accurately detect and segment the most salient obje...|
|||...As a specific example, here we focus on salient object detection models [4, 6, 7, 16], although the proposed measure is general and can be used for other purposes....|
|||...It is necessary to point out that the salient object is not necessary to be foreground object [18]....|
|||...Almost all salient objection detection methods output non-binary maps....|
|||...In the field of salient object detection,  researchers are concerned more about the foreground object structures....|
|||...For high-level vision tasks such as salient object detection, the evaluation of the object-level similarity is crucial....|
|||...So, it is important to assign a higher value to a SM with salient object being uniformly detected (i.e., similar saliency values across the entire object)....|
|||...Most of the images in this dataset contain more than one salient object with low contrast....|
|||...Saliency model comparison  Establishing that our Structure-measure offers a better way to evaluate salient object detection models, here we compare 10 state-of-the-art saliency models on 4 datasets (P...|
|||...In IEEE CVPR,  Frequency-tuned salient region detection....|
|||...What is a salient object?...|
|||...a dataset and a baseline model for salient object detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Local background enclosure for rgb-d salient object detection....|
|||...Deeply supervised salient object detection with short connections....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Deep contrast learning for salient object  detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliencyrank: Two-stage manifold ranking for salient object detection....|
||23 instances in total. (in iccv2017)|
|61|Kuang-Jui_Hsu_Unsupervised_CNN-based_co-saliency_ECCV_2018_paper|...1  Introduction  Co-saliency detection refers to searching for visually salient objects repetitively appearing in multiple given images....|
|||...elated work  2.1 Single-image saliency detection  Single-image saliency detection is to distinguish salient objects from the background by either unsupervised [20,21,22,23,24,25] or supervised [26,27,...|
|||...These approaches can handle well images with single salient objects....|
|||...However, they may fail when the scenes are more complex, for example when multiple salient objects are presented with intra-image variations....|
|||...2.2 Co-saliency detection  Co-saliency detection discovers common and salient objects across multiple images using different strategies....|
|||...The former detects the salient regions in a single image, without considering whether the detected regions are commonly present  UCCDGO  5  (cid:1835)(cid:3041)      (cid:1859)(cid:3046) (cid:1859)(c...|
|||...Cosaliency detection, finding the salient co-occurrence regions, can then be carried out by performing and integrating the two tasks on a graph whose two types of edges respectively correspond to the ...|
|||...The resultant co-saliency map, highlighting the co-occurrence and salient regions, is yielded by Sn = gs(In)  gc(In) = Ss n, where  denotes the element-wise multiplication operator....|
|||...3.2 Unary term s  This term aims to identify the salient regions in a single image....|
|||...nts the importance saliency values of maps Ss of pixel i. Pixels in map  Sn can be divided into the salient and non-salient groups by using the mean value of  Sn as the threshold....|
|||...Rn(i) takes the value 1   if pixel i belongs to the salient group, and  otherwise....|
|||...In this way, the salient and non-salient groups contribute equally in Eq....|
|||...Pixels in a superpixel tend to belong to either a salient object or  k=1, where qk  n}K  8  Hsu et al....|
|||...Superpixels in On and Bn are confident to be assigned to either the salient regions or the background....|
|||...ing to other images in the given image set, single-image saliency methods could detect the visually salient objects that do not repetitively appear in other images, such as the orange and the apple in...|
|||...: Real-time salient object detection with a  minimum spanning tree....|
|||...Huang, F., Qi, J., Lu, H., Zhang, L., Ruan, X.: Salient object detection via multiple  instance learning....|
|||...Liu, N., Han, J.: DHSNet: Deep hierarchical saliency network for salient object  detection....|
|||...Zhang, P., Wang, D., Lu, H., Wang, H., Ruan, X.: Amulet: Aggregating multi-level  convolutional features for salient object detection....|
|||...Zhang, D., Han, J., Zhang, Y.: Supervision by fusion: Towards unsupervised learn ing of deep salient object detector....|
|||...Li, G., Yu, Y.: Deep contrast learning for salient object detection....|
|||...Borji, A., Cheng, M.M., Jiang, H., Li, J.: Salient object detection: A benchmark....|
||22 instances in total. (in eccv2018)|
|62|Non-Local Deep Features for Salient Object Detection|...Non-Local Deep Features for Salient Object Detection  Zhiming Luo1  2  ,  ,  3, Akshaya Mishra4, Andrew Achkar4, Justin Eichel4, Shaozi Li1  ,  2 , Pierre-Marc Jodoin3  2Fujian Key Laboratory of Brain...|
|||...Methods using conventional models struggle whenever salient objects are pictured on top of a cluttered background while deep neural nets suffer from excess complexity and slow evaluation speeds....|
|||...A salient object is often defined as a region whose visual features differ from the rest of the image and whose shape follows some a priori criteria [5]....|
|||...With CNNs, the saliency problem has been redefined as a labeling problem where feature selection between salient and non-salient objects is done automatically through gradient descent....|
|||...[43] developed a two tier strategy: each pixel is assigned a saliency based upon a local context estimation in parallel to a global search strategy used to identify the salient regions....|
|||...Architecture of our 4  5 grid-CNN network for salient object detection....|
|||...Details of the proposed deep convolutional network for predicting salient objects (S: Stride, Pad: zero padding)....|
|||...Capturing global context: Detecting salient objects in an image requires the model to capture the global context of the image before assigning saliency to individual small regions....|
|||...The softmax function is used to compute the probability for each pixel of being salient or not....|
|||...HKU-IS: contains 4447 images, most of which have low contrast and multiple salient objects....|
|||...DUT-OMRON: contains 5168 challenging images, each of which contains one or more salient objects with a relatively cluttered background....|
|||...The NLDF maps provides clear salient regions and exhibit good uniformity as compared to the saliency maps from the other deep learning methods (LEGS, MC, MDF and DCL)....|
|||...Many images contain multiple salient objects with low contrast and overlapping boundaries....|
|||...This substantial speedup enables nearly real-time salient object detection while also delivering state-of-the-art performance....|
|||...As shown in Figure 4, the saliency maps generated from NLDFare fairly coarse and the boundary of the salient objects are not well preserved....|
|||...Frequency tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...A simple method for detecting salient regions....|
|||...Saliency unified: A deep architecture for simultaneous eye fixation prediction and salient object segmentation....|
|||...Deep contrast learning for salient object detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Learning to detect a salient object....|
||22 instances in total. (in cvpr2017)|
|63|Jiang_Salient_Region_Detection_2013_ICCV_paper|...In this paper we propose a novel salient region detection algorithm by integrating three important visual cues namely uniqueness, focusness and objectness (UFO)....|
|||...cular, uniqueness captures the appearance-derived visual contrast; focusness reflects the fact that salient regions are often photographed in focus; and objectness helps keep completeness of detected ...|
|||...As such, how to simulate such human capability with a computer, i.e., how to identify the most salient pixels or regions in a digital image which attract humans first visual attention, has become an i...|
|||...Inspired by the above discussion, in this paper we propose integrating two additional cues, focusness and objectness to improve salient region detection....|
|||...Second, intuitively, a salient region usually completes objects instead of cutting them into pieces....|
|||...This suggests us to use object completeness as a cue to boost the salient region detection....|
|||...Combining focusness and objectness with uniqueness, we propose a new salient region detection algorithm, named UFO saliency, which naturally addresses the aforementioned issues in salient region detection....|
|||...or example, shape prior is proposed in [15], context information is exploited in [36], region-based salient object detection is introduced in [16], and manifold ranking approach is introduced for sali...|
|||...Problem Formulation and Method Overview  3, where   R  We now formally define the problem of salient region detection studied in this paper....|
|||...Therefore, it is desirable to estimate the probability of each region belonging to a well identifiable object in order to prioritize the regions in salient region detection....|
|||...Since our computations of F,O and Ur are all on the region level, they can work together to locate the overall structure of salient objects....|
|||...However, they may sometimes miss small local color details that appear salient to human vision as well....|
|||...Frequencytuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Submodular Salient Region Detection....|
|||...Learning to detect a salient object....|
|||...2, 6  [25] M. Movahedi and J. H. Elder Design and perceptual validation of In POCV,  performance measures for salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection  via low rank matrix recovery....|
|||...Automatic salient object  extraction with contextual cue....|
||21 instances in total. (in iccv2013)|
|64|Learning to Detect Salient Objects With Image-Level Supervision|...t.edu.cn  Abstract  Deep Neural Networks (DNNs) have substantially improved the state-of-the-art in salient object detection....|
|||...n this paper, we leverage the observation that imagelevel tags provide important cues of foreground salient objects, and develop a weakly supervised learning method for saliency detection using image-...|
|||...22], DNNs learned from full supervision are more effective in capturing foreground regions that are salient in the semantic meaning, yielding accurate results under complex scenes....|
|||...Image-level tags (left panel) provide informative cues of dominant objects, which tend to be the salient foreground....|
|||...On the other hand, image-level tags provide the category information of dominant objects in the images which are much likely to be the salient foreground....|
|||...It is thus natural to leverage image-level tags as weak supervision to train DNNs for salient object detection....|
|||...Since we focus on generic salient object detection, a new network, named Foreground Inference Net (FIN) is designed....|
|||..., these methods aim to segment objects of the training categories, whereas we aim to detect generic salient objects, which requires generalization to unseen categories at test time and is more challen...|
|||...In [16], binary image tags indicating the existence of salient objects are utilized to train SVMs....|
|||...To our knowledge, we are the first to leverage object category labels for learning salient object detectors....|
|||...For saliency detection, we do not pay special attentions to the object category, and only aim to discover salient object regions of all categories....|
|||...Frequency-tuned salient region detection....|
|||...Weakly supervised learning for salient object de tection....|
|||...Deep contrast learning for salient object  detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency net work for salient object detection....|
|||...Learning to detect a salient object....|
|||...Learning optimal seeds for diffusion-based salient object detection....|
|||...Minimum barrier salient object detection at 80 fps....|
|||...Harf: Hierarchy-associated rich  features for salient object detection....|
||21 instances in total. (in cvpr2017)|
|65|cvpr18-Depth-Aware Stereo Video Retargeting|...tereo video retargeting poses new challenges because stereo video contains the depth information of salient objects and its time dynamics....|
|||...The proposed depth-aware retargeting method reconstructs the 3D scene to obtain the depth information of salient objects....|
|||...We cast it as a constrained optimization problem, where the total cost function includes the shape, temporal and depth distortions of salient objects....|
|||...As a result, the solution can preserve the shape, temporal and depth fidelity of salient objects simultaneously....|
|||...As compared with 2D video retargeting, stereo video retargeting poses new challenges because stereo video contains the depth information of salient objects and its time dynamics....|
|||...However, the depth of a salient object can be distorted....|
|||...ved by traditional 2D video retargeting methods since they do not preserve the depth information of salient objects by analyzing the leftand right-views jointly (see Fig....|
|||...Once salient objects are detected, the algorithm will preserve their depth information as faithfully as possible....|
|||...For stereo videos (especially for those with salient objects or their moving occupying a large portion of a frame), Lin et al....|
|||...That is, shapes of salient 3D objects are preserved at each frame and objects shapes are coherently resized across frames....|
|||...ight-views separately, we seek for a solution that preserves the shape and the depth information of salient objects and their time dynamics in the original content as much as possible....|
|||...Since the human visual system (HVS) fixates a salient 3D object at a time when humans watch a stereo video, other objects are blurred and their other depth can be altered to some extent as long as the...|
|||...e HVS, we propose a depth-aware retargeting solution that not only preserves the shape and depth of salient 3D objects but also scales the whole 3D scene along time coherently....|
|||...on), temporal shape incoherence (i.e., temporal distortion) and loss of the 3D depth information of salient objects (i.e., depth distortion), respectively, S and D are weighting factors....|
|||...The depth distortion is used to preserve the depth information of salient objects in a stereo video....|
|||...Spatio(cid:173)Temporal Shape Distortions  To preserve the shape of salient objects, we define the shape distortion of stereo video as the total shape distortion of all grids....|
|||...The test videos contain various types of motions and salient objects with a large depth range and significant depth variations, imposing great challenges on stereo video retargeting....|
|||...Because the salient objects in the CVW dataset often occupy a large area (more than 50%) of a frame as shown in Fig....|
|||...A good stereo video retargeting method should preserve both the shape and depth attributes of salient 3D objects and ensure temporal coherence of shape and depth....|
|||...mulated and solved to offer an effective stereo video retargeting solution that preserves depths of salient regions, coherently transforms other non-salient regions and achieves spatio-temporal shape ...|
|||...Minimum barrier salient object detection at 80 fps....|
||21 instances in total. (in cvpr2018)|
|66|Li_Saliency_Detection_via_2013_ICCV_paper|...In addition, the proposed algorithm is demonstrated to be more effective in highlighting salient objects uniformly and robust to background noise....|
|||...In computer vision, more emphasis is paid to detect salient objects in images based on features with generative and discriminative algorithms....|
|||...While center-surround contrast-based measures are able to detect salient objects, existing bottom-up approaches are less effective in suppressing background pixels....|
|||...ons unless the object boundary is known), its contrast with the background is less distinct and the salient object is unlikely to be uniformly highlighted....|
|||...Therefore, these methods do not uniformly detect salient objects  1550-5499/13 $31.00  2013 IEEE DOI 10.1109/ICCV.2013.370  2976  2ULJLQDO,PDJH  ,PDJH6HJPHQWV  5HFRQVWUXFWLRQ  3URSDJDWHG  3L[HO/H...|
|||...We propose an algorithm to detect salient objects by dense and sparse reconstruction using the background templates for each individual image, which computes more effective bottom-up contrast-based saliency....|
|||...A context-based propagation mechanism is proposed for region-based saliency detection, which uniformly highlights the salient objects and smooths the region saliency....|
|||...While these assumptions may not always hold, they nev ertheless provide useful visual information which can be utilized to detect salient objects....|
|||...For cluttered scenes, dense appearance models may be less effective in measuring salient objects via reconstruction errors....|
|||...Since salient objects do not always appear at the image center as Figure 5 shows, the center-biased Gaussian model is not effective and may include background pixels or miss the foreground regions....|
|||...It should be noted that Bayesian integration enforces these two maps to serve as the prior and cooperate with each other in an effective manner, which uniformly highlights salient objects in an image....|
|||...Most images in the MSRA and ASD databases have only one salient object and there are usually strong contrast between objects and backgrounds....|
|||...Figure 7(a) shows that the sparse reconstruction error based on background templates achieves better accuracy in detecting salient objects than RC11 [7], while the dense one is comparable with it....|
|||...Frequency tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Learning to  detect a salient object....|
|||...Design and perceptual validation of In POCV,  performance measures for salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection  via low rank matrix recovery....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
||21 instances in total. (in iccv2013)|
|67|Lu_Learning_Optimal_Seeds_2014_CVPR_paper|...Learning optimal seeds for diffusion-based salient object detection  Song Lu  SVCL Lab, UCSD  Vijay Mahadevan  Yahoo Labs  Nuno Vasconcelos SVCL Lab, UCSD  sol050@ucsd.edu  vmahadev@yahoo-inc.com  nun...|
|||...The propagation of the resulting saliency seeds, using a diffusion process, is finally shown to outperform the state of the art on a number of salient object detection datasets....|
|||... this is based on the eye fixation dataset of [9], from which we randomly selected 10 of the top 2% salient patches (according to the ground truth) as positive examples and 10 out of the bottom 40% as...|
|||...SED1 [4] contains 100 images, each with a single salient object....|
|||...SED2 [4] contains 100 images with two salient objects each....|
|||...Both SED1 and SED2 include pixel-wise salient object segmentation groundtruth....|
|||...SOD [33] contains 300 images from the Berkeley segmentation dataset with the salient object boundary annotated....|
|||...Finally, VOC2008 1023 is a subset of the VOC2008 dataset, where each image contains  one or more salient objects plus segmented groundtruth....|
|||...Note how learning optimal salient seeds can improve saliency performance....|
|||...Conclusion  In this work, we presented an approach for salient object detection....|
|||...The saliency of salient seed locations is propagated through the graph via a diffusion process....|
|||...Unlike previous heuristic approaches to seed selection, an optimal set of salient seeds is learned using a large margin formulation of the discriminant saliency principle....|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Random walks on graphs for salient object detection in images....|
|||...Image signature: Highlighting sparse salient regions....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||20 instances in total. (in cvpr2014)|
|68|Wang_GraB_Visual_Saliency_CVPR_2016_paper|...der the same class share some consistency, the salient objects from two images are often found vastly different in terms of visual appearance, especially when the object can be anything....|
|||...Supervised vs. Unsupervised Unsupervised methods [10, 28, 32, 36, 19] aim at separating salient objects by extracting cues from the input image only....|
|||...A good feature descriptor should exhibit high contrast between salient and non-salient regions....|
|||...Compared to the center prior [17, 24] which assumes that the salient object always stays at the center of an image, the boundary prior is more robust, which is validated on several public datasets [32]....|
|||...Without any prior knowledge of size of the salient object, we adopt the L-layer Gaussian pyramid for robustness....|
|||...The cost function is designed to assign 1 to salient region value and 0 to background region....|
|||...A good feature descriptor should exhibit high contrast between salient and non-salient regions....|
|||...The JuddDB dataset [4] is created from the MIT saliency benchmark [16], mainly for checking generality of salient object detection models over real-world scenes with multiple objects and complex background....|
|||...We note that the proposed algorithm uniformly highlights the salient regions and preserves fine object boundaries than other methods....|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...What is a salient object?...|
|||...a dataset and a baseline model for salient object detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...PAMI,  Global contrast based salient region detection....|
|||...A simple method for detecting salient regions....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
|||...Contextual hypergraph modeling for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||20 instances in total. (in cvpr2016)|
|69|Zhu_Saliency_Optimization_from_2014_CVPR_paper|... Wei, Jian Sun Microsoft Research  {yichenw, jiansun}@microsoft.com  Abstract  Recent progresses in salient object detection have exploited the boundary prior, or background information, to assist oth...|
|||...Introduction  Recent years have witnessed rapidly increasing interest in salient object detection [2]....|
|||...The cost function is defined to directly achieve the goal of salient object detection: object regions are constrained to take high saliency using foreground cues; background regions are constrained to...|
|||...In the following we briefly review previous works from the two viewpoints of interest in this paper: the usage of boundary prior and optimization methods for salient object detection....|
|||...The work in [23] treats salient objects as sparse noises and solves a low rank matrix recovery problem instead....|
|||...The work in [9] models salient region selection as the facility location problem and maximizes the sub-modular objective function....|
|||...From human perception, the green region is clearly a salient object as it is large, compact and only slightly touches the image boundary....|
|||...It is hard to tell whether the detected salient regions are really salient....|
|||...Also, although the ideal output of salient object detection is a clean binary object/background segmentation, such as the widely used ground truth in performance evaluation, most previous methods were...|
|||...We model the salient object detection problem as the optimization of the saliency values of all image superpixels....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Center-surround divergence of feaIn ICCV, 2011.  ture statistics for salient object detection....|
|||...Automatic salient ob ject extraction with contextual cue....|
|||...Submodular salient region detec [15] L. Mai, Y. Niu, and F. Liu....|
|||...Global contrast based salient region detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...3, 6  [19] R.Achanta,  S.Hemami,  F.Estrada,  Frequency-tuned salient region detection....|
|||...A unified approach to salient object  detection via low rank matrix recovery....|
|||...Learning to  detect a salient object....|
||19 instances in total. (in cvpr2014)|
|70|Pan_Robust_Kernel_Estimation_CVPR_2016_paper|...(f) Salient edges extracted by the proposed algorithm (shown by Poisson reconstruction)....|
|||...As shown in Figure 1(f), the selected salient edges contain fewer saturated pixels, which accordingly lead to a better estimated blur kernel (Figure 1(q)) and deblurred result with fine textures (Figure 1(i))....|
|||...Central to our method is to select reliable salient edges (See the part in the blue box in Figure 2) that satisfy the linear convolution model (1) for kernel estimation....|
|||...Update Intermediate Latent Image I  Within the MAP framework, the intermediate image I  detect and remove outliers from intermediate salient edges (Section 2.2.2)....|
|||...can be obtained when the blur kernel k is known,  2.2.1  Intermediate salient edge selection  k  min  I  kf (I  k)  Bk1 + cEI (I)....|
|||...ation when the blurred image B contains numerous saturated or clipped pixels, as these outliers are salient and considered as important salient edges for kernel estimation (See Figure 1)....|
|||...To address these issues, we introduce a model to select intermediate salient edges (Section 2.2.1) such that tiny details corresponding to small image gradients are removed....|
|||...2.2.2 Outliers removal from intermediate salient edges  Although the proposed model in (7) can remove tiny details and retain salient edges for kernel estimation, it is based on large gradients and th...|
|||...Removing outliers from salient edges....|
|||...We present a method to remove outliers from the intermediate salient edges S....|
|||...Since M is computed from the blurred input, it cannot be directly applied to the extracted salient edges S....|
|||...As a result, only the salient edges that satisfy the linear convolution model are retained for kernel estimation....|
|||...As saturated areas in Figure 8(a) are salient (e.g., the white blobs), edge selection methods [3, 36] based on large gradients are likely to select these areas (Figure 8(b) and (c)) for kernel estimation....|
|||...Since the linear convolution model (1) does not hold for the pixels in the saturated areas, kernels cannot be estimated well from the salient edges shown in Figure 8(b) and (c)....|
|||...Although the algorithm [27] adopts an adaptive TV denoising step to select salient edges, which is similar to the intermediate edge selection step (7) of the proposed algorithm, this method is not abl...|
|||...Thus, the selected salient edges still contain saturated pixels which accordingly affect kernel estimation....|
|||...Different from existing edge selection methods [3, 27, 36], the proposed algorithm does not necessarily select the salient edges with the largest gra dients when blurred image contains saturated areas....|
|||...As blurred images may contain ambiguous edges [25, 36], these edges are likely to be selected as they are salient [25] and thus affect kernel estimation....|
|||...where the linear convolution model does not hold from extracted salient edges (See Figure 14(d))....|
||19 instances in total. (in cvpr2016)|
|71|Shtrom_Saliency_Detection_in_2013_ICCV_paper|...Detecting the salient features in a point set of an urban scene....|
|||...The most salient points, such as the rosette and the crosses on the towers, are colored in yellow and red....|
|||...The least salient points, belonging to the floor and the feature-less walls, are colored in blue....|
|||...In this paper we present an algorithm for detecting the salient points in unorganized 3D point sets....|
|||...Then, association is applied, grouping salient points and emphasizing the dragons facial features....|
|||...Therefore, points that are close to the foci of attention are more salient than faraway points....|
|||...We propose a novel algorithm that detects salient points in a 3D point set (Figure 1), by realizing the considerations mentioned above....|
|||...First, we propose a novel algorithm for detecting the salient points in large point sets (Sections 3-5)....|
|||...Finally, we wish to look for salient regions, rather than for isolated points [29]....|
|||...The buildings are salient and therefore are colored in orange....|
|||...The trees, of which there are many, are less salient and are colored in green....|
|||...The buildings are found salient and therefore are colored in orange....|
|||...The trees, which are similar in their appearance and of which there are many, are less salient and are thus colored in green....|
|||...Figure 6 demonstrates that our algorithm detects the expected salient regions, such as the fork of Neptune and the fish next to his feet, and the facial features of Max Planck and the dinosaur....|
|||...Our algorithm detects the expected salient regions, e.g., the fork of Neptune and the fish near his feet and the facial features....|
|||...For example, for the head of Igea, both algorithms choose a side-view, but our view presents the side with the salient scar near the mouth....|
|||...The idea is to maximize the area of the viewed salient regions along a path....|
|||...First, we compute a set of candidate locations and pick a subset, Ls, of the most salient locations, similarly to viewpoint selection....|
|||...Modeling attention to salient proto objects....|
||19 instances in total. (in iccv2013)|
|72|Siva_Looking_Beyond_the_2013_CVPR_paper|...sychological knowledge of the human visual system and finds image patches on edges and junctions as salient using local contrast or global unique frequencies....|
|||...This tendency to find things as being salient is intensified by sampling from similar images....|
|||...ies: Our saliency map exhibits a bias towards selecting junctions or the intersection of objects as salient regions, and a secondary bias towards the objects themselves because they occur infrequently...|
|||...Our interest, motivated by applications in object detection, lies in object saliency approaches that can detect salient regions  3237 3237 3239  as potential object locations....|
|||...MSRA [1]) with a single salient object per image, they do not provide means of proposing multiple object locations in an image, and they do not consider the use of other similar images....|
|||...In particular we define a patch as salient if it is uncommon not only in the current image, but also in other similar images drawn from a large corpus of unlabelled images....|
|||... estimator rather than the MAP, giving us greater robustness, and allowing us to potentially detect salient regions using only a single image (see fig....|
|||...We define salient patches, as those belonging to image I, that have the least probability of being sampled from a set of images DI similar to I....|
|||...First, as in [13], immediate context information is included by weighting the saliency value of each pixel by their distance from the high salient pixel locations....|
|||...Then to maximise the overlap with the salient boxes in B that will be suppressed, b is chosen as the box with the closest histogram to SIFT....|
|||...While this allows exact alignment to the true object to be found in the first 3 salient boxes, objects in a lower salient region are missed....|
|||...However, the selection of a box that narrowly misses the object may cause the later rejection of the most salient box containing the object....|
|||...Such approaches typically assume one object per image and select a single salient region....|
|||...While non-maximum suppression ensures that even low salient regions are sampled from, it does not allow for the repeated sampling of high salient regions....|
|||...Global contrast  based salient region detection....|
|||...In MSRA, unique colors often indicate salient objects but on VOC, unique color is often indicative of a small patch of sky (see fig....|
|||...In section 2, we defined as salient patches with a low probability being sampled from a set of similar images DI....|
|||...Frequency tuned salient region detection....|
||18 instances in total. (in cvpr2013)|
|73|cvpr18-Progressive Attention Guided Recurrent Network for Salient Object Detection|...Progressive Attention Guided Recurrent Network for Salient Object Detection  Xiaoning Zhang1 , Tiantian Wang1 , Jinqing Qi1, Huchuan Lu1, Gang Wang2  1Dalian University of Technology, China  2 Alibaba...|
|||...On the other hand, it is observed that most of existing algorithms conduct salient object detection by exploiting side-output features of the backbone feature extraction network....|
|||...During the past two decades, many salient object detection methods have been proposed....|
|||...However, it is of great difficulty for these low-level features based approaches to detect salient objects in complex scenarios....|
|||...In general, salient objects only correspond to partial regions of the input image....|
|||...In Figure 1(a)-(c), we show some examples that spatial attention can highlight the salient object and avoid distractions in the background regions....|
|||...And our channel-wise attention mechanism assigns larger weights to channels which show higher response to salient objects....|
|||...To accurately locate salient objects and obtain sharper boundaries simultaneously, it is necessary to combine multi-level features together....|
|||...e State(cid:173)of(cid:173)the(cid:173)Art  Our algorithm is compared with thirteen state-of-theart salient object detection methods, including DRFI [9],  (a) Input  (b) FCN  (c) CA  (d) CSA (e) GT  F...|
|||...Global contrast based salient region detection....|
|||...Deeply supervised salient object detection with short connections....|
|||...Deep contrast learning for salient object  detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Learning to detect salient objects with image-level supervision....|
|||...A stagewise refinement model for detecting salient objects in images....|
||18 instances in total. (in cvpr2018)|
|74|Qin_Saliency_Detection_via_2015_CVPR_paper|...t  In this paper, we introduce Cellular Automataa dynamic evolution model to intuitively detect the salient object. First, we construct a background-based map using color and space contrast with the c...|
|||...To better distinguish salient objects from background, highlevel information and supervised methods are incorporated to improve the accuracy of saliency map....|
|||...In [51], the salient object can naturally emerge under a Bayesian framework due to the self-information of visual features....|
|||...Considering that salient objects tend to be clustered, we apply Cellular Automata to exploit the intrinsic relationship of neighbors and reduce the difference in similar regions....|
|||...And the salient object can be more easily detected under the influence of neighbors....|
|||...3.2.4 Optimization of State-of-the-Arts  Due to the connectivity and compactness of the object, the salient part of an image will naturally emerge after evolution....|
|||...Moreover, we surprisingly find out that even if the background-based map is poorly constructed, the salient object can still be precisely detected via Singly-layer Cellular Automata, as exemplified in...|
|||...Based upon Cellular Automata, an intuitive updating mechanism is designed to exploit the intrinsic connectivity of salient objects through interactions with neighbors....|
|||...Frequency-tuned salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Global contrast based salient region detection....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Submodular salient region detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...The secrets  of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
||18 instances in total. (in cvpr2015)|
|75|Li_Primary_Video_Object_ICCV_2017_paper|...In recent years, the segmentation of primary image objects, namely image-based salient object detection, has achieved impressive success since powerful models can be directly trained on large image da...|
|||...They may not always be the most salient ones in each separate frame but can consistently pop-out in most video frames (frames and masks are taken from the dataset [17])....|
|||...onvolutional Neural Networks (CCNN) is trained end-to-end on massive images with manually annotated salient objects so as to simultaneously handle two complementary tasks, i.e., foreground and backgro...|
|||...In this manner, the foregroundness branch mainly focuses on detecting salient objects, while the backgroundness branch focuses on suppressing distractors....|
|||...In the training stage, we collect massive images with labeled salient objects from four datasets for image-based  Figure 3....|
|||...(d) and (h) fusion maps. We can see that the foregroundness and backgroundness maps can well depict salient objects and distractors in many frames (see (a)-(d))....|
|||... the proposed approach, denoted as OUR, is compared with 18 state-of-the-art models for primary and salient object segmentation, including:  1) Image-based & Non-deep (7): RBD [46], SMD [23],  MB+ [43...|
|||...From Table 1, we also find that there exist inherent correlations between salient image object detection and primary video object segmentation....|
|||...5, primary objects are often the most salient ones in many frames, which explains the reason that deep models like ELD, RFCN and DCL outperforms many video-based models like NLC, SAG and GF....|
|||...First, primary objects may not always be the most salient ones in all frames (as shown in Fig....|
|||...se to video boundary due to camera and object motion, making the boundary prior widely used in many salient object detection models no valid any more (e.g., the cow in the 1st row of the center column...|
|||...Last but not least, salient object detection needs to distinguish a salient object from a fixed set of distractors, while primary object segmentation needs to consistently pop-out the same primary obj...|
|||...In particular, the performances of both the foregroundness and backgroundness branches outperform all the other 6 deep image-based salient object detection models on VOS....|
|||...Actually, using multiple color spaces have been proved to be useful in detecting salient objects [12], while multiple color spaces make it possible to assess temporal correspondences from several pers...|
|||...Background prior-based salient object detection via deep reconstruction residual....|
|||...Deep contrast learning for salient object  lated convolutions....|
|||...Minimum barrier salient object detection at 80 fps....|
||17 instances in total. (in iccv2017)|
|76|Kolkin_Training_Deep_Networks_ICCV_2017_paper|...We do not sacrifice accuracy, achieving competitive or state of the art accuracy on benchmarks for salient object detection, portrait segmentation, and visual distractor masking....|
|||...We review it below, and also survey the related work on the segmentation tasks on which we evaluate our contributions: salient object detection, distractor detection, and portrait segmentation....|
|||...Salient Object Detection  Traditionally salient object detection models have been constructed by applying expert domain knowledge....|
|||...combines the task of salient object detection with several other vision tasks, demonstrating a general multi-task CNN architecture....|
|||...This task is somewhat similar to salient object detection, but successful algorithms must go beyond simply detecting all salient objects, and model the image at a global level to discriminate between ...|
|||...with three tasks where we can expect spatial sensitivity to be important for quality of the output: salient object detection, portrait segmentation, and distractor detection....|
|||...Widely used for salient object detection....|
|||...All images with at least one of the following attributes: multiple salient objects, salient objects touching boundary, low color contrast, complex background....|
|||...Frequency-tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Deep contrast learning for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
|||...A fast and compact salient score regression network based on fully convolutional network....|
||16 instances in total. (in iccv2017)|
|77|cvpr18-PiCANet  Learning Pixel-Wise Contextual Attention for Saliency Detection|...We also incorporate the proposed models with the U-Net architecture to detect salient objects....|
|||...Traditional saliency models mainly rely on various saliency cues to detect salient objects, including local contrast [15], global contrast [4], and background prior [43]....|
|||...[45] also utilize U-Net based models to incorporate multi-level contexts to detect salient objects....|
|||...Salient Object Detection using PiCANets  In this section, we elaborate our network architecture which adopts PiCANets hierarchically for salient object detection....|
|||...The last one is the DUTS [36] dataset, which is currently the largest salient object detection benchmark dataset....|
|||...We apply PiCANets to detect salient objects in a hierarchical fashion....|
|||...Global contrast based salient region detection....|
|||...Deeply supervised salient object detection with short connections....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
|||...Non-local deep features for salient object detection....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Learning to detect salient objects with image-level supervision....|
|||...A stagewise refinement model for detecting salient objects in images....|
|||...Deep contrast learning for salient object  [39] S. Xie and Z. Tu....|
||16 instances in total. (in cvpr2018)|
|78|cvpr18-Cube Padding for Weakly-Supervised Saliency Prediction in 360° Videos|...[30, 29] focus on providing various visual guidance in VR display so that the viewers are aware of all salient regions....|
|||...[32, 19, 4, 57, 65, 44] focus on detecting salient regions in [31, 23, 11, 43, 42, 8, 60, 59, 53] employ deep images....|
|||...ing aspects: (i) sufficiently large number of search results of 360 video on YouTube, (ii) multiple salient objects in a single frame with diverse categories, (iii) dynamic contents inside the videos ...|
|||...To adopt the similar approach, but also giving the global perspective to viewers to easily capture multiple salient regions without missing hot spots, we adopt HumanEdit interface from [52]....|
|||...In this setting, various positions could be marked as salient regions....|
|||...To avoid the criterion being too loose, only locations on heatmap with value larger than  + 3 were considered salient when creating the binary mask for the saliency evaluation metrics, e.g....|
|||...Consistent Video Saliency  [61] detects salient regions in spatio-temporal structure based on the gradient flow and energy optimization....|
|||...Our temporal model typical predicts smooth saliency map in time and is more effective to salient regions on image boundaries or in the top/bottom part of the image....|
|||...by saliency score, we use AUTOCAM [52] to find a feasible path of salient viewpoints....|
|||...8 shows that the NFoV tracks we generated are able to capture salient viewpoints better than equirectangular....|
|||...We ask  16 viewers to select the saliency map prediction which (1) activates on salient regions more correctly, (2) is smoother across frames....|
|||...Frequency-tuned salient region detection....|
|||...Dhsnet: Deep hierarchical saliency net work for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Learning to detect salient objects with image-level supervision....|
||16 instances in total. (in cvpr2018)|
|79|Shi_PISA_Pixelwise_Image_2013_CVPR_paper|...We further impose a spatial prior term on each of the two contrast measures, which constrains pixels rendered salient to be compact and also centered in image domain....|
|||...Though quite challenging, being able to separate salient objects from the background is a very useful tool for many computer vision and graphics applications such as object recognition [22], content-a...|
|||...(ii) We further impose a spatial prior term on each of the two contrast measures, which constrains pixels rendered salient to be compact and also centered in image domain....|
|||...They are evaluated based on the general contrast prior principle that rare or infrequent visual features in a global image context give rise to high salient values....|
|||...1(second and third rows), using color information only is not adequate to discriminatively describe and detect salient objects or parts of them from the background....|
|||...tial prior term on each of the two contrast measures {U c(p), U g(p)}, constraining pixels rendered salient to be compact and centered in image domain based on intracluster distance which is more comp...|
|||...al prior term  Dc/g(p) based on the cluster i/i that contains p from two aspects: 1) compactness of salient objects defined by the intra-cluster spatial variance, and 2) preference to the image center...|
|||...tion map  S l. It can produce a smoothly varying dense saliency map S without blurring the edges of salient objects....|
|||...The SED1 dataset [2] is exploited recently, and we consider a pixel salient if it is annotated as salient by all subjects....|
|||...Global  contrast based salient region detection....|
|||...Learning to  detect a salient object....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Saliency filters: contrast based filtering for salient region detection....|
|||...For any background regions that have been assigned high saliency values from either of the contrast cues after the modulation of the spatial priors, they remain salient in the final saliency map (see Fig....|
|||...Frequency-tuned salient region detection....|
||15 instances in total. (in cvpr2013)|
|80|Veeriah_Differential_Recurrent_Neural_ICCV_2015_paper|...onventional LSTMs do not consider the impact of spatio-temporal dynamics corresponding to the given salient motion patterns, when they gate the information that ought to be memorized through time....|
|||...To address this problem, we propose a differential gating scheme for the LSTM neural network, which emphasizes on the change in information gain caused by the salient motions between the successive frames....|
|||...However, we observed that for an action recognition task, not all frames contain salient spatio-temporal information which are discriminative to different classes of actions....|
|||...The conventional LSTM fails to capture the salient dynamic patterns, since the gate units do not explicitly consider whether a frame contains salient motion information when they modulate the input an...|
|||...To address this problem, we propose the differential RNN (dRNN) model that learns these salient spatiotemporal representations of actions....|
|||...Differential Recurrent Neural Networks  For an action recognition task, not all video frames contain salient patterns to discriminate between different classes of actions....|
|||...Many spatio-temporal descriptors, such as 3D-SIFT [29] and HoGHoF [19], have been proposed to localize and encode the salient spatio-temporal points....|
|||...They detect and encode the spatio-temporal points related to salient motions of the objects in video frames, revealing the important dynamics of actions....|
|||...In this paper, we develop a novel LSTM model to automatically learn the dynamics of actions, by detecting and integrating the salient spatio-temporal sequences....|
|||...The conventional LSTMs might fail to capture these salient dynamic patterns, because the gate units do not explicitly consider the impact of dynamic structures present in input sequences....|
|||...Now it is obvious that this model is termed differential RNNs (dRNNs) because of the central role of derivatives of states in detecting and capturing the salient spatio-temporal structures....|
|||...Finally, it is worth pointing out that we do not use the derivative of inputs as a measurement of salient dynamics to control the gate units....|
|||...Using derivative of inputs would treat it as a novel salient motion, even though it has already been stored by LSTM....|
|||...From the result, we found that as time evolves, the proposed dRNNs are faster in learning the salient dynamics for predicting the correct action category than the LSTMs....|
|||...The new structure is better at learning the salient spatio-temporal structure....|
||15 instances in total. (in iccv2015)|
|81|Li_Robust_Saliency_Detection_2015_CVPR_paper|... the  importance  of  different  regions of the image, and conduct detailed processes only on  the  salient  objects  that  mostly  related  to  the  current  tasks,  while  neglecting  the  remaining...|
|||...The detection of such salient objects in the image  is  of  significant  importance,  as  it  directs  the  limited  computational  resources  to  faster  solutions  in  the  subsequent image processi...|
|||...In computer vision, salient object detection algorithms  can be categorized into bottom-up approaches [1, 6, 8, 11,  13,  17,  20,  27],  and  top-down  approaches  [4,  7,  9,  28]....|
|||...[11],  the  random  walks  model  has  been  exploited  in  an  automatic  salient-region-extraction  method  to  effectively  detect  the  rough location of the most salient object in an image....|
|||...Given the conspicuous difference of color and contrast  between  the  background  and  the  salient  object,  the  erroneous boundary tends to have distinctive color distribution  compared to the rema...|
|||...ufficient  to  fully  illustrate  the  foreground  information,  especially  in  cases  where  the  salient  object  has  complicated  structure  or  similar  patterns  to  the  background....|
|||...In  other  words, precision is the ratio of retrieved salient pixels to  all pixels retrieved, and recall is the ratio of retrieved salient pixels to all salient pixels in the image....|
|||...[2]  R.  Achanta,  S.  Hemami,  F.  Estrada,  and  S.  Susstrunk,  "Frequency-tuned  salient  region  detection,"  in  Computer  Vision and  Pattern  Recognition,  2009....|
|||...Zhang,  N.  J.  Mitra,  X.  Huang,  and  S.-M. Hu, "Global contrast based salient region detection,"  in Computer Vision and Pattern Recognition (CVPR), 2011  IEEE Conference on, 2011, pp....|
|||... Zhou,  and  I.  Yu-Hua  Gu,  "Superpixel  based  color  contrast  and  color  distribution  driven salient object detection," Signal Processing: Image  Communication, vol....|
|||...[11]  V. Gopalakrishnan, Y. Hu, and D. Rajan, "Random walks  on  graphs  for  salient  object  detection  in  images,"  Image  Processing, IEEE Transactions on, vol....|
|||...1254-1259, 1998.   scene  analysis,"   [18]  H. Jiang, J. Wang, Z. Yuan, T. Liu, N. Zheng, and S. Li,  "Automatic  salient  object  segmentation  based  on  context  and shape prior," in BMVC, 2011, p. 7....|
|||...,  P.  Krahenbuhl,  Y.  Pritch,  and  A.  Hornung,  "Saliency filters: Contrast based filtering for salient region  detection,"  in  Computer  Vision  and  Pattern  Recognition  (CVPR), 2012 IEEE Conf...|
|||...[23]  E.  Rahtu,  J.  Kannala,  M.  Salo,  and  J.  Heikkila,  "Segmenting  salient  objects  from  images  and  videos,"  in  Computer  VisionECCV  2010,  ed:  Springer,  2010,  pp....|
|||...[25]  X. Shen and Y. Wu, "A unified approach to salient object  detection  via  low  rank  matrix  recovery,"  in  Computer  Vision  and  Pattern  Recognition  (CVPR),  2012  IEEE  Conference on, 2012, pp....|
||15 instances in total. (in cvpr2015)|
|82|Yan_Hierarchical_Saliency_Detection_2013_CVPR_paper|...These examples are not special, and exhibit one common problem  that is, when objects contain salient smallscale patterns, saliency could generally be misled by their complexity....|
|||...With our multi-level analysis and hierarchical inference, the model is able to deal with salient small-scale structure, so that salient objects are labeled more uniformly....|
|||...ods physically obtain human attention shift continuously with eye tracking, while the latter set of approaches aim to find salient objects from images....|
|||...The concept of center bias  that is, image center is more likely to contain salient objects than other regions  was employed in [18, 14, 25, 30]....|
|||...Prior work does not consider the situation that locally smooth regions could be inside a salient object and globally salient color, contrarily, could be from the background....|
|||...So pixels close to a natural image center could be salient in many cases....|
|||...Only sufficiently salient objects can be detected in this case....|
|||...Result from layer L1 is the worst since it contains many small struc We have tackled a fundamental problem that small-scale structures would adversely affect salient detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...to detect a salient object....|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object  detection via low rank matrix recovery....|
|||...Content based image retrieval using color boosted salient points and shape features of an image....|
||15 instances in total. (in cvpr2013)|
|83|Ming_Winding_Number_for_2013_CVPR_paper|...ICTA and ANU  Canberra, ACT, Australia xuming.he@nicta.com.au  Abstract  This paper aims to extract salient closed contours from an image....|
|||...Often these salient contours correspond to semantically meaningful contents in the image, such as object boundaries....|
|||...However, salient contour extraction is a challenging task as it involves both region and boundary information, requiring integration of bottom-up image cues and top-down semantic priors....|
|||...This paper aims to develop a more consistent approach to salient contour extraction that tightly integrates both region cues and boundary cues....|
|||...In Section 4, our method is used for integrating region cue and contour cues to achieve salient contour extraction....|
|||...Winding number representation  This section first presents our salient contour extraction problem setting, which is based on superpixel oversegmentation....|
|||...Basic edge and region hypotheses  We formulate the salient contour extraction problem as an energy minimization problem defined on both region and edge hypotheses....|
|||...Generally, the salient contour detection is formulated as the following energy-minimization problem:  2817 2817 2819  E(x, y)  min x,y s.t....|
|||...This term will favor the foreground with a salient boundary....|
|||...The image set contains salient unoccluded horses in the middle of image....|
|||...Our method, which aims to extract salient closed contours is not successful for detecting occluded, obscured, or camouflaged figural objects in this dataset....|
|||...These results match our perception of salient region....|
|||...Our experiments show that evident improvements can be made for the task of salient contour extraction when both region cue and contour cue are employed....|
|||...Segmentation of multiple salient closed contours from real images....|
||14 instances in total. (in cvpr2013)|
|84|Li_Saliency_Detection_on_2014_CVPR_paper|...In this paper, we explore the salient object detection problem by using a completely different input: the light field of a scene....|
|||...In this paper, we explore how to conduct salient object detection using a light field camera....|
|||...This is also inline with the objectness" [16], i.e., a salient region should complete objects instead of cutting them into pieces....|
|||...Instead of directly detecting salient regions, these algorithms aim to first find the background and then use it to prune non-saliency objects....|
|||...In addition, we choose regions with a high FLS as candidate salient objects. Finally, we conduct contrast-based saliency detection on the all-focus image and combine its estimation with the detected f...|
|||...Many saliency detection schemes exploit contrast cues, i.e., salient objects are  Figure 2....|
|||...Recent studies on human perception [18] have shown that depth cue plays a important role in determining salient regions....|
|||...Next, we combine the focusness measure with the location prior to extract the background and the foreground salient candidates....|
|||...[3] suggested that a salient object should be complete instead of being broken into pieces and refer to this property as the objectness....|
|||...Precision corresponds to the percentage of salient pixels that are correctly assigned and recall refers to the fraction of detected salient region w.r.t....|
|||...In CVPR, pages  Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Learning to detect a salient object....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
||14 instances in total. (in cvpr2014)|
|85|Zhang_Learning_Uncertain_Convolutional_ICCV_2017_paper|...In this paper, we propose a novel deep fully convolutional network model for accurate salient object detection....|
|||...To deal with the problem that salient objects may appear in a low-contrast background, Zhao et al....|
|||...he decoder FCN and fine-tune the entire network on the MSRA10K dataset [4], which is widely used in salient object detection community (More details will be described in Section 4)....|
|||...The SED1 has 100 images each containing only one salient object, while the SED2 has 100 images each containing two salient objects....|
|||...Pixel-wise annotation of salient objects was generated by [18]....|
|||...Our saliency maps can reliably highlight the salient objects in various challenging scenarios, e.g., low contrast between objects and backgrounds (the first two rows), multiple disconnected salient ob...|
|||...In addition, our saliency maps provide more accurate boundaries of salient objects (the 1, 3, 6-8 rows)....|
|||...What is a salient object?...|
|||...a dataset and a baseline model for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Center-surround divergence of feature statistics for salient object detection....|
|||...Deep contrast learning for salient object  detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
||13 instances in total. (in iccv2017)|
|86|Deep 360 Pilot_ Learning a Deep Agent for Piloting Through 360deg Sports Videos|...Saliency Detection  Many methods have been proposed to detect salient regions typically measured by human gaze....|
|||...[35, 21, 1, 46, 59, 64, 46] focused on detecting salient regions on images....|
|||...For 360 piloting, we propose a similar baseline which first detects objects using RCNN [50], then select the viewing angle focusing on the most salient object according to a saliency detector [64]....|
|||...Finally, we recruited 5 human annotators, and 3 were asked to label the most salient object for VR viewers in each frame in a set of video segments containing human-identifiable objects....|
|||...[64] to detect the most salient region in a frame....|
|||...Then we emit the most salient box center sequentially as our optimal viewing angle trajectories....|
|||...We aimed at developing a domain-specific agent for the domain where the definition of a most salient object is clear (e.g., skate boarder)....|
|||...However, our algorithm would suffer in the domains where our assumption is violated (containing equally salient objects or no objects at all)....|
|||...Frequency-tuned salient region detection....|
|||...Dhsnet: Deep hierarchical saliency net work for salient object detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Summarizing unconstrained videos using salient montages....|
||13 instances in total. (in cvpr2017)|
|87|Zhang_Multi-Oriented_Text_Detection_CVPR_2016_paper|...First, a Fully Convolutional Network (FCN) model is trained to predict the salient map of text regions in a holistic manner....|
|||...Then, text line hypotheses are estimated by combining the salient map and character components....|
|||...(a) An input image; (b) The salient map of the text regions predicted by the TextBlock FCN; (c) Text block generation; (d) Candidate character component extraction; (e) Orientation estimation by compo...|
|||...Lower level stages capture more local structures, and higher level stages capture more global information; (g) The final salient maps....|
|||...First, a salient map is generated and segmented into several candidate text blocks....|
|||...Our contributions are in three folds: First, we present a novel way for computing text salient map, through learning a strong text labeling model with FCN....|
|||...We show that the local (character components) and the global (text blocks from the salient map) cues are both helpful and complementary to each other....|
|||...In this section, we learn a FCN model, named Text-Block FCN, to label salient regions of text blocks in a holistic way....|
|||...(a) An input image; (b) The character response map, which is generated by the state-of-the-art method [8]; (c) The salient map of text regions, which is generated by the Text-Block FCN....|
|||...In the testing phase, the salient map of text regions, leveraging all context information from different stages, is computed by the trained Text-Block FCN model at first....|
|||...Implementation Details  In the proposed method, two models are used: the TextBlock FCN is used to generate text salient maps and the Character-Centroid FCN is used to predict the centroids of characters....|
|||...To compute the salient map in the testing phase, each image is proportionally resized to three scales, where the heights are 200, 500 and 1000 pixels respectively....|
||12 instances in total. (in cvpr2016)|
|88|Tasse_Cluster-Based_Point_Set_ICCV_2015_paper|...Our approach detects fine-scale salient features and uninteresting regions consistently have lower saliency values....|
|||...The saliency results show that our method is robust enough to detect salient regions such as the eyes, mouth, feet and wings....|
|||...Regions that are similar but spread over a large area are less salient than similar regions that are more compact....|
|||...We are also able to detect fine-scale salient features on the head of the Dragon....|
|||...[31] fail to capture salient regions on the human and glasses models, and in general, do not match high saliency regions in the ground-truth....|
|||...Only our method succeeds in capturing the salient ends of the tentacles on the octopus model....|
|||... method that consistently produces correct saliency maps is spectral mesh saliency [29]; it detects salient regions correctly but similar unsalient regions, like the main bodies of the vases, often ha...|
|||...Our method is robust enough to detect salient regions with very small K (as small as K = 10)....|
|||...Ground truth [6]  Our saliency  Our keypoints  Mesh Saliency [19] Salient points [2]  HKS [30]  Figure 7: Interest point detection: keypoints detected by our algorithm compared to other methods....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Global contrast based salient region detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||12 instances in total. (in iccv2015)|
|89|Margolin_How_to_Evaluate_2014_CVPR_paper|...d  The output of many algorithms in computer-vision is either non-binary maps or binary maps (e.g., salient object detection and object segmentation)....|
|||...a foreground map against a binary ground-truth is common in various computer-vision problems, e.g., salient object detection [10], object segmentation [11], and foreground-extraction [6]....|
|||...non-binary foreground maps were generated for each image using five state-of-the-art algorithms for salient object detection [9, 10, 12, 13, 19] (binary maps are obtained by thresholding the non-binar...|
|||...Frequency tuned salient region detection....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Learning to detect a salient object....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection  via low rank matrix recovery....|
||11 instances in total. (in cvpr2014)|
|90|cvpr18-Revisiting Video Saliency  A Large-Scale Benchmark and a New Model|...Other datasets are either limited in terms of variety and scale of video stimuli [25, 17], or collected for a special purpose (e.g., salient objects in videos [59])....|
|||...There are some salient object detection models [40, 1, 11, 61, 58, 60, 4, 62, 21] that attempt to uniformly highlight salient object regions in images or videos....|
|||...In this paper, we use attention for enhancing intra-frame salient features, thus allowing the LSTM to model dynamic representations more easily....|
|||...Frequency-tuned salient region detection....|
|||...What is a salient object?...|
|||...A dataset and a baseline model for salient object detection....|
|||...Global contrast based salient region detection....|
|||...Deeply supervised salient object detection with short connections....|
|||...The  secrets of salient object segmentation....|
|||...5  Learning to detect a salient object....|
|||...Video salient object detec tion via fully convolutional networks....|
||11 instances in total. (in cvpr2018)|
|91|cvpr18-Gaze Prediction in Dynamic 360° Immersive Videos|...In terms of the image contents, those salient objects easily attract viewers attention....|
|||...ead position Fixation points and Head position  Head position Head position Head position  Annotate salient object in panorama  Without using HMD  Eye tracking in VR  HTC VIVE  Manually labeled boundi...|
|||...At the same time, we find that a viewer is more likely to be attracted by salient objects characterized by appearance and motion, thus we also take into the saliency into our consideration, specifical...|
|||...Most of the models are based on the bottom-up [10] [2] [14] [23], top-down [25] [13] [8] [22] [29], or hybrid approaches to detect salient regions on images....|
|||...Further, [19] propose to extract salient objects in VR videos, but the salient objects are manually annotated with panorama rather than obtained with gaze tracking in immersive VR....|
|||...build a large 360 sports videos dataset by asking viewers manually annotate the salient object with panorama rather than in HMD screen....|
|||...Method  regions  Saliency detection assumes those more salient regions attract viewers attention....|
|||...We can see that gaze points usually coincide with salient points....|
|||...Saliency Encoder Module  As aforementioned, gaze points usually coincide with spatial salient regions and objects with salient motions (large optical flows)....|
|||...Saliency unified: A deep architecture for simultaneous eye fixation prediction and salient object segmentation....|
|||...Dhsnet: Deep hierarchical saliency network for salient object detection....|
||11 instances in total. (in cvpr2018)|
|92|cvpr18-Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering|...In this work, we propose a combined bottom-up and topdown attention mechanism that enables attention to be calculated at the level of objects and other salient image regions....|
|||...These mechanisms improve performance by learning to focus on the regions of the image that are salient and are currently based on deep neural network architectures....|
|||...Our approach enables attention to be calculated at the level of objects and other salient image regions (right)....|
|||...To generate more human-like captions and question answers, objects and other salient image regions are a much more natural basis for attention [10, 35]....|
|||...The bottom-up mechanism proposes a set of salient image regions, with each region represented by a pooled convolutional feature vector....|
|||...We first present an image captioning model that takes multiple glimpses of salient image regions during caption generation....|
|||...Comparatively few previous works have considered ap 1http://www.panderson.me/up-down-attention  plying attention to salient image regions....|
|||...to identify salient image regions, which are filtered with a classifier then resized and CNN-encoded as input to an image captioning model with attention....|
|||...ly-sized set of k image features, V = {v1, ..., vk}, vi  RD, such that each image feature encodes a salient region of the image....|
|||...rying scales and aspect ratios  each aligned to an object, several related objects, or an otherwise salient image patch....|
|||...Our approach enables attention to be calculated more naturally at the level of objects and other salient regions....|
||11 instances in total. (in cvpr2018)|
|93|Byrne_Nested_Motion_Descriptors_2015_CVPR_paper|...Furthermore, this structure enables an elegant visualization of salient motion using the reconstruction properties of the steerable pyramid....|
|||...This descriptor provides a representation of salient motion that is invariant to global camera motion, without requiring an explicit optical flow estimate....|
|||...This motion could be due to the salient motion of a foreground object, or due to the global motion of the camera....|
|||...The nested motion descriptor represents salient motion in video....|
|||...This saliency map shows salient responses in red and non-salient in blue....|
|||...(top left) Salient motion for HMDB basketball dribbling using NMD with log spiral normalization....|
|||...The log-spiral normalization suppresses the significant camera motion in the scene focusing on the salient motion of the rock climbers....|
|||...Motion Visualization Results  In this section, we show qualitative results applying the visualization of salient motion captured by the NMD as described in section 3.7....|
|||...The colors encode the saliency map such that red is salient and blue is not-salient....|
|||...Observe that the salient motion extracted using this technique highlight the small motions of dribbling the basketball and not the large motions due to the camera....|
|||...Summary  In this paper, we introduced the nested motion descriptor for representation of salient motion....|
||11 instances in total. (in cvpr2015)|
|94|Quanlong_Zheng_Task-driven_Webpage_Saliency_ECCV_2018_paper|...For each task, we shade two salient semantic components as key components, which are used in our task-driven data synthetic approach....|
|||...Our component saliency ratio tells whether a semantic component under a particular task is more salient (> 1), equally salient (= 1) or less salient (< 1), as compared with the average saliency....|
|||...This is perhaps because that those free-viewing saliency models tend to fire at almost all the salient regions in a webpage, thereby generating a more uniform saliency distribution that is more likely...|
|||...Grad-CAM fails to locate salient regions for each task.The free viewing saliency models (i.e., SalNet, SALICON, VIMGD) simply highlight all the salient regions, oblivious to task conditions....|
|||...Borji, A., Cheng, M., Jiang, H., Li, J.: Salient object detection: A survey....|
|||...Guanbin Li, Yuan Xie, L., Yu, Y.: Instance-level salient object segmentation....|
|||...He, S., Jiao, J., Zhang, X., Han, G., Lau, R.: Delving into salient object subitizing  and detection....|
|||...Kruthiventi, S., Gudisa, V., Dholakiya, J., Venkatesh Babu, R.: Saliency unified: A deep architecture for simultaneous eye fixation prediction and salient object segmentation....|
|||...Li, G., Yu, Y.: Deep contrast learning for salient object detection....|
|||...Liu, N., Han, J.: DHSNet: Deep hierarchical saliency network for salient object  detection....|
|||...Xiao, H., Feng, J., Wei, Y., Zhang, M., Yan, S.: Deep salient object detection with dense connections and distraction diagnosis....|
||11 instances in total. (in eccv2018)|
|95|Yang_Du_Interaction-aware_Spatio-temporal_Pyramid_ECCV_2018_paper|...Thirdly, spatio-temporal detection [18,19] based methods also have good performance so it is promising to use attention to detect salient spatio-temporal regions in videos....|
|||...It utilizes multi-scale feature maps to accurately focus on the salient regions....|
|||...Visualization of salient receptive fields in different frames from appearance (RGB) and motion (Flow) streams....|
|||...ows 5 frames from videos where blue,green and red regions respectively corresponds to the center of salient receptive fields by using 1 scale, 2 scales and 3 scales to obtain attention of different la...|
|||...    Firstly, we visualize the salient receptive fields of K input frames contributing i ....|
|||...Namely for one frame, salient receptive fields to the fixed position (wm, hm) in F centered at the positions satisfied {(wj, hj) Awm hm,kj wj hj > threshold, kj = 1, .., K, wm = 1, hm = 1}....|
|||...Then we set a threshold 0.5 to show salient attention regions for 5 input frames, as shown in Figure 4....|
|||...Moreover, we also shows the obtained salient regions by three methods which respectively use feature maps of 1 scale, 2 scales, and 3 scales to obtain the multi-scale attention scores A....|
|||...Secondly, for one fixed input frame we visualize the salient receptive fields i ....|
|||...y, contributing to different positions (wm, hm) in attention feature maps F for one position, every salient spatial regions centered at the positions satisfied {(wj, hj) Awm hm,kj wj hj > threshold, k...|
|||...Visualization of salient receptive fields for different positions in attention feature maps from appearance (RGB) stream where 3-scale attention is used....|
||11 instances in total. (in eccv2018)|
|96|cvpr18-Diversity Regularized Spatiotemporal Attention for Video-Based Person Re-Identification|...Instead of averaging full frame features across time, we propose a new spatiotemporal approach which learns to detect a set of K diverse salient image regions within each frame (superimposed heatmaps)....|
|||...[11] designed a spatial-temporal segmentation method to extract visual cues and employed color and salient edges for foreground detection....|
|||...We employ temporal attention to assign weights to different salient regions on a per-frame basis to take full advantage of discriminative image regions....|
|||...2) to better handle video re-identification by automatically organizing the data into sets of consistent salient subregions....|
|||...3.2) to generate a diverse set of discriminative spatial gated visual features each roughly corresponding to a specific salient region of a person (Sec....|
|||...Multiple Spatial Attention Models  We employ multiple spatial attention models to automatically discover salient image regions (body parts or accessories) useful for re-identification....|
|||...a grid structure), our approach automatically identifies multiple disjoint salient regions in each image that consistently occur across multiple training videos....|
|||...Each spatial attention model discovers a specific salient image region and generates a spatial gated feature (Fig....|
|||...(3)  Each gated feature represents a salient part of the input image (Fig....|
|||...To encourage the spatial attention models to focus on different salient regions, we design a penalty term which measures the overlap between different receptive fields....|
|||...Q, on the other hand, allows large salient regions like upperbody while discouraging receptive fields from overlapping....|
||11 instances in total. (in cvpr2018)|
|97|Huang_SALICON_Reducing_the_ICCV_2015_paper|...Since saliency prediction aims at localizing the salient regions, the neural responses in mid and high layers might be more informative than the responses at the last layer....|
|||...This convolutional layer has only one filter, that detects whether the responses in Yc correspond to a salient region or not....|
|||...The multi-scale DNN can detect salient regions of different sizes....|
|||...In the fine scale, the DNN detects salient regions of small size, while in the coarse scale, the center of large salient regions stands out....|
|||...It can be seen that by combining features from both scales, our saliency map correctly highlights salient regions of different sizes....|
|||...Pixels in the saliency map are evaluated using a groundtruth label that indicates whether the pixel is salient or nonsalient....|
|||...PASCAL-S [25]: This recent dataset contains 850 images from the PASCAL VOC 2010 dataset [9] with eye fixations from 8 viewers, as well as salient object labeling....|
|||...Our saliency maps are very localized in the salient regions compared to the rest of the methods....|
|||...Observe that our architecture can detect salient regions in different sizes (row 1, 4 and 8)....|
|||...Image signature: Highlight ing sparse salient regions....|
|||...The  secrets of salient object segmentation....|
||11 instances in total. (in iccv2015)|
|98|Cao_Look_and_Think_ICCV_2015_paper|...As a result, only salient regions related with the concept PANDA are captured in visualizations....|
|||...As demonstrated in [24], visualization of Convolutional Neural Network shows semantically meaningful salient object regions and helps understand working mechanism of CNNs....|
|||...Comparing against original gradient and Deconv, the feedback visualization captures more accurate salient area of the target object....|
|||... and Thinking Twice, which eliminate noisy or cluttered background and makes the network focused on salient regions....|
|||...However, compared with Deconv-like approaches, our feedback model is more efficient in capturing salient regions for each specific class while suppress those irrelevant object areas at the same time a...|
|||...Moreover, although both VggNet and GoogleNet produce very similar image classification accuracies, GoogleNet better captures the salient object areas than VggNet....|
|||...We find that VggNet performs quite better than AlexNet, especially in capturing salient object details, suggesting the benefit of usage of small convolutional filters and deeper architecture....|
|||..., our model obtains an initial guess of a set of most probable object classes, we then identify the salient object regions from the predicted top-ranked labels using the feedback neural nets, and recl...|
|||...Our method obtains 38.8% localization error, and significantly outperforms Oxford (44.6%), suggesting that in terms of capturing attention and localizing salient objects, our feedback net is better....|
|||...After feedback, the network determines potential salient ROIs, reconsider these salient regions as input and make correct classifications, e.g., Ibizan hound and mask with high confidence scores....|
||10 instances in total. (in iccv2015)|
|99|Gong_Saliency_Propagation_From_2015_CVPR_paper|...Due to the interactions between the teacher and learner, the uncertainty of original difficult regions is gradually reduced, yielding manifest salient objects with optimized background suppression....|
|||...For example, Maybank [19] proposed a probabilistic definition of salient image regions for image matching....|
|||...tom-up methods use low-level cues, such as contrast and spectral information, to recognize the most salient regions without realizing content or specific prior knowledge about the targets....|
|||...hich significantly confuses the propagation; and 2) the generated convex hull completely misses the salient object....|
|||...Global contrast based salient region detection....|
|||...Geodesic saliency propagation for image salient region detection....|
|||...Automatic salient object segmentation based on context and shape prior....|
|||...Learning In Computer Vision and Pattern  to detect a salient object....|
|||...A probabilistic definition of salient regions for  image matching....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||10 instances in total. (in cvpr2015)|
|100|Li_A_Weighted_Sparse_2015_CVPR_paper|...Introduction  Human visual system can rapidly identify salient objects that mostly attract attention in a 3D scene....|
|||...At the core of the problem is to develop an effective feature contrast measure to separate salient objects from the background and tremendous efforts have been focused on extract color, texture, depth...|
|||...Certainly when the number of superpixels is too small, the salient and non-salient regions will merge and the performance of our approach will ultimatly degrade....|
|||...The PSU Stereo Saliency Benchmark (SSB) contain 1000 pairs of stereoscopic images and corresponding salient object masks for the left images....|
|||...Global contrast based salient region detection....|
|||...Learning to detect a salient object....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection viIn Proceedings of the 2012 a low rank matrix recovery....|
||10 instances in total. (in cvpr2015)|
|101|Kuen_Recurrent_Attentional_Networks_CVPR_2016_paper|...Introduction  Saliency detection refers to the challenging computer vision task of identifying salient objects in imagery and segmenting their object boundaries....|
|||...Besides, conventional CNNs downsize feature maps over multiple convolutional and pooling layers and lose detailed information for our problem of densely segmenting salient objects....|
|||...convolutionaldeconvolutional network (CNN-DecNN) in semantic segmentation [36], in this paper, we adapt the network to detect salient objects in an end-to-end fashion....|
|||...Deconvolutional Networks for Salient Object  Detection  Conventionally, CNNs downsize feature maps over multiple convolutional and pooling layers, to construct spatially compact image representations....|
|||...It is challenging because some of its images do not contain any salient object....|
|||...For each image, there are two salient objects....|
|||...However, the proposed method tends to fail to detect salient objects which are mostly made up of background-like colors and textures (e.g., sky: third image, soil: fourth image)....|
|||...Frequency-tuned salient region detection....|
|||...Global contrast based salient region detection....|
|||...Learning to detect a salient object....|
||10 instances in total. (in cvpr2016)|
|102|Pan_Deblurring_Text_Images_2014_CVPR_paper|...We discuss the relationship with other deblurring algorithms based on edge selection and provide insight on how to select salient edges in a more principled way....|
|||...Much success of the state-ofthe-art algorithms [6, 16, 3, 18, 12, 10, 20] can be attributed to the use of learned prior from natural images and the selection of salient edges for kernel estimation....|
|||...We present analysis on the relationship with other methods based on salient edges, and show that the proposed algorithm generates reliable intermediate results for kernel estimation without any ad-hoc...|
|||...(g) intermediate salient edges of [20]....|
|||...(h) intermediate salient edges using only Pt(x)....|
|||...(j) our intermediate salient edges, i.e., g in (11)....|
|||...By using (10) and (11) in the proposed algorithm, pixels with small intensity values or tiny structures can be removed while salient edges are retained....|
|||...This prior can help preserve more salient edges in the intermediate latent image rather than destroy the salient edges  (e.g., Figure 4(j))....|
|||...The result demonstrates that the use of prior P (x) can also preserve salient edges and removes tiny details in natural images, thereby facilitating kernel estimation in natural images....|
|||...As the priors of the state-of-the-art methods are developed to exploit salient edges for motion  0510152025303540im01im02im03im04im05im06im07im08im09im10im11im12im13im14im15Blurred imageCho and LeeXu ...|
||10 instances in total. (in cvpr2014)|
|103|Zhao_Unsupervised_Salience_Learning_2013_CVPR_paper|...g, xgwang}@ee.cuhk.edu.hk  Abstract  Human eyes can recognize person identities based on some small salient regions....|
|||...Intuitively, if a body part is salient in one camera view, it is usually also salient in another camera view....|
|||...2) Distinct patches are considered as salient only when they are matched and distinct in both camera views....|
|||...For example, a person only with salient upper body and a person only with salient lower body must have different identities....|
|||...Therefore they could not pick up salient regions as shown in Figure 1....|
|||...Following the shared goal of abnormality detection and salience detection, we redefine the salient patch in our task as follows: Salience for person re-identification: salient patches are those posses...|
|||...Illustration of salient patch distribution....|
|||...If the distribution of the reference set well relects the test scenario, the salient patches can only find limited number (k = Nr) of visually similar neighbors, as shown in Figure 3(a), and then scor...|
|||...0 <  < 1 is a proportion parameter relecting our expectation on the statistical distribution of salient patches....|
|||...Our unsupervised learning method better captures the salient regions....|
||10 instances in total. (in cvpr2013)|
|104|cvpr18-Categorizing Concepts With Basic Level for Vision-to-Language|...Specifically, a salient concept category is firstly generated by intersecting the labels of ImageNet and the vocabulary of MSCOCO dataset....|
|||...he observation from human early cognition that children make fewer mistakes on the basic level, the salient category is further refined by clustering concepts with a defined confusion degree which mea...|
|||...Specifically, a salient concept (SaC) category is firstly proposed that contains candidate basic level concepts by matching ImageNet [40] classes with image captions from MSCOCO....|
|||...Categorizing Salient Concepts  Since basic level concepts are objects frequently appeared in daily lives so that children can easily access to them during their early learning....|
|||...Considering that MSCOCO provides only 91 stuffs and obviously it cannot cover as many salient concepts as needed, these concepts in image captions are aligned directly with the annotations in the larg...|
|||...Afterwards, these filtered 9,566 words are matched to ImageNet annotations and each match is considered as an extracted salient concept....|
|||...Finally, a Salient Concept (SaC) Category is generated which contains 1,689 concepts....|
|||...Therefore, a rough set of objects is extracted which contains salient concepts from human annotated datasets....|
|||...The results imply that the SaC Category provides partial salient semantics for word embedding of language model but does not greatly promote the optimization of visual representations, while the BaC C...|
|||...By intersecting classes in ImageNet and MSCOCO, a salient concept category is obtained....|
||10 instances in total. (in cvpr2018)|
|105|Shi_Chen_Boosted_Attention_Leveraging_ECCV_2018_paper|...ting natural language and visual content, without prior knowledge on the visual content in terms of salient regions (i.e., stimulus-based attention), the computed visual attention can fail to concentr...|
|||...As shown in Figure 1, a model with only top-down attention focuses on non-salient regions in the background (Figure 1(c)) and does not capture salient objects in the image, i.e., bulldog and teddy bea...|
|||...Therefore, we propose that the visual stimuli can be a reasonable source for locating salient regions in image captioning, which can also complement top-down attention that relates to specific tasks....|
|||...es based on task-specific top-down signals from natural language while at the same time focusing on salient regions highlighted by task-independent stimulus....|
|||... objects of interest (i.e., cake, police car, man, remote and boy), it typically covers part of the salient regions displayed in the captioning attention maps....|
|||...Specifically, with the use of ReLU activation ensures nonnegativity, to highlight salient regions in the saliency map Wsal needs to construct the correlations between filters and stimulus-based attention (i.e....|
|||...Furthermore, the results also indicate that the model with the proposed Boosted Attention method is capable of capturing multiple salient objects within images....|
|||...In this case, top-down attention tends to play a minor role on discriminating the salient regions related to the task....|
|||...Scenario III: Stimulus-based attention fails to distinguish salient objects with irrelevant background....|
|||...Stimulus-based attention provides prior knowledge on salient regions within the visual scenarios and plays a complementary role to the top-down attention computed by the image captioning models....|
||10 instances in total. (in eccv2018)|
|106|Eunji_Chong_Connecting_Gaze_Scene_ECCV_2018_paper|...int on a smartphone screen, [23] predicts fixation on an object given that the person is looking at salient object within the frame, [7, 30] predict eye contact given that the camera is located near t...|
|||...In (a) subjects are looking at a salient object in the scene, in (b) the subject is looking somewhere outside of the frame and in (c) the subject is looking at or around the camera....|
|||...In Figure 1 (a), the subjects are looking at a salient object in the scene, while in (b) the subject is looking somewhere outside of the scene, and (c) they are looking at the camera....|
|||...ging the finding from [21, 4, 5] which indicate that annotators very often agree on which object is salient in the scene....|
|||...A purely saliency based approach would also fail: notice that there are salient objects in (b), an American flag, and (c), a mug, which can confound such an approach....|
|||...sual attention prediction is influenced by the task of visual saliency since people tend to look at salient objects inside a scene, yet it is distinct because we consider cases where the subject is no...|
|||...For example, when we interpret a persons attention from an image, we infer their gaze direction and consider whether there are any salient objects in the image along the estimated direction....|
|||...Also, when the subject is closer to the camera than some salient object in the background, the method sometimes estimates those as fixation candidate due to the lack of scene depth understanding....|
|||...Borji, A., Cheng, M.M., Jiang, H., Li, J.: Salient object detection: A benchmark....|
|||...: The secrets of salient object segmentation....|
||10 instances in total. (in eccv2018)|
|107|Exploiting Saliency for Object Segmentation From Image Level Labels|...If the image contains a single label dog, chances are that the image is about a dog, and that the salient object of the image is a dog....|
|||...If two locally salient dogs appear in the image, both will be labelled as foreground....|
|||...The second difficulty, clearly visible in the examples of figure 4, is that the salient object might not belong to a category of interest (shirt instead of person in figure 4b) or that the method fail...|
|||...MRSA provides bounding boxes (from multiple annotators) of the main salient element of each image....|
|||...Although having been trained with images with single salient objects, due to its convolutional nature the network can predict multiple salient regions in the Pascal images (as shown in figure 7)....|
|||...Global contrast based salient region detection....|
|||...Deepsaliency: Multi-task deep neural network model for salient object detection....|
|||...The secrets of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Minimum barrier salient object detection at 80 fps....|
||10 instances in total. (in cvpr2017)|
|108|Instance-Aware Image and Sentence Matching With Selective Multimodal LSTM|...el, named selective multimodal Long ShortTerm Memory network (sm-LSTM), that can recurrently select salient pairs of image-sentence instances, and then measure and aggregate their local similarities w...|
|||...In contrast, our model can automatically select salient pairwise image-sentence instances, and sequentially aggregate their local similarities to obtain global similarity....|
|||...[39] develop an attention-based caption model which can automatically learn to fix gazes on salient objects in an image and generate the corresponding annotated words....|
|||...For image and sentence matching, the words of sentence cannot be used as supervision information since we also have to select salient instances from the sentence to match image instances....|
|||...Ideally, T should be equal to the number of salient pairwise instances appearing in the image and sentence....|
|||...nstance(cid:173)aware Saliency Maps  To verify whether the proposed model can selectively attend to salient pairwise instances of image and sentence at different timesteps, we visualize the predicted ...|
|||...In addition, sm-LSTM-att always finishes attending to salient instances within the first two steps, and does not focus on meaningful instances at the third timestep any more....|
|||...Different from it, sm-LSTM focuses on more salient instances at all three timesteps....|
|||...It is mainly attributed to the fact that salient instances mostly appear in the cental regions  (a) 1-st timestep (b) 2-nd timestep (c) 3-rd timestep  Figure 6....|
|||...Our main contribution is proposing a multimodal context-modulated attention scheme to select salient pairwise instances from image and sentence, and a multimodal LSTM network for local similarity meas...|
||10 instances in total. (in cvpr2017)|
|109|Yang_PatchCut_Data-Driven_Object_2015_CVPR_paper|...Similar situations exist in salient object segmentation [28, 25, 8] in that most of algorithms work well when the images have high foreground-background color contrast, but work poorly in cluttered images....|
|||...Note that salient object segmentation masks may not be binary as subjects may disagree on the choice of salient objects as shown in Figure 10....|
|||...On the other hand, we use all the images in the training set to build our example database, and collect salient object segmentation ground truth in a similar way as in [22]....|
|||...e use the semantic labeling provided by [27] as full segmentation, and ask 6 subjects to select the salient object regions by clicking on them, so the saliency value for each segment is defined as the...|
|||...In this experiment, we present results for salient object segmentation using the PASCAL VOC 2010 dataset [10]....|
|||...Image  GT  CPMC GBVS  GBVS PatchCut soft  GBVS PatchCut  Figure 10: Comparing salient object segmentation results on PASCAL....|
|||...Global contrast based salient region detection....|
|||...The  secrets of salient object segmentation....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||9 instances in total. (in cvpr2015)|
|110|Li_Pairwise_Geometric_Matching_2015_CVPR_paper|...It serves to eliminate unreliable correspondences between salient points in a given pair of images, and is typically performed by analyzing the consistency of spatial transformations between the image...|
|||...An analysis of the state-of-the-art reveals that these approaches and methods are typically centered around the idea of detecting and verifying correspondences between salient points in a given pair of images....|
|||...(b) global rotation and scale relations between images encoded in the transformation of the matched salient points from individual correspondences, (c) rotation and scale relations between vectors for...|
|||...between visual feature statistics measured in different images around found salient points....|
|||...n for spatial verification, namely the rotation and scaling relations between the vectors formed by salient points involved in correspondences....|
|||...This representation typically involves detection of salient points in the image and representation of these points by suitable feature vectors describing local image regions around these points....|
|||...For instance, in the SIFT [17] scheme, which is widely deployed for this purpose, salient points are detected by a Difference of Gaussians (DOG) function applied in the scale space....|
|||...Given the images F and  F , and their salient points with indexes i and m and represented by feature vectors fi and  fm, respectively, we define the initial set C of correspondences cim between them a...|
|||...Local descriptors and visual words: we use Hessianaffine detector [18] to detect salient points and compute SURF descriptors [3] for these points....|
||9 instances in total. (in cvpr2015)|
|111|Zhang_Co-Saliency_Detection_via_2015_CVPR_paper|...Thus,  with the goal of effectively identifying common and salient                                                              * Corresponding author....|
|||...The proposed  FRA  strategy  the  overall  agreement  between  salient  regions  over  the  whole  image  and the whole  group with containing two phases, i.e....|
|||...But they are  different: co-saliency detection only focuses on detecting  common  salient  objects  while  the  similar  but  non-salient  background  might  in  co-segmentation....|
|||...A  unified  approach  to  salient  object   detection via low rank matrix recovery....|
|||...Automatic salient object segmentation based on context and  shape prior....|
|||...Background Prior Based Salient Object Detection via Deep  Reconstruction  Residual....|
|||...Frequency-tuned salient region detection....|
|||...Global  contrast  based  salient  region  detection....|
|||...Automatic salient  object extraction with contextual cue and its applications to  recognition  and  alpha  matting....|
||9 instances in total. (in cvpr2015)|
|112|Mai_Saliency_Aggregation_A_2013_CVPR_paper|...y  S(p) = P (yp = 1 S1(p), S2(p), .., Sm(p))   1 Z  m(cid:2)  i=1  (Si(p)),  (1)  value 1 if p is a salient pixel and 0 otherwise, and Z is a constant....|
|||...We also assign a binary random variable yp, which indicates whether the pixel is salient or not....|
|||...Like the pixel-wise aggregation method, we associate each node with a saliency feature vector x(p) = (S1(p), S2(p), , Sm(p)) and a binary random label yp, 1 for salient and 0 for non-salient....|
|||...Because aggregation is based solely on the saliency maps from individual methods, when all the individual methods fail to identify a salient region in an image, saliency aggregation will usually fail too....|
|||...In IEEE CVPR,  Frequency-tuned salient region detection....|
|||...In IEEE  Global contrast based salient region detection....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
||9 instances in total. (in cvpr2013)|
|113|Reflection Removal Using Low-Rank Matrix Completion|...e the warping is estimated by dominant transmitted scene, Tks are well matched to  5439  sume that salient edges belonging to the transmission image and the reflection image rarely appear at the same...|
|||...From (4), when p lies on a salient edge of the transmission image, it is highly probable that multiple warped images have a consistent large value of  T (p)  and negligible small   Rk(p) s at p. Howev...|
|||... using [16], where we see that the undesired reflection artifacts are successfully removed, but the salient contours and textures of the transmission image are also blurred....|
|||...Also, when p lies on a salient edge of the transmission image T ,   Ik(p) s have almost same large values across the multiple images and thus k(p)s become close to 1 for all the multiple images....|
|||...On the contrary, when p comes from a salient edge in one of the multiple reflection images,   Ik(p)  becomes large in only one image while  Gmin(p)  is relatively small, which results in a low reliabi...|
|||...We observe that the homogeneous regions and the salient edges of the transmission image are assigned high reliability values, while the edges of the reflection images are assigned low values, for exam...|
|||...Note that, some pixels located on the salient edges of the transmission image have relatively low reliability values in one image, but the corresponding pixels in another image usually have high relia...|
|||...However, as iteratively completing the low-rank matrix X, the resulting gradient map highlights most of the salient edges in the transmission image faithfully as shown in Fig....|
|||...5(c) include the salient edges of the transmission images as well as the reflection images together, however, the resulting optimal gradient maps preserve the edges of the transmission  5442  (a)  (b...|
||9 instances in total. (in cvpr2017)|
|114|Rudoy_Learning_Video_Saliency_2013_CVPR_paper|...Furthermore, accuracy and computation speed are improved by restricting the salient locations to a carefully selected candidate set....|
|||...Hence they typically focus on the single most salient point of each frame [24]....|
|||...In this way, we handle interframe dynamics of the gaze transitions, along with withinframe salient locations....|
|||...This method can find salient regions both in images and in videos....|
|||...Focusing on a sparse candidate set of salient locations allows us to model and learn these transitions explicitly with a relatively small computational effort....|
|||...Furthermore, some candidates capture less salient regions, such as the two bars....|
|||...Thus, next we incorporate motion cues into our salient candidate set....|
|||...These candidates cover most of the semantically salient regions in the frame....|
|||...Modeling gaze dynamics  Having extracted a set of candidates we next wish to select the most salient one....|
||9 instances in total. (in cvpr2013)|
|115|Li_Harvesting_Mid-level_Visual_2013_CVPR_paper|...In addition, for the airplane and the caterpillar, the salient windows naturally correspond to the parts....|
|||...In our experiment, the top 50 salient windows are used as the instances of a positive bag directly....|
|||...For large salient windows with  850850850852852  Figure 2....|
|||...For example, for the images of beach, the salient windows only cover patterns such as birds, trees, and clouds (see the salient windows of the beach image in Fig....|
|||...To avoid missing non-salient regions for a word, besides using  the salient windows, we also randomly sample some image patches from non-salient regions....|
|||...Each bag constructed in this way thus consists of patches from both salient and non-salient regions....|
|||...Top five salient windows for images from 12 words....|
|||...Except for the words sky, beach, and yard, the patterns of interest can be covered by a few top salient windows....|
|||...For objects such as caterpillar, bicycle, and conductor, parts can be captured by the salient windows....|
||9 instances in total. (in cvpr2013)|
|116|Automatic Discovery, Association Estimation and Learning of Semantic Attributes for a Thousand Categories|...We utilize online text corpora to automatically discover a salient and discriminative vocabulary that correlates well with the human concept of semantic attributes....|
|||... natural textual description that not only accounts for discrimination but also mines a diverse and salient vocabulary which correlates well with the human concept of semantic attributes....|
|||...Instead, we use textual description at the category level in form of encyclopedia entries to extract a salient and diverse set of attributes....|
|||... 1) We automatically analyze the articles in order to extract an attribute vocabulary with the most salient and discriminative words to describe these categories....|
|||...only colors or parts); and 3) represent salient semantic concepts understandable by humans....|
|||...While the top ranked topics capture salient concepts like music and dogs, the low ranked ones are obscure and have no particular theme....|
|||...Saliency An important aspect of semantic attributes is that they represent salient words with relatively clear semantic concepts, e.g....|
|||...C() favors salient words which will have a cost close to 1 while it punishes junk words which have a higher probability to appear in junk topics....|
|||...Attribute prediction  Having selected a set of salient attributes, we evaluate here the performance of our model in predicting these at 619  Figure 4: The ranking performance of the attribute embeddin...|
||9 instances in total. (in cvpr2017)|
|117|Perra_Adaptive_Eye-Camera_Calibration_2015_CVPR_paper|... of the device in a simple way by utilizing the users natural gaze in previous frames and observing salient areas of interest within the scene....|
|||...On the contrary, our approach finds the PoR by relating the users gaze to automatically detected salient regions of interest within the scene, breaking the requirement for a known scene geometry and k...|
|||...Our method takes this concept further by using salient areas found within the real world as an indication of the users gaze direction during our calibration process....|
|||...We expect that users will subconsciously fixate upon salient regions of the environment for multiple frames, meaning that the eye-device transformation will remain approximately constant for several frames....|
|||...Hence, our system leverages salient features in the environment and it can automatically calibrate itself using gaze data collected during normal operation of the device....|
|||...Recently, much work has been done to improve the detection of salient regions [12, 7, 24, 6] in an image that closely correspond to the points where an average user is most likely inclined to look....|
|||...solute differences in the x and y directions between the projected visual axis, vj , and the nearby salient interest point, sj , for all frames within a short window of frames preceding time t. For a ...|
|||...We used the specific class of faces for the salient region detection, but this setup could easily be generalized using saliency maps [12, 7, 24], which would also impose similar constraints for the ey...|
|||...Image signature: Highlighting sparse salient regions....|
||9 instances in total. (in cvpr2015)|
|118|Huang_Automatic_Thumbnail_Generation_ICCV_2015_paper|... thumbnails, have largely been ignored in previous methods, which instead are designed to highlight salient content while disregarding the effects of downsizing....|
|||...Many of them operate by extracting a rectangular region that contains the most visually salient part of a photograph....|
|||...Another way to remove less salient image content is through image retargeting [4, 26, 36, 28], which downsizes images through operations such as seam carving, local image warping, and rearranging spat...|
|||...The saliency weight puts greater emphasis on salient regions, whose color properties are more critical to preserve....|
|||...Saliency Ratio A thumbnail is more representative if it contains more of the salient content of the original photo....|
|||...It can be seen that the saliencybased method (SOAT) discards less salient parts of images, but may also remove important contextual information, making the thumbnails less suitable as an image index....|
|||...SOAT may also be affected by salient background regions....|
|||...Global contrast based salient region detection....|
||8 instances in total. (in iccv2015)|
|119|Cheng_BING_Binarized_Normed_2014_CVPR_paper|...[11, 14] proposed a salient object detection and segmentation method based on region contrast analysis and iterative graph based segmentation....|
|||...Such salient object segmentation for simple images achieved great success in image scene analysis [15, 58], content aware image editing [13, 56, 60], and it can be used as a cheap tool to process larg...|
|||...Frequency-tuned salient region detection....|
|||...Efficient salient region detection with soft image abstraction....|
|||...Global contrast based salient region detection....|
|||...The secrets  of salient object segmentation....|
|||...Learning to detect a salient object....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||8 instances in total. (in cvpr2014)|
|120|cvpr18-A Common Framework for Interactive Texture Transfer|... We propose a method that extracts salient structure regions and conveys structure information in the source image to the target....|
|||...Thus, it is difficult for internal patches with salient structure to be correctly synthesized by only relying on semantics....|
|||...As shown in Figure 3,  three main steps constitute our pipeline including salient internal structure extraction, structure propagation and guided texture transfer....|
|||...Internal Salient Structure Extraction  Some salient texture details in the internal region of a source semantic map are prone to being lost or suffering disorder in the synthesized target image....|
|||...Saliency detection is performed to mark the salient regions of the source stylized image, which contain complex textural structure or curvilinear structure such as an edge or contour in an image....|
|||..., and N i p is the coordinate of pixel i related to p.  cos   sin  sin   3.3.3 Structure Guide  The salient structure in Ssty ignored by semantic map is pre-projected as Tstruct....|
|||...Parameter l and  in Equation (1) control the salient degree of structural pixels....|
|||...More specifically, we introduced a structure guidance acquired by automatically extracting salient regions and propagating structure information....|
||8 instances in total. (in cvpr2018)|
|121|Le_PDM-ENLOR_Learning_Ensemble_2013_CVPR_paper|...Specifically, a set of salient reference points are first selected to be used as explanatory variables of the regression models....|
|||...[29] proposed to detect the salient points based on prior knowledge about the contrast of the contour and reconstruct the full shape from the detection of salient points....|
|||...Similar to [29], the PASM-CTX and PDM-ENLOR detect the salient points and reconstructs the shape based on the guidance of the salient points to account for the large errors in detection....|
|||...However, our methods learn the set of salient points from training data in advance and exploits information from additional supporting salient points, which may not belong to the boundaries....|
|||...In addition, PDM-ENLOR uses salient points selectively in the ensemble of multiple models to provide further flexibility at local level....|
|||...The similarity-saliency score of a salient point should be high (the higher score, the more salient) and greater than 1.0 (i.e., the average similarity in NC is higher than that in NP )....|
|||...A set of salient points are selected using the following steps....|
|||...A set of selected salient reference points is used to construct the models to minimize the errors in fitting due to unreliable model points....|
||8 instances in total. (in cvpr2013)|
|122|Ahmed_Semantic_Object_Selection_2014_CVPR_paper|...Unlike saliency methods, our method can select objects that are small and potentially not salient in the input image as well as objects in images with several salient objects....|
|||...In [28], the saliency is determined by optimizing an energy function which encourages pixels to be salient if they are contained in regions that have high contrast to all other regions....|
|||... address the general semantic object selection problem where the object of interest is not the most salient object....|
|||...The closeness in performance is due to the fact that the MSRC dataset contains images with a salient target object and uniform background....|
|||...This acts as a boon to Object Discoverys approach which is tuned to work well in cases where object is the most salient object in the image....|
|||...  To prove our claim that our method is more general and works well when the object is not the only salient object in the image, we test the performance of our method on the Object Discovery dataset....|
|||...Since the images contain objects which are not salient in the image (more realistic images), our approach performs better than Object Discovery....|
|||...Global contrast based salient region detection....|
||8 instances in total. (in cvpr2014)|
|123|cvpr18-Emotional Attention  A Study of Image Sentiment and Visual Attention|...Building on the emotion prioritization effect, we propose a deep neural network (DNN) that learns the relative importance of the salient regions within an image....|
|||...(b, c) Images illustrate how objects in strong emotions (outlined in blue), such as the crying face and broken card, are more salient than neutral/less emotional stimuli (outlined in gray)....|
|||...ual saliency: As suggested in [43, 11], NSS and IG take into account the relative importance of the salient regions, thus are the best evaluation measures for contextual saliency....|
|||...This suggests the effectiveness of learning the relative weights of salient regions inside an image through the proposed subnetwork....|
|||...To show this, we identify all images with co Figure 10: The most salient patches predicted by N-CASNet (yellow sqaure) and CASNet (red square)....|
|||...CASNet correctly prioritizes the most salient faces within an image (top row), people/body parts over other objects (middle row), and the most salient non-human objects....|
|||...ritization effect, we develop a novel DNN (CASNet) that encodes the relative importance of multiple salient regions and accounts for contextual importance for human attention....|
|||...Automatic image annotation by using concept-sensitive salient objects for image content representation....|
||8 instances in total. (in cvpr2018)|
|124|Fong_Interpretable_Explanations_of_ICCV_2017_paper|...(2)  [15]s This formulation provides an interpretation of saliency maps, which visualize the gradient S1(x0) = f (x0) as an indication of salient image regions....|
|||...7, top;   [0.2, 0.6] tends to cover the salient part identified by the learned mask)....|
|||...These results show that our method finds a small salient area that strongly impacts the network....|
|||...For all 76 classes, the mean average intensity of eyes were lower and thus more salient than that of feet (see supplementary materials for class-specific results)....|
|||...Localization and pointing  From qualitatively examining learned masks for different animal images, we noticed that faces appeared to be more salient than appendages like feet....|
|||...Because the deletion game is meant to discover minimal salient part and/or spurious correlation, we do not expect it to be particularly competitive on localization and pointing but tested them for com...|
|||...Second, for energy thresholding [2], we threshold heatmaps by the percentage of energy their most salient subset covered with   [0 : 0.05 : 0.95]....|
|||...We noticed qualitatively that our method did not produce salient heatmaps when objects were very small....|
||8 instances in total. (in iccv2017)|
|125|Zhou_Time-Mapping_Using_Space-Time_2014_CVPR_paper|...temporally resample based on frame importance, and (3) temporal filters to enhance the rendering of salient motion....|
|||...To accomplish this goal, we devised methods for: (1) predicting what is interesting or salient in the video, (2) retiming to retain as many salient frames as possible while minimizing time distortion,...|
|||...In other words, a spatio-temporal region may be salient because its color or motion is different from its neighbors....|
|||...The goal is to slightly slow down the action to focus on highly salient regions at the cost of slightly speeding up other portions....|
|||...r high-speed video is to combine the advantages of box and delta filters, i.e., retain the blur for salient motion, while keeping foreground as clear in the original as possible....|
|||...Automatic salient object  segmentation based on context and shape prior....|
|||...Saliency filters: Contrast  based filtering for salient region detection....|
|||...Segmenting salient objects from  images and videos....|
||8 instances in total. (in cvpr2014)|
|126|cvpr18-Salience Guided Depth Calibration for Perceptually Optimized Compressive Light Field 3D Display|...Contrast Enhanced Salient Segmentation  After completing the preprocessing work for super-pixel segmentation that is focusness map generation chosen and color image chosen, we integrate focusness back...|
|||...This indicates that our algorithm is capable of locating the most salient regions with a high confidence....|
|||...Note that the simulated result of the proposed initialization optimization (IO) method is better than any other fixed configuration for the salient object....|
|||...Global contrast based salient region detection....|
|||...Learning to detect a salient object....|
|||...A simple method for detecting salient regions....|
|||...A unified approach to salient object detection via low rank matrix recovery....|
|||...Fusing disparate object signatures for salient object detection in video....|
||8 instances in total. (in cvpr2018)|
|127|Borji_Analysis_of_Scores_2013_ICCV_paper|...Some works have compared salient object detection and region-of-interest algorithms [42]....|
|||...A closely related field to saliency modeling is salient region detection....|
|||...While the goal of the former is to predict locations that grab attention, the latter attempts to segment the most salient object or region in a scene....|
|||...Evaluation is often done by measuring precision-recall of saliency maps of a model against ground truth data (explicit saliency judgments of subjects by annotating salient objects or clicking on locations)....|
|||...[45] showed that initial fixations were more likely to be on emotional objects than more visually salient neutral ones....|
|||...A separate analysis over the Kootstra dataset showed that models have difficulty in saliency detection over nature stimuli where there are less distinctive and salient objects (See supplement)....|
|||...(10 bins), vectorized saliency map of size 2015 (a vector of size 1300) and coordinates of ten most salient points obtained by applying IOR to the saliency map (a vector of size 1  20)....|
|||...Modeling attention to salient proto-objects....|
||8 instances in total. (in iccv2013)|
|128|Niu_Hierarchical_Multimodal_LSTM_ICCV_2017_paper|... embedding that maps not only full sentences and whole images but also phrases within sentences and salient regions within images into a multimodal embedding space....|
|||...0 31.7 29.6 42.5 43.7 44.7 42.4 64 68.1  500 25 29 15 14 12.4 15 5 4  ticular, after detecting some salient image regions/object proposals, we can extract the visual features from them, and retrieve s...|
|||...Generally, after detecting some salient image regions/object proposals, our model can retrieve subtle and detailed phrases to describe them....|
|||...As a result, 4467 salient regions and 18724 corresponding phrases are collected in total....|
|||...In contrast, our method targets a salient image region (e.g., which is marked by red box), and produce detailed and subtle descriptions such as a white and gray cat with a strip tail....|
|||...Note that (d) is a failure example, it is mainly due to that the salient regions do not cover the objects mentioned in its caption....|
|||... embedding, which can jointly learn the embeddings of all the sentences, their phrases, images, and salient image regions....|
|||...Bedsides, our method can produce detailed and diverse phrases to describe image salient regions....|
||8 instances in total. (in iccv2017)|
|129|cvpr18-Learning Superpixels With Segmentation-Aware Affinity Loss|...d recent deep learning [23, 13, 9] frameworks with applications to a wide range of problems such as salient object detection [29, 32, 13], and semantic segmentation [23, 9], to name a few....|
|||...erformance improvements in vision tasks that rely on superpixels, such as semantic segmentation and salient object detection....|
|||...For this study, we choose existing semantic segmentation and salient object detection techniques, and replace the superpixels used in those techniques with our SEAL-ERS superpixels....|
|||...Table 2: Superpixels for salient object detection....|
|||...0.5070  0.1877  0.4897  0.1633  0.5237  0.1841  0.4955  stituting SLIC superpixels used in existing salient object detection algorithms with our superpixels....|
|||...We experiment with two salient object detection algorithms: Saliency Optimization (SO) [32] and Graph-based Manifold Ranking (GMR) [29]....|
|||...These results on semantic segmentation and salient object detection demonstrate the potential of using learned superpixels for downstream vision tasks....|
|||...Numerous salient object detection algorithms are based on superpixels....|
||8 instances in total. (in cvpr2018)|
|130|Wang_Saliency-Aware_Geodesic_Video_2015_CVPR_paper|...Based on this argument, we opt to use the geodesic distance to discriminate the visually salient regions from backgrounds and measure their likelihoods for foreground....|
|||...P recision is defined as the percentage of salient pixels correctly assigned, while recall measures the percentage of salient pixel detected....|
|||...ious saliency maps highlight salient regions in images....|
|||...The minimum recall value of the proposed method does not drop to zero because the corresponding saliency maps are able to effectively detect the salient region with strong response....|
|||...Moreover, our saliency method achieves the best performance up to a precision rate above 0.8, which indicates our saliency maps are more precise and responsive to the salient regions....|
|||...In most cases, saliency methods [13, 28, 14] for video are able to accurately locate the salient objects, which perform better than the method [31] for image saliency detection....|
|||...Saliency filters: Contrast based filtering for salient region detection....|
||7 instances in total. (in cvpr2015)|
|131|cvpr18-Tags2Parts  Discovering Semantic Regions From Shape Tags|...With the rise of deep neural networks, researchers observed that neurons in a classification network often activate on salient objects [19]....|
|||...To correct this, we simply mirror inferred salient regions on both sides of the symmetry plane....|
|||...Generally, detection of larger salient parts is aided by larger average pooling kernels (Figure 5)....|
|||...In these validation experiments, we test if our WU-Net architecture successfully detects salient parts that distinguish one category of shapes from another....|
|||...es only on a userselected tag (our simple implementation uses weighted average distance between the salient voxels, after aligning centroids of salient regions)....|
|||...Third, we demonstrate that our method facilitates better thumbnail creation (Figure 10) by focusing on salient regions that correspond to specific tags....|
|||...ng to leverage natural language processing in addition to geometric analysis to automatically infer salient shape tags and corresponding parts from free-form shape descriptions provided by people....|
||7 instances in total. (in cvpr2018)|
|132|Temporal Attention-Gated Model for Robust Sequence Classification|...Our proposed model first employs an attention module to extract the salient frames from the noisy raw input sequences, and then learns an effective hidden representation for the top classifier....|
|||...TAGMs attention module automatically localizes the salient observations which are relevant to the final decision and ignore the irrelavant (noisy) parts of the input sequence....|
|||...fication models, TAGM benefits from the following advantages:   It is able to automatically capture salient parts of the input sequences thereby leading to better performance....|
|||...Inspired by this setup, our TAGM model also employs a gate to filter out the noisy time steps and preserve the salient ones....|
|||...The red lines indicate the ground-truth of salient segments....|
|||...Our model still shows encouraging results since it is quite a challenging task for TAGM to capture salient sections for 20 events with complex scenes simultaneously....|
|||...Figure 6 shows some examples where TAGM correctly locates the salient subsequences by the attention weights....|
||7 instances in total. (in cvpr2017)|
|133|Hadfield_Hollywood_3D_Recognizing_2013_CVPR_paper|...These interest points detect salient image locations, for example using separable linear filters [7] or spatio-temporal Harris corners [13]....|
|||...Firstly salient points are detected in using a range of detection schemes which incorporate the depth information, as discussed in section 5....|
|||...Next, feature descriptors are extracted from these salient points, using extensions of 2 well known techniques, discussed in detail in section 7....|
|||...Interest Point Detection  The additional information present in the depth data may be exploited during interest point extraction, in order to detect more salient features, and discount irrelevant detections....|
|||...Descriptors are extracted only in salient regions (found through interest point detection) and are composed of a Histogram of Oriented Gradients (HOG) G, concatenated with a Histogram of Oriented Flow...|
|||...This provides a descriptor  of the visual appearance and local motion around the salient point at I(u, v, w)....|
|||...Interest Point Threshold Results  Different interest point operators produce very different response strengths, meaning the optimal threshold for extracting salient points varies....|
||7 instances in total. (in cvpr2013)|
|134|Fried_Finding_Distractors_In_2015_CVPR_paper|...Computational saliency methods can be roughly divided into two groups: human fixation detection [13, 11, 15] and salient object detection [7, 6, 22, 20]....|
|||...Another interesting work [28] focused on detecting and de-emphasizing distracting texture regions that might be more salient than the main object....|
|||...For instance, many methods just try to crop around the most salient object....|
|||...We also detect features that distinguish main subjects from salient distractors that might be less important, such as objects near the image boundary....|
|||...This comparison is rather unfair since saliency methods try to predict all salient regions and not just distractors....|
|||...Fusing generic objectness and visual saliency for salient object detection....|
|||...Global contrast based salient region detection....|
||7 instances in total. (in cvpr2015)|
|135|Learning Video Object Segmentation From Static Images|...Without the additional input channel, this pixel labelling convnet was trained offline as a salient object segmenter and fine-tuned online to capture the appearance of the object of interest....|
|||...This model obtains competitive results (72.5 mIoU) on DAVIS, since the object to segment is also salient for this dataset....|
|||...ithout using guidance from the previous frame mask as these two datasets have a weaker bias towards salient objects compared to DAVIS....|
|||...As one could expect, MaskTrack+Flow+CRF better discriminates cases involving color ambiguity and salient motion....|
|||...Global contrast based salient region detection....|
|||...The  secrets of salient object segmentation....|
|||...Design and perceptual validation of performance measures for salient object segmentation....|
||7 instances in total. (in cvpr2017)|
|136|Matzen_BubbLeNet_Foveated_Imaging_ICCV_2015_paper|...In this manner, our approach efficiently learns both network weights as well as the most salient parts of each training image....|
|||...We then cluster these salient image regions to derive a set of visual elements for a given dataset....|
|||...These regions are identified as salient for a given category [27, 29]....|
|||... classification performance; an additive method like ours is more suitable for identifying the most salient regions....|
|||...Recent work has used deep learning to attend to salient regions of an image [22, 15], and applied this to applications such as object recognition [2] and automatic caption generation [25]....|
|||...Now that we have a CNN trained on bubble images, we can use it to discover salient patches....|
|||...The performance drops significantly, and much more if the occluder is guided by our bubble method than by random guessing, suggesting we are locating salient features....|
||7 instances in total. (in iccv2015)|
|137|Khatoonabadi_How_Many_Bits_2015_CVPR_paper|...elves, which would only consume the power and bandwidth necessary to transmit video when faced with salient or anomalous events....|
|||...This is defined with respect to a binary classification problem, where salient blocks of 16  16 pixels belong to class 1 and non-salient blocks to class 0....|
|||...the block is labeled salient if o(n) > 0:5 and non-salient otherwise....|
|||...Final saliency map  The procedure above produces the most probable, a posteriori, map of salient block labels....|
|||...To emphasize the locations with higher probability of attracting attention, the OBDL of a block declared salient (non-salient) by the MRF is increased (decreased) according to the OBDLs in its neighborhood....|
|||...In this way, a block n labeled as salient by the MRF inference is assigned a saliency equal to the largest feature value within its neighborhood, weighted by its distance from n. On the other hand, fo...|
|||...Comparing salient point detectors....|
||7 instances in total. (in cvpr2015)|
|138|Amir_Sadeghian_CAR-Net_Clairvoyant_Attentive_ECCV_2018_paper|...st visual attention model that can predict the future trajectory of an agent while attending to the salient regions of the scene....|
|||...Related work from Xu and Gregor [17, 23] introduces attention based models that learn to attend the salient objects related to the task of interest....|
|||...Then, a visual attention module computes a context vector ct representing the salient areas of the image to attend at time t. Finally,  CAR-Net: Clairvoyant Attentive Recurrent Network  5  in the rec...|
|||...The vector (ht) is then applied to feature vectors A (through a function fatt), resulting in a context vector ct+1 that contains the salient image features at time step t + 1:  ct+1 = fatt(A, (ht))....|
|||...On racing track datasets (Car-Racing and F1), we expect the region of the road close to the car to contain salient semantic elements....|
|||...It shows that attending only close to the agent would not capture all salient semantics so attention grids reach ahead....|
|||...We observe that both single and multi-source attention mechanisms are consistent with the predicted positions over time, as they attend to the salient parts of the scene e.g., the curve in front of the car....|
||7 instances in total. (in eccv2018)|
|139|cvpr18-Context Contrasted Feature and Gated Multi-Scale Aggregation for Scene Segmentation|...ct detection [45, 46, 15, 37,  Figure 1: Scene segmentation refers to labeling each pixel including salient objects, inconspicuous objects and stuff....|
|||...However, the various forms of objects/stuff (e.g salient or inconspicuous, foreground or background) and the existence of multi-scale objects (e.g the multi-scale cows in third image) make it challeng...|
|||... every pixel to one of many classes including stuff and object classes, thus not only the dominated salient objects but  2393  also the stuff and inconspicuous objects should be parsed well....|
|||...Meanwhile, due to the various forms of objects/stuff in scene segmentation, a pixel may belong to salient object, inconspicuous object or stuff....|
|||... directly applying DCNN on scene segmentation, inconspicuous objects and stuff will be dominated by salient objects and its information will be somewhat weakened or even disregarded, which is contradi...|
|||...However, contexts often have smooth representation and are dominated by features of salient objects, which is harmful for labeling inconspicuous objects and stuff....|
|||...A robust segmentation network should be able to handle huge scale variation of objects and detect inconspicuous objects/stuff from images overwhelmed by other salient objects....|
||7 instances in total. (in cvpr2018)|
