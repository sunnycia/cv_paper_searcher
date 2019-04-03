|Index|Title|sentenct|
|---|---|---|
|0|STD2P_ RGBD Semantic Segmentation Using Spatio-Temporal Data-Driven Pooling|Lstm-cf: Unifying context modeling and fusion with lstms for rgb-d scene labeling.|
|||Semanticfusion: Dense 3d semantic mapping with convo-lutional neural networks.|
|1|cvpr18-Context Contrasted Feature and Gated Multi-Scale Aggregation for Scene Segmentation|However,inprevious works, such as [38, 21, 40, 51, 7, 43], the scoremaps of skip layers are integrated via a simple sum fusionand hence the different importance of different scales areignored.|
|||As a selectionmechanism is embedded in the multi-scale fusion, moreskip layers can participate in the aggregation to providerich information for selection.|
|||The seminalwork FCN [38] introduced the skip layers to locally classifymulti-scale feature maps and aggregate their predictionsvia sum fusion.|
|||With gated sum fusion, the networkcan exploit more skip layers from richer scale features inDCNN and customize a suitable integration of differentscale features.|
|||However, in previous works such as [38, 21, 40, 51, 7],the score maps of skip layers are mainly integrated viasum fusion that does not take into account the individualdifferences of these inputs.|
|||Sum fusion can only non-selectively collectthe score maps from different skiplayers, but some of them may not be appropriate orIf these score maps are aggregatedeven be harmful.|
|||With gated sum fusion, the networkcan customize a suitable aggregation choice of score mapsaccording to the information of images, corresponding tochoose which scale of feature is better and more desirable.|
|||More importantly, with gated sum fusion, we can addmore skip layers to extract richer scale information withoutposing problem of inapposite results.|
|||Sum fusion dose not take intoaccount the individual characteristic of different inputsand could only indiscriminately fuse all the inputs.|
|2|Joseph_DeGol_Improved_Structure_from_ECCV_2018_paper|Moulon, P., Monasse, P., Marlet, R.: Global fusion of relative motions for ro-bust, accurate and scalable structure from motion.|
|3|cvpr18-Enhancing the Spatial Resolution of Stereo Images Using a Parallax Prior|Stereo fusion: Combining re-fractive and binocular disparity.|
|4|cvpr18-Deeply Learned Filter Response Functions for Hyperspectral Reconstruction|There are also anumber of fusion based super-resolution algorithms to boostthe spatial resolution by using a high-resolution grayscale[18, 35] or RGB [8, 20, 1, 22, 23, 2] image.|
|5|cvpr18-Im2Flow  Motion Hallucination From Static Images for Action Recognition|Convolutionaltwo-stream network fusion for video action recognition.|
|6|cvpr18-Learning Superpixels With Segmentation-Aware Affinity Loss|For comparison,we include a baseline model that predicts both horizontaland vertical affinities, uses 7  7 filters instead of 1  7 andwithout using Canny edge fusion.|
|7|Hyo_Jin_Kim_Hierarchy_of_Alternating_ECCV_2018_paper|Existing methodsorganize classes into coarse categories, either based on the semantic hierarchy[12,15,22,58] or the confusion matrix of a trained classifier [53,57].|
|||[37] use the confusion matrix of a trained classifier to groupclasses into coarse categories.|
|||3.2 Discovering the areas of confusionWe want to partition the input data based on their high-level appearance fea-tures, and not by their categorization, thus allowing samples belonging to theHierarchy of Alternating Specialists for Scene Recognition7same class to fall into different clusters.|
|||We also evalu-ated the performance of fused features, one with early fusion that concatenatestwo representations before the last fully connected layer, and the other with latefusion where the predictions of the two architectures are averaged.|
|||The earlyfusion did not yield competitive classification accuracy.|
|||On the other hand, thelate fusion (Fusion in Table 1 and 2) achieves better performance than using eachrepresentation separately, however, does not reach the classification accuracy ofour proposed alternating architecture.|
|8|Group-Wise Point-Set Registration Based on Renyi's Second Order Entropy|Tasks such as image registration, facerecognition, object tracking, image stitching or 3D objectfusion all require registration of features/point-sets.|
|9|cvpr18-Parallel Attention  A Unified Framework for Visual Object Discovery Through Dialogs and Queries|Mutan:Multimodal tucker fusion for visual question answering.|
|10|Kejie_Li_Efficient_Dense_Point_ECCV_2018_paper|The fusion of multiple partial-view point clouds escalatesthe noise.|
|||[21] have realized that the fusion of multiplepartial surfaces generates noisy points and thus developed a multi-view con-sistency supervision based on binary masks and depths to address this issue.|
|11|cvpr18-Bidirectional Attentive Fusion With Context Gating for Dense Video Captioning|We solve this problem by representing each eventwith an attentive fusion of hidden states from the proposalmodule and video contents (e.g., C3D features).|
|||We detail the fusion methods in Section 3.2.2.|
|||proposed to learn a task-driven fusion modelby dynamically fusing complementary features from multi-ple channels (appearance, motion) [49].|
|||Then we describe a noveldynamic fusion method.|
|||We denote H as context vectors,E as event clip features, TDA as temporal dynamic at-tention fusion, and CG as context gate, respectively.|
|||Qualitative dense-captioning analysis for model withoutor with event clip fusion.|
|||The fusion mechanism allows the sys-tem to pay more attention to current event while simulta-neously referring to contexts, and thus can generate moresemantic-related sentences.|
|||In contrast, the system with-out event clip fusion generally tends to make more seman-tic mistakes, either incorrect (Fig.|
|||ConclusionIn this paper we identified and handled two challengeson the task of dense video captioning, which are contextfusion and event representation.|
|||Task-driven dynamic fusion: Reducing ambiguity in video description.|
|12|Chenglong_Li_Cross-Modal_Ranking_with_ECCV_2018_paper|Keywords: Visual tracking, Information fusion, Manifold ranking, Softcross-modality consistency, Label optimization1IntroductionThe goal of RGB-T tracking is to estimate the states of the target object invideos by fusing RGB and thermal (corresponds the visible and thermal in-frared spectrum data, respectively) information, given the initial ground truthbounding box.|
|||Cross-Modal Ranking for Robust RGB-T Tracking3First, we propose a general scheme for effective multimodal fusion.|
|||The RG-B and thermal modalities are heterogeneous with different properties, and thehard consistency [11,4] between these two modalities may be difficult to performeffective fusion.|
|||The proposed cross-modality consistency is general, and can be applied toother multimodal fusion problems.|
|||It is beneficial to the effectivefusion of visible and thermal information in our method.|
|||4 shows that our tracker per-forms well against the state-of-the-art RGB-T methods, which suggest that theproposed fusion approach is effective.|
|||SGT [5] is better than our tracker in PRmainly due to adaptive fusion of different modalities by introducing modalityweights, but performs weaker than ours in SR.RGBT210 Evaluation.|
|||: The effect of pixel-level fusion on object tracking in multi-sensorsurveillance video.|
|||Wu, Y., Blasch, E., Chen, G., Bai, L., Ling, H.: Multiple source data fusion viasparse representation for robust visual tracking.|
|13|Disentangled Representation Learning GAN for Pose-Invariant Face Recognition|Different from prior work, our fusion isconducted within a unified framework.|
|||3, com-paring (ii) and (iii), using the coefficients learned by thenetwork for representation fusion is superior over the con-ventional score averaging.|
|||Further, theproposed fusion scheme via learnt coefficients is superiorto the averaged cosine distances of representations.|
|||We extend GAN with a fewdistinct novelties, including the encoder-decoder structuredgenerator, pose code, pose classification in the discrimina-tor, and an integrated multi-image fusion scheme.|
|14|Chenyang_Si_Skeleton-Based_Action_Recognition_ECCV_2018_paper|Then, we calculate the relationrepresentation rtk at time step t via:k = rt1rtk + stk(6)The residual design of Eqn.6 aims to add the relationship features between eachpart based on the individual part features, so that the representations containthe fusion of both features.|
|15|cvpr18-Multi-Shot Pedestrian Re-Identification via Sequential Decision Making|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|16|cvpr18-Two Can Play This Game  Visual Dialog With Discriminative Question Generation and Answering|For instance, in [6], three models are formulated basedon - late fusion, attention based hierarchical LSTM, andmemory networks.|
|||A baseline for simple models is setusing the late fusion architecture.|
|||While late fusion hasa simple architecture, the other two complex models ob-tain better performance.|
|||Similarly, to obtain an embedding for a question-answer pair, we use a question and an answer LSTM to en-code all question-answer pairs in the history set H. Uponencoding the question and the answer of a question-answer5756Figure 2: Overview of the proposed approach: Joint similar-ity scoring of answer option and fusion of all input features.|
|||We performboth fusion and similarity scoring together using a multi-layer perceptron network.|
|||As mentioned before,unlike previous methods, we perform similarity scoring andfeature fusion jointly.|
|||Network TrainingTo describe training more formally, let Fw(Oi) denotethe score for answer option i obtained from the similarityscoring + fusion network, and let w denote all the param-eters of the architecture illustrated in Fig.|
|||Our similarity scoring + fusion (SF) performs bestin all three scenarios.|
|||This in-cludes models proposed in [6], based on late fusion (LF), hi-erarchical LSTM net (HRE), and memory networks (MN).|
|||Mutan:Multimodal tucker fusion for visual question answering.|
|18|cvpr18-PIXOR  Real-Time 3D Object Detection From Point Clouds|Incomparison, we build a YOLO-like [27] baseline detectorwith a customized backbone network on ATG4D, and addobject anchors and multi-scale feature fusion to further im-prove the performance.|
|19|cvpr18-Transferable Joint Attribute-Identity Deep Learning for Unsupervised Person Re-Identification|Importantly, we designa progressive knowledge fusion mechanism by introducingan Identity Inferred Attribute (IIA) regularisation space formore smoothly transferring the global identity informationinto the local attribute feature representation space.|
|||To establish a channelfor knowledge fusion, we introduce the Identity InferredAttribute (IIA) space (Figure 1(c)) designed for transfer-ring the re-id discriminative information from the IdentityBranch to the Attribute Branch where two-source infor-mation is synergistically integrated in a smoother manner.|
|||Instead, we present an alternative pro-gressive scheme for more effective multi-source knowledgefusion as described below (see evaluations in Sec.|
|||Identity Inferred Attribute Space We introduce an inter-mediate Identity Inferred Attribute (IIA) Space for achiev-ing the knowledge fusion learning on attribute and identitylabels in a softer manner (Fig.|
|||The IIA space isjointly learned with the two branches while being exploitedto perform information transfer and fusion from the identitybranch to the attribute branch simultaneously.|
|||This schemeallows for both consistent and cumulative knowledge fusionin the whole training course.|
|||This suggests the overall performance advantagesof the proposed TJ-AIDL in the capability of multi-source(attribute and identity) information extraction and fusion forcross-domain unsupervised re-id matching.|
|||Comparisons to Alternative Fusion MethodsWe compare the TJ-AIDL with two multi-source fusionmethods: (a) Independent Supervision: Independently train adeep CNN model for either attribute or identity label in thesource domain and use the concatenated feature vectors ofthe two models for re-id matching in the target domain.|
|||Table 2 shows that: (1) The TJ-AIDL outperforms bothalternative fusion methods.|
|||Comparing different multi-source fusion methods.|
|||We also compared the TJ-AIDLmodel with popular multi-supervision fusion methods andprovided detailed component analysis with insights into theperformance gain of our model design.|
|20|Hongmei_Song_Pseudo_Pyramid_Deeper_ECCV_2018_paper|Recently, with the popularity ofdeep neural network, various deep learning based image salient object detec-tion models were proposed, e.g., multi-stream network with embedded super-pixels [22, 25], recurrent module [26, 40], and multi-scale and hierarchical featurefusion [16, 51, 36], etc.|
|||DB-LSTM per-forms better than ConvLSTM and B-ConvLSTM due to its deeper fusion ofbidirectional information.|
|21|cvpr18-Defocus Blur Detection via Multi-Stream Bottom-Top-Bottom Fully Convolutional Network|Finally, we design a fusion and recurrent reconstructionnetwork to recurrently refine the preceding blur detectionmaps.|
|||Then, we develop a fusion and recursivereconstruction network (FRRNet) to recursively refine thepreceding blur detection maps.|
|||[8] propose a spatially-varyingblur detection method based on a high-frequency multi-scale fusion and sort transform of gradient magnitudes todetermine the level of blur at each location.|
|||Finally, FRRNet consisting of the fusion network(FNet) and recursion reconstruction network (RRNet) is used torefine the predicted DBD maps, generating the final DBD map.|
|||Then, the fusion and recurrent reconstructionnetwork (FFRNet) is described in Section 3.2.|
|||Comparison of multi-stream DBD map fusion results.|
|||F) [20], spectral and spatial approach (SS) [27], deep andhand-crafted features (DHCF) [17], kernel-specific featurevector (KSFV) [16], local binary patterns (LBP) [31] andhigh-frequency multi-scale fusion and sort transform of gra-dient magnitudes (HiFST) [8].|
|22|Baris_Gecer_Semi-supervised_Adversarial_Learning_ECCV_2018_paper|Convolutional fusion network for faceverification in the wild.|
|23|Zhenyu_Zhang_Joint_Task-Recursive_Learning_ECCV_2018_paper|Actually, themethods in [32, 53] used the depth ground truth as the input, and carefullydesigned some depth-RGB feature fusion strategies to make the segmentationprediction better benefit from the depth ground truth.|
|||Cheng, Y., Cai, R., Li, Z., Zhao, X., Huang, K.: Locality-sensitive deconvolutionnetworks with gated fusion for rgb-d indoor semantic segmentation.|
|||Li, Z., Gan, Y., Liang, X., Yu, Y., Cheng, H., Lin, L.: Lstm-cf: Unifying contextmodeling and fusion with lstms for rgb-d scene labeling.|
|||Seong-Jin, P., Ki-Sang, H., Seungyong, L.: Rdfnet: Rgb-d multi-level residualfeature fusion for indoor semantic segmentation.|
|24|Primary Object Segmentation in Videos Based on Region Augmentation and Reduction|Discovering primary objects in videos by saliencyfusion and iterative appearance estimation.|
|25|Lan_Wang_PM-GANs_Discriminative_Representation_ECCV_2018_paper|To incorporate all feature maps into a high-level representation,the sum fusion model in [9] is applied to compute the sum of T feature maps atthe same spatial location i, j and feature channel d:Inf , .|
|||It consists of a fully-connected layer followed by a softmaxlayer which takes the fusion of the feature map of both the partially-availabledata channel and generated missing channel and finally outputs the category-level confidences.|
|||To fuse these two feature maps, a convolutional fusion modelin [9] is applied to automatically learn the fusion weights:fconv = fcat  f + b,(7)where f are filters with dimensions 1  1  2D  D, and fcat denotes the stack oftwo feature maps at the same spatial locations (i, j) across the feature channelsd:f (i,j,2d)cat= f (i,j,d)Inf,f (i,j,2d1)cat= f (i,j,d)g,where fg denotes the generated fake feature map G(fInf , z).|
|||In the case of two-modality fusion, we directly concatenate the featuresof infrared channel and RGB channel.|
|||The evaluation results of different features on different channels and theirfusion on the proposed datasetMethodInfrared Channel OrgDescriptor Accuracy(%)55%RGB ChannelFusion61%49%Optical Flow 69.67%MHIOrgOptical Flow 78.66%65.33%MHIOrg55.33%Optical Flow 80.67%MHI68.67%As shown in Table 1, the performances of different representations for bothinfrared and RGB channels and their combined results are listed.|
|||In two modalities fusion, the 3D-CNNfeatures after optical flows in RGB channel can effectively boost the performanceof using the infrared channel only.|
|||Forsingle modality, we utilize the 3D ConvNet part and the predictor part withoutfusion model for training and testing.|
|||And for the case of real infrared and RGBchannel fusion, we directly input the real feature map of RGB channel to thefusion model instead of using generated ones.|
|||Moreover, the fusion of infrared and generated RGBrepresentations achieves an Accuracy of 78%.|
|||Although it performs worse thanthe original RGB channels and the fusion of infrared and RGB channels, it onlyutilizes the information of infrared channel in the testing process.|
|||Evaluation results on the discriminative ability of transferable modalityData ModalitiesInfrared channelRGB channelGenerated RGBInfrared + RGB channelsInfrared channel + Generated RGB 78%Accuracy (%)71.67%79.33%76.67%82.33%In order to analyze the intra-class performance, the confusion matrices aredrawn in Fig.|
|||The results illustrated in confusion matrices using the proposed methodseen in Table 3, the generated fake RGB representations outperform the originalinfrared ones, which shows the robust transferability of PM-GANs.|
|||We apply the discriminative code layer and the secondfusion strategy for feature extraction, and train a K-nearest neighbor classifier(KNN) [4] using the provided Gaussian kernel function for classification.|
|||Bronstein, M.M., Bronstein, A.M., Michel, F., Paragios, N.: Data fusion throughcross-modality metric learning using similarity-sensitive hashing.|
|26|Mohammadreza_Zolfaghari_ECO_Efficient_Convolutional_ECCV_2018_paper|The architecture is based on merging long-term content al-ready in the network rather than in a post-hoc fusion.|
|||This principle is much related to the so-calledearly or late fusion used for combining the RGB stream and the optical flowstream in two-stream architectures [8].|
|||Partial observation not onlycauses confusion in action prediction, but also requires an extra post-processingstep to fuse scores.|
|||Hori, C., Hori, T., Lee, T.Y., Zhang, Z., Harsham, B., Hershey, J.R., Marks, T.K.,Sumi, K.: Attention-based multimodal fusion for video description.|
|||Zhang, X., Gao, K., Zhang, Y., Zhang, D., Li, J., Tian, Q.: Task-driven dynamicfusion: Reducing ambiguity in video description.|
|27|Zi_Jian_Yew_3DFeat-Net_Weakly_Supervised_ECCV_2018_paper|H ansch, R., Weber, T., Hellwich, O.: Comparison of 3d interest point detectors anddescriptors for point cloud fusion.|
|28|cvpr18-Super SloMo  High Quality Estimation of Multiple Intermediate Frames for Video Interpolation|By apply-ing the visibility maps to the warped images before fusion,we exclude the contribution of occluded pixels to the inter-polated intermediate frame to avoid artifacts.|
|||By applying the visibility maps to the warped im-ages before fusion, we exclude the contribution of occludedpixels to the interpolated intermediate frame, reducing ar-tifacts.|
|29|cvpr18-WILDTRACK  A Multi-Camera HD Dataset for Dense Unscripted Pedestrian Detection|Fuseddnn: A deep neural network fusion approach to fast and ro-bust pedestrian detection.|
|30|Themos_Stafylakis_Zero-shot_keyword_search_ECCV_2018_paper|Transformer [16]), combinations of CTC and attention,gating neural networks, as well as novel fusion approaches [1724].|
|31|Shuhan_Chen_Reverse_Attention_for_ECCV_2018_paper|Such multi-level feature fusion schemes also play animportant role in semantic segmentation [18,19], edge detection [20], skeleton de-tection [21,22].|
|||Nevertheless, the existing archaic fusions are still incompetent forsaliency detection under complex real-world scenarios, especially when dealingwith multiple salient objects with diverse scales.|
|||[11] designed short connections for multi-scale featurefusion, while in Amulet [13], multi-level convolutional features were aggregatedadaptively.|
|||[33] designed an attention mask to highlightthe prediction of the reverse object class, which then be subtracted from theoriginal prediction to correct the mistakes in the confusion area for semanticsegmentation.|
|||Based on this observation,multi-level features fusion is a common choice to capture their complementarycues, however, it will degrade the confident prediction of deep layers when com-bining with shallow ones.|
|||Differentwith HED [5] and DSS [11], there is no fusion layer included in our approach.|
|32|Parsing Images of Overlapping Organisms With Deep Singling-Out Networks|Fast fusion moves formulti-model estimation.|
|33|Point to Set Similarity Based Deep Feature Learning for Person Re-Identification|The deep architecture is constituted of a globalsub-network, a local sub-network and a fusion sub-network,such that different body parts are first discriminately learnedin the global sub-network and local sub-network, and thenfused in the fusion sub-network.|
|||Deep ArchitectureThe proposed P2S metric is combined with our proposedpart-based deep CNN to implement an end-to-end frame-work for both feature learning and fusion.|
|||3, the proposed deep architecture is consisted of threesub-networks: global sub-network, local sub-network andfusion sub-network.|
|||The deep feature learning and fusion neural network.|
|||This architecture is comprised of three sub-networks: global sub-network,local sub-network and fusion sub-network.|
|||Fusion sub-network The third part of our network isa fusion sub-network, which is consisted of four teams offully connected layers.|
|||ConclusionIn this paper, we propose a novel person re-identificationmethod by point to set (P2S) similarity comparison in apart-based deep CNN to perform integrated feature learn-ing and fusion.|
|||The deep architecture learns the global fea-tures, local features and fused features in the global sub-network, local sub-network and fusion sub-network, respec-tively.|
|34|cvpr18-Rolling Shutter and Radial Distortion Are Features for High Frame Rate Multi-Camera Tracking|A spline-based trajectory representation for sensor fusion and rollingshutter cameras.|
|35|Yunchao_Wei_TS2C_Tight_Box_ECCV_2018_paper|VGG16 and VGG-M) fusion.|
|36|cvpr18-Multi-Evidence Filtering and Fusion for Multi-Label Classification, Object Detection and Semantic Segmentation Based on Weakly Supervised Learning|From left to right: (a) Image level stage: fuse the object heatmaps H and the imageattention map Ag to generate object instances R for the instance level stage, and provide these two maps for information fusion at the pixellevel stage.|
|||During the fusion, for each object class, the at-tention proposals Ra which cover more than 0.5 of any pro-posals in Rh are preserved.|
|||Inthe pixel level stage, we still perform multi-evidence fil-tering and fusion to integrate the inference results from allthese component networks to obtain the pixelwise probabil-ity map indicating potential object categories at every pixel.|
|37|cvpr18-CBMV  A Coalesced Bidirectional Matching Volume for Disparity Estimation|Deep stereo fusion: combiningmultiple disparity hypotheses with deep-learning.|
|38|Learning Adaptive Receptive Fields for Deep Image Parsing Network|While in multi-paths version, layers from BN to interpo-lation layer are paralleled, followed by a summation oper-ation as feature fusion.|
|39|cvpr18-Baseline Desensitizing in Translation Averaging|Global fusion of rela-tive motions for robust, accurate and scalable structure frommotion.|
|40|cvpr18-Pyramid Stereo Matching Network|Moreover, stacked multiple encoder-decoder net-works such as [5] and [20] were introduced to improve fea-ture fusion.|
|||pooling to compress features into four scales and is fol-lowed by a 1  1 convolution to reduce feature dimension,input conv0_1 conv0_2 conv0_3 conv1_x conv2_x conv3_x conv4_x branch_1 branch_2 branch_3 branch_4 fusion 3Dconv0 3Dconv1 3Dstack1_1 3Dstack1_2 3Dstack1_3 3Dstack1_4 3Dstack2_1 3Dstack2_2 3Dstack2_3 3Dstack2_4 3Dstack3_1 3Dstack3_2 3Dstack3_3 3Dstack3_4 output_1 output_2 output_3 SPP module CNN (cid:885)(cid:885),(cid:885)(cid:884) (cid:885)(cid:885),(cid:885)(cid:884) (cid:885)(cid:885),(cid:885)(cid:884) (cid:885)(cid:885),(cid:885)(cid:884)(cid:885)(cid:885),(cid:885)(cid:884) (cid:885)(cid:885)(cid:885),6(cid:886)(cid:885)(cid:885),6(cid:886) (cid:883)6(cid:885)(cid:885),(cid:883)(cid:884)8(cid:885)(cid:885),(cid:883)(cid:884)8 (cid:885), dila = 2 (cid:885)(cid:885),(cid:883)(cid:884)8(cid:885)(cid:885),(cid:883)(cid:884)8 (cid:885), dila= 4 6(cid:886)6(cid:886) avg.|
|||The left and right input stereo images are fed to two weight-sharing pipelinesconsisting of a CNN for feature maps calculation, an SPP module for feature harvesting by concatenating representations from sub-regionswith different sizes, and a convolution layer for feature fusion.|
|41|Daniel_Castro_From_Face_Recognition_ECCV_2018_paper|Choi, J.Y., De Neve, W., Ro, Y.M., Plataniotis, K.: Automatic face annotationin personal photo collections using context-based unsupervised clustering and faceinformation fusion.|
|43|Damien_Teney_Visual_Question_Answering_ECCV_2018_paper|The division is arbitrarily placed after the fusion of the question and imageembeddings.|
|44|cvpr18-Viewpoint-Aware Attentive Multi-View Inference for Vehicle Re-Identification|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|45|Hierarchical Multimodal Metric Learning for Multimodal Classification|A Heterogeneous Multi-Metric Learning algorithmproposed in [43] for multi-sensor fusion essentially ex-tended the LMNN algorithm [36] for multi-metric learn-ing.|
|||Confusion matrix for Instance recognition result.|
|||Confusion matrices of classification results based on the pro-posed algorithm are shown in Figure 3 for instance recognitionexperiment and in Figure 4 for the 8th trial of category recognitionexperiment.|
|||Confusion matrix for 8th trial category recognition re-sult.|
|||Confusion matrix for scene recognition result.|
|||Heteroge-neous multi-metric learning for multi-sensor fusion.|
|||Discriminative multi-modalfeature fusion for rgbd indoor scene recognition.|
|||Modality and com-ponent aware feature fusion for rgb-d scene classification.|
|46|cvpr18-Human Pose Estimation With Parsing Induced Learner|In particular, we ignore the bias param-eters in the adaptive convolution layer, due to the residualfeature fusion strategy explained in the next part.|
|||We first evaluate our proposed Parsing InducedLearner (PIL) based on VGG16 and compare it with var-ious popular strategies (including feature fusion throughadding, multiplying and concatenating) on exploiting pars-ing features for pose estimation, in order to demonstrateits efficacy.|
|||This demonstrates naive feature fusionis not an effective way of utilizing parsing informationas expected.|
|47|cvpr18-Fast and Furious  Real Time End-to-End 3D Detection, Tracking and Motion Forecasting With a Single Convolutional Net|We investigate twodifferent ways to exploit the temporal dimension on our 4Dtensor: early fusion and late fusion.|
|||We do not utilize 3Dconvolutions on our single frame representation as this op-(a) Early fusion(b) Later fusionFigure 4: Modeling temporal information[25] with each layer number of feature maps reduced byhalf.|
|||We use the same number of convolutionlayers and feature maps as in the early fusion model, butinstead perform 3D convolution with kernel size 3  3  3for 2 layers without padding on temporal dimension, whichreduces the temporal dimension from n to 1, and then per-form 2D spatial convolution with kernel size 3  3 for otherlayers.|
|||For both our early-fusion and late-fusion models, wetrain from scratch using Adam optimizer [13] with a learn-ing rate of 1e-4.|
|||2, using temporal informationwith early fusion gives 3.7% improvement on mAP at IoU3574Single 5 Frames Early Laster w/ F w/ T IoU 0.5 IoU 0.6 IoU 0.7 IoU 0.8 IoU 0.9 Time [ms]XXXXXXX89.8191.4992.0192.02X X X 93.24X X86.2788.5789.3789.3490.5477.2080.9082.3381.5583.1052.2857.1458.7758.6161.616.338.178.939.6211.83911293030Table 2: Ablation study, on 144  80 region with vehicles having 3 number 3D pointsfor association and 0.9 score for thresholding both methods.|
|||While later fusion uses the same information as earlyfusion, it is able to get 1.4% extra improvement as it canmodel more complex temporal features.|
|48|Chang_Liu_Linear_Span_Network_ECCV_2018_paper|Robust object skeleton detection requires to explore rich rep-resentative visual features and effective feature fusion strategies.|
|||State-of-the-art approaches rootin effective multi-layer feature fusion, with the motivation that low-level featuresfocus on detailed structures while high-level features are rich in semantics [5].|
|||: A segmentation-free approach for skeletonization of gray-scaleimages via anisotropic vector diffusion.|
|49|CDC_ Convolutional-De-Convolutional Networks for Precise Temporal Action Localization in Untrimmed Videos|Convolutionaltwo-stream network fusion for video action recognition.|
|50|Heng_Wang_Scenes-Objects-Actions_A_Multi-Task_ECCV_2018_paper|8J. Ray, H. Wang, D. Tran, Y. Wang, M. Feiszli, L. Torresani and M. PaluriModel # params FLOPsRes2D 11.5M 2.6GRes3D 33.2M 81.4GI3D12.3M 13.0GInputRGBRGBRGB44.1Optical flow 29.7Late fusion 48.748.0Optical flow 39.4Late fusion 51.545.4Optical flow 34.0Late fusion 49.4Scenes Objects Actions SOA23.026.816.721.527.632.227.333.623.632.137.7 30.924.530.320.529.235.428.522.814.624.725.920.227.422.616.324.4Table 2: Three models trained with different inputs on SOA.|
|||For late fusion of RGBand optical flow streams, we uniformly sample 10 clips from a given video, andextract a 512-dimensional feature vector from each clip using the global averagepooling layer of the trained model.|
|||Late fusion has been shown to be very effectiveSOA video dataset9Fig.|
|||5: Tree structure recoveredfrom confusion matrix.|
|||The mAPof late fusion is about 2  4% higher than each individual input in Table 2.|
|||To further understand the performance of the model, we construct a confusionmatrix.|
|||All these combinations are accumulated to compute thefinal confusion matrix.|
|||To find meaningful structures from the confusion matrix,we recursively merge the two classes with the biggest confusion.|
|||SOA video dataset11MethodsUCF101 HMDB51 Kinetics CharadesActionVLAD+iDT [8]I3D (two-stream) [4]MultiScale TRN [44]S3D-G [42]ResNeXt-101 (64f) [10]SOA(optical flow)SOA(late fusion)93.698.0-96.894.586.590.769.880.7-75.970.265.667.0-75.7-77.265.159.167.921.0-25.2--16.116.9Table 4: Compare the effectiveness of pre-training on SOA with the state of theart.|
|||For late fusion, we follow the same procedure described in Section 3.3 bycombining the RGB results from Table 3 with the optical flow results listed inthis table.|
|||Long, X., Gan, C., de Melo, G., Liu, X., Li, Y., Li, F., Wen, S.: Multimodal keylessattention fusion for video classification (2018)25.|
|51|A Dataset and Exploration of Models for Understanding Video Data Through Fill-In-The-Blank Question-Answering|Experiments and DiscussionFirst in Section 5.1 we describe 5 baseline models whichinvestigate the relative importance of 2D vs. 3D features,as well as early vs. late fusion of text information (by ini-tializing the video encoder with the question encoding andthen finetuning).|
|||into a classification task is a broadly applicableidea, useful for benchmarking models; (2) exploring spatio-temporal attention; (3) determining which factors contributemost to improvement of video model performance - increas-ing data, refinement of existing architectures, developmentof novel spatio-temporal architectures, etc; (4) further in-vestigation of multimodal fusion in video (e.g.|
|52|Weidi_Xie_Comparator_Networks_ECCV_2018_paper|(a) Confusion matrix(b) Sampling histogramFig.|
|||Navaneeth, B., Jingxiao, Z., Hongyu, X., Jun-Cheng, C., Carlos, C., Rama, C.:Deep heterogeneous feature fusion for template-based face recognition.|
|53|Semantic Regularisation for Recurrent Image Annotation|(c) The image CNN output feature layer is integrated with the LSTM output via late fusion [22, 31].|
|||[22] combine word embedding and image fea-tures via output fusion (Fig.|
|||CNN-RNN: A CNN-RNN model whichuses output fusion (Fig.|
|54|cvpr18-Very Large-Scale Global SfM by Distributed Motion Averaging|Seeing double with-out confusion: Structure-from-motion in highly ambiguousscenes.|
|||Global fusion of rela-tive motions for robust, accurate and scalable structure frommotion.|
|55|Recurrent Modeling of Interaction Context for Collective Activity Recognition|The two types of CNN-based featuresare further combined in a regularized feature fusion networkfor video event classification.|
|||Note that results of others arecalculated from corresponding original confusion matrix in[22, 5, 17, 15].|
|||Confusion matrix on Collective Activity Dataset [6] (re-gard W alking and Crossing as the same class M oving).|
|||(a)Confusion matrix of baseline [17]; (b) Confusion matrix of ourmethod.|
|||The confusion matrix of our method is also illustratedin Figure.|
|||Confusion matrix on Chois New Dataset [5] obtainedby using our hierarchical recurrent interactional context encodingmodel.|
|56|Chen_Sun_Actor-centric_Relation_Network_ECCV_2018_paper|This results in confusionof similar actions and interactions, such as jumping and shooting a basketball.|
|||To combine RGB and opticalflow input modalities, we use early fusion at the Mixed 4f block instead of latefusion at the logits layer.|
|57|Fast Person Re-Identification via Cross-Camera Semantic Binary Transformation|Data fusion through cross-modality metric learning us-ing similarity-sensitive hashing.|
|58|Scalable Person Re-Identification on Supervised Smoothed Manifold|Diffusion processes for retrievalrevisited.|
|||TPAMI,Query specific rank fusion for image retrieval.|
|||Query-adaptive late fusion for image search and person re-identification.|
|59|Danfeng_Hong_Joint__Progressive_ECCV_2018_paper|2) University of Houston Image: The second hyperspectral cube was providedfor the 2013 IEEE GRSS data fusion contest acquired by ITRES-CASI sensorwith size of 3491905144.|
|60|Yu_Liu_Transductive_Centroid_Projection_ECCV_2018_paper|Notice that we found more than one hundredwrong annotations in this dataset, which introduce significant confusion for recallrate on some small false positive rate (FPR  1e-3), so we remove these pairsin evaluation5.|
|||Zhao, H., Tian, M., Sun, S., Shao, J., Yan, J., Yi, S., Wang, X., Tang, X.:Spindlenet: Person re-identification with human body region guided feature de-composition and fusion.|
|61|FASON_ First and Second Order Information Fusion Network for Texture Recognition|We propose an effective fusion architecture - FASONthat combines second order information flow and first or-der information flow.|
|||This allows us to extend our fusionarchitecture to combine features from multiple convolutionlayers, which captures different style and content informa-tion.|
|||This multiple level fusion architecture further im-proves recognition performance.|
|||Our experiments show theproposed fusion architecture achieves consistent improve-ments over state-of-the-art methods on several benchmarkdatasets across different tasks such as texture recognition,indoor scene recognition and fine-grained object classifica-tion.|
|||The contribution of our work is two fold: We design a deep fusion architecture that effectivelycombines second order information (calculated from abilinear model) and first order information (preservedthrough our leaking shortcut) in an end-to-end deepnetwork.|
||| We extend our fusion architecture to take advantage ofthe multiple features from different convolution layers.|
|||In section 3, we presentthe design of our fusion architecture, give details and expla-nations of each building block.|
|||FASONIn this section, we describe our proposed frameworkFASON (First And Second Order information fusion Net-work).|
|||We first introduce the basic components of our firstorder and second order fusion building blocks.|
|||Then, wedescribe our final deep architecture with multiple level fea-ture fusion.|
|||First order information fusion by gradient leak(cid:173)ing(x) =sign(x)p|x|ksign(x)p|x|k2(2)3.2.|
|||Given two randomly sampled mapping vectors h  Ndwhere each entry is uniformly drawn from {1, 2,    , c},and s  {+1, 1}d where each entry is filled with either +1or 1 with equal probability, the sketch function is definedas:where(x, s, h) = [C1, C2,    , Cc]Cj = Xs(i)  x(i)i:h(i)=j(3)(4)Now we introduce our core building block of the firstorder and second order information fusion.|
|||Figure 1: The core building block of our first and secondorder information fusion architecture compared to originalbilinear model.|
|||First and second order fusion with multiplelevels of convolutional featuresOne benefit of our fusion framework is that we can fusemore convolutional features into bilinear layers and conductan effective end-to-end training as shown in Figure 2.|
|||We investigate two major network architectures: a sin-gle fusion at conv5 level (equivalent to features generatedfrom conv5 4 of VGG-19 network) and a multiple fusionat conv4, conv5 layers (equivalent to features generatedfrom conv4 4 and conv5 4 of VGG-19 network).|
|||For faircomparison, we also conduct experiments using typical bi-linear networks without fusion on these same two setups.|
|||Figure 3: The detailed configurations of our first and sec-ond order information fusion architectures.|
|||Effectiveness of fusionWe first evaluate the effectiveness of our fusion archi-tecture by comparing two networks with and without firstorder information fusion on single (conv5) and multipleFigure 4: Comparison of learning curves on bilinear modelusing single level of convolutional feature (conv5) with andwithout our first order information fusion on DTD dataset.|
|||Figure 5: Comparison of learning curves on bilinear modelusing two levels of convolutional features (conv4+conv5)with and without our first order information fusion on DTDdataset.|
|||As shown by the plots of testing accuracy (Figure 4band Figure5b), our architecture with first order informationfusion clearly outperforms the bilinear network without fu-sion in both training stages.|
|||We also evaluate the effectiveness of our fusion architec-ture on different deep networks such as VGG-16 and VGG-19.|
|||Our fusionarchitectures gives consistent improvements over a baselinebilinear network.|
|||The performance further boosts with ourmultiple layer fusion when two level of convolutional layersconv4 and conv5 are combined.|
|||Com-bining the improvements from first and second order infor-mation fusion with multi-layer feature fusion, we obtain a2% improvement from a strong bilinear CNN baseline forboth VGG-16 and VGG-19.|
|||VGG-16 VGG-19conv5conv5+fusionImprovementsconv5+conv4conv5+conv4+fusionImprovements72.4573.09+0.6472.8774.47+1.6072.8273.62+0.8073.3174.57+1.26Table 1: Effectiveness comparison across different net-work architecture on one training and testing split on DTDdataset.|
|||We compare the performance ofour fusion architecture with several state-of-the-art meth-ods such as [3, 4, 20] on the DTD dataset.|
|||We also report thefusion results from previous work that use multiple features.|
|||We also evaluate the performance of our fusion architec-ture on KTH-T2b dataset with several state-of-the-art meth-ods such as [30, 3, 4, 20].|
|||We compare our fusion architectures with the standardVGG network (using the weights learned form the DTDdataset for classification task) on different combinations ofcontent and style images in Figure 6.|
|||Further-more, our multi-layer fusion architecture is even better thanour single-layer fusion architecture.|
|||Ex-periments show that our fusion architecture consistently im-proves over the standard bilinear networks.|
|62|cvpr18-Temporal Hallucinating for Action Recognition With Few Still Images|As spatialand temporal features are complementary to represent dif-ferent actions, we apply spatial-temporal prediction fusionto further boost performance.|
|||Finally, we perform spatial-temporal fusion as our final prediction for query image, dueto the fact that spatial and temporal characteristics of ac-tions are often complementary.|
|||Step3: We perform spatial and temporal score fusionfor video bag, where each video has a fused score vectorsf use.|
|||We examine the classification accuracy of temporalpredicting (TP) in temporal memory module, spatial pre-dicting (SP) in spatial memory module, and HVM (spatial-temporal fusion).|
|||Using this5320ActionConfusion BlowingCandlesBrushingTeeth RidingHorseRidingBike AmericanFootballPlayingHandballWEB101VOCDIFF20BaselineOur HVM20 mistaken images6 mistaken images103 mistaken images62 mistaken images20 mistaken images11 mistaken imagesTable 3.|
|||Confusion Reduction by HVM.|
|||Due tocomplementary properties of spatial and temporal features,we apply spatial-temporal prediction fusion to further en-hance the performance.|
|63|cvpr18-Eliminating Background-Bias for Robust Person Re-Identification|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|64|Konstantinos-Nektarios_Lianos_VSO_Visual_Semantic_ECCV_2018_paper|: Incremental densesemantic stereo fusion for large-scale semantic scene reconstruction.|
|65|StyleBank_ An Explicit Representation for Neural Image Style Transfer|Because of the explicit representation, we can more con-veniently control style transfer and create new interestingstyle fusion effects.|
|||More specifically, we can either linearlyfuse different styles altogether, or produce region-specificstyle fusion effects.|
|||Style FusionWe provide two different types of style fusion:linearfusion of multiple styles, and region-specific style fusion.|
|||Figure 9 shows suchlinear fusion results of two styles with variant fusion weightwi.|
|||Then region-can be described as F = Pmspecific style fusion can be formulated as Equation (6):eF = Xmi=1Ki  (Mi  F ),(6)where Ki is the i-th filter bank.|
|||Figure 10 shows such a region-specific style fusion resultwhich exactly borrows styles from two famous paintings ofPicasso and Van Goph.|
|||Region-specific style fusion with two paintings of Pi-casso and Van Gophm, where the regions are automatically seg-mented with K-means method.|
|||The decoupling allows faster training(for multiple styles, and new styles), and enables new in-teresting style fusion effects, like linear and region-specificstyle transfer.|
|66|Deep Cross-Modal Hashing|Data fusion through cross-modality metriclearning using similarity-sensitive hashing.|
|67|End-To-End Learning of Driving Models From Large-Scale Video Datasets|We also investigate below another temporal fusion ap-proach, temporal convolution, instead of LSTM to fuse thetemporal information.|
|||In the TCNN configuration we study using temporalconvolution as the temporal fusion mechanism.|
|68|Spatially-Varying Blur Detection Based on Multiscale Fused and Sorted Transform Coefficients of Gradient Magnitudes|We propose a robust spatially-varying blur detec-tion method from a single image based on a novel high-frequency multiscale fusion and sort transform (HiFST) ofgradient magnitudes to determine the level of blur at eachlocation in an image.|
|||Our work is based on a multiscale transform decom-position followed by the fusion and sorting of the high-frequency coefficients of gradient magnitudes.|
|||High(cid:173)frequency multiscale fusion and sorttransformThe Discrete Cosine Transform (DCT) has emerged asone of the most popular transformations for many com-puter vision and image compression applications.|
|||Proposed spatially(cid:173)varying blur detectionIn the following, we present in details our spatially-varying blur detection approach which is based on thefusion, sorting, and normalization of multiscale high-frequency DCT coefficients of gradient magnitudes to de-tect blurred and unblurred regions from a single image with-out having any information about the camera settings or theblur type.|
|||We proposed an effective blur detection method based on ahigh-frequency multiscale fusion and sort transform, whichmakes use of high-frequency DCT coefficients of the gra-dient magnitudes from multiple resolutions.|
|||Shape fromdefocus via diffusion.|
|||Depth from diffusion.|
|69|Timo_von_Marcard_Recovering_Accurate_3D_ECCV_2018_paper|Specifically, our approach has three steps: initialization, association and datafusion.|
|||Given those associations, in the data fusionstep, we define an objective function and jointly optimize for the 3D poses of thefull sequence, the per-sensor heading errors, the camera pose and translation.|
|||4 Video Inertial Poser (VIP)In order to perform accurate 3D human motion capture with hand-held videoand IMUs we perform three subsequent steps: initialization, pose candidate as-sociation and video-inertial fusion.|
|||Pons-Moll, G., Baak, A., Helten, T., M uller, M., Seidel, H.P., Rosenhahn, B.:Multisensor-fusion for 3d full-body human motion capture.|
|||Zheng, Z., Yu, Tao, L.H., Guo, K., Dai, Q., Fang, L., Liu, Y.: Hybridfusion: Real-time performance capture using a single depth sensor and sparse imus.|
|70|cvpr18-SPLATNet  Sparse Lattice Networks for Point Cloud Processing|FusionNet [18] combines shapeclassification scores from a volumetric and a multi-viewnetwork, yet this fusion happens at a late stage, after thefinal fully connected layer of these networks, and does notjointly consider their intermediate local and global featurerepresentations.|
|||ing edge preserving diffusion.|
|||Oct-NetFusion: Learning depth fusion from data.|
|71|Dynamic Facial Analysis_ From Bayesian Filtering to Recurrent Neural Network|Multilayer and mul-timodal fusion of deep neural networks for video classifica-tion.|
|72|Seeing Into Darkness_ Scotopic Visual Recognition|Real-time classification and sensor fusion with a spiking deep belief net-work.|
|73|cvpr18-Rotation Averaging and Strong Duality|Global fusion of rela-tive motions for robust, accurate and scalable structure frommotion.|
|74|Lequan_Yu_EC-Net_an_Edge-aware_ECCV_2018_paper|Xu, D., Anguelov, D., Jain, A.: PointFusion: Deep sensor fusion for 3D boundingbox estimation.|
|75|Learning Cross-Modal Deep Representations for Robust Pedestrian Detection|The two ROI feature maps are concate-nated and given as input to the convolutional fusion layer.|
|76|cvpr18-Learning Deep Models for Face Anti-Spoofing  Binary or Auxiliary Supervision|We advance [5] ina number of aspects, including fusion with temporal super-vision (i.e., rPPG), finer architecture design, novel non-rigidregistration layer, and comprehensive experimental support.|
|77|Detailed, Accurate, Human Shape Estimation From Clothed 3D Scan Sequences|Since the model factors pose andshape, all cloth alignment templates live in a common un-posed space; we call the union of these unposed alignmentsthe fusion scan.|
|||Since the cloth should lie outside the bodyfor all frames we minimize the single-frame objective usingthe fusion scan as input and obtain an accurate shape tem-plate (fusion shape) for the person.|
|||Finally, to obtain thepose and the time varying shape details, we optimize againthe single objective function using the fusion shape as a reg-ularizer.|
|||The fusion scan is the union of the frame wiseunposed alignments.|
|||From the fusion scan c) we obtain the fusion shape d).|
|||The unionof the unposed templates creates the fusion scan (Fig.|
|||We use it to estimate a single shape, that we call the fusionshape (Fig.|
|||Since all temporal information is fusedinto a single fusion scan, we can estimate the fusion shapeusing the same single-frame objective function.|
|||Using thefusion shape template as a prior, we can accurately estimatethe pose and shape of the sequence.|
|||Hence we gather all templatesand treat them as a single point cloud that we call the fusionscan SFu = {Tk.|
|||(7)The obtained fusion shapes are already quite accurate be-cause the fusion scan carves the volume where the nakedshape should lie.|
|||Pose and Shape TrackingFinally we use the fusion shape to perform tracking reg-ularizing the estimated shapes to remain close to the fu-sion shape.|
|||We achieve that by coupling the estimationstowards the fusion shape instead of the SMPL model shapespace.|
|||[45]fusion meshdetailedhipsYang et al.|
|||[45]fusion meshdetailedshoulders millYang et al.|
|||[45]fusion meshdetailed0000517.292.582.520000521.022.812.750000518.772.562.49t-shirt, long pants000570003217.9413.762.532.392.362.440003215.772.662.630003218.022.742.720005717.872.662.550005716.502.462.370322317.902.432.270322321.842.542.400322318.152.422.260009618.682.892.830009621.662.712.640009619.022.922.850011415.422.382.310011418.052.652.560011414.782.692.59soccer outfit0000516.772.502.440000522.522.652.580000518.742.892.830003216.962.632.590003216.812.632.590003217.882.872.820005718.522.372.280005719.552.582.500005715.802.372.280322320.412.282.170322322.032.502.380322319.472.442.33Avrg.|
|||[45],fusion shape (ours), detailed shape (ours).|
|||1 we show the numerical results obtained by [45],our fusion mesh, and our detailed mesh.|
|||The proposed fusionshape accurately recovers the body shape, while the detailedshape is capable of capturing the missing details.|
|||The single-frameobjective computation takes 10 seconds per frame, fusionmesh is computed in 200 seconds.|
|||Dynamicfusion:Reconstruction and tracking of non-rigid scenes in real-time.|
|78|A Dual Ascent Framework for Lagrangean Decomposition of Combinatorial Problems|Other settings, given in thesupplement, may turn it to other popular message passingtechniques like MPLP [28] or min-sum diffusion [64].|
|||We compare against state-of-the-art multicut al-gorithms implemented in the OpenGM [41] library, namely(i) the branch-and-cut based solver MC-ILP [44] utilizingthe ILP solver CPLEX [2], (ii) the heuristic primal fusionmove algorithm CC-Fusion [9] with random hierarchicalclustering and random watershed proposal generator, de-noted by the suffixes -RHC and -RWS and (iii) the heuristicprimal Cut, Glue & Cut solver CGC [10].|
|||Diffusion algorithmsand structural recognition optimization problems.|
|79|Efficient Diffusion on Region Manifolds_ Recovering Small Objects With Compact CNN Representations|Efficient Diffusion on Region Manifolds:Recovering Small Objects with Compact CNN RepresentationsAhmet Iscen1 Giorgos Tolias2 Yannis Avrithis1 Teddy Furon1 Ondrej Chum21Inria Rennes2VRG, FEE, CTU in Prague{ahmet.iscen,ioannis.avrithis,teddy.furon}@inria.fr{giorgos.tolias,chum}@cmp.felk.cvut.czAbstractQuery expansion is a popular method to improve thequality of image retrieval with both conventional and CNNrepresentations.|
|||This work focuses on diffusion, a mechanismthat captures the image manifold in the feature space.|
|||Thediffusion is carried out on descriptors of overlapping im-age regions rather than on a global image descriptor likein previous approaches.|
|||We perform diffusion through asparse linear system solver, yielding practical query timeswell below one second.|
|||Diffusion on a synthetic dataset in R2.|
|||Contour lines correspond to ranking scoresafter diffusion.|
|||On theother hand, diffusion [39, 64, 13] is based on a neighbor-hood graph of the dataset that is constructed off-line andefficiently uses this information at query time to search onthe manifold in a principled way.|
|||We make the following contributions: We introduce a regional diffusion mechanism, whichhandles one or more query vectors at the same cost.|
||| In diffusion mechanisms [39, 64, 13], query vectors areusually part of the dataset and available at the indexingstage.|
|||Searching in parallel in more than one manifolds via dif-fusion and using the nearest neighbors of unseen queries areillustrated in Figure 1.|
|||Sections 2and 3 discuss related work and background respectively, fo-cusing on diffusion mechanisms.|
|||We also review the concept of diffusionin computer vision and image retrieval in particular.|
|||This is unlike our regional diffusion mechanism, which hasa fixed cost with respect to the number of query regions.|
|||Diffusion.|
|||We are focusing on diffusion mechanisms,which propagate similarities through a pairwise affinity ma-trix [13, 39].|
|||Diffusion is used for retrieval of general scenes or shapesof particular objects [28, 14, 60, 13].|
|||Diffusion with regional similarity has been investi-gated before, but only to define image level affinity [62], toaggregate local features [15], or to handle bursts [16].|
|||Donoser and Bischof [13] review a number of diffusionmechanisms for retrieval.|
|||Ranking with diffusionDiffusion in the work of Donoser and Bischof [13] de-notes a mechanism spreading the query similarities over themanifolds composing the dataset.|
|||This is only weakly re-lated to continuous time diffusion process or random walkson graph.|
|||Diffusion.|
|||We focus on a particular diffusion mechanism that, givenan initial vector f 0, iterates according tof t = Sf t1 + (1  )y.|
|||A diffusion mechanismalso appears in seeded image segmentation [20], wherequery points correspond to labeled pixels (seeds) anddatabase points to the remaining unlabeled pixels.|
|||Diffusion interpolatesfd from fq by minimizing, w.r.t.|
|||MethodThis section describes our contributions on image re-trieval: handling new query points not in the dataset, search-ing for multiple regions with a single diffusion mechanism,and efficiently computing the solution.|
|||Handling new queriesIn prior work on diffusion, a query point q is consideredto be contained in the dataset X [63, 13].|
|||Figure 1 shows a toy 2-dimensional example of diffusion,where the k-nearest neighbors to each query point taken intoaccount in (7) are depicted.|
|||Regional diffusionThe diffusion mechanism described so far is applica-ble to image retrieval when database and query images areglobally represented by single vectors.|
|||We call this globaldiffusion in the rest of the paper.|
|||Unlike the traditionalrepresentation with local descriptors [49, 40], global dif-fusion fits perfectly with the early CNN-based global fea-tures [4, 30, 43].|
|||Fortunately, diffusion as defined in section 3 can alreadyhandle multiple queries.|
|||We callthis mechanism regional diffusion.|
|||Our work is inspired by the analysis in thework of Grady [20] that we apply to the diffusion mech-anism of Zhou et al.|
|||Diffusion.|
|||Given this definition of y, diffusion is nowperformed on dataset X , jointly for all query points in Q.Affinities of multiple query points are propagated in thegraph in a single process at no additional cost compared tothe case of a single query point.|
|||Figure 1 illustrates the diffusion on single and multiplequery points.|
|||After diffusion, each image is associated with sev-eral elements of the ranking score vector f , one for eachpoint x in X  X .|
|||Diffusion is an iterative solver.|
|||It improves convergence, be it for CG or diffusion (3).|
|||It has been used for random walkproblems [20], but not diffusion-based retrieval accordingto our knowledge.|
|||Instead of ranking the fulldataset, diffusion re-ranks an initial search.|
|||Then we apply diffusion only on the top rankedimages.|
|||The cost ofthis step is not significant compared to the actual diffusion.|
|||Retrieval performance (mAP) of regional diffusion withsum and generalized max pooling (GMP), with  = 1 in (14).|
|||10090807060PAmDiffusion, regionalDiffusion, globalBaseline, regionalBaseline, globalwith many variations such as different scales, rotations, andocclusions.|
|||The diffusion parameter  is always 0.99, as in the workof Zhou et al.|
|||We vary the number of nearest neighbors k forconstructing the affinity matrix and evaluate performancefor both global and regional diffusion.|
|||[44], where image regions3http://people.rennes.inria.fr/Ahmet.Iscen/diffusion.html1050100k2005001000Figure 2.|
|||mAP performance for global and regional diffusion onOxford5k; baselines are R-MAC and R-match respectively.|
|||The performance stays stableover a wide range of k. The drop for low k is due to very fewneighbors being retrieved (where regional diffusion is moresensitive), whereas for high k, it is due to capturing morethan the local manifold structure (where regional diffusionis superior).|
|||We set k = 200 for regional diffusion, and k = 50 forglobal diffusion for the rest of our paper.|
|||regional) diffusion,measured on INSTRE.|
|||We set k = 200 for the query as wellin the case of the regional diffusion, while for the global onek = 10 is needed to achieve good performance.|
|||We evaluate the two pooling strategies after re-gional diffusion in Table 1.|
|||Efficient diffusion with conjugate gradient.|
|||We comparethe iterative diffusion (3) to our conjugate gradient solution.|
|||The average query time on Oxford5k including all stagesfor global baseline, regional baseline, global diffusion andregional diffusion without truncation is 0.001s, 0.321s,0.02s, and 0.664s, respectively.|
|||We therefore not only offer space208298PAm940.7s0.6s3.1s2.6sOxf5k (CG)Oxf5k (PR)Par6k (CG)Par6k (PR)9010203050 70 100iterationsFigure 3. mAP performance of regional diffusion vs. number ofiterations for conjugate gradient (CG) and iterative diffusion (3).|
|||Labels denote diffusion time.|
|||Symbol  denotesglobal diffusion, and  to the default number of regions (21) perimage.|
|||Average diffusion time in seconds is shown in text labels.|
|||improvements but also better performance,mainly in thecase of regional diffusion.|
|||Large scale diffusionWe now focus on the large scale solutions of Section 4.4.|
|||Methodm  d INSTRE Oxf5k Oxf105k Par6k Par106kGlobal descriptors - nearest neighbor searchCroW [30]R-MAC [43]R-MAC [19]NetVLAD [1]5125122,0484,096-47.762.6-68.277.783.971.663.270.180.8-Global descriptors - query expansionR-MAC [43]+AQE [8]R-MAC [43]+SCSM [48]R-MAC [43]+HN [42]Global diffusionR-MAC [19]+AQE [8]R-MAC [19]+SCSM [48]Global diffusion5125125125122,0482,0482,04857.360.164.770.370.571.480.585.485.379.985.789.689.187.179.780.5-82.788.387.387.479.884.193.879.788.489.492.094.195.395.496.571.076.889.9-83.584.5-92.592.792.595.4Regional descriptors - nearest neighbor searchR-match [44]R-match [44]21512212,04855.571.081.588.176.585.786.194.979.991.32.4k12821512551221512212,04852,048212,04874.760.477.580.077.188.489.6Regional descriptors - query expansion84.078.684.790.389.690.094.2HQE [51]81.0R-match [44]+AQE [8]Regional diffusion93.0Regional diffusion92.692.5R-match [44]+AQE [8]Regional diffusion95.8Regional diffusion95.3Table 2.|
|||Regional diffusion with 5 regions uses GMM.|
|||Regional diffusion on the fulldataset takes 13.9s for Oxford105k, which is not practical.|
|||We therefore rank images according to the aggregated re-gional descriptors, which is equivalent to the R-MAC rep-resentation [52], and then perform diffusion on a short-list.|
|||Statistics computed on INSTRE over allqueries for global and regional diffusion.|
|||Query examples from INSTRE, Oxford, and Paris datasets and retrieved images ranked by decreasing order of ranking differencebetween global and regional diffusion.|
|||We measure precision at the position where each image is retrieved and report this under each imagefor global and regional diffusion.|
|||The performance ofthe full database diffusion is nearly attained by re-rankingless than 10% of the database.|
|||The entire truncation and dif-fusion process on Oxford105k takes 1s, with truncation andre-normalization taking only a small part of it.|
|||Regional diffusion significantly outperforms all othermethods in all datasets.|
|||Global diffusion performs well onParis because query objects almost fully cover the imagein most of the database entries.|
|||The improve-ments of regional diffusion are in this case much larger.|
|||Regional diffusion was not possible before.|
|||Incontrast to prior work, we use the closed form solution ofthe diffusion iteration, obtained by the conjugate gradientmethod.|
|||Diffusion processes for retrievalrevisited.|
|||Diffusion-on-manifold aggrega-tion of local features for shape-based 3d model retrieval.|
|||ondiffusion aggregation for image retrieval.|
|||Learning optimalseeds for diffusion-based salient object detection.|
|||Re-ranking by multi-feature fusion with diffusion for image retrieval.|
|||Locallyconstrained diffusion process on locally densified distancespaces with applications to shape retrieval.|
|||Query specific fusion for image retrieval.|
|80|Maria_Klodt_Supervising_the_new_ECCV_2018_paper|Moulon, P., Monasse, P., Marlet, R.: Global fusion of relative motions for ro-bust, accurate and scalable structure from motion.|
|82|Unified Embedding and Metric Learning for Zero-Exemplar Event Detection|Then, at thebottom, the network fV learns to embed the videofeature x as zv  Z such that the distance be-tween (cid:0)zv, zt(cid:1) is minimized, in the learned metricspace Z.self-paced reranking [23], pseudo-relevance feedback [24],event query manual intervention [25], early fusion of fea-tures (action [26, 27, 28, 29, 30] or acoustic [31, 32, 33]) orlate fusion of concept scores [17].|
|||Zero-shot event detectionusing multi-modal fusion of weakly supervised concepts.|
|83|Filip_Radenovic_Deep_Shape_Matching_ECCV_2018_paper|ComponentTrain/Test: Edge filteringTrain: Query binarizationTest: MirroringTest: Multi-scaleTest: DiffusionOO(cid:4)F(cid:4)NetworkFF(cid:4)(cid:4)(cid:4)(cid:4)(cid:4)F(cid:4)(cid:4)(cid:4)F(cid:4)(cid:4)(cid:4)(cid:4)F(cid:4)(cid:4)(cid:4)(cid:4)(cid:4)mAP25.927.938.441.943.445.646.168.9is measured via the matching accuracy at the top K retrieved images, denotedby acc.@K.|
|||Finally, we boost the recall of our sketch-based retrievalby global diffusion, as recently proposed by Iscen et al.|
|||This approach is later improved [32] by adding anattention module with a coarse-fine fusion (CFF) into the architecture, andby extending the triplet loss with a higher order learnable energy function(HOLEF).|
|||Hand-crafted methodsMethodGF-HOG [11]S-HELO [12]HLR+S+C+R [14]GF-HOG extended [15]PerceptualEdge [16]LKS [17]AFM [19]AFM+QE [19]Dim mAPn/a 12.21296 12.4n/a 17.1n/a 18.23780 18.41350 24.5243 30.4755 57.9CNN-based methodsMethodSketch-a-Net+EdgeBox [20]Shoes network [22]Chairs network [22]Sketchy network [23]Quadruplet network [24]Triplet no-share network [26] EdgeMACSketch-a-Net+EdgeBox+GraphQE [20] EdgeMAC+DiffusionDim mAP5120 27.0256 29.9256 29.81024 34.01024 32.2128 36.2512 46.1n/a 32.3n/a 68.9Quadruplet network [24] tackles the problem in a similar way as Sketchy net-work, however, they use ResNet-18 [65] architecture with shared weights for bothsketch and image branches.|
|||This holds for both plain search withthe descriptors, and for methods using re-ranking techniques, such as queryexpansion [66] and diffusion [59].|
|||Iscen, A., Tolias, G., Avrithis, Y., Furon, T., Chum, O.: Efficient diffusion onregion manifolds: Recovering small objects with compact CNN representations.|
|||: Query specific fusion forimage retrieval.|
|84|Beyond Instance-Level Image Retrieval_ Leveraging Captions to Learn a Global Visual Representation for Semantic Retrieval|Unsupervised vi-sual and textual information fusion in cbmir using graph-based methods.|
|85|Temporal Residual Networks for Dynamic Scene Recognition|Convolutionaltwo-stream network fusion for video action recognition.|
|86|Tsung-Yu_Lin_Second-order_Democratic_Aggregation_ECCV_2018_paper|Arsigny, V., Fillard, P., Pennec, X., Ayache, N.: Log-euclidean metrics for fastand simple calculus on diffusion tensors.|
|||Dryden, I.L., Koloydenko, A., Zhou, D.: Non-euclidean statistics for covariancematrices, with applications to diffusion tensor imaging.|
|87|Jingyi_Zhang_Generative_Domain-Migration_Hashing_ECCV_2018_paper|Inspired by the mix-up operation [58], in order to further reduce the domaindiscrepancy, we propose a feature fusion method that employs a linear mix-up ofGenerative Domain-Migration Hashing13Fig.|
|||Besides the linear embedding,we also evaluated other fusion strategies such as concatenation and the Kroneckerproduct.|
|||However, none of these fusion methods is helpful.|
|||Bronstein, M.M., Bronstein, A.M., Michel, F., Paragios, N.: Data fusion throughcross-modality metric learning using similarity-sensitive hashing.|
|88|cvpr18-Geometry-Aware Scene Text Detection With Instance Transformation Network|The ITN consists of three parts: convolutional feature extraction and fusion layers shown in theleft where feature maps of three different scales are fused; in-network transformation embedding module shown in the gray dashed boxwhere instance-level affine transformations are predicted and embedded; and multi-task learning shown in the right where classification,transformation regression and coordinate regression are jointly optimized.|
|||Instance Transformation NetworkThe ITN is an end-to-end detection network whichtakes in an input image I and output word-level or line-level quadrilateral detections D. Each detection contain-s a quadrilateral d represented by four clockwise cor-ners (starting from the left-top vertex) in the form of(d1x, d1y, d2x, d2y, d3x, d3y, d4x, d4y) and its confidence s-core s. As depicted in Figure 3, the ITN includes main-ly three parts: convolutional feature extraction and fusion,in-network transformation embedding introduced in Sec-tion 3.1 and multi-task learning.|
|||Feature fusion To tackle with multi-scale targets, we fusefeature maps of different scales.|
|89|Jue_Wang_Learning_Discriminative_Video_ECCV_2018_paper|While recur-rent networks, such as LSTM and GRU, have shown promising results on videotasks [60, 33, 3], training them is often difficult, and so far their performancehas been inferior to models that look at parts of the video followed by a latefusion [8, 41].|
|90|cvpr18-Generating a Fusion Image  One's Identity and Another's Shape|We propose a newGAN-based network that generates a fusion image with theidentity of input image x and the shape of input image y.|
|||MethodsThe goal of our work is to learn a mapping function thatgenerates a fusion image from two input images given frommultiple unlabeled sets.|
|||When our network has two input images x = (Ix, Sx) andy = (Iy, Sy), our goal is to generate the following newfusion image:G(x = (Ix, Sx), y = (Iy, Sy)) = (Ix, Sy)(1)Thus, the output is a fusion image that has the same iden-tity as x, and the same shape as y.|
|91|Memory-Augmented Attribute Manipulation Networks for Interactive Fashion Search|The fusion is learned through a fully-connectedlayer in AMNet.|
|92|cvpr18-Deep Unsupervised Saliency Detection  A Multiple Noisy Labeling Perspective|To effectively leverage these noisy butinformativesaliency maps, we propose a novel perspective to the prob-lem: Instead of removing the noise in saliency labeling fromunsupervised saliency methods with different fusion strate-gies [35], we explicitly model the noise in saliency maps.|
|||To the best of our knowledge, [35] is the first and onlydeep method that learns saliency without human anno-tations, where saliency maps from unsupervised saliencymethods are fused with manually designed rules in combin-ing intra-image fusion stream and inter-image fusionstream to generate the learning curriculum.|
|||A simple fusion of the multiple labels (training with averag-ing, treating as multiple labels) will also not work due to thestrong inconsistency between labels.|
|||Supervision by fusion: To-wards unsupervised learning of deep salient object detector.|
|93|cvpr18-Stacked Latent Attention for Multimodal Reasoning|Building on top of this, we reinforce the technique ofusing attention for multimodal fusion, and propose a twinstream stacked latent attention model which is able to cap-ture positional information from both textual and visualmodalities.|
|||We further build on top ofthis to develop a twin stream stacked latent attention modelfor the fusion of knowledge in different modalities.|
|||It is another task which investigates the fusionof vision and text for multimodal reasoning where attentionmodels have demonstrated large impact [24][25][22][9].|
|||Mu-tan: Multimodal tucker fusion for visual question answering.|
|94|Revisiting Metric Learning for SPD Matrix Based Visual Representation|Inmedical image analysis, SPD matrices have long been usedfor diffusion tensor data [1], and correlation matrix and in-verse covariance matrix have been employed to model theinteraction of brain regional imaging signals [8, 25].|
|||Log-euclidean metrics for fast and simple calculus on diffusiontensors.|
|||Non-euclideanstatistics for covariance matrices, with applications to dif-fusion tensor imaging.|
|96|Liangliang_Ren_Collaborative_Deep_Reinforcement_ECCV_2018_paper|Kim, D.Y., Jeon, M.: Data fusion of radar and image measurements for multi-object tracking via kalman filtering.|
|97|Unrolling the Shutter_ CNN to Correct Motion Distortions|A spline-based trajectory representation for sensor fusion and rollingshutter cameras.|
|98|Tianwei_Lin_BSN_Boundary_Sensitive_ECCV_2018_paper|To get final proposals set, we need to make score fusion to get final confidencescore, then suppress redundant proposals based on these score.|
|||Score fusion for retrieving.|
|||(4)After score fusion, we can get generated proposals set p = {n = (ts, te, pf )}Npn=1,where pf is used for proposals retrieving.|
|||Feichtenhofer, C., Pinz, A., Zisserman, A.: Convolutional two-stream network fusion forvideo action recognition.|
|99|cvpr18-Squeeze-and-Excitation Networks|Our winning en-try comprised a small ensemble of SENets that employeda standard multi-scale and multi-crop fusion strategy to ob-tain a 2.251% top-5 error on the test set.|
|100|Wenhao_Jiang_Recurrent_Fusion_Network_ECCV_2018_paper|In this paper, to exploit the complementary information from multipleencoders, we propose a novel recurrent fusion network (RFNet) for theimage captioning task.|
|||The fusion process in our model can exploit theinteractions among the outputs of the image encoders and generate newcompact and informative representations for the decoder.|
|||Multiple CNNs are employed as encoders and arecurrent fusion procedure is inserted after the encoders to form better representationsfor the decoder.|
|||The fusion procedure consists of two stages.|
|||In this paper, to exploit complementary information from multiple encoders,we propose a recurrent fusion network (RFNet) for image captioning.|
|||1, introduces a fusion procedure between the encodersand decoder.|
|||The fusion procedure per-forms a given number of RNN steps and outputs the hidden states as thoughtvectors.|
|||Our fusion procedure consists of two stages.|
|||2.3 Ensemble and Fusion LearningOur RFNet also relates to information fusion, multi-view learning [29], and en-semble learning [30].|
|||For the input fusion, the most simple wayis to concatenate all the representations and use the concatenation as input ofthe target model.|
|||For theoutput fusion, the results of base learners for individual views are combined toform the final results.|
|||The common ensemble technique in image captioning isregarded as an output fusion technique, combining the output of the decoder ateach time step [18, 19, 24].|
|||For the intermediate fusion, the representations fromdifferent views are preprocessed by exploiting the relationships among them toform input for the target model.|
|||Our method can be regarded as a kind ofintermediate fusion methods.|
|||The fusion stage I contains M review components.|
|||The fusion stage II is a review component thatperforms the multi-attention mechanism on the multiple sets of thought vectors fromfusion stage I.|
|||The parameters of LSTM units in the fusion procedure are all different.|
|||4 Our MethodIn this section, we propose our recurrent fusion network (RFNet) for imagecaptioning.|
|||The fusion process in RFNet consists of two stages.|
|||The fusionprocedure of RFNet consists of two stages, specifically fusion stage I and II.|
|||The hidden states and memory cellsafter the last step of fusion stage I are aggregated to form the initial hidden stateand the memory cell for fusion stage II.|
|||4.2 Fusion Stage IFusion stage I takes M sets of annotation vectors as inputs and generates M setsof thought vectors, which will be aggregated into one set of thought vectors infusion stage II.|
|||At time step t, the hidden states of the m-th review component areth(m), c(m)t = LSTM(m)twheret1 ,att-fusion-IA(m), h(m)Ht, f (m,t)h(1)t1...h(M )t1Ht =(8)(9)(10)tis the concatenation of hidden states of all review components at the previoustime step, f (m,t)att-fusion-I(, ) is the attention model for the m-th review component,and LSTM(m)(, ) is the LSTM unit used by the m-th review component at timestep t. Stage I can be regarded as a grid LSTM [33] with independent attentionmechanisms.|
|||In our model, the LSTM unit LSTM(m)(, ) can be different fordifferent t and m. Hence, M  T1 LSTMs are used in fusion stage I.|
|||In fusion stage I, the interactions among review components are realized viaEq.|
|||4.3 Fusion Stage IIThe hidden state and memory cell of fusion stage II are initialized with h(1)T1and c(1)T1.|
|||At each time step, the concatenation of contextvectors is calculated as: zt =...f (1,t)f (2,t)att-fusion-IIB(1), ht1att-fusion-IIB(2), ht1att-fusion-IIB(M ), ht1f (M,t),(12)where f (m,t)att-fusion-II(, ) is an attention model.|
|||The hidden states of fusion stage II are collected to form the thought vectorset:C = {hT1+1,    , hT1+T2 } ,(14)which will be used as the input of the attention model in the decoder.|
|||Recurrent Fusion Network for Image Captioning94.4 DecoderThe decoder translates the information generated by the fusion procedure intonatural sentences.|
|||The initial hidden state and memory cell are inherited fromthe last step of fusion stage II directly.|
|||(6), the complete loss function of our model is expressedas:Lall = L +M + 1Ld(C) +Mm=1LdB(m) ,(19)where  is a trade-off parameter, and B(m) and C are sets of thought vectorsfrom fusion stages I and II.|
|||With the recurrent fusion strategy, RFNet can extractuseful information from different encoders to remedy the lacking of informationabout objects and attributes in the representations.|
|||Ablation study of fusions stage I and II.|
|||To study the effects of the two fusionstages, we present the performance of the following models: RFNetI denotes RFNet without fusion stage I, with only the fusion stageII preserved.|
||| RFNetII denotes RFNet without fusion stage II.|
||| RFNetinter denotes RFNet without the interactions in fusion stage I.|
|||Eachcomponent in the fusion stage I is independent.|
|||Therefore, withthe specifically designed recurrent fusion strategy, our proposed RFNet providesthe best performance.|
|||Ablation study of fusion stages I and II in our RFNet.|
|||0110CIDEr105.2107.2107.3100104.76 ConclusionsIn this paper, we proposed a novel recurrent fusion network (RFNet), to exploitcomplementary information of multiple image representations for image caption-ing.|
|||In the RFNet, a recurrent fusion procedure is inserted between the encodersand the decoder.This recurrent fusion procedure consists of two stages, and eachstage can be regarded as a special RNN.|
|||Wang, J., Jiang, W., Ma, L., Liu, W., Xu, Y.: Bidirectional attentive fusion withcontext gating for dense video captioning.|
|101|cvpr18-Ring Loss  Convex Feature Normalization for Face Recognition|This allows for score fusiontechniques to be developed.|
|||Since the protocol is template matching, we utilizethe same template score fusion technique we utilize in theIJB-A results with K = 2.|
|||Deep heterogeneous feature fusion for template-based face recognition.|
|102|cvpr18-Video Person Re-Identification With Competitive Snippet-Similarity Aggregation and Co-Attentive Snippet Embedding|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|103|cvpr18-Probabilistic Joint Face-Skull Modelling for Facial Reconstruction|A regional method for cranio-facial reconstruction based on coordinate adjustments and anew fusion strategy.|
|104|cvpr18-Unsupervised Sparse Dirichlet-Net for Hyperspectral Image Super-Resolution|[27]further improved the fusion results by introducing a sparseconstraint.|
|||Mtf-tailored multiscale fusion of high-resolution ms and panimagery.|
|||Hyperspectral and multispectral image fusion based on asparse representation.|
|||Hyperspec-tral and multispectral data fusion: A comparative review ofthe recent literature.|
|||Coupled nonnegativematrix factorization unmixing for hyperspectral and multi-spectral data fusion.|
|||Syn-thesis of multispectral images to high spatial resolution: Acritical review of fusion methods based on remote sensingphysics.|
|||Aerial vehi-cle tracking by adaptive fusion of hyperspectral likelihoodmaps.|
|105|cvpr18-Polarimetric Dense Monocular SLAM|For robustness and accuracy, fusion algorithms use RGB-D cameras [27, 6, 40], which however increase power con-sumptions and limit the operating capabilities to short-rangeindoor scenes.|
|||Typical results with surface fusion Figure 6 shows ourresults on various indoor and outdoor scenes, where thefirst row shows some sample input frames, and the secondand third rows are the camera trajectories and fused densemesh respectively.|
|||We do notinclude Remode[30] because it tends to generate a sparseset of reliable points, which leads to poor fusion.|
|||Bundlefusion: Real-time globally consistent 3d reconstruc-tion using on-the-fly surface reintegration.|
|||Massively parallelmultiview stereopsis by surface normal diffusion.|
|||Kinectfusion: Real-time dense surfacemapping and tracking.|
|||Monofusion: Real-time 3d reconstruction ofsmall scenes with a single web camera.|
|106|Fine-To-Coarse Global Registration of RGB-D Scans|Bundlefusion: Real-time globally con-sistent 3d reconstruction using on-the-fly surface re-integration.|
|||Real-time 3d reconstructionin dynamic scenes using point-based fusion.|
|||Kinectfusion: Real-timedense surface mapping and tracking.|
|||Real-time large scaledense RGB-D SLAM with volumetric fusion.|
|107|A Minimal Solution for Two-View Focal-Length Estimation Using Two Affine Correspondences|Global fusion of rela-tive motions for robust, accurate and scalable structure frommotion.|
|108|Marie-Morgane_Paumard_Image_Reassembly_Combining_ECCV_2018_paper|Table 2: Accuracy for different fusion strategies, for the 8-classes classificationproblem on ImageNet validation.|
|||Ben-younes, H., Cadene, R., Cord, M., Thome, N.: Mutan: Multimodal tuckerfusion for visual question answering.|
|109|Minjun_Li_Unsupervised_Image-to-Image_Translation_ECCV_2018_paper|Moreover, to properly ex-ploit the information from the previous stage, an adaptive fusion block isdevised to learn a dynamic integration of the current stages output andthe previous stages output.|
|||To beneit more from multi-stage learning, we also introduce an adaptivefusion block in the reinement process to learn the dynamic integration of thecurrent stages output and the previous stages output.|
|||Secondly,we introduce a novel adaptive fusion block to dynamically integrate the cur-rent stages output and the previous stages output, which outperforms directlystacking multiple stages.|
|||Diferent form the existing works, this work exploits stacked image-to-imagetranslation networks combined with a novel adaptive fusion block to tackle theunsupervised image-to-image translation problem.|
|||2G2 consists of two parts: a newly initialized image translation network GTand an adaptive fusion block GF2 .|
|||Illustration of the linear combination in an adaptive fusion block.|
|||The fusionblock applies the fusion weight map  to ind defects in the previous result y1 andcorrect it precisely using y2 to produce a reined output y2.|
|||Besides simply using y2 as the inal output, we introduce an adaptive fusionblock GF2 to learn a dynamic combination of y2 and y1 to fully utilize the entiretwo-stage structure.|
|||Speciically, the adaptive fusion block learns a pixel-wiselinear combination of the previous results:GF(5)where  denotes element-wise product and   (0, 1)HW represents the fusionweight map, which is predicted by a convolutional network hx:2 (y1, y2) = y1  (1  x) + y2  x,(6)Figure 4 shows an example of adaptively combining the outputs from two stages.|
|||The adaptive fusion block is a simple 3-layer convolutional network,which calculates the fusion weight map  using two Convolution-InstanceNorm-ReLU blocks followed by a Convolution-Sigmoid block.|
|||To examine the efectiveness of the proposed fusion block, we compare itwith several variants: 1) Learned Pixel Weight (LPW), which is our proposedfusion block; 2) Uniform Weight (UW), in which the two stages are fused withthe same weight at diferent pixel locations y1(1  w) + y2w, and during trainingw gradually increases from 0 to 1; 3) Learned Uniform Weight (LUW), which issimilar to UW, but w is a learnable parameter instead; 4) Residual Fusion (RF),Unsupervised Image-to-Image Translation with SCAN11Table 2.|
|||FCN Scores and Segmentation Scores of several variants of the fusion blockon the Cityscapes dataset.|
|||which uses a simple residual fusion y1 + y2.|
|||It can be observed that our proposed LPW fusion yields the best performanceamong all alternatives, which indicates that the LPW approach can learn betterfusion of the outputs from two stages than approaches with uniform weights.|
|||Distributions of fusion weights over all pixels in diferent epochs.|
|||Dashedarrows indicate the average weights of fusion maps.|
|||4.6 Visualization of Fusion Weight DistributionsTo illustrate the role of the adaptive fusion block, we visualize the three aver-age distributions of fusion weights (x in Equation 5) over 1000 samples fromCityscapes dataset in epoch 1, 10, and 100, as shown in Figure 9.|
|||We observedthat the distribution of the fusion weights gradually shifts from left to right.|
|||It indicates a consistent increase of the weight values in the fusion maps, whichimplies more and more details of the second stage are bought to the inal output.|
|||Class IoUMethodSCAN Stage-1 128SCAN Stage-2 128-256 w/o Skip,FusionSCAN Stage-2 128-256 w/o SkipSCAN Stage-2 128-256 w/o FusionSCAN Stage-2 128-2560.4570.5130.5930.6130.6370.1880.1860.1840.1940.2010.1240.1250.1360.1370.157 SCAN w/o Adaptive Fusion Block: remove the inal adaptive fusion blockin the Stage-2 model , denoted by SCAN w/o Fusion.|
||| SCAN w/o Skip Connection and Adaptive Fusion Block: remove both theskip connection from the input to the translation network and the adaptivefusion block in the Stage-2 model , denoted by SCAN w/o Skip, Fusion.|
|||Table 4 shows the results of the ablation study, in which we can observe thatremoving the adaptive fusion block as well as removing the skip connection bothdowngrade the performance.|
|||Note thatthe fusion block only consists of three convolution layers, which have a relativelysmall size compared to the whole network.|
|||Thus, theimprovement of the fusion block does not simply come from the added capacity.|
|||Therefore, we can conclude that using our proposed SCAN structure, whichconsists of the skip connection and the adaptive fusion block, is critical forimproving the overall translation performance.|
|110|Learning a Deep Embedding Model for Zero-Shot Learning|(2) Asimple yet effective multi-modality fusion method is devel-oped in our neural network model which is flexible and im-portantly enables end-to-end learning of the semantic spacerepresentation.|
|||(ii) A multi-modality fusion method is further de-veloped to combine different semantic representations andto enable end-to-end learning of the representations.|
|||Score-level fusion is perhaps the simpleststrategy [14].|
|||Multiple semantic space fusionAs shown in Fig.|
|||More specifically, we map different semantic representa-tion vectors to a multi-modal fusion layer/space where theyare added.|
|||For multiple semantic space fusion, the multi-modal fusionlayer output size is set to 900 (see Fig.|
|111|cvpr18-Deformable GANs for Pose-Based Human Image Generation|In order to move informa-tion according to a specific spatial deformation, we decom-pose the overall deformation by means of a set of localaffine transformations involving subsets of joints, then wedeform the convolutional feature maps of the encoder ac-cording to these transformations and we use common skipconnections to transfer the transformed tensors to the de-coders fusion layers.|
|||For instance, on the Market-1501dataset, the G2R human confusion is one order of mag-nitude higher than in [12].|
|112|A New Representation of Skeleton Sequences for 3D Action Recognition|They are fed into five LSTMs for feature fusion and clas-sification.|
|||More specifically, the feature maps are pro-cessed with temporal mean pooling with kernel size 14  1,i.e., the pooling is applied over the temporal, or row dimen-sion, thus to generate a compact fusion representation fromall temporal stages of the skeleton sequence.|
|113|cvpr18-Learning to Localize Sound Source in Visual Scenes|Learning joint statistical models for audio-visual fusionand segregation.|
|114|KillingFusion_ Non-Rigid 3D Reconstruction Without Correspondences|On the one hand, this setting is easier for data fusion into thecumulative model.|
|115|Bidirectional Multirate Reconstruction for Temporal Modeling in Videos|[19] proposed a convolu-tional temporal fusion network, but it is only marginally bet-ter than the single frame baseline.|
|||We combine the three scores by average fusion.|
|||Note that each encoded representa-tion has the same feature vector length as the GoogLeNetmodel, and we use late fusion to combine the scores of thethree scales.|
|||Comparison with other models without fusion.|
|116|Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks|The two channels are then combined by latefusion and the whole network is end-to-end trainable.|
|||Here the fusion is performed by combining thesoftmax class posteriors from the two nets.|
|||We plot and compare the confusion matrices of our two-stream RNN and the temporal RNN on the SBU Interactiondataset in Figure 6.|
|||Comparison of confusion matrices on the SBU Interac-tion dataset.|
|117|cvpr18-A Twofold Siamese Network for Real-Time Object Tracking|In order tomake the semantic features suitable for the correlation op-eration, we insert a fusion module, implemented by 1  1ConvNet, after feature extraction.|
|||The fusion is performedwithin features of the same layer.|
|||The feature vector forsearch region X after fusion can be written as g(fs(X)).|
|||In the semantic branch, we only train the fusion mod-ule and the channel attention module.|
|||S1 is a basic version with onlyS-Net and fusion module.|
|||In the future,we plan to continue exploring the effective fusion of deepfeatures in object tracking task.|
|118|cvpr18-Video Based Reconstruction of 3D People Models|Free-form methods typically use multi-view cameras, depthcameras or fusion of sensors and reconstruct surface ge-ometry quite accurately without using a strong prior on theshape.|
|||One way tomake fusion and tracking more robust is by using multiplekinects [21, 47] or multi-view [63, 38, 16]; such methodsachieve impressive reconstructions but do not register all8388frames to the same template and focus on different appli-cations such as streaming or remote rendering for telepres-ence, e.g., in the holoportation project [47].|
|||In [23, 69]they pre-scan a template and insert a skeleton and in [78]they use a skeleton to regularize dynamic fusion.|
|||In contrast, we usea single RGB video of a moving person, which makes theproblem significantly harder as geometry can not be directlyunwarped as it is done in depth fusion papers.|
|||Kinectfusion: real-time 3d reconstruction and inter-action using a moving depth camera.|
|||Dynamicfusion:Reconstruction and tracking of non-rigid scenes in real-time.|
|||Kinectfusion: Real-time dense surface map-ping and tracking.|
|||Doublefusion: Real-time cap-ture of human performance with inner body shape from adepth sensor.|
|||Bodyfusion: Real-time capture of human mo-tion and surface geometry using a single depth camera.|
|119|Natalia_Neverova_Two_Stream__ECCV_2018_paper|Propagation of the context-based informationfrom visible to invisible parts that are entirely occluded at the present viewis achieved through a fusion mechanism that operates at the level of latentrepresentations delivered by the individual encoders, and injects global posecontext in the individual encoding through a concatenation operation.|
|||The fusion layer concatenates these obtained encodingsinto a single vector which is then down-projected to a 256-dimensional globalpose embedding through a linear layer.|
|||Typical failures of keypoint-based pose transfer (top) in comparison with Dense-Pose conditioning (bottom) indicate disappearance of limbs, discontinuities, collapseof 3D geometry of the body into a single plane and confusion in ordering in depth.|
|||Qualitatively, exploiting the inpainted representation has two advantages overthe direct warping of the partially observed texture from the source pose to thetarget pose: first, it serves as an additional prior for the fusion pipeline, and, sec-ond, it also prevents the blending network from generating clearly visible sharpartifacts that otherwise appear on the boarders of partially observed segmentsof textures.|
|120|cvpr18-Mask-Guided Contrastive Attention Model for Person Re-Identification|The compared methods include the body part-basedand pose-based methods, such as Spindle-Net [44], Deeply-Learned Part-Aligned Representations (DLPAR) [45], aswell as the fusion version of MSCAN [24].|
|||These methodstend to remove the background clutters and fusion the fea-tures of the body regions.|
|||Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|121|cvpr18-Multimodal Visual Concept Learning With Weakly Supervised Techniques|A possible explanation is that the seman-tically identical labels of the baseline usually consist of amore clean set, while the confusion introduced to the modelwith semantically similar labels rises.|
|||This confusion is compensated partially by eitherPLMIL or FSMIL.|
|||Multimodalsaliency and fusion for movie summarization based on aural,visual, and textual attention.|
|122|Thomas_Robert_HybridNet_Classification_and_ECCV_2018_paper|Ben-Younes, H., Cad`ene, R., Thome, N., Cord, M.: Mutan: Multimodal tuckerfusion for visual question answering.|
|123|Rui_Yu_Hard-Aware_Point-to-Set_Deep_ECCV_2018_paper|Zhao, H., Tian, M., Sun, S., Shao, J., Yan, J., Yi, S., Wang, X., Tang, X.: Spindlenet: Person re-identification with human body region guided feature decompositionand fusion.|
|124|cvpr18-End-to-End Weakly-Supervised Semantic Alignment|However, in [10] the matching is learned atthe object-proposal level and a non-trainable fusion step isnecessary to output the final alignment making the methodnon end-to-end trainable.|
|125|HU_Jian-Fang_Deep_Bilinear_Learning_ECCV_2018_paper|Comparison with other fusion and bilinear schemes.|
|||Here, we compareour bilinear learning framework with other fusion and bilinear schemes.|
|||As can be seen, our model offers distinct advantages over the hard-coded non-learning fusion methods (e.g., max and mean).|
|||Thusour bilinear model offers learning capability towards better fusion.|
|||By examining the results obtained bythe data-driven fusion schemes (e.g., FCN, linear-SVM, MCB and multi-kernelTable 5.|
|||Comparison with other fusion schemes, which used our feature netowrks.|
|||Block number 1Accuracy234583.8% 84.4% 85.4% 85.1% 84.9%learning (MKL)), we can see that data-driven fusion can achieve better resultsthan the hard-coded ones.|
|||Our method is also not sensitive to the order of fusion.|
|126|Xin_Li_Contour_Knowledge_Transfer_ECCV_2018_paper|SBFtrains the desirable deep saliency model by automatically generating reliablesupervisory signals from the fusion process of weak saliency models.|
|127|Slow Flow_ Exploiting High-Speed Cameras for Accurate and Diverse Optical Flow Reference Data|Optical flow with geometricocclusion estimation and fusion of multiple frames.|
|128|cvpr18-Focal Visual-Text Attention for Visual Question Answering|Mu-tan: Multimodal tucker fusion for visual question answering.|
|129|Deep Quantization_ Encoding Convolutional Activations With Deep Generative Model|MethodAccuracyTwo-stream ConvNet [29]C3D (3 nets) [33]Factorized ST-ConvNet [32]Two-stream + LSTM [45]Two-stream fusion [4]Long-term temporal ConvNet [34]Key-volume mining CNN [49]TSN (3 modalities) [42]IDT [40]C3D + IDT [33]TDD + IDT [41]Long-term temporal ConvNet + IDT [34]FV-VAE-pool5FV-VAE-pool5 optical flowFV-VAE-res5cFV-VAE-(pool5 + pool5 optical flow)FV-VAE-(res5c + pool5 optical flow)FV-VAE-(res5c + pool5 optical flow) + IDT88.1%85.2%88.1%88.6%92.5%91.7%93.1%94.2%85.9%90.4%91.5%92.7%83.9%89.5%86.6%93.7%94.2%95.2%VAE and GA are then with the same dimension of 4,096.|
|||For fair comparison, two basic and widely adoptedmodalities, i.e., video frame and optical flow image, areconsidered as inputs to our visual representation frameworkand late fusion is used to combine classifier scores on thetwo modalities.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|130|Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper|tensor fusion, whichis used to reduce the overhead of small operations) are switched on.|
|132|Guo_Lu_Deep_Kalman_Filtering_ECCV_2018_paper|Table 2: Accuracy for different fusion strategies, for the 8-classes classificationproblem on ImageNet validation.|
|||Ben-younes, H., Cadene, R., Cord, M., Thome, N.: Mutan: Multimodal tuckerfusion for visual question answering.|
|133|Learning Object Interactions and Descriptions for Semantic Image Segmentation|recent advanced image segmentation system, incorporat-ing ResNet-101, multi-scale fusion, and CRF smoothinginto a unified framework.|
|||To identify the usefulness ofIDW dataset, IDW-CNN only inherits ResNet-101 fromDeepLab-v2, yet removing the other components such asmulti-scale fusion and CRF in DeepLab-v2.|
|||Toidentify the usefulness of IDW, for all experiments, IDW-CNN employs ResNet-101 of DeepLab-v2 as backbonenetwork, yet removing any pre- and post-processing suchas multi-scale fusion and CRF.|
|||Most of theseapproaches employed pre- and post-processing methodssuch as multiscale fusion and CRF to improve performance,while IDW-CNN does not.|
|134|Re-Sign_ Re-Aligned End-To-End Sequence Modelling With Deep Recurrent CNN-HMMs|The main building blocks of this archi-tecture are Inception modules which are fusion of multipleconvolutional layers with different receptive fields appliedto the output of a 1x1 convolution layer which serves asa dimensionality reduction tool.|
|||This is possible by fusion of two LSTMunits, one of which processes the sequence from the begin-ning and towards the end, while the other does the samefrom the end and towards the beginning.|
|135|Florian_Strub_Visual_Reasoning_with_ECCV_2018_paper|is an arbitrary fusion mecha-nism (concatenation, element-wise product, etc.).|
|||g can be any fusion mechanism that facilitates selecting the rele-vant context to attend to; here we use a simple dot-product following [33], thusg(ck, st) = ck  st .|
|136|Instance-Level Salient Object Segmentation|Both SSR-Net and MSVGG perform much better than VGG16, whichrespectively demonstrates the effectiveness of the refine-ment module and attention based multiscale fusion in MSR-Net.|
|137|Ming_Sun_Multi-Attention_Multi-Class_Constraint_ECCV_2018_paper|On the other hand,the idea of multi-scale feature fusion or recurrent learning has become increas-ingly popular in recent works.|
|||And the recurrent attentionCNN [10] alternates between the optimization of softmax and pairwise rankinglosses, which jointly contribute to the final feature fusion.|
|138|Learning Category-Specific 3D Shape Models From Weakly Labeled 2D Images|Silhouette and stereo fusionfor 3d object modeling.|
|139|Task-Driven Dynamic Fusion_ Reducing Ambiguity in Video Description|Task-Driven Dynamic Fusion: Reducing Ambiguity in Video DescriptionXishan Zhang12, Ke Gao1, Yongdong Zhang12, Dongming Zhang1, Jintao Li1,and Qi Tian31 Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China2 University of Chinese Academy of Sciences, Beijing, China3 Department of Computer Science, University of Texas at San Antonio{zhangxishan,kegao,zhyd,dmzhang,jtli}@ict.ac.cn,qitian@cs.utsa.eduAbstractIntegrating complementary features from multiple chan-nels is expected to solve the description ambiguity problemin video captioning, whereas inappropriate fusion strate-gies often harm rather than help the performance.|
|||Existingstatic fusion methods in video captioning such as concate-nation and summation cannot attend to appropriate featurechannels, thus fail to adaptively support the recognition ofvarious kinds of visual entities such as actions and object-s. This paper contributes to: 1)The first in-depth study ofthe weakness inherent in data-driven static fusion methodsfor video captioning.|
|||2) The establishment of a task-drivendynamic fusion (TDDF) method.|
|||It can adaptively choosedifferent fusion patterns according to model status.|
|||Extensive experimentsconducted on two well-known benchmarks demonstrate thatour dynamic fusion method outperforms the state-of-the-artresults on MSVD with METEOR scores 0.333, and achievessuperior METEOR scores 0.278 on MSR-VTT-10K.|
|||Com-pared to single features, the relative improvement derivedfrom our fusion method are 10.0% and 5.7% respectivelyon two datasets.|
|||While differ-ent fusion methods such as concatenation and summationhave been used in video captioning, the relative increaseobtained by fusing multiple-channel visual features is only0.1%1.7% [11] or even -0.7% [26].|
|||It reveals that existingvisual fusion strategies in video captioning have not madefull use of each channel of features and their correlation.|
|||Therefore,in the data-driven fusion method such as feature concate-nation, the appearance features are often enhanced, whilemotion features are suppressed.|
|||Such static fusion modelscannot adaptively support the recognition of three differentkinds of visual entities, which result in description ambigu-ity, including recognition error and detail deficiency.|
|||To alleviate description ambiguity, we propose a task-driven dynamic fusion approach which can adaptively at-tend to certain visual cues based on the current model status,so that the generated visual representation will be most rele-vant to the current word.|
|||The fusion model consists of threedifferent fusion patterns, which support the recognition ofthree kinds of visual entities separately.|
|||Three different fusion patterns aredesigned to support the recognition of appearance-centric,motion-centric and correlation-centric entities.|
|||The fusionmodel learns to dynamically choose one of the three fusionpatterns appropriately according to task status.|
|||In summary, we make the following contributions: In-depth study of the weakness inherent in data-drivenstatic fusion methods for video captioning.|
|||Existingstatic fusion methods cannot adaptively support therecognition of various kinds of visual entities, whichresults in description ambiguity, including recognitionerror and detail deficiency.|
||| A task-driven dynamic fusion (TDDF) model is pro-posed to adaptively choose different fusion patterns ac-cording to task status.|
|||The dynamic fusion model canattend to certain visual cues that are most relevant tothe current word.|
||| Extensive experiments conducted on two well-knownvideo captioning benchmarks, MSVD and MSR-VTT-10K demonstrate that our dynamic fusion methodachieves noticeable gains by appropriately integratingmultiple-channel features.|
|||Our proposed in-depthstudy of the fusion of motion and appearance informationin video captioning generates a joint representation by pro-moting individual feature channels and correlating compli-mentary features according to task status.|
|||Feature Fusion: All the existing feature fusion meth-ods in video caption are static fusion, which means thevisual fusion model is not affected by the previous gen-erated target words.|
|||The work includes score-level de-cision fusion [31, 28] and early-stage feature combina-tion [11, 19, 3, 22, 26].|
|||Decision fusion is achieved byaveraging a set of network predictors.|
|||However, the deci-sion fusion is not data-driven, since discrepant predictioncapabilities on different samples of the individual featuresare neglected.|
|||Our proposed dynamic fusion model can adaptively choosedifferent fusion patterns according to task status.|
|||Illustration of task-driven dynamic fusion (TDDF) in video captioning.|
|||However, there is a significant difference betweenour dynamic fusion and the wildly used attention mechanis-m.The attention mechanism deals with homogeneous fea-tures extracted from different samples (frames or regions).|
|||Our dynamic fusion deals with heterogeneous features evenfrom the same sample.|
|||In dynamic fusion, not the feature contentbut the kind of the feature will determines its relevance.|
|||Our dynamic fusion is built upon attention and extendsit one step further, which automatically determines whetherthe appearance of the dog, or the movement of the dog, orthe combination should be focused.|
|||We further enhance the encoder part by adding the task-driven dynamic fusion layer as shown in Figure.|
|||Illustration of the task-driven dynamic fusion unit.|
|||Layer 3 performs concatenation fusion.|
|||Layer 4 performs dynamicfusion, choosing appropriate pathway relevant to the current word.|
|||We first introduce two kinds of basicshallow fusion functions: concatenation fusion and sum ormax fusion.|
|||1) Concatenation fusion.|
|||The fusion function isVMS() = WF([VM(), VS()] )= WFl VM() + WFr VS()(11)where motion features and appearance features are concate-nated together [VM(), VS()]  R+ .|
|||The concatenation fusionis widely applied in multi-modal learning [17] and recentinception module in GoogLeNet [25, 24].|
|||The concatena-tion fusion is capable of modeling correlations within andacross features.|
|||However, the fusion parameters are fixedonce learned.|
|||2) Sum or max fusion.|
|||The fusion function is theelement-wise sum VMS() = VM()  +VS() or theelement-wise max VMS() = max{VM(), VS()}.|
|||These parameter-free fusion functions usually applied tofeatures of same kind so the element-wise addition or max3716is reasonable.|
|||Sum fusion is applied to the shortcut connec-tion in Residual Network [10] and the combination layersin FractalNet [14].|
|||However, different from concatenationfusion, sum or max fusion can hardly model the correlationbetween different dimensions of heterogeneous features.|
|||1 and ()1 VM() +()3) Dynamic fusion.|
|||We propose a fusion functionthat is the element-wise weighted-sum of feature channelsVMS() = ()2 VS().|
|||Therefore, the sum ormax fusion can be transformed to a special case of the dy-namic weighted-sum fusion.|
|||The idea of dynamic fusion is sim-ilar to the idea of attention mechanism [33, 36, 35] in thesense that both of them deal with how well the inputs arerelated to the target words.|
|||Dynam-ic fusion deals with heterogeneous feature channel  with =  (ht1, ) :=  (ht1).|
|||The fusion weights aredetermined by the type of the feature  instead of the con-tent of the feature.|
|||Similar to the sum fusion, the element-wise weighted-sum hardly models the correlation betweenmultiple features, and it is not reasonable to do element-wise addition for heterogeneous features.|
|||In design of task-driven dynamic fusion (TDDF) unit, wetake advantage of the above three kinds of fusion functions,which are related and complimentary.|
|||Then, the followed concatenation fusion Layer 3 is used tocombine the refined motion and appearance features, andgenerates the correlation pathway.|
|||At last, we apply a dy-namic fusion Layer 4 on the top of motion, appearance andcorrelation pathways.|
|||The three pathways correspond tothree different fusion patterns that are designed to supportthe recognition of appearance-centric, motion-centric andcorrelation-centric entities in video description.|
|||Dynamicfusion Layer 4 learns to adaptively choose one of the threefusion patterns according to task status through a dynamicweighing mechanism.|
|||Performance evaluation on MSVD(%)CIDEr(%)ROUGE-L(%)BLEU4(%)--(4.6%)(1.7%)(1.3%)(3.3%)(3.6%)-(3.7%)(4.8%)-0.5630.5420.6520.5580.6540.6630.6020.517---(15.8%)(-0.9%)(16.1%)(17.7%)(6.9%)--0.658(6.0%)--0.6750.6670.6800.6750.6810.6870.684------(0.7%)(0%)(0.9%)(1.8%)(1.3%)----(10.0%)0.730(29.7%)0.697(3.3%)0.4160.4120.4280.4170.4380.4520.4400.4190.4530.4990.4380.458--(2.9%)(0.2%)(5.3%)(8.6%)(5.8%)-(8.6%)(2.2%)-(10.1%)*fusion method has relatively (%) improvement over the best single featuresGoogLeNet [25].|
|||The proposed task-drivendynamic fusion unit is shown in Figure 3, Layer 2 and Lay-er 3 are fully connected layers with tanh function as activa-tion.|
|||Experimental ResultsBaseline Methods: First, we compare our task-drivendynamic visual fusion method (TDDF) with single featuremethods, denoted as VGG, GoogLeNet, and C3D.|
|||Then,as stated in Section 3.3, our TDDF unit takes advantage ofstatic fusion, so we compare to these methods: concatena-tion fusion denoted as CON, sum fusion denoted as SUMand max fusion denoted as MAX.|
|||C3D+Res [26] inves-tigates multimodal fusion.|
|||Though the whole frameworkincorporates audio modality, we compare with their visualfusion results.|
|||As some work also fuses multiple featuresand reports the results before and after fusion, we presenttheir relative improvement by fusion methods.|
|||Our task-driven dynamic visual fusion methodachieves the best METEOR and CIDEr scores among all themethods.|
|||by all the fusion method.|
|||The baseline static fusion methods also have im-provement over the single feature methods.|
|||Although CON, MAX-3 and SUM-3 considered the feature correlation through aconcatenation fusion layer, they still perform worse than ourfusion method.|
|||Theirrelative improvement obtained by the fusion method is lessthan our fusion method.|
|||Therefore, the fusion methods in thetask of video caption is worth exploring.|
|||Thoughs() is not the final fusion weights a(), it serves as a auto-matic switch to give us intuition in what features the modelis focusing on to predict the current word.|
|||ConclusionExisting static fusion methods cannot adaptively supportthe recognition of various kinds of visual entities, so therelative increase obtained by fusing multiple-channel visu-al features is limited.|
|||In this paper, we propose a task-driven dynamic visual fusion method for video caption-ing, which achieves state-of-the-art performance on pop-ular benchmarks.|
|||Our method adaptively chooses differ-ent fusion patterns according to task status.|
|||Three differ-ent fusion patterns are designed to support the recognitionof three visual entities respectively, including appearance-centric, motion-centric and correlation-centric entities.|
|||Thedynamic fusion model can attend to certain visual cues thatare most relevant to the current word, thus reducing ambi-guity in video description.|
|||Our task-driven dynamic fusionmethod can be added on any encoder-decoder based videocaptioning architecture, so any further improvement on re-lated architectures will promote the overall performance.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|||A parallel-fusionrnn-lstm architecture for image caption generation.|
|140|cvpr18-Fight Ill-Posedness With Ill-Posedness  Single-Shot Variational Depth Super-Resolution From Shading|Super-resolutionkeyframe fusion for 3D modeling with high-quality textures.|
|||In Geometry-driven diffusion in computer vision, pages135146.|
|141|Not Afraid of the Dark_ NIR-VIS Face Recognition via Cross-Spectral Hallucination and Low-Rank Embedding|1To avoid confusion, in this paper we will refer to the deep neural net-work used for feature extraction as DNN and to the convolutional neuralnetwork used to hallucinate full-resolution VIS images from NIR as CNN.|
|||A local-coloring method fornight-vision colorization utilizing image analysis and fusion.|
|142|cvpr18-Glimpse Clouds  Human Activity Recognition From Unstructured Feature Points|Fusing modalities is tradi-tionally done as late [40], or early fusion [57].|
|143|cvpr18-Recurrent Pixel Embedding for Instance Grouping|To compare quantitatively tothe state-of-the-art, we learn a fusion layer that combinesembeddings from multiple levels of the feature hierarchyfine-tuned with a logistic loss to make a binary prediction.|
|144|cvpr18-VoxelNet  End-to-End Learning for Point Cloud Based 3D Object Detection|There are also several multi-modal fusion methods thatcombine images and LiDAR to improve detection accu-racy [10, 16, 5].|
|145|cvpr18-Functional Map of the World|Images were gathered in pairs,consisting of 4-band or 8-band multispectral imagery in thevisible to near-infrared region, as well as a pan-sharpenedRGB image that represents a fusion of the high-resolutionpanchromatic image and the RGB bands from the lower-resolution multispectral image.|
|146|Yapeng_Tian_Audio-Visual_Event_Localization_ECCV_2018_paper|Ourexperiments support the following findings: joint modeling of auditoryand visual modalities outperforms independent modeling, the learned at-tention can capture semantics of sounding objects, temporal alignmentis important for audio-visual fusion, the proposed DMRN is effectivein fusing audio-visual features, and strong correlations between the twomodalities enable cross-modality localization.|
|||Keywords: audio-visual event, temporal localization, attention, fusion1IntroductionStudies in neurobiology suggest that the perceptual benefits of integrating visualand auditory information are extensive [9].|
|||Furthermore, we investigate several audio-visual featurefusion methods and propose a novel dual multimodal residual fusion network thatachieves the best fusion results.|
|||Our extensive experiments support the following find-ings: modeling jointly over auditory and visual modalities outperforms modelingindependently over them, audio-visual event localization in a noisy conditioncan still achieve promising results, the audio-guided visual attention can wellcapture semantic regions covering sounding objects and can even distinguishaudio-visual unrelated videos, temporal alignment is important for audio-visualfusion, the proposed dual multimodal residual network is effective in addressingthe fusion task, and strong correlations between the two modalities enable cross-modality localization.|
|||Feature fusion isone of the most important part for multimodal learning [8], and many differ-ent fusion models have been developed, such as statistical models [15], MultipleKernel Learning (MKL) [19,44], Graphical models [20,38].|
|||3: (a) Audio-visual event localization framework with audio-guided visual attentionand multimodal fusion.|
|||One timestep is illustrated, and note that the fusion networkand FC are shared for all timesteps.|
|||4.2 and a novel dual multi-modal residual fusion network in Sec.|
|||4.1 Audio-Visual Event Localization Networkt , ..., vkOur network mainly consists of five modules: feature extraction, audio-guidedvisual attention, temporal modeling, multimodal fusion and temporal labeling(see Fig.|
|||To better incorporate the two modalities, wetAudio-Visual Event Localization in Unconstrained Videos7introduce a multimodal fusion network (see details in Sec.|
|||The audio-visual representation ht is learned by a multimodal fusion network with audioand visual hidden state output vectors hvt as inputs.|
|||(b) Dual mul-timodal residual network for audio-visual feature fusionGiven that attention mechanism hasshown superior performance in many appli-cations such as neural machine translation[7] and image captioning [57,34], we use itto implement our audio-guided visual attention (see Fig.|
|||4.3 Audio-Visual Feature FusionOur fusion method is designed based on the philosophy in [51], which processesmultiple features separately and then learns a joint representation using a mid-dle layer.|
|||Given audio and visual features hat and hvtcompute the updated audio and visual features:from LSTMs, the DMRN willt + f (hat , hvt )) ,hat = (hahvt = (hvt + f (hat , hvt )) ,(5)(6)t and hvwhere f () is an additive fusion function, and the average of hais usedas the joint representation ht for labeling the video segment.|
|||Simply, we canstack multiple residual blocks to learn a deep fusion network with updated hatand hvt as inputs of new residual blocks.|
|||We argue that the network becomes harder to train with increasing parametersand one block is enough to handle this simple fusion task well.|
|||We empirically find that late fusion (fusion after tem-poral modeling) is much better than early fusion (fusion before temporal mod-eling).|
|||Temporal modeling by LSTMs can implicitly learn certain alignmentswhich can help make better audio-visual fusion.|
|||We compare our fusion method: DMRN with several network-based multi-modal fusion methods: Additive, Maxpooling (MP), Gated, Multimodal Bilinear(MB), and Gated Multimodal Bilinear (GMB) in [28], Gated Multimodal Unit(GMU) in [4], Concatenation (Concat), and MRN [29].|
|||Three different fusionstrategies: early, late and decision fusions are explored.|
|||Here, early fusion meth-ods directly fuse audio features from pre-trained CNNs and attended visualfeatures; late fusion methods fuse audio and visual features from outputs of twoLSTMs; and decision fusion methods fuse the two modalities before SoftmaxAudio-Visual Event Localization in Unconstrained Videos11Fig.|
|||To further enhance the performance of DMRN, we also introduce a vari-ant model of DMRN called dual multimodal residual fusion ensemble (DMRFE)12Y. Tian, J. Shi, B. Li, Z. Duan, and C. XuTable 1: Event localization prediction accuracy (%) on AVE dataset.|
|||Table 2 shows event lo-calization performance of different fusion methods.|
|||WeAudio-Visual Event Localization in Unconstrained Videos13Table 2: Event localization prediction accuracy (%) of different feature fusion methodson AVE dataset.|
|||Table 2 shows audio-visual event localization predictionaccuracy of different multimodal feature fusion methods on AVE dataset.|
|||OurDMRN model in the late fusion setting can achieve better performance than allcompared methods, and our DMRFE model can further improve performance.|
|||We also observe that late fusion is better than early fusion and decision fu-sion.|
|||The superiority of late fusion over early fusion demonstrates that temporalmodeling before audio-visual fusion is useful.|
|||We know that auditory and visualmodalities are not completely aligned, and temporal modeling can implicitlylearn certain alignments between the two modalities, which is helpful for theaudio-visual feature fusion task.|
|||The decision fusion can be regard as a type oflate fusion but using lower dimension (same as the category number) features.|
|||14Y. Tian, J. Shi, B. Li, Z. Duan, and C. XuThe late fusion outperforms the decision fusion, which validates that process-ing multiple features separately and then learning joint representation using amiddle layer rather than the bottom layer is an efficient fusion way.|
|||Our systematic study well supportsour findings: modeling jointly over auditory and visual modalities outperformsindependent modeling, audio-visual event localization in a noisy condition is stilltractable, the audio-guided visual attention is able to capture semantic regionsof sound sources and can even distinguish audio-visual unrelated videos, tempo-ral alignments are important for audio-visual feature fusion, the proposed dualresidual network is capable of audio-visual fusion, and strong correlations exist-ing between the two modalities enable cross-modality localization.|
|||: Gated multimodalunits for information fusion.|
|||: Learning joint statisticalmodels for audio-visual fusion and segregation.|
|147|Peiliang_LI_Stereo_Vision-based_Semantic_ECCV_2018_paper|: Incremental dense seman-tic stereo fusion for large-scale semantic scene reconstruction.|
|||McCormac, J., Handa, A., Davison, A., Leutenegger, S.: Semanticfusion: Dense 3dsemantic mapping with convolutional neural networks.|
|149|cvpr18-LIME  Live Intrinsic Material Estimation|This temporalfiltering and fusion approach is particularly useful for ourenvironment map estimation strategy (see Section 7), sinceit helps in integrating novel lighting directions sampled bythe object as the camera pans during the video capture.|
|||Confusion matrix of shininess prediction for classification(left) and regression of log-shininess (right).|
|||This becomes more evidentwhen we look at the distribution of the estimation accuracy forshininess over the classification bins in the confusion matrixin Figure 7.|
|||The confusion matrix for the classification task(left) is symmetric at the diffuse and specular ends, whereasfor the regression (right) it is more asymmetric and biasedtowards specular predictions.|
|||In case of a video as input, we integrate the low-and high-frequency lighting estimates of multiple time stepsinto the same environment map using the filtering and fusiontechnique described in Section 5.5.|
|150|A Message Passing Algorithm for the Minimum Cost Multicut Problem|They are grouped below intothree categories: primal feasible local search algorithms,linear programming algorithms and fusion algorithms.|
|||The fusion processcan either rely on column generation [47], binary quadraticprogramming [14] or any algorithm for solving integerLPs [13].|
|||Our approach combines the efficiency of localsearch with the lower bounds of LPs and the subproblems offusion, as we show in experiments with large and diverse in-stances of the problem (Sec.|
|151|AMC_ Attention guided Multi-modal Correlation Learning for Image Search|Late fusion (LF) and multi-modal inter-attentionnetworks (MTN) are applied on multi-modalities.|
|||(3) Late fusion network (LF) first calculates the simi-larity scores between the input query and each modality.|
|||We further applies thelate fusion network (LF) on two attention-guided modali-2649Query: snooki baby bumpVisual: 0.6534Language: 0.3466Query: snooki baby bumpVisual: 0.7128Language: 0.2872Query: silk twist hair stylesVisual: 0.5028Language: 0.4972Query: silk twist hair stylesVisual: 0.5631Language:  0.4369transport, white, attractive, buyer, object, elegance,young, glamour, activity, arm, speaker, woman,shopper, photomodel,seated, pregnant, appearance, paint, drinking, pretty, smile ...attractive, art, sunglasses, breakage, elegance, young, industrial, computer, cafe, belly, woman, candy, women, camera, cars, stroll, paint, singer, american, person, tourist, arrival, people ...nature, white, art, guard, color, rodent, event, attractive, little, heritage, dance, glamour, long, god, young, veil, hair, haircut, woman, eye, cut, hairstyle ...white, hair, lips, shawl, human, attractive, expression, glamour, lovely, american, young, woman, woman, eye, makeup, hairstyle ...Figure 4.|
|||The AMC Full model achieves thebest performance, with 0.6-1.0% increase in R@k scorescompared to late fusion model, 1.8-4.9% increase in R@kscores compared to DSSM-Key and 3.8-7.1% increase inR@k scores compared to DSSM-Img.|
|||Late fusion (LF) andinter-attention (MTN) networks are applied on multi-modalities.|
|||Same as the denotation in Sec 5.1, weapply latefusion (LF) and inter-attention (MTN) mecha-nisms to combine features from image modality and key-word modality (Key).|
|||In Table 6, we first combine keywordand image modalities using latefusion (Skip-Vgg-Key-LF).|
|152|cvpr18-Deep Marching Cubes  Learning Explicit Surface Representations|More formally, let D  RN N N denote a (discretized)signed distance field obtained using volumetric fusion [8] orpredicted by a neural network [10, 35] where N denotes thenumber of voxels along each dimension.|
|||As ShapeNet mod-els comprise interior faces such as car seats, we rendereddepth images and applied TSDF fusion at a high resolution(128  128  128 voxels) for extracting clean meshes andoccupancy grids.|
|||Oct-NetFusion: Learning depth fusion from data.|
|153|Person Search With Natural Language Description|This demonstrates that, the modal-ity fusion between image and word before or after LSTMhas little impact on the person search performance.|
|154|Turning an Urban Scene Video Into a Cinemagraph|Dynamicfusion:Reconstruction and tracking of non-rigid scenes in real-time.|
|155|Pengfei_Zhang_Adding_Attentiveness_to_ECCV_2018_paper|Someapproaches explore the temporal dynamics of the sequential frames by simply av-eraging/multiplying the scores/features of the frames for fusion [32,44,7].|
|156|Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild|The commonly used methodsinclude SVM, nearest neighbor, LDA, DBN and decision-level fusion on these classifiers [46].|
|||Confusion matrixes for cross-database experiments usingHOG features.|
|||Multiclass support vector machine(mSVM) and confusion matrix were used as the classifica-tion method and the assessment criteria respectively.|
|||The metric is the mean diagonal valueof the confusion matrix.|
|||Instead,we use the mean diagonal value of the confusion matrix asthe ultima metric.|
|||Results declined to55.98%, 58.45% and 65.12% respectively when using themean diagonal value of the confusion matrix as metric.|
|||Results declined to 28.84%, 33.65% and 35.76%respectively when using the mean diagonal value of theconfusion matrix as metric.|
|||The metric is the mean diagonal value of the confusion matrix.|
|157|Visual Dialog|requiresWhat is she doing?for Visual Dialog with 3 novel encoders Late Fusion: that embeds the image, history, and ques-tion into vector spaces separately and performs a latefusion of these into a joint embedding.|
|||VisDial can be viewed as afusion of reading comprehension and VQA.|
|||The recurrent block Rt embeds the ques-tion and image jointly via an LSTM (early fusion), embedseach round of the history Ht, and passes a concatenationof these to the dialog-RNN above it.|
|158|Quality Aware Network for Set to Set Recognition|So therepresentation of a set is a fusion of each images features,weighted by their quality scores.|
|159|cvpr18-Seeing Voices and Hearing Faces  Cross-Modal Biometric Matching|The only attempt we can find to solve asimilar task to the one proposed here (but only for videos,8428concatsoftmaxmodality shared weights modality specific weights meanmeanconcatN21mean faceN+2+1concatN21(1) Static(2) Dynamic fusion(3) N-wayFigure 2: The three main networks architectures used in this paper.|
|||From left to right: (1) The static 3-stream CNN architecture consistingof two face sub-networks and one voice network, (2) a 5-stream dynamic-fusion architecture with two extra streams as dynamic feature sub-networks, and finally (3) the N-way classification architecture which can deal with any number of face inputs at test time due to the conceptof query pooling (see Sec.|
|||The three streams are then combined througha fusion layer (via feature concatenation) and fed intomodality-shared fully connected layers on top.|
|||The fusionlayer is required to enable the network to establish a corre-spondence between faces and voices.|
|||As a consequence of using concatenation as a fusionlayer in our base architecture, the number of face streamscannot be adjusted during inference.|
|||These inputs are then fed intothe fusion architecture.|
|||Experiment 4: RGB + SDI fusion: In this experiment weuse the dynamic fusion architecture (figure 2, middle).|
|||Experiment 5: RGB + MDI fusion: We also use densesampling in order to obtain multiple aligned RGB and dy-Dynamic Formulation1.|
|||RGB + MDI FusionAI (Total)79.2  0.176.9  0.679.9  0.282.4  0.384.3  0.2Table 3: Results using different dynamic formulations for thedynamic matching F-V task; SDI: Single dynamic Image; MDI:Multiple Dynamic Images; The best performance is achieved us-ing both RGB and MDI fusion.|
|||These inputs are thenfed into the fusion architecture, and results are ensembled attest time (this is done in a similar manner to experiment 1,but over the dynamic images as well).|
|160|cvpr18-Single Image Dehazing via Conditional Generative Adversarial Network|Single image dehazing bymulti-scale fusion.|
|162|Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper|Our proposed continuousfusion layer encode both discrete-state image features as well as continu-ous geometric information.|
|||This fusion usually happens at a coarse level, with significant resolution loss.|
|||The proposed continuous fusion layer is capable of encoding denseaccurate geometric relationships between positions under the two modalities.|
|||Our proposed continuous fusion layer canbe considered as a special case that connects points between different modalities.|
|||Since most self-driving cars are equipped with both LIDAR andcameras, sensor fusion between these modalities is desirable in order to furtherboost performance.|
|||This fusing operation is non-trivial, as image features happen at dis-crete locations; thus, one needs to interpolate to create a dense BEV feature4M. Liang, B. Yang, S. Wang and R. UrtasunResNetBlock...ResNetBlockResNetBlockCamera StreamFusion LayersContinuous Fusion...Multi-scale fusionDetection HeaderContinuous FusionContinuous FusionAddAddAddResNetBlock...ResNetBlockLIDAR Stream+ResNetBlock+Fig.|
|||Continuous fusion layers are usedto fuse the image features onto the BEV feature maps.|
|||We design the continuous fusion layer to bridge multiple interme-diate layers on both sides in order to perform multi-sensor fusion at multiplescales.|
|||After that we propose a deep multi-sensor detection architecture usingthis new continuous fusion layer.|
|||2: Continuous fusion layer: given a target pixel on BEV image, we firstextract K nearest LIDAR points (Step 1); we then project the 3D points ontothe camera image plane (Step 2-Step 3); this helps retrieve corresponding imagefeatures (Step 4); finally we feed the image feature + continuous geometry offsetinto a MLP to generate feature for the target pixel (Step 5).|
|||Continuous Fusion Layer: Our proposed continuous fusion layer exploits con-tinuous convolutions to overcome the two aforementioned problems, namely thesparsity in the observations and the handling of the spatially-discrete featuresin camera view image.|
|||Given the input camera image feature map and a setof LIDAR points, the target of the continuous fusion layer is to create a denseBEV feature map where each discrete pixel contains features generated fromthe camera image.|
|||One difficulty of image-BEV fusion is thatnot all the discrete pixels on BEV space are observable in the camera.|
|||Comparison against Standard Continuous Convolution: Compared against stan-dard parametric continuous convolution [36], the proposed continuous fusionlayer utilizes MLP to directly output the target feature, instead of outputtingweights to sum over features.|
|||We use four continuous fusion layers to fuse multiplescales of image features into BEV network from lower level to higher level.|
|||Fusion Layers: Four continuous fusion layers are used to fuse multi-scale imagefeatures into the four residual groups of the BEV network.|
|||We initialize the image network with ImageNet pre-trained weights and initializethe BEV network and continuous fusion layers using Xavier initialization [14].|
|||Note thatthere is no direct supervision on the image stream; instead, error is propagatedalong the bridge of continuous fusion layer from the BEV feature space.|
|||Wetrain a 3D multi-sensor fusion detection model, where all seven regression termsin Equation 3 are used.|
|||We compare our continuous fusion model with a LI-DAR only model (LIDAR input), a sparse fusion model (no KNN pooling) anda discrete fusion model (no geometric feature).|
|||Our detector runs at > 15 frames per second, much faster than allother LIDAR based and fusion based methods.|
|||4.2 Ablation Study on KITTIContinuous fusion has two components which enables the dense accurate fusionbetween image and LIDAR.|
|||We investigate thesecomponents by comparing the continuous fusion model with a set of derivedmodels.|
|||The first derived model is a LIDAR BEV only model, which uses the BEVstream of the continuous fusion model as its backbone net and the same de-tection header.|
|||All continuous fusion models significantly outperform the BEVmodel in all six metrics, which demonstrates the great advantage of our model.|
|||The second derived model is a discrete fusion model, which has neither KNNpooling nor geometric feature.|
|||Continuous fusion models outperform the discrete fusion model in all metrics.|
|||For BEV detection, the discrete fusion model even has similar scores as the BEVDeep Continuous Fusion11VehiclePedestrianBicyclistn/an/aModelAP0.5 AP0.7 AP0.3 AP0.5 AP0.3 AP0.591.35 79.37 n/an/a93.26 81.41 78.87 72.46 70.97 57.63Ours (Continuous Fusion) 94.94 83.89 82.32 75.34 74.08 59.83Ours (BEV only)PIXORTable 3: Evaluation of multi-class BEV object detection on TOR4D dataset.|
|||We compare the continuous fusion model with the BEV baseline, and a recentLIDAR based detector PIXOR [37].|
|||When geometric feature is removed from MLP input, the performance ofthe continuous fusion model significantly drops.|
|||However, even when offsets areabsent, the continuous fusion model still outperforms the discrete one, whichjustifies the importance of interpolation by KNN pooling.|
|||Continuous fusion layer has two hyper-parameters, the maximum neighbordistance d and number of nearest neighbors k. Setting a threshold on the dis-tance to selected neighbors prevents propagation of wrong information from faraway neighbors.|
|||Our continuous fusion model, itsBEV baseline, and PIXOR [37] are compared.|
|||When x is very small, the fusionmodel and LIDAR models have similar performance.|
|||The fusion model achievesmore gains for long range detection.|
|||Evaluation results We compare the continuous fusion model with two baselinemodels.|
|||One is a BEV model which is basically the BEV stream of the fusionmodel.|
|||Our continuous fusion model significantly outperforms the other two LIDARbased methods on all classes(Table 3).|
|||Thecontinuous fusion model outperforms BEV and PIXOR [37] at most ranges, andachieves more gains for long range detection.|
|||The BEV andimage pairs and detected bounding boxes by our continuous fusion model areshown.|
|||High resolution images can be readily incorporatedinto our model, thanks to the flexibility of continuous fusion layer.|
|163|Zero-Shot Classification With Discriminative Semantic Representation Learning|To investigate the reason, we pro-duced the confusion matrices for the DSRL prediction re-sults without label propagation on the four datasets, whichare presented in Figure 1.|
|||We can see that the confusionmatrix on the aPY dataset contains more noise than on theother datasets, which suggests large prediction uncertain-ties.|
|||36.06-44.150.3446.230.5350.352.9756.290.4451.291.4271.520.7976.330.8379.120.5377.380.0687.220.2730.190.5930.410.2041.780.5250.260.0457.140.0782.170.7682.501.3283.830.2982.000.0085.400.2257.010.6258.870.7263.771.0866.480.1470.260.5024681012123456789105101520253035404550123456789102468101212345678910510152025303540455012345678910(a) aPY(b) AwA(c) CUB(d) SUNFigure 1: Visualization of confusion matrix from the DSRL prediction results on the four datasets.|
|||1[32] S. Wu, S. Bondugula, F. Luisier, X. Zhuang, and P. Natara-jan. Zero-shot event detection using multi-modal fusion ofweakly supervised concepts.|
|164|cvpr18-Dense 3D Regression for Hand Pose Estimation|Our contribution can be summarized as follows: we formulate 3D hand pose estimation as a dense re-gression through a pose re-parameterization that canleverage both 2D surface geometric and 3D coordinateproperties; we provide a non-parametric post-processing methodaggregating pixel-wise estimates to 3D joint coordi-nates; this post-processing explicitly handles the holis-tic estimation and ensures consensus between the 2Dand 3D estimates; we implement several baselines to investigate fusionstrategies for holistic regression and 2D joint detectionin a multi-task setup; such an analysis has never car-ried out before for hand pose estimation and providesvaluable insights to the field.|
|||This type of fusion schemeis translation invariant and can better generalize to differ-ent combinations of finger gestures.|
|||Impact of fusion strategies To further explore betterstrategies for fusion of 2D detection and 3D regression,we design an alternative method using the identical net-work architecture as detection+coordinate regression(seeFig.|
|||Our method provides a better fusion scheme be-tween 2D detection and 3D regression than previous state-of-the-art and the various baselines.|
|165|cvpr18-A Unifying Contrast Maximization Framework for Event Cameras, With Applications to Motion, Depth, and Optical Flow Estimation|These images canserve as input to more complex processing algorithms suchas visual-inertial data fusion, object recognition, etc.|
|166|An Empirical Evaluation of Visual Question Answering for Novel Objects|The image feature can beVGG, INC (Inception), EF (Early fusion of VGG, INC)or LF (Late fusion of VGG, INC), the auxilliary data canbe none (baseline), text (BookCorpus pre-trained AE)or text+im (BookCorpus + WeakPaired data pre-trainedAE) and the vocabulary can be train (only words fromtrain data of novel split), oracle (oracle case), gen (gen-eral case) or gen(exp) (vocabulary expansion in generalcase).|
|||+6.2% and +4.2% in A1.i OEQ and MCQ vs.+11.1% and +8.5% in A2.i OEQ and MCQ, both withearly fusion of VGG and Inception features, respectively,indicating that it is more difficult to improve performancefor more saturated methods.|
|167|Ubernet_ Training a Universal Convolutional Neural Network for Low-, Mid-, and High-Level Vision Using Diverse Datasets and Limited Memory|As in [29, 56] we keep the task-specific memory and computation budget low by applyinglinear operations within these skip layers, and fuse skip-layer results through additive fusion with learned weights.|
|||MethodMFMethodODS OIS AP0.764MDF [33]0.793FCN [34]DCL [34]0.815DCL + CRF [34] 0.822Ours, 1-Task0.8350.823Ours, 7-Task0.790 0.808 0.811HED-fusion [56]Multi-Scale [29]0.809 0.827 0.861Multi-Scale +sPb [29] 0.813 0.831 0.8660.815 0.835 0.862Ours, setup of [29]Ours, 1-Task0.791 0.809 0.8490.785 0.805 0.837Ours, 7-TaskMethodMean Median 11.25 22.5 3064.0 73.922.2VGG-Cascade [17]70.0 77.819.8VGG-MLP [2]61.2 68.2VGG-Design [53]26.9Ours, 1-Task  = 50 21.465.9 76.9Ours, 1-Task  = 523.360.8 72.7Ours, 1-Task  = 159.7 71.923.9Ours, 7-Task  = 126.752.0 65.938.647.942.035.331.129.824.215.312.014.815.617.618.122.0Table 6:Saliencyestimation: MaximalF-measure (MF) onPASCAL-S [35].|
|168|What Is and What Is Not a Salient Object_ Learning Salient Object Detector by Ensembling Linear Exemplar Regressors|As a result, their fusion can adaptively handle theSOD tasks in various scenarios, and the usage of shape de-scriptor and high-objectness proposals ensure the well sup-pression of non-salient objects.|
|||However, the scores of each linear ex-emplar regressor may fall in different dynamic ranges sothat their direct fusion will lead to inaccurate saliency maps(see Fig.|
|||The enhancement-based fusion strategy forcombining exemplar scores makes the learned salient de-tector emphasize more on the most relevant linear exemplar43264147Figure 6.|
|||Different fusion strategy for SOD.|
|||(a) Image, (b) ground-truth, (c) direct fusion by computing the maximum saliency value,(d) direct fusion by computing the mean saliency value, (e) en-hanced fusion by computing the mean saliency value after an en-hancement operation using a sigmoid function.|
|||In the third experiment, we test various fusion strategiesof exemplar scores.|
|||We compare 3 different fusion waysas shown in Fig.|
|||We find thatthe weighted F-measure of using the max and mean value ofraw exemplar scores as the final saliency value of a proposalis 0.540 and 0.588, while the weighted F-measure of usingenhancement-based fusion is 0.649.|
|169|Scene Flow to Action Map_ A New Representation for RGB-D Based Action Recognition With Convolutional Neural Networks|Previ-ous works have considered the depth and RGB modalitiesas separate channels and extract features for later fusion.|
|||Differently from theoptical flow-based late fusion methods on RGB and depth595data, scene flow extracts the real 3D motion and also explic-itly preserves the spatial structural information contained inRGB and depth modalities.|
|||Multiply(cid:173)Score Fusion for ClassificationAfter construction of the several variants of SFAM, wepropose to adopt one effective late score fusion method,namely, multiply-score fusion method, to improve the finalrecognition accuracy.|
|||The multiply score fusion methodis compared with the other two commonly used late scorefusion methods, average and maximum score fusion on bothdatasets.|
|||The other methods, WHDMM+SDI [52, 1], ex-tracted features and conducted classification with ConvNetsfrom depth and RGB individually and adopted multiply-score fusion for final recognition.|
|||Interestingly, the proposed variants ofSFAM are complementary to each other and can improveeach other largely by using multiply-score fusion.|
|||Bag of visualwords and fusion methods for action recognition: Compre-hensive study and good practice.|
|170|Johannes_Schoenberger_Learning_to_Fuse_ECCV_2018_paper|proposals and the WTA step should be replaced by a more general fusion step.|
|||Based on this insight, we formulate the fusion step as the task of selecting thebest amongst all the scanline optimization proposals at each pixel in the image.|
|||SGM-Forest uses this fusion method instead of SGMs sum-based aggregation and WTA steps and our results shows that it consistentlyoutperforms SGM in many different settings.|
|||In contrast, we use regular scanlineoptimization but propose a learning-based fusion step using random forests.|
|||Stereo matching has been solved by combining multiple disparity maps usingMRF fusion moves [24, 4, 44].|
|||UnlikeMRF fusion moves [24], our fusion method is not general.|
|||To overcome this problem, wepropose a novel fusion method to robustly compute the disparity dp from themultiple scanline costs Lr(p, d).|
|||4 Learning To Fuse Scanline Optimization SolutionsWe start by analyzing some difficult examples for scanline optimization in orderto motivate our fusion method and then describe the method in detail.|
|||This insight formsthe basis of our fusion model which is described next.|
|||The main challengefor robust and accurate scanline fusion is to identify the scanlines which agree onthe correct estimate.|
|||In our proposed approach, we cast the fusion of scanlinesas a classification problem that chooses the optimal estimate from the given setof candidate scanlines.|
|||Both methods perform worse thanbaseline SGM, underlining the need for a more sophisticated fusion approach.|
|||While the biggest accuracy improve-ment stems from the initial fusion step (see Table 1), the final filtering furtherimproves the results by eliminating spatially inconsistent outliers.|
|||In contrast to most learning-based methods, we demonstrate thatour learned fusion approach is general and extremely robust across different do-mains and settings: SGM-Forest performs well outdoors when trained on indoorscenes, handles different image resolutions, disparity ranges and diverse match-ing costs, and consistently outperforms baseline SGM by a large margin.|
|||5.5 Limitations and Future WorkOur current SGM and random forest implementation is CPU-based and is notreal-time capable since we buffer all scanline cost volumes before fusion.|
|||Poggi, M., Mattoccia, S.: Deep stereo fusion: combining multiple disparity hypothe-ses with deep-learning.|
|171|cvpr18-Action Sets  Weakly Supervised Action Segmentation Without Ordering Constraints|Convolutionaltwo-stream network fusion for video action recognition.|
|172|Chi_Li_A_Unified_Framework_ECCV_2018_paper|To learn discriminative pose features,we integrate three new capabilities into a deep Convolutional Neural Network(CNN): an inference scheme that combines both classification and pose regressionbased on a uniform tessellation of the Special Euclidean group in three dimensions(SE(3)), the fusion of class priors into the training process via a tiled class map,and an additional regularization using deep supervision with an object mask.|
|||1: Illustration of different learning architectures for single-view object pose estimation: (a)each object is trained on an independent network; (b) each object is associated with one outputbranch of a common CNN root; and (c) our network with single output stream via class priorfusion.|
|||This exacerbates the complexity ofview fusion when multiple correct estimates from single views do not agree on SE(3).|
||| We present a multi-view fusion framework which reduces single-view ambiguitybased a voting scheme.|
|||This can be mainly attributed to our class fusiondesign in learning discriminative class-specific feature so that similar objects can bewell-separated in feature space (e.g.|
|||We first introduce a single-view pose estimation network with threeinnovations: a new bin & delta pose representation, the fusion of tiled class map intoconvolutional layers and deep supervision of object mask at intermediate layer.|
|||:Kinectfusion: real-time 3d reconstruction and interaction using a moving depth camera.|
|173|cvpr18-A Robust Method for Strong Rolling Shutter Effects Correction Using Lines With Automatic Feature Selection|A spline-based trajectory representation for sensor fusion and rollingshutter cameras.|
|174|cvpr18-Learning Answer Embeddings for Visual Question Answering|Image and ques-tion features are then inputed into the SAN structure forfusion.|
|||They usually explore better architec-tures for extracting rich visual information [32, 2], or betterfusion mechanisms across multiple modalities [9, 31, 30].|
|||We notice that our proposed PMC model is orthogonal toall those recent advances in multi-modal fusion and neuralarchitectures.|
|||Mutan:Multimodal tucker fusion for visual question answering.|
|175|Emotion Recognition in Context|Doubt/Confusion: difficulty to understand or decide; thinking aboutdifferent options13.|
|||Some examplesare {anticipation, engagement, confidence}, {affection,happiness, pleasure}, {doubt/confusion, disapproval, an-noyance}, {yearning, annoyance, disquietment}.|
|||The model consists of two modules forextracting features and a fusion network for jointly estimat-ing the discrete categories and the continuous dimensions.|
|||The architecture consists of threemain modules: two feature extractors and a fusion module.|
|||Finally, the third moduleis a fusion network that takes as input the image and bodyfeatures and estimates the discrete categories and the con-tinuous dimensions.|
|||Features extracted from these two modules are combinedusing a separate fusion network.|
|||This fusion module firstuses a global average pooling layer on each feature map toreduces the number of features, and then, a first fully con-nected layer acts as a dimensionality reduction layer for theset of concatenated pooled features.|
|||Doubt/Confusion13.|
|||Joint facial ac-tion unit detection and feature fusion: A multi-conditionallearning approach.|
|176|Deep Sketch Hashing_ Fast Free-Hand Sketch-Based Image Retrieval|Particularly, natural images and their correspondingsketch-tokens are fed into a heterogeneous late-fusionnet, while the CNNs for sketches and sketch-tokensshare the same weights during training.|
|||3, theDSH framework includes the following two parts:1) Cross-weight Late-fusion Net: A heterogeneoustermed C1-net with two parallel CNNs is developed,inputconv1pooling1conv2pooling2conv3conv4conv5pooling3fc afc bhash C1inputconv1pooling1conv2 1conv2 2pooling2conv3 1conv3 2pooling3fc afc bhash C2-111133553333333333771111-141433333333333333771111C1-Net(NaturalImage)C2-Net(Free-handsketch/Sketch-tokens )-42121112111-32112112111-00201111000-0011011000032272279655559627272562727256131338413133841313384131325677409611102411m 1112002006463636431311283131128313112815152561515256151525677409611102411m 11Net (Bottom) and C2-Net (Middle).|
|||For natural images andtheir sketch-tokens, we form the deep hash function BI =sign(F1(O1; 1, 2)) from the cross-weight late-fusion32864Figure 3.|
|||Data fusion through cross-modality metric learningusing similarity-sensitive hashing.|
|177|Xiaoqing_Ye_3D_Recurrent_Neural_ECCV_2018_paper|Both of these strategies capture multi-scale local structures at the expenseof indirected and complex fusion strategy, as well as extra computation.|
|||Li, Z., Gan, Y., Liang, X., Yu, Y., Cheng, H., Lin, L.: Lstm-cf: Unifying contextmodeling and fusion with lstms for rgb-d scene labeling.|
|178|End-To-End Instance Segmentation With Recurrent Attention|Post-processing based on templatematching and instance fusion produces the instance identi-ties.|
|||Moreover,their bottom-up instance fusion method plays a crucial role(omitting this leads to a steep performance drop); this likelyhelps segment smaller objects, whereas our box networkdoes not reliably detect distant cars.|
|179|cvpr18-Dual Skipping Networks|Comparably, each sub-network ofBilinear-CNN uses the pre-trained VGG network and thereis no information flow between two networks until the finalfusion layer.|
|180|Nikolaos_Karianakis_Reinforced_Temporal_Attention_ECCV_2018_paper|Zhao, H., Tian, M., Sun, S., Shao, J., Yan, J., Yi, S., Wang, X., Tang, X.: Spindlenet: Person re-identification with human body region guided feature decomposi-tion and fusion.|
|181|cvpr18-Harmonious Attention Network for Person Re-Identification|We finally add the scaling layer for automaticallylearning an adaptive fusion scale in order to optimally com-bining the channel attention described next.|
|182|Wei_Liu_Learning_Efficient_Single-stage_ECCV_2018_paper|For the IoU threshold of 0.5, thiskind of score fusion is considerably better than both C1 and C2.|
|||Du, X., El-Khamy, M., Lee, J., Davis, L.: Fused dnn: A deep neural network fusionapproach to fast and robust pedestrian detection.|
|183|Changqian_Yu_BiSeNet_Bilateral_Segmentation_ECCV_2018_paper|In pursuit of better accuracy without loss of speed, we also research thefusion of two paths and refinement of final prediction and propose Feature FusionModule (FFM) and Attention Refinement Module (ARM) respectively.|
|||BiSeNet7Feature fusion module: The features of the two paths are different in level offeature representation.|
|||NVIDIA Titan XNVIDIA Titan XPMethod640360ms1280720 192010806403601280720 19201080fps msfps msfps msfps msfps msfpsSegNet [1] 69ENet [25]714.6135.4289213.546.863746Ours1Ours25 203.5 12 82.3 24438129.447.9211.621.641.423------------4 285.2 8 124.1 18295205.778.81357.334.4Ablation for feature fusion module: Based on the Spatial Path and Context Path,we need to fuse the output features of these two paths.|
|184|Semantically Coherent Co-Segmentation and Reconstruction of Dynamic Scenes|Incremental dense se-mantic stereo fusion for large-scale semantic scene recon-struction.|
|185|Zhao_Chen_Estimating_Depth_from_ECCV_2018_paper|Whelan, T., Kaess, M., Johannsson, H., Fallon, M., Leonard, J.J., McDonald, J.:Real-time large-scale dense rgb-d slam with volumetric fusion.|
|186|Yaojie_Liu_Face_De-spoofing_ECCV_2018_paper|The accuracy of different outputs of the proposed architecture and their fusions.|
|||Different fusion methods In the proposed architecture, three outputs can be utilizedfor classification: the norms of either the 0\1 map, the spoof noise pattern or the depthmap.|
|||Table 2 shows the performance ofeach output and their fusion with maximum and average.|
|||It shows that the fusion ofspoof noise and depth map achieves the best performance.|
|||Hence, for the rest of experiments, we report performance from the averagefusion of the spoof noise N and the depth map D, i.e., score = (kNk1 +(cid:13)(cid:13)(cid:13)Advantage of each loss function We have three main loss functions in our proposedarchitecture.|
|||The confusion matrices of spoof mediums classification based on spoof noise pattern.|
|187|Ruohan_Gao_Learning_to_Separate_ECCV_2018_paper|: Learning joint statisticalmodels for audio-visual fusion and segregation.|
|188|cvpr18-Multi-Cue Correlation Filters for Robust Visual Tracking|Although feature-level fusion methods [26, 31, 38] havebeen widely used or extended to boost the performance,there still leaves room for improvement.|
|||In HCF [31] orother methods [56, 38] that follow such a fusion strategy,the initial weight of high-level features is usually high suchthat semantic features play the dominant role in general.|
|||Therefore, the feature-level fusion4844approach sometimes still fails to fully explore the relation-ship of multiple features.|
|||Furthermore, it is quite difficult tohandle various challenging variations using a single model,and relying on a certain feature-level fusion strategy limitsthe model diversity to some extent.|
|||Since it is quite diffi-cult to design a satisfying feature-level fusion method thatsuits various challenging scenes, it is intuitive to design anadaptive switch mechanism to achieve better performance,which can flexibly switch to the reliable tracker depend-ing on what kind of challenging factors it is expert at han-dling.|
|||In other words, the performance of a single trackercan sometimes be unstable but the decision-level fusion ofthe outputs from multiple trackers can enhance the robust-ness effectively.|
|||Different from the DCF based methods mentionedabove, our algorithm considers not only feature-level fusionbut also decision-level fusion to better explore the relation-ship of multiple features, and adaptively selects the expertthat is suitable for a particular tracking task.|
|||In[19], a partition fusion framework is proposed to cluster re-liable trackers for target state prediction.|
|||Spe-cially, the fusion methods [19, 24] by analyzing the for-ward and backward trajectories require each tracker to runat least twice; (2) the trackers are just regarded as inde-pendent black boxes and their fusion result in each framedoes not feedback to the trackers [1, 44, 19], which fails tomake full use of the reliable fusion outputs; (3) if the fu-sion tracker number increases, the dynamic programmingbased fusion methods [25, 1] still bring obvious computa-tional burden (e.g., O(T N 2) for T frames and N trackers).|
|||Since HOG and ColorNames are bothlow-level features, we do not conduct coarse-to-fine fusionand just simply concatenate them to construct DCF.|
|||The MCCT-PSR (MCCTH-PSR) repre-sents the tracker adopts the fusion method and uses PSRmeasurement for adaptive update (Sec.|
|||MCCT-H MLDF SSAT TCNN C-COT CSR-DCF ECO MCCTAccuracyFailure RateEAO0.571.240.3050.540.480.960.830.311 0.321 0.3250.571.040.520.850.3310.510.850.3380.580.540.730.720.374 0.393feature-level fusion but also decision-level fusion to fullyexplore the strength of multiple features.|
|||A superior tracking ap-proach: Building a strong tracker through fusion.|
|||Multi-trackerpartition fusion.|
|189|cvpr18-Learning Patch Reconstructability for Accelerating Multi-View Stereo|Depth map based meth-ods [10, 12] compute individual depth maps for each viewfollowed by a fusion step to merge the depth maps into afinal 3D representation.|
|||We address this issue by introducing a Coarse-to-Fineplane Diffusion strategy (CFD, Figure 6) inspired by Wu etal.|
|||Massively parallelmultiview stereopsis by surface normal diffusion.|
|191|Attentional Push_ A Deep Convolutional Network for Augmenting Image Salience With Shared Attention Modeling in Social Scenes|We also included the resultsfor using the augmented BMS network, fed with SalNetsaliency during testing to illustrate the performance withsub-optimal information fusion.|
|192|Action Unit Detection With Region Adaptation, Multi-Labeling Learning and Optimal Temporal Fusing|Many proposed approaches face challengingproblems in dealing with the alignments of different face re-gions, in the effective fusion of temporal information, and intraining a model for multiple AU labels.|
|||3) An LSTM-based temporalfusion recurrent net(LSTM Net) is proposed to fuse static CNN features, whichmakes the AU predictions more accurate than with staticimages only.|
|||Experimental resultsare included in Section 5 where we evaluate our proposedapproach in terms of regions cropping, multi-label learningand temporal fusion, and performance comparison againstbaseline approaches are also given.|
|||Due to the fusion of both spatial CNNand temporal features, the AU detection performance in thiswork has improved significantly compared to existing ap-proaches.|
|||The current best network for temporal fusionis the Long Short Term Memory (LSTM) network [10].Asa recurrent net, it can memorize the previous features andstates, which can help current feature learning and estima-tion.|
|||To see if LSTM isuseful in AU detection, we have conducted experiments tocompare LSTM-based temporal fusion versus static imageAU prediction.|
|||That impliesthat the global information does have an important impacton the fusion learning.|
|||Comparison of static image and temporal fusion in AU detection on BP4DTable 2.|
|||To obtain the spatiotemporal fusion features, the last layerfeatures of the CNN and LSTM nets are concatenated.|
|||ConclusionIn this paper, we looked into three essential prob-lems, the region adaption learning, temporal fusion andsingle/multi-label AU learning, in AU detection and pro-posed a novel approach to address these problems.|
|||We finally explored the LSTM-based temporal fusion approach, which boosted the AU de-tection performance significantly compared to static image-based approaches.|
|193|FusionSeg_ Learning to Combine Motion and Appearance for Fully Automatic Segmentation of Generic Objects in Videos|FusionSeg: Learning to combine motion and appearance for fully automaticsegmentation of generic objects in videosSuyog Dutt JainBo XiongKristen GraumanUniversity of Texas at Austinsuyog@cs.utexas.edu, bxiong@cs.utexas.edu, grauman@cs.utexas.eduhttp://vision.cs.utexas.edu/projects/fusionseg/AbstractWe propose an end-to-end learning framework for seg-menting generic objects in videos.|
|||Each convolutional layer except the first 7 7 convolutional layer and our fusion blocks is a residual block [16],adapted from ResNet-101.|
|||Motion StreamOur complete video segmentation architecture consistsof a two-stream network in which parallel streams for ap-pearance and motion process the RGB and optical flow im-ages, respectively, then join in a fusion layer (see Fig.|
|||Another alternative would directly train the joint model thatcombines both motion and appearance, whereas we firstpre-train each stream to make it discover convolutionalfeatures that rely on appearance or motion alone, followedby a fusion layer (below).|
|||We can then train the fusion model with very limited an-notated video data, without overfitting.|
|194|cvpr18-COCO-Stuff  Thing and Stuff Classes in Context|Sensorfusion for semantic segmentation of urban scenes.|
|195|cvpr18-Jointly Optimize Data Augmentation and Network Training  Adversarial Data Augmentation in Human Pose Estimation|ankle, elbow, wrist), and left-right confusion.|
|||Interestingly, thepose network could handle the left-right confusions after theadversarial training.|
|||Towards real-time detection and tracking of spatio-temporal features: Blob-filaments in fusion plasma.|
|196|Chen_Zhu_Fine-grained_Video_Categorization_ECCV_2018_paper|We have explored three approachesto achieve the fusion.|
|||Our fusion result is achieved by adding RGB and flow scoresdirectly.|
|||Our method surpasses TSN on both RGB and optical flow by significantmargins, but the fusion result is a bit lower, which might due to sampling thesame frames for both RGB and flow at validation.|
|||We have already given ablation results in Table 3, and qualitative resultsdemonstrating a reduction of confusion in Fig.|
|||We have also computed theconfusion maps of our model with 73.7 accuracy and a TSN with 72.5 accuracyon Kinetics validation sets RGB frames.|
|||Our model has a systematicallylower confusion than the TSN model.|
|||13Fine-grained Video Categorization with RRAwaiting in liltossing contripe jumpiticklingwhistliyawnngingneitakng a showersword fightingthrowng batasting foodlliiiiswng dancingsweepng floorstretchng legstretchng armswngng legsiiiistickng tongue outsomersaultingsprayngiiiswngng on somethngisneezingsmokngslappngsniffingsingngiiisign language interpretingshooting goal (soccer)shooting basketballrock scissors paperiirecordng musicpumpng fistrobot dancingrippng paperiilpayng basketballlpasteringiiishakng handssalsa dancingshakng headshavng legsiimakng a sandwichmovng furnituremassagng feetmakng a cakeiiparkourfinger snappngexercising armeating hotdogfacepantingeating doughnutsliieating chpseating cakedrop kickngdrinkng shotsdrinkng beeriiidrinkngiiihockey stophgh kickheadbuttingfixng hairljaughngoggnghuggngiiidancing gangnam styedancing charlestondancing macarenalidong aerobicsiicrackng neckcleanng floorcookng eggiclappngiilchangng wheeceebratinglcryngicartwheeiiibendng metabendng backbrushng hairlnglilbeatboxngappaudngiianswering questionsair drummngiicatchng or throwng basebaillTrue Labelpassing American footbaipetting anmal (not cat)ll (not in game)Ground Truth Ours TSN Highest Confusion Ours TSN Ours TSN0.456 0.344 playing harmonica0.106 0.079 0.350 0.265beatboxing0.420 0.307 applauding0.079 0.072 0.341 0.235celebrating0.467 0.393 gymnastics tumbling 0.065 0.075 0.402 0.318cartwheeling0.201 0.257 0.339 0.1780.540 0.435 scrambling eggscooking egg0.125 0.114 0.205 0.1240.330 0.238 drinking beerdrinkingdrinking shots 0.253 0.169 drinking beer0.087 0.097 0.166 0.072 Top 3Confidenceours 1ours 2ours 3TSN 1TSN 2TSN 3Fig.|
|||Left: confusion maps of our model and TSN on Kinetics.|
|||Darker color indicates higher confidence,Column 1 and 4 are names of ground truth actions and actions with highest confusion.|
|||Both modelson most wrong classes , demonstrating the reduction of confusion.|
|||AccordingTo demonstrate the reduction of confusion brought by our model, in Fig.|
|||6 weto the annotations, the training and validation sets of Youtube-Birds each hasshow some of TSN and our models top-3 average confidences from the confusion28.94% and 29.26% of their frames containing the subject, while the trainingmatrix on confusing classes of the Kinetics dataset.|
|197|ER3_ A Unified Framework for Event Retrieval, Recognition and Recounting|In addition, several work [18, 50, 46, 51]also explore multiple features fusion strategies to furtherimprove the recognition performance.|
|||(vgg+res) denotes thelater fusion result.|
|||As shown in Table 4.5,the fusion result (MA+RNet-(vgg+res)) can further boostthe recognition performance (mAP = 87.1) and outperformsprevious work.|
|||MA+RNet-(vgg+res)denotes the result fused with audio and motion information usingadaptive fusion method [45].|
|||Democraticdiffusion aggregation for image retrieval.|
|||Multi-stream multi-class fusion of deep networks for video classi-fication.|
|198|Kyoungoh_Lee_Propagating_LSTM_3D_ECCV_2018_paper|2(e) shows the proposed p-LSTM which consists of one LSTM networkand one depth fusion layer.|
|||Then, the estimated3D joints are merged into the input 2D pose in the depth fusion layer of the first6K. Lee et al.|
|||A1: Algorithm of p-LSTMsInput: X (2D pose)Output: Y (3D pose)Variablesk: index of the p-LSTMK: number of the p-LSTMY k: output of the kth LSTM networkX k: output of the kth depth fusion layerLSTMk: kth LSTM networkDepthk: kth depth fusion layerFC: fully connected layerYPred: output of 3D posePropagating Connection: To re-flect the joint interdependency (JI)into our method, the body part basedstructural connectivity is carefullydealt with.|
|||To preventthe initial 2D pose from disappearing, each p-LSTM uses the input 2D pose asancillary data and merges it with its own output in the depth fusion layer.|
|||Inother words, the pose depth cue is created by integrating the 2D with 3D posesin the depth fusion layer, as shown in Fig.|
|||2(d), and createsthe pose depth cue for each depth fusion layer of the p-LSTMs.|
|||One stageof p-LSTMs consists of 9 LSTM blocks, 9 depth fusion layers and 2 FCs.|
|||How to set the propagating direction: The pose depth cue is createdthrough a depth fusion layer of a p-LSTM.|
|199|cvpr18-Structure Inference Net  Object Detection Using Scene-Level Context and Instance-Level Relationships|Pie charts:fraction of detections that are correct (Cor) or false positive dueto poor localization (Loc), confusion with similar objects (Sim),confusion with other VOC objects (Oth), or confusion with back-ground or unlabeled objects (BG).|
|||Conclusionconcatenationmax-poolingmean-poolingmean-poolingmean-pooling2221370.270.470.569.869.6fective fusion of the two separated updated hidden state hsand he of nodes respectively obtained by the modules ofScene and Edge.|
|200|Zorah_Laehner_DeepWrinkles_Accurate_and_ECCV_2018_paper|: Dynamicfusion: Reconstruction and trackingof non-rigid scenes in real-time.|
|||Tung, T., Nobuhara, S., Matsuyama, T.: Complete multi-view reconstruction ofdynamic scenes from probabilistic fusion of narrow and wide baseline stereo.|
|201|Pose-Aware Person Recognition|Fusion typeAverage poolingMax poolingElementwise multi-plicationConcatenationPose-aware weights(I)83.57%80.54%81.71%84.44%(II)87.78%85.51%86.32%87.62%89.05%Table 6: Comparison of different fusion schemes for com-bining (i) features during joint training (using frontalPSM) and (ii) pose-aware classifier scores during testing.|
|||A pose-aware fusion strategy is proposedto combine the classifiers using weights obtained from apose estimator.|
|202|ArtTrack_ Articulated Multi-Person Tracking in the Wild|Interestingly, BU-sparse+temporaloutperforms BU-full + temporal: longer-range connectionssuch as, e.g., head to ankle, may introduce additional con-fusion when information is propagated over time.|
|203|Hengshuang_Zhao_ICNet_for_Real-Time_ECCV_2018_paper|We provide in-depth analysis of our frameworkand introduce the cascade feature fusion unit to quickly achieve high-quality segmentation.|
|||Then cascade feature fusion unit andcascade label guidance strategy are proposed to integrate medium and high reso-lution features, which refine the coarse semantic map gradually.|
||| The developed cascade feature fusion unit together with cascade label guid-ance can recover and refine segmentation prediction progressively with a lowcomputation cost.|
|||2, along with the cascade feature fusion unit and cascade label guidance,for fast semantic segmentation.|
|||CFF stands for cascade feature fusion detailedin Sec.|
|||Instead it takes cascade image inputs (i.e., low-, medium- and highresolution images), adopts cascade feature fusion unit (Sec.|
|||Lightweighted CNNs (green dotted box) are adopted in higher resolution branches;different-branch output feature maps are fused by cascade-feature-fusion unit(Sec.|
|||Because weights and computation (in 17 layers) can be shared be-tween low- and medium-branches, only 6ms is spent to construct the fusion map.|
|||3.3 Cascade Feature FusionTo combine cascade features from different-resolution inputs, we propose a cascade fea-ture fusion (CFF) unit as shown in Fig.|
|||Cascade feature fusion.|
|||Newlyintroduced cascade-feature-fusion unit and cascade label guidance strategy in-tegrate medium and high resolution features to refine the coarse semantic mapgradually.|
|||Effectiveness of cascade feature fusion unit (CFF) and cascade label guid-ance (CLG).|
|||Cascade Structure We also do ablation study on cascade feature fusion unitand cascade label guidance.|
|||Compared tothe deconvolution layer with 3  3 and 5  5 kernels, with similar inferenceefficiency, cascade feature fusion unit gets higher mIoU performance.|
|||Comparedto deconvolution layer with a larger kernel with size 77, the mIoU performanceis close, while cascade feature fusion unit yields faster processing speed.|
|||With proposed gradual feature fusion steps and cascade label guid-4 https://youtu.be/qWl9idsCuLQICNet for Real-Time Semantic Segmentation13Fig.|
|||The major contributions include the new framework for sav-ing operations in multiple resolutions and the powerful fusion unit.|
|204|cvpr18-pOSE  Pseudo Object Space Error for Initialization-Free Bundle Adjustment|Global fusion of rela-tive motions for robust, accurate and scalable structure frommotion.|
|206|cvpr18-Appearance-and-Relation Networks for Video Classification|[19] first tested deep networks with different tem-poral fusion strategies on a large-scale and noisily-labeleddataset (Sports-1M) and achieved lower performance thantraditional features [44].|
|||[49] designed a temporal segmentnetwork (TSN) to perform sparse sampling and temporalfusion, which aims to learn from the entire video.|
|||Intuitively, the spatial and temporal features arecomplementary for action recognition and this fusion stepaims to compress them into a more compact representation.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|207|cvpr18-Flow Guided Recurrent Neural Encoder for Video Salient Object Detection|Videosaliency detection via spatial-temporal fusion and low-rankcoherency diffusion.|
|208|Yang_Shen_Egocentric_Activity_Prediction_ECCV_2018_paper|[8] using the observed gaze; (b) Two-stream CNN results with object-cnn, SVM-fusion and joint training [22]; (c) 2D and 3D Ego ConvNet results (H: Hand mask, C:Camera/Head motion, M: Saliency map) [28].|
|||Confusion matrix of our proposed method for activity prediction, best viewedin color.|
|||Theconfusion matrices (using two-stream LSTM with attention) are shown in Fig.|
|||The methods we show are fusionLSTMs of our method, motion-object joint training of Ma et al.|
|209|Dong_Li_Recurrent_Tubelet_Proposal_ECCV_2018_paper|We apply late fusionon the scores from different channels to obtain the final action score of a tubelet.|
|||Similar to RTP, the two-stream pipeline is employed and late fusionstrategy is utilized to fuse the two streams.|
|||Moreover, the fusion on any two orall three channels could further improve the results, indicating that the threechannels are complementary.|
|211|cvpr18-Deep End-to-End Time-of-Flight Imaging|Specifically,we treat depth generation as a multi-channel image fusionproblem, where a desired depth map is the weighted com-bination of the same scene measured at multiple [i, j]illumination-sensor configurations.|
|||Kinectfusion: real-time 3d reconstruction and inter-action using a moving depth camera.|
|212|cvpr18-Disentangled Person Image Generation|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|213|Deeply Aggregated Alternating Minimization for Image Restoration|In[9], a nonlinear diffusion-reaction process was modeled byusing parameterized linear filters and regularization func-tions.|
|||The TGV model [11] uses an anisotropic dif-fusion tensor that solely depends on the RGB image.|
|||On learning optimized re-action diffusion processes for effective image restoration.|
|214|cvpr18-IVQA  Inverse Visual Question Answering|Theyare further combined by a score fusion network, whose out-put is utilised as the confidence of final prediction.|
|||Beforethe VQA-iVQA fusion, the VQA model alone can achievea validation accuracy of 57.85, while after the final modelreaches an accuracy of 58.86, where the performance gain ismainly from the challenging number type (improved from34.94 to 38.71).|
|215|Huang_Predicting_Gaze_in_ECCV_2018_paper|The late fusion modulecombines the results of saliency prediction and attention transition to generatea final gaze map.|
|||Since our task is different from [12], we modify the kernel sizes of the fusion part,which can be seen in detail in Section 3.7.|
|||3.5 Late FusionWe build the late fusion module (LF) on top of the saliency prediction moduleand the attention transition module, which takes Gst as input and outputsthe predicted gaze map Gt.|
|||3.6 TrainingFor training gaze prediction in saliency prediction module and late fusion mod-ule, the ground truth gaze map G is given by convolving an isotropic Gaussianover the measured gaze position in the image.|
|||The late fusion module consists of 4 convolution layers fol-lowed by sigmoid activation.|
|||After training the attention transitionmodule, we fix the saliency prediction and the attention transition module totrain the late fusion module in the end.|
||| SP+AT d: The late fusion on top of SP and AT d. SP+AT s: The late fusion on top of SP and AT s.Quantitative results of different settings are shown in Figure 4.|
|||However, SP+AT d withthe late fusion module can still improve the performance compared with SP andAT s, even with the context learned from different tasks.|
|216|MIML-FCN+_ Multi-Instance Multi-Label Learning via Fully Convolutional Networks With Privileged Information|Note that the results shownfor [32] in Table 2 is a fusion of their system and VeryDeep,but our MIML-FCN+BB still achieves better performance.|
|217|cvpr18-Blind Predicting Similar Quality Map for Image Quality Assessment|Quality Maps FusionWe evaluate two different fusion schemes for combin-ing multi types predicted quality maps.|
|||SRCC and PLCC comparison for different fusion schemesand multi predicted quality maps combinations on TID2013Single streamMulti streamsSRCCPLCCSRCCPLCCFg MD S Fg MD S0.8530.8620.8730.8850.8250.8420.8730.8610.8250.8590.8210.854S Fg MD0.8550.8800.8340.868s are given in Table 4.|
|218|3D Face Morphable Models _In-The-Wild_|Kinectfusion: real-time 3d reconstruction and inter-action using a moving depth camera.|
|||Kinectfusion: Real-time dense surface map-ping and tracking.|
|219|Qingnan_Fan_Learning_to_Learn_ECCV_2018_paper|: Fason: First and second order information fusionnetwork for texture recognition.|
|220|cvpr18-Weakly Supervised Action Localization by Sparse Temporal Pooling Network|Convolutionaltwo-stream network fusion for video action recognition.|
|221|Shi_Yan_DDRNet_Depth_Map_ECCV_2018_paper|Inspired by the recentprogress on depth fusions [19, 26, 11], we generate reference depth maps from thefused 3D model.|
|||With fusion, heavy noise present in single depth map can bereduced by integrating the truncated signed distant function (TSDF).|
|||From thisperspective, our denoising net is learning a deep fusion step, which is able toachieve better depth accuracy than heuristic smoothing.|
|||These fusion methods are able to effectively reduce the noises in the scanningby integrating the TSDF.|
|||Recent progresses have extended the fusion to dynam-ic scenes [26, 11].|
|||The scan from these depth fusion methods can achieve veryclean 3D reconstruction, which improves the accuracy of the original depth map.|
|||Based on this observation, we employ depth fusion to generate a training datafor our denoising net.|
|||By feeding lots of the fused depth as our training datato the the network, our denoising net effectively learns the fusion process.|
|||For denoising part, a function D mapping a noisydepth map Din to a smoothed one Ddn with high-quality low frequency is learnedby a CNN with the supervision of near-groundtruth depth maps Dref , createdfrom a state of the art of dynamic fusion.|
|||To achieve this, we use the non-rigid dynamic fusion pipelineproposed by [11], which is able to reconstruct complete and good quality geome-tries of dynamic scenes from single RGB-D camera.|
|||Then we runthe non-rigid fusion pipeline [11] to produce a complete and improved mesh, anddeform it using the estimated motion to each corresponding frame.|
|||Three post-fusion convolutional layers is introduced to learn a betterchannel coupling.|
|||The temporal window in fusion systems would smoothout noise, but it will also wipe out high-frequency details.|
|||The time in TSDFfusion blocks the whole system from tracking detailed motions.|
|||We proposed a near-groundtruthtraining data generation pipeline, based on the depth fusion techniques.|
|||Enabledby the separation of low/high frequency parts in network design, as well as thecollected fusion data, our cascaded CNNs achieves state-of-the-art result in real-time.|
|||In: ECCV Workshop on multi-camera & multi-modal sensorfusion (2008)7.|
|||Izadi, S., Kim, D., Hilliges, O., Molyneaux, D., Newcombe, R., Kohli, P., Shotton,J., Hodges, S., Freeman, D., Davison, A., Fitzgibbon, A.: Kinectfusion: Real-time3d reconstruction and interaction using a moving depth camera.|
|||Lindner, M., Kolb, A., Hartmann, K.: Data-fusion of pmd-based distance-information and high-resolution rgb-images.|
|||: Dynamicfusion: Reconstruction and trackingof non-rigid scenes in real-time.|
|||: Kinectfusion: Real-time dense surface mappingand tracking.|
|||Or El, R., Rosman, G., Wetzler, A., Kimmel, R., Bruckstein, A.M.: Rgbd-fusion:Real-time high precision depth recovery.|
|||Riegler, G., Ulusoy, A.O., Bischof, H., Geiger, A.: Octnetfusion: Learning depthfusion from data.|
|||Yu, T., Guo, K., Xu, F., Dong, Y., Su, Z., Zhao, J., Li, J., Dai, Q., Liu, Y.:Bodyfusion: Real-time capture of human motion and surface geometry using asingle depth camera.|
|||Yu, T., Zheng, Z., Guo, K., Zhao, J., Dai, Q., Li, H., Pons-Moll, G., Liu, Y.: Dou-blefusion: Real-time capture of human performance with inner body shape from adepth sensor.|
|222|cvpr18-Egocentric Activity Recognition on a Budget|It can be seen that if we consider methods indi-vidually without any type of sensor fusion or extra featuressuch as the ones from optical flow, our methods have thehighest overall accuracy.|
|223|Hong-Min_Chu_Deep_Generative_Models_ECCV_2018_paper|Lin, G., Liao, K., Sun, B., Chen, Y., Zhao, F.: Dynamic graph fusion label propagation forsemi-supervised multi-modality classification.|
|224|cvpr18-Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation|A generative model for image segmentationbased on label fusion.|
|225|cvpr18-Defense Against Adversarial Attacks Using High-Level Representation Guided Denoiser|[8] as-sess the defending performance of a set of preprocessingtransformations on MNIST digits [17], including the per-turbations introduced by image acquisition process, fusionof crops and binarization.|
|226|A Combinatorial Solution to Non-Rigid 3D Shape-To-Image Matching|for non-metric pairwise terms [27, 55], orfusion moves [34]).|
|227|cvpr18-Residual Dense Network for Image Super-Resolution|Localfeature fusion in RDB is then used to adaptively learn moreeffective features from preceding and current local featuresand stabilizes the training of wider network.|
|||After fully ob-taining dense local features, we use global feature fusionto jointly and adaptively learn global hierarchical featuresin a holistic way.|
|||After extracting multi-level local densefeatures, we further conduct global feature fusion (GFF) toadaptively preserve the hierarchical features in a global way.|
|||The accumulated features are then adaptivelypreserved by local feature fusion (LFF).|
||| We propose global feature fusion to adaptively fusehierarchical features from all RDBs in the LR space.|
|||2, our RDN mainly consists four parts:shallow feature extraction net (SFENet), redidual denseblocks (RDBs), dense feature fusion (DFF), and finally theup-sampling net (UPNet).|
|||After extracting hierarchical features with a set of RDBs,we further conduct dense feature fusion (DFF), which in-cludes global feature fusion (GFF) and global residuallearning (GRL).|
|||Our RDB contains dense con-nected layers, local feature fusion (LFF), and local resid-ual learning, leading to a contiguous memory (CM) mecha-nism.|
|||Local feature fusion is then applied to adaptively fusethe states from preceding RDB and the whole Conv layersin current RDB.|
|||We name this operation as local featurefusion (LFF) formulated asFd,LF = H dLF F ([Fd1, Fd,1,    , Fd,c,    , Fd,C]) , (7)where H dLF F denotes the function of the 1  1 Conv layerin the d-th RDB.|
|||Dense Feature FusionAfter extracting local dense features with a set of RDBs,we further propose dense feature fusion (DFF) to exploithierarchical features in a global way.|
|||Our DFF consists ofglobal feature fusion (GFF) and global residual learning.|
|||Global feature fusion is proposed to extract the globalfeature FGF by fusing features from all the RDBsFGF = HGF F ([F1,    , FD]) ,(9)where [F1,    , FD] refers to the concatenation of feature-maps produced by residual dense blocks 1,    , D. HGF Fis a composite function of 1  1 and 3  3 convolution.|
|||All the otherlayers before global feature fusion are fully utilized withour proposed residual dense blocks (RDBs).|
|||We would alsodemonstrate the effectiveness of global feature fusion inSection 5.|
|||Implementation DetailsIn our proposed RDN, we set 3  3 as the size of allconvolutional layers except that in local and global featurefusion, whose kernel size is 1  1.|
|||Shallow feature extraction layers,local and global feature fusion layers have G0=64 filters.|
|||While in RDN, we combine dense connectedlayers with local feature fusion (LFF) by using local resid-ual learning, which would be demonstrated to be effective2475in Section 5.|
|||Last not theleast, we adopt global feature fusion to fully use hierarchi-cal features, which are neglected in DenseNet.|
|||Our RDB allow largergrowth rate by using local feature fusion (LFF), which sta-bilizes the training of wide network.|
|||Instead we use globalfeature fusion (GFF) and global residual learning to ex-tract global features, because our RDBs with contiguousmemory have fully extracted features locally.|
|||Ablation investigation of contiguous memory (CM), lo-cal residual learning (LRL), and global feature fusion (GFF).|
|||Ablation InvestigationTable 1 shows the ablation investigation on the effects ofcontiguous memory (CM), local residual learning (LRL),and global feature fusion (GFF).|
|||We find that local fea-ture fusion (LFF) is needed to train these networks prop-erly, so LFF isnt removed by default.|
|||This is mainly because RDN uses hier-archical features through dense feature fusion.|
|||The local feature fusion (LFF) not only stabi-lizes the training wider network, but also adaptively controlsthe preservation of information from current and precedingRDBs.|
|||Moreover, we propose global feature fusion(GFF) to extract hierarchical features in the LR space.|
|||Byfully using local and global features, our RDN leads to adense feature fusion and deep supervision.|
|||An edge-guided image interpolation al-gorithm via directional filtering and data fusion.|
|228|Real-Time Video Super-Resolution With Spatio-Temporal Networks and Motion Compensation|Specifically, we discuss theuse of early fusion, slow fusion and 3D convolutions forthe joint processing of multiple consecutive video frames.|
|||We study different treatments of the temporaldimension with early fusion, slow fusion and 3D convolu-tions, which have been previously suggested to extend clas-sification from images to videos [23, 37].|
||| Comparing early fusion, slow fusion and 3D con-volutions as alternative architectures for discoveringspatio-temporal correlations.|
|||We restrict our analysis to standard architec-tural choices and do not further investigate potentially ben-eficial extensions such as recurrence [24], residual connec-tions [15, 16] or training networks based on perceptual loss4779(a) Early fusion(b) Slow fusion(c) 3D convolutionFigure 2: Spatio-temporal models.|
|||In early fusion (a), the temporal depth of the networks input filtersmatches the number of input frames collapsing all temporal information in the first layer.|
|||In slow fusion (b), the first layersmerge frames in groups smaller than the input number of frames.|
|||(3)2.2.1 Early fusionMethods preprocessing I LR with bicubic upsamplingbefore mapping from LR to HR impose that the output num-ber of filters is nL1 = 1 [6, 22].|
|||An illustration of early4780fusion is shown in Fig.|
|||2.2.2 Slow fusionAnother option is to partially merge temporal information ina hierarchical structure, so it is slowly fused as informationprogresses through the network.|
|||This architecture, termed slow fusion,has shown better performance than early fusion for videoclassification [23].|
|||2b we show a slow fusion net-work where D0 = 5 and the rate of fusion is defined bydl = 2 for l  3 or dl = 1 otherwise, meaning that ateach layer only two consecutive frames or filter activationsare merged until the networks temporal depth shrinks to 1.|
|||Note that early fusion is an special case of slow fusion.|
|||2.2.33D convolutionsAnother variation of slow fusion is to force layer weights tobe shared across the temporal dimension, which has com-putational advantages.|
|||(8)bias & activation2|{z}In measuring the complexity of slow fusion networks withweight sharing we look at steady-state operation where theoutput of some layers is reused from one frame to the fol-lowing.|
|||Using SF, E5, S5, andS5-SW to refer to single frame networks and 5 frame inputnetworks using early fusion, slow fusion, and slow fusionwith shared weights, we show in Table 2 results for 7 and 9layer networks.|
|||As seen previously, early fusion networks attain a higheraccuracy at a marginal 3% increase in operations relativeto the single frame models, and as expected, slow fusionarchitectures provide efficiency advantages.|
|||Slow fusion isfaster than early fusion because it uses fewer features in theinitial layers.|
|||(8), slow fusion uses dl = 2in the first layers and nl = 24/Dl, which results in feweroperations than dl = 1, nl = 24 as used in early fusion.|
|||While the 7 layer network sees a considerable decreasein accuracy using slow fusion relative to early fusion, the 9layer network can benefit from the same accuracy while re-ducing its complexity with slow fusion by about 30%.|
|||Thissuggests that in shallow networks the best use of network re-sources is to utilise the full network capacity to jointly pro-cess all temporal information as done by early fusion, butthat in deeper networks slowly fusing the temporal dimen-sion is beneficial, which is in line with the results presentedby [23] for video classification.|
|||Using 7 layers withE5 nevertheless shows better performance and faster opera-tion than S5-SW with 9 layers, and in all cases we foundthat early or slow fusion consistently outperformed slowfusion with shared weights in this performance and effi-ciency trade-off.|
|||Motion compensated video SRIn this section, the proposed frame motion compensationis combined with an early fusion network of temporal depthD0 = 3.|
|||This results in a network that will4783Figure 4: CDVL 3 SR using single frame models (SF) andmulti frame early fusion models (E3-7).|
|||This ensures that the number of features per hidden layerin early and slow fusion networks is always the same.|
|||Spatio(cid:173)temporal video SR3.2.1 Single vs multi frame early fusionFirst, we investigate the impact of the number of inputframes on complexity and accuracy without motion com-pensation.|
|||We compare single frame models (SF) againstearly fusion spatio-temporal models using 3, 5 and 7 inputframes (E3, E5 and E7).|
|||The increase in complexityfrom early fusion is marginal because only the first layercontributes to an increase of operations.|
|||3.2.2 Early vs slow fusionHere we compare the different treatments of the temporaldimension discussed in Section 2.2.|
|||We assume networkswith an input of 5 frames and slow fusion models with fil-timised with Eq.|
|||Results for 3SR on CDVL are compared in Table 3 against a single frame(SF) model and early fusion without motion compensation(E3).|
|||To demon-strate its benefits in efficiency and quality we evaluate twoearly fusion models: a 5 layer 3 frame network (5L-E3) anda 9 layer 3 frame network with motion compensation (9L-E3-MC).|
|||# Layers6SFE3E3-MC37.71837.84237.928737.78037.88937.961837.81237.95638.019937.80037.98038.060Table 3: PSNR for CDVL 3 SR using single frame (SF)and 3 frame early fusion without and with motion compen-sation (E3, E3-MC).|
|||The early fusion motion compensated SR network (E3-MC) is initialised with a compensation and a SR networkpretrained separately, and the full model is then jointly op-3.4.2 Efficiency comparisonThe complexity of methods in Table 4 is determined by net-work and input image sizes.|
|||ConclusionIn this paper we combine the efficiency advantages ofsub-pixel convolutions with temporal fusion strategies topresent real-time spatio-temporal models for video SR.|
|229|Social Scene Understanding_ End-To-End Multi-Person Action Localization and Collective Activity Recognition|Convolu-tional two-stream network fusion for video action recogni-tion.|
|230|Dong_Yang_Proximal_Dehaze-Net_A_ECCV_2018_paper|Ancuti, C.O., Ancuti, C., De Vleeschouwer, C., Bekaert, P.: Color balance and fusion forunderwater image enhancement.|
|||: Gated fusion network forsingle image dehazing.|
|||Tripathi, A., Mukhopadhyay, S.: Single image fog removal using anisotropic diffusion.|
|231|Shervin_Ardeshir_Integrating_Egocentric_Videos_ECCV_2018_paper|In the fusion step (Sec 3.4), we combine visual and geometrical reasoning tonarrow down the search space and generate a set of candidate (ls,  ) pairs.|
|||After fusion is the over-14S. Ardeshir, and A. Borjiall performance after combining the re-identification method with our geometrical andspatiotemporal reasoning.|
|||After fusion shows the performance of our method if we replace our two stream network with themethods mentioned above.|
|232|Vassileios_Balntas_RelocNet_Continous_Metric_ECCV_2018_paper|One possible solution tothis, would be to actively enforce some notion of dissimilarity between the retrievednearest neighbours, therefore ensuring that the fusion operates on a more diverse set ofproposals.|
|233|DESIRE_ Distant Future Prediction in Dynamic Scenes With Interacting Agents|An RNN scene context fusion module jointly captures pastmotion histories, the semantic scene context and interactionsamong multiple agents.|
|||iWe achieve the goal by having an RNN that takes follow-ing input xt at each time step:xt = h(vi,t), p(yi,t; (I)), r(yi,t; yj\i,t, h Yj\i)i(4)iwhere vi,t is a velocity of Y (k)at t,  is a f c layer witha ReLU activation that maps the velocity to a high dimen-sional representation space, p(yi,t; (I)) is a pooling oper-ation that pools the CNN feature (I) at the location yi,t,r(yi,t; yj\i,t, h Yj\i) is the interaction feature computed bya fusion layer that spatially aggregates other agents hiddenvectors, similar to SocialPooling (SP) layer [3].|
|||Trajecto-ries of each agent are represented using RNN encoders andare combined together through a fusion layer within thearchitecture.|
|||Recurrent neural networks for driver activity anticipation viasensory-fusion architecture.|
|234|cvpr18-Pixels, Voxels, and Views  A Study of Shape Representations for Single View 3D Object Shape Prediction|We expect that improved depth fusion and meshreconstruction would likely yield even better results.|
|235|cvpr18-A Hybrid l1-l0 Layer Decomposition Model for Tone Mapping|Thanks to the rapid development ofhigh dynamic range (HDR) techniques in the past decade, theintact information of the scene can be recorded in a radiancemap by bracketed exposure fusion technique [2, 7].|
|||assumedthat the illumination is piecewise-smooth and proposed anonlinear diffusion based method for illumination estima-tion [19].|
|||Contrast enhancement bynonlinear diffusion filtering.|
|||Fast and effective l0 Gra-dient minimization by region fusion.|
|236|cvpr18-High Performance Visual Tracking With Siamese Region Proposal Network|GOTURN[13] adopts the Siamese network as feature extractor anduses fully connected layers as the fusion tensor.|
|237|cvpr18-Tangent Convolutions for Dense Prediction in 3D|We build a full confusion matrixbased on the entire test set, and derive the final scores fromit.|
|||LSTM-CF: Unifying context modeling and fusion with LSTMs forRGB-D scene labeling.|
|||Oct-NetFusion: Learning depth fusion from data.|
|||Incremental dense se-mantic stereo fusion for large-scale semantic scene recon-struction.|
|238|Zerong_Zheng_HybridFusion_Real-Time_Performance_ECCV_2018_paper|Our method combines non-rigid surface tracking and volumetric fusion to simultaneously recon-struct challenging motions, detailed geometries and the inner humanbody of a clothed subject.|
|||Significant fusion artifacts are reduced using a newconfidence measurement for our adaptive TSDF-based fusion.|
|||On the other end of the spectrum, the recent trend of using a single depthcamera for dynamic scene reconstruction [25, 12, 10, 32] provides a very conve-nient and real-time approach for performance capture combined with online non-rigid volumetric depth fusion.|
|||Combining IMUs with depth sensors within a non-rigid depth fusion frame-work is non-trivial.|
|||Moreover, previous tracking&fusion methods [25, 46] may generateseriously deteriorated reconstruction results for challenging motions and occlu-sions due to the wrongly fused geometry, which will further affect the trackingperformance, and vice versa.|
|||We thus propose a simple yet effective scheme thatjointly models the influence of body-camera distance, fast motions and occlu-sions in one metric, which guides the TSDF (Truncated Signed Distance Field)fusion to achieve robust and precise results even under challenging motions (seeFig.1).|
||| Adaptive Geometry fusion.|
|||To address the problem that previous TSDFfusion methods are vulnerable in some challenging cases (far body-cameradistance, fast motions, occlusions, etc.|
|||), we propose an adaptive TSDF fu-sion method that considers all the factors above in one tracking confidencemeasurement to get more robust and detailed TSDF fusion results.|
|||DoubleFusion [46] leveraged parametric body model(SMPL [18]) in non-rigid surface integration to improve the tracking, loop clo-sure and fusion performance, and achieved the state-of-the-art single-view humanperformance capture results.|
|||In summary,our pipeline performs hybrid motion tracking, adaptive geometry fusion, volu-metric shape-pose optimization and sensor calibration sequentially, as shown inFig.2.|
||| Adaptive Geometry Fusion To improve the robustness of the fusion step,we propose an adaptive fusion method that utilizes tracking confidence toadjust the weight of TSDF fusion adaptively.|
||| Volumetric Shape-Pose Optimization We perform volumetric shape-pose optimization after adaptive geometry fusion.|
|||Besides voxel collision, thesurface fusion still suffers from inaccurate motion tracking, which is a factor thatprevious fusion methods do not consider.|
|||Since the TSDFfusion step only needs node graph to perform non-rigid deformation [25], wemerge the two types of motion tracking confidence together to get a more ac-curate estimation of hybrid tracking confidence for each node.|
|||Sincethe quality of depth input is inversely proportional to body-camera distance andthe low quality depth will significantly deteriorate the tracking and fusion per-formance, the tracking confidence of all nodes declines when the body is far fromthe camera (Fig.|
|||For a voxel v, D(v) denotes the TSDF value of the voxel, W(v)denotes its accumulated fusion weight, d(v) is the projective signed distance10Z. Zheng et al.|
|||function (PSDF) value, and (v) is the fusion weight of v at current frame:(v) = XxkN (v)Ctrack (xk) , (v) =(0(v)(v) <  ,otherwise.|
|||Themajority of the running time is spent on the joint motion tracking (23 ms) andthe adaptive geometric fusion (6 ms).|
|||More-over, the erroneous motion tracking performance will lead to erroneous surfacefusion results (ghost hands and legs).|
|||With the per-frame calibration optimiza-tion algorithm, our system can generate accurate motion tracking and surfacefusion results as shown in Fig.8(d).|
|||We also evaluate the effectiveness of the adap-tive geometric fusion method.|
|||ing scenarios for detailed surface fusion, which include far body-camera distance,body-part occlusion and fast motion.|
|||We then compare our adaptive geometryfusion method against previous fusion method used in [26, 10, 45, 46].|
|||In Fig.9,the results of the previous fusion method are presented on the left side of eachsub-figure, while the reconstruction results with adaptive fusion are shown onthe right.|
|||4, the fusion weights in our system can be auto-matically adjusted (set to a very small value or skip the fusion step) in all thesituations, resulting in more plausible and detailed surface fusion results.|
|||Evaluation of adaptive fusion under far body-camera distance (a), occlusions(b) and fast motions (c).|
|||In each sub-figure, the left mesh is fused by previous fusionmethod and the right one is fused using our adaptive fusion method.|
|||Dou, M., Davidson, P., Fanello, S.R., Khamis, S., Kowdle, A., Rhemann, C.,Tankovich, V., Izadi, S.: Motion2fusion: Real-time volumetric performance cap-ture.|
|||: Dynamicfusion: Reconstruction and trackingof non-rigid scenes in real-time.|
|||: Dynamicfusion: Reconstruction and trackingof non-rigid scenes in real-time.|
|||Pons-Moll, G., Baak, A., Helten, T., M uller, M., Seidel, H.P., Rosenhahn, B.:Multisensor-fusion for 3d full-body human motion capture.|
|||Yu, T., Guo, K., Xu, F., Dong, Y., Su, Z., Zhao, J., Li, J., Dai, Q., Liu, Y.:Bodyfusion: Real-time capture of human motion and surface geometry using asingle depth camera.|
|||Yu, T., Zheng, Z., Guo, K., Zhao, J., Dai, Q., Li, H., Pons-Moll, G., Liu, Y.: Dou-blefusion: Real-time capture of human performance with inner body shape from adepth sensor.|
|239|cvpr18-Language-Based Image Editing With Recurrent Attentive Models|Inspired by the obser-vation aforementioned, we introduce a recurrent attentivefusion module in our framework.|
|||The fusion module takesas input the image features that encode the source image viaa convolutional neural network, and the textual features thatencode the natural language expression via an LSTM, andoutputs the fused features to be upsampled by a deconvolu-tional network into the target image.|
|||In the fusion module,recurrent attentive models are employed to extract distincttextual features based on the spatial features from differentregions of an image.|
|||A high-level diagram of our model, composed of a convolutional image encoder, an LSTM text encoder, a fusion module, adeconvolutional upsampling layer, with an optional convolutional discriminator.|
|||The framework is composed of a convolutional imageencoder, an LSTM text encoder, a fusion network thatgenerates a fusion feature map by integrating image and textfeatures, a deconvolutional network that generates pixel-wise outputs (the target image) by upsampling the fusionfeature map, and an optional convolutional discriminatorused for training colorization models.|
|||Recurrent attentive fusion module The fusion networkfuses text information in U into the M  N image featuremap V , and outputs an M  N fusion feature map, witheach position (image region) containing an editing featurevector, O = {oi : i = 1, .|
|||The fusion network is devised to mimic the human imageediting process.|
|||For each region in the source image vi, thefusion network reads the language feature map U repeat-edly with attention on different parts each time until enoughediting information is collected to generate the target imageregion.|
|||Note that Ot is the intermediate output ofthe fusion feature map at time step t.i; tg)).|
|||Each termi-nation gate generates a binary random variable accordingto the current internal state of its image region:  ti p(ftg(sti = 1, the fusion process for the imageregion vi stops at t, and the editing feature vector for thisimage region is set as oi = oti.|
|||When all terminate gates aretrue, the fusion process for the entire image is completed,and the fustion network outputs the fusion feature map O.|
|||k<tthe probability of stopping the fusion process at the i-thimage region of the feature map at time t.Inference Algorithm 1 describes the stochastic infer-ence process of the fusion network.|
|||The fusion networkoutputs for each image region vi an editing feature vectoroi at the ti-th step, where ti is controlled by the ith termi-nation gate, which varies from region to region.|
|||It takes as input the M N fusionfeature map O produced by the fusion module, and unsam-ples from O to produce a H  W  De editing map E ofthe same size as the target image, where De is the number ofclasses in segmentation and 2 (ab channels) in colorization.|
|||In the fusion network, the attention model has 16 units, theGRU cells use 16 units, and the termination gate uses alinear map on top of the hidden state of each GRU cell.|
|||Inthe fusion network, the attention model uses 512 units andthe GRU cells 1, 024 units, on top of which is a classifierand an upsampling layer similar to the implementation inSection 4.1.|
|||We attribute the superior performance to theunique attention mechanism used by our fusion network.|
|||In the fusion network, theattention model uses 128 units and the GRU cells 128units.|
|||The image encoder is composed of 2 deconvolu-tional layers, each followed by 2 convolutional layers,to upsample the fusion feature map to the target imagespace of 256  256  2.|
|||At the heart ofthe proposed framework is a fusion module that uses recur-rent attentive models to dynamically decide, for each regionof an image, whether to continue the text-to-image fusionprocess.|
|240|Alex_Locher_Progressive_Structure_from_ECCV_2018_paper|Jiang, N., Tan, P., Cheong, L.F.: Seeing double without confusion: Structure-from-motion in highly ambiguous scenes.|
|||Moulon, P., Monasse, P., Marlet, R.: Global fusion of relative motions for robust,accurate and scalable structure from motion.|
|241|cvpr18-End-to-End Deep Kronecker-Product Matching for Person Re-Identification|[5] proposed a network consisting of multiple branchesfor learning multi-scale features and one feature fusionbranch.|
|||Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|242|Cross-Modality Binary Code Learning via Fusion Similarity Hashing|Few methods consider topreserve the fusion similarity among multi-modal in-stances instead, which can explicitly capture their hetero-geneous correlation in cross-modality retrieval.|
|||In this pa-per, we propose a hashing scheme, termed Fusion Simi-larity Hashing (FSH), which explicitly embeds the graph-based fusion similarity across modalities into a commonHamming space.|
|||Inspired by the fusion by diffusion, ourcore idea is to construct an undirected asymmetric graphto model the fusion similarity among different modalities,upon which a graph hashing scheme with alternating opti-mization is introduced to learn binary codes that embedssuch fusion similarity.|
|||We argue thatsuch fusion similarity are more important for measuring the7380   The Framework of our proposed Fusion SimilarityFigure 1.|
|||Hashing (FSH).FSH explicitly embeds the graph based fusion sim-ilarity across modalities into a common Hamming space.|
|||In particular, it is shown that binary code learningby fusion similarity is more robust to noise compared withthat by indirectly preserving intra- and inter-modal similar-ity respectively.|
|||Thebiggest concern lies in the efficiency issue in building thefusion model, i.e., the fusion graph, which typically needsrelaxation on the eigen decomposition of the graph Lapla-cian, resulting in significant performance degeneration withthe growth of hash bits.|
|||To address the above problems, we propose a novelcross-modality hashing method, termed Fusion SimilarityHashing (FSH), which makes the attempt towards directlypreserving the fusion similarity from the multiple modal-ities to a common Hamming space.|
|||Such fusion similar-ity is robust to noise in capturing multi-modal relationshipamong instances.|
|||Different from the existing work of cross-modality hashing [12, 23], we argue that it is the fusion sim-ilarity, rather than the individual intra-modal similarity, thatshould be preserved in the common Hamming space.|
|||Tothat effect, an asymmetrical fusion graph is built, which si-multaneously captures the intrinsic relations according toheterogeneous and homogenous data with a low storagecost afterwards.. After that, we design an efficient objec-tive function to learn accurate binary codes and the corre-sponding hash functions in an alternating optimizing way.|
|||FSH first builds the similarity matrix ineach modality, and then combining them to construct a ma-trix that reflect the fusion similarity.|
|||To handle such problem, we propose analternating optimization algorithm, which also updates thefusion parameters, so as to find the optimal fusion graph togenerate more discriminated hash codes.|
|||Asmentioned, a graph matrix G is constructed to measure thefusion similarity among training instances, where G(, )indicates the affinity between instance  and .|
|||We do this by minimizing the quanti-zation error between the fusion similarity matrix G and theHamming similarity matrix G , which can be written as:G  G 2 ,(2)where    is the Frobenius norm of the matrix.|
|||Therefore, the key issue falls in the construction qual-ity of the fusion similarity matrix G  R.|
|||Inspiredby the Neighbor Set Similarity (NSS) [1], it is straight-forward to define our fusion similarity in the following:Given two instances with two modalities  = { , }and  = {}, the bi-modal NSS in the -th modality , 7381can be defined as:S  (),  () =12   ()(, ), ( )(3)where   () returns the -nearest neighbor index numbersaccording to the -th modal similarity measure.|
|||And thefusion similarity G(, ) across such two modalities can bedefined via:G(, )= {S1 1 (),  2 (), S2 2 (),  1 ()},(4)It is quite intuitive to extend the above bi-modal fusion sim-ilarity to multi-modal case, i.e.|
|||We then use the SIAG to rewrite the fusion similarity:G(, ) =L(, ),1=112 =L(, ) =S ,(),  ,(),(6)S ,(),  ,() =12   (, ),where    index numbers of -nearest anchors in the -th modality.|
|||Such hash func-tion learning can be easily integrated into the overall cross-modality similarity persevering, which is rewritten as:B,W=B B G2min +B B G2+ =1 B  (X)2(8).. B  {1, 1}, =1  = 1, 0    1,where  is a tradeoff parameter to control the weights be-tween minimizing the binary quantization and preservingthe fusion similarity.|
|||Therefore, to handle this prob-lem, the symmetric fusion similarity matrix G can be ap-proximated by Cholesky decomposition G = UU , whereU  R, and U can be represented as the low-rank ap-proximated to the high-dimensional matrix G. Althoughsuch decomposition makes the storage efficiency, it is stilltoo complexity to calculate such decomposition.|
|||minThen, we assume Bs = (B G)  {1, 1} to be thebinary anchors, and U to be the affinity matrix that mea-sures the fusion similarity between data points and anchorsbinary codes Bs.|
|||We then omit thefirst constant item and replace the third item with matrixA =  ) isthe fusion similarity at the -th modality among anchorpoints.|
|||The newfusion similarity matrix G is updated upon the definition inEq.6, which combines each modal asymmetric similarity toconstruct the matrix that reflect more accurate fusion simi-larity.|
|||For the asymmetric similarity matrix G R, its multiplication G G  R approximately es-timates the fusion similarity of anchor points, which is de-fined as A = Proof.|
|||Except these, we further compare the FSHwith a simple fusion graph construction, which just fuse theanchor graph in each modality, and we refer this as FSH-Swhich is commonly a strong baseline for both two cross-modality retrieval tasks4.|
|||It is worth noting that, for bothtasks, our FSH shows advantage on precision with all hashbits, which is mainly due to the fact that more accurate sim-ilarity got from fusion similarity with SIAG can find moreoptimal binary codes from multi-modality.|
|||We further com-pare the simple fusion way, named FSH-S, with the base-lines [12, 21, 5, 10].|
|||The results in Fig.2 and Tab.1 showthat the simple fusion model can also achieve second best5The percentage of mAP growth is obtained by the means of improve-ment on all hash bits.|
|||The Precision curves and Recall curves of all the algo-rithms on two benchmark when hash bit is 64.performance, which demonstrate that simple or complexityfusion similarity should be preserved in binary codes.|
|||How-ever, FSH is overall better than FSH-S. As shown in [1],NSS is robust to noise with different similarity measures,and such similar scheme in fusion construction makes theproposed FSH more robust for cross-modality retrieval task.|
|||This demonstrates that the fusion similarity has advan-tageous to produce more distinguished binary code on themodality with weak expression power, which subsequentlyenhances the performance of single-modality retrieval.|
|||It is shown that a large size ofanchors with small parameter  will bring more noise tothe fusion graph, which decreases the performance of theproposed FSH.|
|||As a conclusion, for the proposedFSH, asymmetric fusion graph with little anchor points canenhance the performance of cross-modality retrieval, whichsolves the large-scale problem of binary code learning effi-ciently.|
|||To this end, a fusiongraph is constructed to define the similarity among multi-modality instances.|
|||In this framework, com-bining neighbor set similarity with sample important anchorgraph can be embedded to the fusion graph matrix, leadingto the learning of more discriminative binary codes.|
|||The core idea is to directly preserve the fusion sim-This work is supported by the National Key TechnologyR&D Program (No.|
|||Unsu-pervised metric fusion by cross diffusion.|
|||Beyond diffusion process:Neighbor set similarity for fast re-ranking.|
|||Data fusion through cross-modality metric learningusing similarity-sensitive hashing.|
|243|Shitala_Prasad_Using_Object_Information_ECCV_2018_paper|It can in fact be consideredas a decision level fusion, because the final results from faster RCNN, whichare the bounding boxes of the objects and text, are fused with the knowledgegraph information.|
|244|cvpr18-Actor and Action Video Segmentation From a Sentence|We then explore a fusion of RGB and Flow streams by com-puting a weighted average of the response maps from eachstream.|
|245|Yuge_Shi_Action_Anticipation_with_ECCV_2018_paper|MethodELSTM [39]Within-class Loss [28]DP-SVM [42]S-SVM [42]Where/What [43]Context-fusion [16]fm+VGG16fm+kSVM+GAN+VGG16fm+Inception V3fm+RBF+GAN+Inception V3OthersOursAccuracy55%33%5%5%10%28%63%67%70%73%Table 2: Comparison of our model againststate-of-the-arts on UT-Interaction datasetfor action anticipation.|
|||MethodELSTM [39]Within-class Loss [28]Context-fusion [16]Cuboid Bayes [35]I-BoW [35]D-BoW [35]Cuboid SVM [38]BP-SVM [26]OursAccuracy84%48%45%25%65%70%32%65%97%8% improvement from baseline ELSTM, which is purely influenced by the implemen-tation of Feature Mapping RNN .|
|||Jain, A., Singh, A., Koppula, H.S., Soh, S., Saxena, A.: Recurrent neural networks for driveractivity anticipation via sensory-fusion architecture.|
|246|Video2Shop_ Exact Matching Clothes in Videos to Online Shopping Images|These features are then fed into the similarity net-work to perform pair-wise matching between clothing re-gions from videos and shopping images, in which a recon-figurable deep tree structure is proposed to automaticallylearn the fusion strategy.|
|||The proposed ap-proach attempts to allocate fusion nodes to summarize thesingle similarity located in different viewpoints.|
|||There are twotypes of nodes involved in the tree structure, i.e., similaritynetwork node (SNN) and fusion node (FN), correspondingto the leaves and the branches in the tree.|
|||After that, these results are passedto fusion nodes, which generate a scalar output controllingthe weight of similarity fusion.|
|||These fusion nodes will bepassed layer by layer to fuse the results of internal results.|
|||Here, eachlow-level fusion node is connected to a specific SNN.|
|||Theoutput of the low-level fusion node gij is a weighted scorenormalized by the scores of all fusion nodes connecting tothe same top-level fusion node:gij =(4)ei,jPi ei,jSimilarly, for the top-level fusion node F Nj , an interme-T (xj), where xjdiate variable j is computed as: j = vjis an average pooling vector from multiple low-level fusionnodes, which are connected to F Nj , vj is the parameters ofthis fusion node.|
|||The fusion score gj is normalized by thescores of all top-level fusion nodes as: gj = ejPj ej .|
|||Withsuch a tree structure, for each mini-batch, the parameters offusion nodes are updated in the forward pass.|
|||Once the sim-ilarity network converges, the fusion strategy is obtained.|
|||The learning is implemented ina two-step iteration approach, where similar network nodesand fusion nodes will be mutually enhanced.|
|||The featurerepresentation network and similar network nodes are firstlearned, and then the fusion nodes are learned when similarnetwork nodes are fixed.|
|||Once the individual SNN is calculat-ed, the fusion scores of all fusion nodes will be generatedwith a a tree-like structure.|
|||In this network, multiple low-level fusion nodes are connected to a higher-level fusion n-ode, which forms a tree-like structure.|
|||Thelow-level fusion nodes refer to the leaves while the top-levelis the side of the root.|
|||For a low-level fusion node F Nij ,( yklog (yk)+(1  yk) log (1  yk))+ kWik2(5)where Wi is the parameters of i-th SNN, yk is the outputof single similarity network with xk as the input, which isdefined in Eqn.|
|||gj and gij are the fusion scores of higher and low-level fusion nodes.|
|||6 is that thesimilarity of all similar network nodes are passed to multi-ply layers of fusion nodes to generate the results of globalsimilarity.|
|||6,the posterior probabilities of fusion nodes are defined.|
|||Thefusion scores of top-level gj and low-level gij are referredas prior probabilities, since they are computed without theknowledge of corresponding output of SNN yi (as calculat-ed in Eqn.|
|||With Bayes rule, the posteriorprobabilities at the top-level fusion nodes and low-level n-odes are denoted as follows:hj =gj Pi gijpi(y)Pj gj Pi gijpi(y)hij =gijpi(y)Pi gij pij(y)(7)(8)andWith these posterior probabilities, a gradient descent learn-ing algorithm is developed for Eqn.|
|||The log likelihoodfunction of a training sample is obtained as:l = lnXjgj Xigijpi(y)(9)In this case, by differentiating l with respect to the param-eters, the following gradient descent learning rules for theparameters of top-level and low-level fusion nodes are ob-tained as: vj = (hj  gj)xj(t) vij = hj(hij  gij)xij(10)(11)where  is a learning rate.|
|||vj and vij are the parameters ofhigh-level and low-level fusion nodes, respectively.|
|||Theseequations denote a batch learning algorithm to train fusionnodes (i.e.|
|||To form a deeper tree, each SNNis expanded recursively into a fusion node and a set of sub-SNN networks.|
|||In our experiment, we have five-level deeptree structure and the number of fusion nodes in each levelis 32, 16, 8, 4, 2, respectively.|
|||7-8;6: Train fusion nodes as Eqn.|
|||Structure Selection of Similarity NetworkTo investigate the structure of similarity network, wevary the number of levels and the fusion nodes in similari-ty network, while keeping all other common settings fixed.|
|||We evaluate two types of architectures: 1) Homogeneousbranches: all fusion nodes have the same number of branch-es; 2) Varying branches: the number of branches is incon-sistent across layers.|
|||For homogeneous setting, one-levelflat structure with 32 fusion nodes to hierarchical structurewith five levels (62 fusion nodes) are tested.|
|||As the training proceeds, theparameters in the fusion nodes begin to grow in magnitude,which means that the weights of fusion nodes are becomingmore and more reasonable.|
|||However, the improvement is not obvious after 4 e-pochs, since the weights of fusion nodes tend to be stable.|
|||As the training proceeds, the parame-ters in the fusion nodes begin to grow in magnitude.|
|||Whenthe fusion notes begin to take action, the performance of thesystem is boosted.|
|||We also notice that the general perfor-mance is increased when more levels of fusion nodes are in-volved.|
|||It indicates that the similar network be-comes stable when the levels of fusion nodes are more thanthree.|
|||Performance of Similarity Learning NetworkIn order to verify the effectiveness of our similarity net-work, we compare the performance of the proposed methodwith other methods when fusion nodes are not included.|
|||This is mainly because AsymNet can handlethe temporal dynamic variety existing in videos, and it inte-grates discriminative information of video frames by auto-matically adjusting the fusion strategy.|
|247|Visual-Inertial-Semantic Scene Representation for 3D Object Detection|There is also work that focuses on scene understandingfrom visual sensors, specifically video [37, 2, 42, 57, 6, 72],although none integrates inertial data, despite a resurgentinterest in sensor fusion [73].|
|||(1) is a diffusion aroundthe mean/mode gt|t, xt|t; if the covariance Pt|t is small, itcan be further approximated: Givengt|t, xt|t = arg maxgt,xpSLAM (gt, x|yt),(3).|
|||Semantic fusion: Dense 3d semantic mapping with convo-lutional neural networks.|
|||Robust filtering forvisual inertial sensor fusion.|
|||Incremental dense semantic stereofusion for large-scale semantic scene reconstruction.|
|248|cvpr18-Improving Occlusion and Hard Negative Handling for Single-Stage Pedestrian Detectors|Improving Occlusion and Hard Negative Handlingfor Single-Stage Pedestrian DetectorsJunhyug NohSoochan Lee Beomsu Kim Gunhee KimDepartment of Computer Science and EngineeringSeoul National University, Seoul, Korea{jh.noh, soochan.lee}@vision.snu.ac.kr, {123bskim, gunhee}@snu.ac.krhttp://vision.snu.ac.kr/projects/partgridnetAbstractWe propose methods of addressing two critical issuesof pedestrian detection: (i) occlusion of target objects asfalse negative failure, and (ii) confusion with hard nega-tive examples like vertical structures as false positive fail-ure.|
|||For reducing con-fusion with hard negative examples, we introduce averagegrid classifiers as post-refinement classifiers, trainable inan end-to-end fashion with little memory and time overhead(e.g.|
|||[24]systemically break down, we are interested in two criticalissues: (i) occlusion of target objects (as false negative fail-ure cases), and (ii) confusion with hard negative examples(as false positive failures).|
|||For reducing the con-fusion with hard negative examples, we introduce averagegrid classifiers as post-refinement classifiers, trainable in anend-to-end manner without large time and memory over-heads.|
|||(1) We propose an approach to address the two critical is-sues of pedestrian detection: (i) occlusion of objects, and (ii)confusion with hard negative examples.|
|||ConclusionWe addressed the two critical issues of pedestrian detec-tion: occlusion and confusion with hard negative examples.|
|249|cvpr18-Learning for Disparity Estimation Through Feature Constancy|3) Multi-scalefusion features(i.e., up 1a2a andup 1b2b).|
|||Using the full-resolution multi-scale fusion features as skipconnection features, we are able to estimate initial dispar-ity of full resolution.|
|||The multi-scale fusion features arealso used to calculate the reconstruction error, as will be de-scribed in Sec.|
|||For the sake of computa-tional efficiency, the multi-scale fusion features (describedin Sec.|
|||The second feature constancy term re iscalculated as the reconstruction error of the initial disparity,i.e., the absolute difference between the multi-scale fusionfeatures (Sec.|
|250|cvpr18-Weakly Supervised Coupled Networks for Visual Sentiment Analysis|Approximating dis-crete probability distribution of image emotions by multi-modal features fusion.|
|251|Learning Detailed Face Reconstruction From a Single Image|Real-time height map fusion using differentiable rendering.|
|252|cvpr18-Structured Attention Guided Convolutional Neural Fields for Monocular Depth Estimation|MethodError(lower is better)rellog10rmsAccuracy(higher is better) < 1.25  < 1.252  < 1.2530.168 1.072 5.101Front-end CNN (w/o multiple deep supervision)Front-end CNN (w/ multiple deep supervision)0.152 0.973 4.9020.143 0.949 4.825Multi-scale feature fusion with naive concatenation0.134 0.895 4.733Multi-scale feature fusion with CRFs (w/o attention model)Multi-scale feature fusion with CRFs (w/ attention model)0.127 0.869 4.636Multi-scale feature fusion with CRFs (w/ structured attention model) 0.122 0.897 4.6770.7410.7820.7950.8030.8110.8180.9320.9310.9390.9420.9500.9540.9810.9740.9780.9800.9820.985et al.|
|||Monocular depth estimation withhierarchical fusion of dilated cnns and soft-weighted-sum in-ference.|
|253|Auston_Sterling_ISNN_-_Impact_ECCV_2018_paper|ISNN: Impact Sound Neural Network5Early attempts at multimodal fusion in neural networks focused on increas-ing classification specificity by combining the individual classification results ofseparate input streams [41].|
|||Multimodal Audio-Visual Network (ISNN-AV) Our audio-visual net-work, as shown in Figure 1, consists of our audio-only network combined witha visual network based on VoxNet [4] using either a concatenation, addition,multiplicative fusion, or bilinear pooling operation.|
|||Multiplicative fusion calculates element-wise products between in-puts, while projecting the interactions into a lower-dimensional space to reducedimensionality [46].|
|||Thismethod builds on the basic idea of multiplicative fusion by performing a sequenceof pooling and regularization steps after the initial element-wise multiplication.|
|||Ourmultimodal networks combined VoxNet with either ISNN-A or SoundNet8 andwere merged through either concatenation (MergeCat), element-wise addition(MergeAdd), multiplicative fusion (MergeMultFuse) [46], or multimodal factor-ized bilinear pooling (MergeMFB) [45].|
|||a large objectwill produce a deeper sound than a small object), and the multimodal fusion ofthose cues produces higher accuracy.|
|||Similarly, ModelNet40 produces opti-mal results using ISNN-AV with multiplicative fusion on MN40osm, at 93.24 %.|
|||(c) Our method hasbeen able to correctly classify impact sounds with voxel data across ModelNet40classes, as displayed by the MN40osm confusion matrix, for instance.|
|254|cvpr18-PiCANet  Learning Pixel-Wise Contextual Attention for Saliency Detection|F i denotes a fusion feature map and F iatt denotes its attended contextual feature map.|
|255|cvpr18-Detect-and-Track  Efficient Pose Estimation in Videos|Convolutionaltwo-stream network fusion for video action recognition.|
|256|cvpr18-Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet |Convolutionaltwo-stream network fusion for video action recognition.|
|257|Xuan_Chen_Focus_Segment_and_ECCV_2018_paper|The finalmulti-label segmentation result is the fusion of the predictions given by the fourbinary classifiers, as well as the whole-tumor classifier.|
|258|UntrimmedNets for Weakly Supervised Action Recognition and Detection|Convolutionaltwo-stream network fusion for video action recognition.|
|259|Lei_Zhou_Learning_and_Matching_ECCV_2018_paper|Particularlyfor the point cloud data generally co-registered with camera images [2629], thefusion of multiple image views, which has reported success on various tasks [3033], is expected to further improve the discriminative power of 3D local descrip-tors.|
|||1) We arethe first to leverage the fusion of multiple image views for the description of 3Dkeypoints when tackling point cloud registration.|
|||Multi-view fusion.|
|||The multi-view fusion technique is used to integrate in-formation from multiple views into a single representation.|
|||A more general strategy of multi-view fusionis view pooling [3133, 40], which aggregates the feature maps of multiple viewsvia element-wise maximum operation.|
|||3 Multi-View Local Descriptor (MVDesc)In this section, we propose to learn multi-view descriptors (MVDesc) for 3D key-points which combine multi-view fusion techniques [3033] with patch descriptorlearning [1721].|
|||Specifically, we first propose a new view-fusion architecture tointegrate feature maps across views into a single representation.|
|||Second, we buildthe MVDesc network for learning by putting the fusion architecture above mul-tiple feature networks [45].|
|||3.1 Multi-View FusionCurrently, view pooling is the dominant fusion technique used to merge featuremaps from different views [3133, 40].|
|||To verify the advantage of our FRN over the widely-usedview pooling [3133] in multi-view fusion, we remove the Fuseption branch fromour MVDesc network and train with the same data and configuration.|
|||First, it demonstrates the advantage of our FRNover view pooling [3133, 70, 40] in terms of multi-view fusion.|
|||First, a multi-view descriptor, named MVDesc, has been proposed for theencoding of 3D keypoints, which strengthens the representation by applying thefusion of image views [3133] to patch descriptor learning [1721].|
|||Dai, A., Niener, M., Zollh ofer, M., Izadi, S., Theobalt, C.: Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration.|
|260|Slawomir_Bak_Domain_Adaptation_through_ECCV_2018_paper|Zhao, H., Tian, M., Sun, S., Shao, J., Yan, J., Yi, S., Wang, X., Tang, X.: Spindlenet: Person re-identification with human body region guided feature decompositionand fusion.|
|261|cvpr18-Non-Local Neural Networks|7799modelbackbonemodalitytop-1 valtop-5 valtop-1 testtop-5 testI3D in [7]2-Stream I3D in [7]RGB baseline in [3]3-stream late fusion [3]3-stream LSTM [3]3-stream SATT [3]NL I3D [ours]RGBRGB + flowInceptionInceptionInception-ResNet-v2 RGBInception-ResNet-v2 RGB + flow + audioInception-ResNet-v2 RGB + flow + audioInception-ResNet-v2 RGB + flow + audioResNet-50ResNet-101RGBRGB72.175.773.074.977.177.776.577.790.392.090.991.693.293.292.693.371.174.289.391.3------------avg test80.282.8-----83.8Table 3.|
|262|Wei_Dong_Probabilistic_Signed_Distance_ECCV_2018_paper|We propose a novel 3D spatial representation for data fusionand scene reconstruction.|
|||Many representations built upon appropriate mathematical models are de-signed for robust data fusion in such a context.|
|||Volumetric grids lack flexibility to some extent, hence corre-sponding data fusion can be either oversimplified using weighted average [20], ormuch time-consuming in order to maximize joint distributions [27].|
|||Our framework is able to perform reliable depth data fusion and reconstructhigh-quality surfaces in real-time with more details and less noise, as depictedin Fig.1.|
|||Incremental 3D data fusion is built upon less ad-hoc probabilistic computa-tions in a parametric Bayesian updating fashion, contributes to online surfacereconstruction, and benefits from iteratively recovered geometry in return.|
|||While these systems performonline depth fusion, they usually require offline MarchingCubes [17] to output fi-nal mesh models; [14,24,6] incorporates online meshing modules in such systems.|
|||Instead of utilizing volumetric spatial representations, [13] proposes point-basedfusion that maintains light-weight dense point cloud or surfels as 3D maps.|
|||Uncertainty-aware data fusion.|
|||[20,19]use weight average of Truncated SDF in the data fusion stage by consideringa per-voxel Gaussian distribution regarding SDF as a random variable.|
|||The pipeline consists of iterative operations of datafusion and surface generation.|
|||As we have discussed, evaluation of inlier ratio was performed,increasing total time of the fusion stage.|
|||When we come to meshing, we find that by taking the advantage of PSDFfusion and inlier ratio evaluation, unnecessary computations can be avoided andPSDF method runs faster than TSDF, as plotted in Fig.8(d).|
|||The meshing stageis the runtime bottleneck of the approach, in general the saved time compensatefor the cost in fusion stage, see Fig.8(e) and Table.2.|
|||Built upon a hybrid datastructure, our framework can iteratively generate surfaces from the volumetricPSDF field and update PSDF values through reliable probabilistic data fusionsupported by reconstructed surfaces.|
|263|cvpr18-Learning Semantic Concepts and Order for Image and Sentence Matching|To handle this, we design a gated fusion unit thatcan selectively balance the relative importance of semanticconcepts and context.|
|||As illustratedin Figure 2 (d), after obtaining the normalized context vec-tor x  RI and concept score vector p  RK , their fusionby the gated fusion unit can be formulated as:bp = kWlpk2, bx = kWgxk2, t = (Ulp + Ugx)v = t  bp + (1  t) bxwhere kk2 denotes the l2-normalization, and v  RH is thefused representation of semantic concepts and global con-text.|
|||3) sum and gate are two different ways that com-bine semantic concepts and context via feature summationand gated fusion unit, respectively.|
|||7) Using the proposed gat-ed fusion unit (as cnp + ctx) performs better, due to theeffective importance balancing scheme.|
|||8) The best perfor-mance is achieve by the cnp + ctx + gen, which combinesthe 10-cropped extracted context with semantic concepts vi-a the gated fusion unit, and exploits the sentence generationfor semantic order learning.|
|||Learning arecurrent residual fusion network for multimodal matching.|
|264|cvpr18-ScanComplete  Large-Scale Scene Completion and Semantic Segmentation for 3D Scans|The TSDF is gener-ated from depth frames following the volumetric fusion ap-proach of Curless and Levoy [3], which has been widelyadopted by modern RGB-D scanning methods [21, 10, 23,4579Figure 1.|
|||Bundlefusion: Real-time globally consistent 3d reconstruc-tion using on-the-fly surface reintegration.|
|||Kinectfusion: real-time 3d reconstruction and inter-action using a moving depth camera.|
|||Kinectfusion: Real-time dense surface map-ping and tracking.|
|||Oct-netfusion: Learning depth fusion from data.|
|||Elasticfusion: Dense slam without a posegraph.|
|265|CHUNLUAN_ZHOU_Bi-box_Regression_for_ECCV_2018_paper|: Fused DNN: A deep neural net-work fusion approach to fast and robust pedestrian detection.|
|266|Hyper-Laplacian Regularized Unidirectional Low-Rank Tensor Recovery for Multispectral Image Denoising|Data fusion: definitions and architectures: fusion of imagesof different spatial resolutions.|
|267|Yang_Du_Interaction-aware_Spatio-temporal_Pyramid_ECCV_2018_paper|Deep ConvNets [27] used thefusion of different layers and were trained on a large scale dataset such as Sports-1M.|
|||(3)F is a fusion function for yj of all layers in the pyramid.|
|||Here we respectivelyinvestigate three fusion functions, element-wise maximum, element-wise sum andelement-wise multiplication.|
|||4.1 Evaluations of the Proposed Attention LayerWe investigate our interaction-aware spatio-temporal pyramid attention on thefollowing five parts: (1) layer position of feature maps used for aggregation,CONV+Classification LossATTENTIONRELUCONV12KK group of feature mapsOne group of aggregated feature maps12KSptaio-Temporal PyramidFlow or RGB is used as inputSalientregionsK++Attention LossInteractive LossISTPAN for Action Classification9(2) different fusion functions F of feature maps of pyramid, (3) numbers oflayers in pyramid, (4) loss functions with ablated regularization items, and (5)the generality of our layer applied in different deep networks, including populararchitectures VGGNet-16 [9], BN-Inception [52] and Inception-ResNet-V2 [10].|
|||We explore different fusion functions F of feature maps of pyramid evalu-ated on RGB stream.|
|||Table 1(c) lists the comparison results of different fusionstrategies.|
|||Element-wise multiplication performs better than other candidatefunctions, and is therefore selected as a default fusion function.|
|||Evaluations of (a) position of the top layer of pyramid; (b) different scaleswith Inception-ResNet-V2 (I-R-V2); (c) fusion functions with 3 scales; (d) loss functionswith I-R-V2 and our attention layer; and 3) our layer on VGGNet-16, BN-Inceptionand Inception-ResNet-V2.|
|||Block A (35  35  320)Block B (17  17  1088)Block (Inception-ResNet-V2) RGB Flow85.8% 83.5%86.1% 83.7%86.3% 84.0%85.5% 83.4%Block C (8  8  2080)FC (1536)Scale RGB Flow1 scale 86.3% 84.0%2 scales 86.8% 84.8%3 scales 87.3% 85.5%4 scales 86.5% 85.0%(c) Performance of different fusion functions.|
|||Fusion Function (F )Accuracy #RGBElement-wise MaximumElement-wise SumElement-wise Multiplication85.7%86.4%87.3%Stream linterRGBFlowlattn no loss87.8% 87.5% 87.3%86.1% 85.7% 85.5%Late fusion 94.7% 94.4% 94.2%(e) Performance of the proposed attention layer on popular networks.|
|||Stream/(linter + lattn) VGGNet-16 BN-Inception Inception-ResNet-V2RGBRGB (3 scales)FlowFlow (3 scales)Late fusionLate fusion (3 scales)80.4%83.8%85.5%87.1%90.7%92.8%84.5%86.7%87.2%87.9%92.0%94.6%85.2%88.2%83.1%86.5%92.6%95.1%by improving the performance 0.9%/1.0%/0.9% on RGB/Flow/Late fusion (3scales).|
|||The late fusion approach means that the predictionscores of the RGB and Flow streams are averaged as the final video classification,as other methods [26,28,39,40] do.|
|||For VGGNet-16, our attention network re-spectively promotes 3.4%/1.6%/2.1% on RGB/Flow/Late fusion on the UCF101split 1.|
|||For BN-Inception, our model respectively promotes 2.2%/0.7%/2.6% onRGB/Flow/Late fusion.|
|||For Inception-Resnet-V2, our model respectively pro-motes 3.0%/3.4%/2.5% on RGB/Flow/Late fusion.|
|268|cvpr18-Dynamic Feature Learning for Partial Face Recognition|(4)(4) applies a sum fusion among reconstruction errorEq.|
|269|OctNet_ Learning Deep 3D Representations at High Resolutions|35828316332364312832563InputResolution01020304050607080Memory[GB]OctNetDenseNet8316332364312832563InputResolution0246810121416Runtime[s]OctNetDenseNet8316332364312832563InputResolution0.860.880.900.920.94AccuracyOctNet1OctNet2OctNet38316332364312832563InputResolution0.860.880.900.920.94AccuracyOctNetDenseNetVoxNet3836132334683323Figure 9: Confusion Matrices on ModelNet10.|
|||Taking a closer look atthe confusion matrices in Fig.|
|||Anoctree-based approach towards efficient variational rangedata fusion.|
|||Kinectfusion: Real-time dense surface map-ping and tracking.|
|270|cvpr18-DeLS-3D  Deep Localization and Segmentation With a 3D Semantic Map|The uniqueness of our design is a sen-sor fusion scheme which integrates camera videos, motionsensors (GPS/IMU), and a 3D semantic map in order toachieve robustness and efficiency of the system.|
|||We show thatpractically, pose estimation solely relying on images likePoseNet [25] may fail due to street view confusion, and itis important to fuse multiple sensors.|
|||Outdoor navigation of a mobile robot between build-ings based on dgps and odometry data fusion.|
|271|Simultaneous Super-Resolution and Cross-Modality Synthesis of 3D Medical Images Using Weakly-Supervised Joint Convolutional Sparse Coding|Patch-based synthesis suffers frominconsistencies introduced during the fusion process thattakes place in areas where patches overlap.|
|||An edge-guided image interpolation al-gorithm via directional filtering and data fusion.|
|272|cvpr18-Low-Latency Video Semantic Segmentation|The fusion process takes the concatenationof both features as input, and sends it through a convolutionlayer with 3  3 kernels and 256 channels.|
|273|Harmonic Networks_ Deep Translation and Rotation Equivariance|[17] do the same for a broaderclass of global image transformations, and propose a novelper-pixel pooling technique for output fusion.|
|||A finalfusion layer is created by taking a weighted linear combinationof the side-connections, this is the final output.|
|274|Xinyu_Gong_Neural_Stereoscopic_Image_ECCV_2018_paper|Employing the proposed view loss, our network is ableto coordinate the training of both the paths and guide the feature aggregationblock to learn the optimal feature fusion strategy for generating view-consistentstylized stereo image pairs.|
||| A feature aggregation block is proposed to learn a proper feature fusionstrategy for improving the view consistency of the stylized results.|
|275|cvpr18-Motion-Appearance Co-Memory Networks for Video Question Answering|Multi-layer contextual facts are dynamically constructedvia a soft attention fusion process, which computes a weightedaverage facts according to the attention.|
|||We believe the reason is that the attention-based fact fusionoptimizes the ensemble process by using weighted averageof the contextual facts, and avoids just using only one ofthem, which may make the facts sub-optimal.|
|||There are two co-memory variants shown in Table 5:co-memory (w/o DFE) uses co-memory attention withT = 2 memory update, but not dynamic fact ensemble;co-memory (full) uses co-memory attention with T = 2memory update and dynamic fact ensemble (soft fusion) on3-layer contextual facts.|
|276|Angela_Dai_3DMV_Joint_3D-Multi-View_ECCV_2018_paper|Dai, A., Niener, M., Zollh ofer, M., Izadi, S., Theobalt, C.: Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration.|
|||McCormac, J., Handa, A., Davison, A., Leutenegger, S.: Semanticfusion: Dense3d semantic mapping with convolutional neural networks.|
|||Newcombe, R.A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A.J.,Kohi, P., Shotton, J., Hodges, S., Fitzgibbon, A.: Kinectfusion: Real-time densesurface mapping and tracking.|
|||Riegler, G., Ulusoy, A.O., Bischof, H., Geiger, A.: Octnetfusion: Learning depthfusion from data.|
|||: Incremental dense seman-tic stereo fusion for large-scale semantic scene reconstruction.|
|277|cvpr18-Avatar-Net  Multi-Scale Zero-Shot Style Transfer by Feature Decoration|encInversely, the decoder progressively generates intermediatefeatures dl = Dl+1(dl+1) starting from z, in which dl isdecfurther updated by fusing with the corresponded encodedfeatures el via the style adaptive feature fusion module.|
|||In the end, the stylized image  xcs is inverted by thedecoded module Ddec(zcs) with multiple style fusion mod-ules that progressively modify the the decoded features dlcsunder the guidance of multi-scale style patterns els, froml = L to 1, as shown in Fig.|
|278|cvpr18-Style Aggregated Network for Facial Landmark Detection|[72] utilize a coarse searchover a shape space with diverse shapes to overcome the poor380convresidual...residualdeconvfaces in the aggregatedstylestyle-aggregated face generation modulefaces in theoriginal stylesoriginal streamgenerative streamfusion stream1-vnoc2-vnoc3-vnoc4-vnoclkcob-vnocfeature extraction partstage-1 FC structurenoitanetacnocnoitanetacnocconcatenationstage-2 stage-3 FC structureFC structurestage-1 outputsstage-2 outputsfacial landmark prediction modulefinal heat-map outputsFigure 3.|
|280|Global Context-Aware Attention LSTM Networks for 3D Action Recognition|Multi-modal featurefusion for action recognition in rgb-d sequences.|
|281|cvpr18-Deep Layer Aggregation|To overcome these bar-Figure 1: Deep layer aggregation unifies semantic and spa-tial fusion to better capture what and where.|
|||The keyaxes of fusion are semantic and spatial.|
|||Semantic fusion, oraggregating across channels and depths, improves inferenceof what.|
|||Spatial fusion, or aggregating across resolutions andscales, improves inference of where.|
|||Deep layer aggregationcan be seen as the union of both forms of fusion.|
|||Densely connected networks (DenseNets) [19] are thedominant family of architectures for semantic fusion, de-signed to better propagate features and losses through skipconnections that concatenate all the layers in stages.|
|||Ourhierarchical deep aggregation shares the same insight on theimportance of short paths and re-use, and extends skip con-nections with trees that cross stages and deeper fusion thanconcatenation.|
|||Feature pyramid networks (FPNs) [30] are the dominantfamily of architectures for spatial fusion, designed to equal-ize resolution and standardize semantics across the levels ofa pyramidal feature hierarchy through top-down and lateralconnections.|
|||Our iterative deep aggregation likewise raisesresolution, but further deepens the representation by non-linear and progressive fusion.|
|||Finally, we iteratively aggregate these stages to learna deep fusion of low and high level features.|
|||DLA vs. DenseNet compares DLA with the dominant ar-chitecture for semantic fusion and feature re-use.|
|||While these networks canaggressively reduce depth and parameter count by feature re-use, concatenation is a memory-intensive fusion operation.|
|282|Guojun_Yin_Zoom-Net_Mining_Deep_ECCV_2018_paper|The proposed hROI, deROIi operationsdiffer from conventional feature fusion operations (channel-wise concatenation orsummation).|
|||The conventional feature fusion operations are implemented by channel-wise concatenation in SCA-M cells here.|
|283|Deep Learning of Human Visual Sensitivity in Image Quality Assessment Framework|Image Pro-ment using multi-method fusion.|
|284|Joint Registration and Representation Learning for Unconstrained Face Identification|score level fusion strategy for the probe templates.|
|||To optimally use the probe template data at classi-fication, we employ a fusion strategy (details in Sec.|
|||Note thatjmrepresents the rows of the confusion matrix m corre-jsponding to each media representation.|
|||The VB al-gorithm for decision fusion works by iteratively updatingthe hidden output variables (actual labels y) and the modelparameters (, p).|
|||Second,its capability to synthesize multitude of information in thetemplate media with proposed decision level fusion scheme.|
|285|The Misty Three Point Algorithm for Relative Pose|Their approach treats depth maps as a by-product of estimating the scene radiance, whereas our pro-posed method provides a fusion of their two separate meth-ods.|
|288|Fast Haze Removal for Nighttime Image Using Maximum Reflectance Prior|To overcome the above difficulties, someworks adopt various new techniques, such as color transfer[23], illumination correction [29], glow removal [17] andimage fusion [1], to resolve the issues associated with hazeremoval from single nighttime image (see Sec.|
|||[1] estimate the localairlight by applying a local maximum on patches of darkchannel, and then use the multi-scale fusion approach toobtain a visibility enhanced image.|
|||Night-time dehazing by fusion.|
|289|Yumin_Suh_Part-Aligned_Bilinear_Representations_ECCV_2018_paper|Zhao, H., Tian, M., Sun, S., Shao, J., Yan, J., Yi, S., Wang, X., Tang, X.: Spindle net: Per-son re-identification with human body region guided feature decomposition and fusion.|
|||Zheng, L., Wang, S., Tian, L., He, F., Liu, Z., Tian, Q.: Query-adaptive late fusion for imagesearch and person re-identification.|
|290|cvpr18-Compressed Video Action Recognition|Our approach is simple and fast, without using RNNs,complicated fusion or 3D convolutions.|
|||To combine them, we tried var-ious fusion strategies, including mean pooling, maximumpooling, concatenation, convolution pooling, and bilinearpooling, on both middle layers and the final layer, but withlimited success.|
|||There are several reasonable candidatesfor such a fusion, e.g.|
|||+ denotes score fusion of mod-els.|
|||[16] and TLE [5] consider morecomplicated fusions and pooling.|
|||Our method uses much faster 2D CNNs plus simplelate fusion without additional supervision, and still signifi-cantly outperforms these methods.|
|||[16]ResNet-50 [12] (from ST-Mult [8])ResNet-152 [12] (from ST-Mult [8])C3D [39]Res3D [40]TSN (RGB-only) [44]*TLE (RGB-only) [5]I3D (RGB-only) [2]*MV-CNN [49]P3D ResNet [27]Attentional Pooling [10]CoViARWith optical flowiDT+FV [42]Two-Stream [33]Two-Stream fusion [9]LRCN [6]Composite LSTM Model [35]ActionVLAD [11]ST-ResNet [7]ST-Mult [8]I3D [2]*TLE [5]L2STM [37]ShuttleNet [30]STPN [45]TSN [44]CoViAR + optical flow65.482.383.482.385.885.787.984.586.488.6-90.4-88.092.582.784.392.793.494.293.493.893.694.494.694.294.9-48.946.751.654.9-54.249.8--52.259.157.259.465.444.066.966.468.966.468.866.266.668.969.470.2Table 6: Accuracy on UCF-101 [34] and HMDB-51 [18].|
|||Here we train a temporal-stream network using 7segments with BN-Inception [14], and combine it with ourmodel by late fusion.|
|||Again, our method simplytrains 2D CNNs separately without any complicated fusionor RNN and still outperforms these models.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|291|cvpr18-CNN in MRF  Video Object Segmentation via Inference in a CNN-Based Higher-Order Spatio-Temporal MRF|This algorithm proceeds by alternating between atemporal fusion step and a feed-forward CNN step.|
|||The entire inferencealgorithm alternates between a temporal fusion step anda feed-forward pass of the CNN.|
|||The algorithm alternates betweena temporal fusion operation and a feed-forward CNNto progressively refine the segmentation results.|
|||(11) in step 2 only considers spatial dependencies foreach frame c. The two steps are essentially performingtemporal fusion and mask refinement, respectively.|
|||We add additional skip connectionsfrom intermediate pooling layers to a final output convo-lutional layer to enable multi-level feature fusion.|
|||The number of inner iterations (i.e.,the ICM iterations in temporal fusion) is set to L = 5.|
|||The temporal fusion step in our algorithm isperformed locally and the runtime is almost ignorable.|
|||TF represents the temporal fusion step in our algorithm,while MR represents the mask refinement step.|
|||By performing infer-ence in the MRF model, we developed an algorithm thatalternates between a temporal fusion operation and a maskrefinement feed-forward CNN, progressively inferring theresults of video object segmentation.|
|292|Lai_Jiang_DeepVS_A_Deep_ECCV_2018_paper|Most early saliency prediction methods [16, 20, 26, 34]relied on the feature integration theory, which is composed of two main steps: featureextraction and feature fusion.|
|||In addition to feature extrac-tion, many works have focused on the fusion strategy to generate video saliency maps.|
|||Other advancedmethods [9, 19, 41] applied phase spectrum analysis in the fusion model to bridge thegap between features and video saliency.|
|293|Youngjae_Yu_A_Joint_Sequence_ECCV_2018_paper|A Joint Sequence Fusion Model for VideoQuestion Answering and RetrievalYoungjae YuJongseok Kim Gunhee KimDepartment of Computer Science and Engineering,Seoul National University, Seoul, Korea{yj.yu,js.kim}@vision.snu.ac.kr, gunhee@snu.ac.krhttp://vision.snu.ac.kr/projects/jsfusion/Abstract.|
|||4.1 LSMDC Dataset and TasksThe LSMDC 2017 consists of four video-language tasks for movie understandingand captioning, among which we focus on the three tasks in our experiments:10Y. Yu , J. Kim and G. KimTasksMetricsDatasetLSTM-fusionSA-G+SA-FC7 [12]LSTM+SA-FC7 [12]C+LSTM+SA-FC7 [12]VSE-LSTM [19]EITanque [30]SNUVL [29]CT-SAN [9]Miech et al.|
|||A Joint Sequence Fusion Model for Video VQA and Retrieval11DatasetMultiple-ChoiceAccuracyLM38.352.8LSTM-fusion55.855.1SA-G+SA-FC7 [12]59.1LSTM+SA-FC7 [12]56.360.2C+LSTM+SA-FC7 [12] 58.167.3VSE-LSTM [19]63.065.463.1SNUVL [29]66.163.5ST-VQA-Sp.Tp [11]65.563.7EITanque [30]66.4CT-SAN [9]63.876.169.0MLB [45]68.764.7JSTfc79.772.1JSTlstm74.468.3JSTmax80.0JSTmean70.279.269.4JSFusion-noattention75.668.7JSFusion-VGG-noaudio72.5JSFusion-noaudio82.9JSFusion73.5 83.4Fill-in-the-BlankText-only BLSTM [34]Text-only Human [34]GoogleNet-2D + C3D [34]Ask Your Neurons [46]Merging-LSTM [35]SNUVL [29]CT-SAN [9]LR/RL LSTMs [36]LR/RL LSTMs (Ensemble) [36]MLB [45]JSTfcJSTlstmJSTmaxJSTmeanJSFusion-noattentionJSFusion-VGG-noaudioJSFusion-noaudioJSFusionHuman [34]Accuracy32.030.235.733.234.238.041.940.943.541.642.943.741.344.244.544.245.2645.5268.7Table 3.|
|||As one naive variantof our model, we test a simple LSTM baseline (LSTM-fusion) that only car-ries out the Hadamard product on a pair of final states of video and languageLSTM encoders.|
|||That is, (LSTM-fusion) is our JSFusion model that has neitherJST nor CHD, which are the two main contributions of our model.|
|||We train(LSTM-fusion) in the same way as done for the JSFusion model in section 3.5.|
|||As easily expected, the performance of (LSTM-fusion) is significantly worse thanour JSFusion in all the tasks.|
|294|cvpr18-Towards Human-Machine Cooperation  Self-Supervised Sample Mining for Object Detection|RFCN+Flip&Rescale means the fusionof them.|
|295|cvpr18-Attention-Aware Compositional Network for Person Re-Identification|Then, part feature alignment andweighted fusion are performed in AFC module, given at-tention maps and visibility scores from PPA.|
|||Implementation DetailsIn AFC, GoogleNet is utilized as base network for globalcontext feature extraction, and two additional 1x1 con-volution layers are used for part weight estimation and fi-nal feature fusion, respectively.|
|||Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|296|cvpr18-Gesture Recognition  Focus on the Hands|Sparse fusioncombines softmax scores according to the gesture type.|
|||For every gesture type, the sparse fusionlayer learns the relative importance of the different spatialregions and data modalities.|
|||They used a combination ofrank pooling, LSTMs and temporal streams, and fused thestreams using average fusion.|
|||Other participants of the chal-lenge [31, 34] also used C3Ds and some form of LSTM fortemporal fusion.|
|||It has threemain components: 1) a focus of attention mechanism, 2) 12separate global and focused channels, and 3) a fusion mech-anism.|
|||Finally, fusion occursthrough a sparse network that learns which channels are im-portant for each gesture.|
|||Section 3.2 explains the sparse fusion network that com-bines information across channels.|
|||With 12 channels,the concatenated feature vector would be over 24,000 ele-ments long, and the fusion layer as a whole would have tolearn over 6 million weights.|
|||To measure the effectiveness of spatial attention chan-nels and gesture-based fusion relative to other techniques,we compare the recognition accuracy of FOANet as shownin Figure 2 to those of previous systems on the ChaLearnIsoGD (this section) and NVIDIA (next section) data sets.|
|||To learn the weights of the fusion layer, the softmaxscores of different channels of training data are precom-puted.|
|||All thescores are stacked together and are multiplied by the fusionlayer weights and the diagonal of the resulting matrix is ex-tracted.|
|||As already stated, focus of attention and sparse networkfusion are the keys to our method.|
|||To evaluate the contri-bution of sparse network fusion, we replace it with averagefusion, i.e.|
|||The average fusion version of FOANetachieves better results than previous methods (67.38% vs64.40% on validation set and 70.37% vs 67.71% on test set),as shown in Table 2.|
|||Therefore, sparse network fusion im-proves performance by 11.7%.|
|||With averaging as the fusionmechanism, the best performance was achieved by a subsetof 7 of the 12 channels: 3 RGB flow channels, 2 depth focuschannels, the RGB right hand channel, and the depth flowright hand channel.|
|||We see adifferent pattern with sparse network fusion, however.|
|||Byusing only 7 channels with sparse network fusion, the ac-curacy decreases to 77.31% on the validation set and 78.9%on the test set.|
|||With sparse network fusion the system learnswhich channels to include for each gesture type, with the re-sult that sparse network fusion benefits from the presence ofchannels that hurt performance when averaging channels.|
|||Un-fortunately, this method doesnt perform on par with sparsenetwork fusion or even simply averaging the softmax out-5240FusionSparseAverageConcatenationValidTest12 Channels7 Channels12 Channels7 Channels80.9667.3856.0377.3169.0655.2982.0770.3759.4478.9071.9358.84Table 2.|
|||Comparison of fusion strategies.|
|||Accuracies are shownfor FOANet using sparse network fusion, channel averaging, andconcatenation for 12 channels (maximal for sparse nets) and 7channels (optimal for averaging).|
|||However, overall perfor-mance is best when all channels are combined, suggestingthat the left hand is important for two-handed gestures andthat sparse network fusion is able to learn when to pay at-tention to the left hand.|
|||Next we combine channels from different modalities us-ing sparse network fusion, as shown in Table 4.|
|||From thefirst two fusion columns, we can see that the combinationof focus channels is better than the combination of globalchannels.|
|||In fact, the fusion of focus channels is the bestcombination, short of combining all channels.|
|||We also notice that the fusion of RGBand RGB flow nets is better than the fusion of depth anddepth flow nets on validation set.|
|||Next, we see that the fusion ofRGB and depth channels performs on par with the fusionof RGB flow and depth flow channels.|
|||The accuracy of FOANet drops to 85.26% when sparsenetwork fusion is replaced by average fusion, emphasizingthe importance of sparse network fusion even in domainswith only one hand and no significant background changes.|
|||Thecurrent architecture does not address temporal fusion in asophisticated way.|
|297|Hang_Zhao_The_Sound_of_ECCV_2018_paper|For validation samples of each category, we find the strongest activatedchannel, and then sort them to generate a confusion matrix.|
|||9 shows the(a) visual and (b) audio confusion matrices from our best model.|
|||(a) Visual and (b) audio confusion matrices by sorting channel activations withrespect to ground truth category labels.|
|||The binarymasking model gives the highest correct rate, lowest error rate, and lowest con-fusion (percentage of Both), indicating that the binary model performs sourceseparation perceptively better than the other models.|
|298|Ruoxi_Deng_Learning_to_Predict_ECCV_2018_paper|In a refinement module, the mask-encoding is fused withthe side-output features and then reduces its channels by a factor of 2 and doubleits resolution to prepare for the fusion in the next refinement module.|
|||Ours-w/o-FL refers to our methodwithout the fusion loss.|
|||HED-FL refers to HED trained via the proposedfusion loss.|
|||Moreover, we trained two versionsof the proposed network via the balanced cross-entropy loss (Ours-w/o-FL) andthe proposed fusion loss (Ours), respectively.|
|||Lastly, both the quantitative and the qualitative results havedemonstrated the effectiveness of the proposed fusion loss.|
|||By simply using theproposed fusion loss, the ODS f-score (before NMS) of our network is increasedLearning to predict crisp boundaries11(a) Input Image(b) GT(c) Ours(d) RCF [13](e) CED [45]Fig.|
|||MethodSE [26]OursRCF-VOC-aug [13]FPS281/240DeepContour [11]DeepEdge [10]Canny [19]gPb-UCM [5]HFL [47]HED [12]CEDN [17]ODS OIS.611 .676.729 .7552.5.743 .7631/30.757 .776.753 .772 1/1000.767 .788.788 .808.788 .804MIL+G-DSN+MS+NCuts [27] .813 .831.806 .823.811 .830.794 .811.803 .820.815 .833.800 .816.808 .824.815 .8345/6301013010301010303010CED-MS-VOC-aug [45]RCF-MS-VOC-aug [13]CED [45]CED-MS [45]Ours-VOC-augOurs-MS-VOC-augImproving the crispness of HED As mentioned in Section 3.2, the proposedfusion loss plays a key role in our method in terms of generating sharp bound-12Deng et al.|
|||One may ask a question:Does the fusion loss only work on the convolutional encoder-decoder network?|
|||Similar to the ablation experiments, we evaluate two versions of HED: one istrained by means of the proposed fusion loss, the other is applying the balancedcross-entropy loss.|
|||Perona, P., Malik, J.: Scale-space and edge detection using anisotropic diffusion.|
|299|CNN-SLAM_ Real-Time Dense Monocular SLAM With Learned Depth Prediction|Thisis the case of volumetric fusion approaches such as KinectFusion [21], as well as dense SLAM methods based onRGB-D data [30, 11], which, in addition to navigation andmapping, can also be employed for accurate scene recon-The first two authors contribute equally to this paper.|
|||We alsoshow qualitative results of our joint scene reconstructionand semantic label fusion in a real environment.|
|||Since the CNN is trained to provide semantic labels in ad-dition to depth maps, semantic information can be also as-sociated to each element of the 3D global model, through aprocess that we denote as semantic label fusion.|
|||4.2) and accuracy of semantic label fusion(Subsec.|
|||Also, the CNN-based depth prediction and semantic segmentation are runon the GPU, while all other stages are implemented on theCPU, and run on two different CPU threads, one devoted toframe-wise processing stages (camera pose estimation anddepth refinement), the other carrying out key-frame relatedprocessing stages (key-frame initialization, pose graph op-timization and global map and semantic label fusion), so toallow our entire framework to run in real-time.|
|||In all our experiments,we used the CNN model trained on the indoor sequencesof the NYU Depth v2 dataset [25], to test the generaliza-tion capability of the network to unseen environments; alsobecause this dataset includes both depth ground-truth (rep-resented by depth maps acquired with a Microsoft Kinectcamera) and pixel-wise semantic label annotations, neces-sary for semantic label fusion.|
|||Finally, we alsocompare our method to the one in [16], that uses the CNN-predicted depth maps as input for a state-of-the-art depth-based SLAM method (point-based fusion[11, 27]), basedon the available implementation from the authors of [27]4.|
|||Interestingly, the pose accuracy of our technique is onaverage higher than that of LSD-SLAM even after apply-ing bootstrapping, implying an inherent effectiveness of theproposed depth fusion approach rather than just estimatingthe correct scaling factor.|
|||The results of reconstruction and semantic label fusion on the office sequence (top, acquire by our own) and one sequence(kitchen 0046) from the NYU Depth V2 dataset [25] (bottom).|
|||Additional qualitative results in terms of pose andreconstruction quality as well as semantic label fusion areincluded in the supplementary material.|
|||Incremental dense se-mantic stereo fusion for large-scale semantic scene recon-struction.|
|||Real-time large scale dense6251RGB-D SLAM with volumetric fusion.|
|300|cvpr18-Learning to Understand Image Blur|Kernel fusion for better image deblurring.|
|301|Rajvi_Shah_View-graph_Selection_Framework_ECCV_2018_paper|Seeing double without confusion:In Proceedings IEEEStructure-from-motion in highly ambiguous scenes.|
|||Global fusion of relative motions forrobust, accurate and scalable structure from motion.|
|302|XU_YANG_Shuffle-Then-Assemble_Learning_Object-Agnostic_ECCV_2018_paper|(1) is a naive model and thereare fruitful ways of combining xi and xj in the literature, such as appendingindependent MLPs for each RoI [60], the union RoI [28], and even the fusionwith textual features [21], our feature learning can be seamlessly incorporatedinto any of them.|
|303|cvpr18-A Weighted Sparse Sampling and Smoothing Frame Transition Approach for Semantic Fast-Forward First-Person Videos|They assign scores tothe video segments by using late fusion of spatial and tem-poral deep convolution neural networks (DCNNs).|
|304|cvpr18-CarFusion  Combining Point Tracking and Part Detection for Dynamic 3D Reconstruction of Vehicles|Our fusion cost combines the complementary strengthsof the structured and unstructured points using rigidity con-straints.|
|||We evaluate our reconstruction pipeline at its progressivestages: car-centric RANSAC (cRANSAC), temporal inte-gration using only the structured points (TcRANSAC), andthe fusion of both structured points and unstructured trackWe evaluate our framework on a traffic scene capturedwith six Samsung Galaxy 6, ten iPhone 6, and six Go-1http://www.cs.cmu.edu/ILIM/projects/IM/CarFusion/1910Length of No ofTraj141442Traj234172202StraightTurningMulticRANSACRMSEPretrained MVB8.526.955.312.248.947.45T-cRANSACRMSEPretrained MVB17.812.514.37.15.834.47No ofTraj141442No ofTraj112101414CarFusionRMSEPretrained MVB16.815.517.42.53.12.2Table 1: Reprojection error of the reconstructed tracks at different stages of the pipeline.|
|||The number of trajectories using cRANSAC andT-cRANSAC is fixed to the number of parts, while with point fusion we have a combination of structured and unstructuredtracks.|
|||The TcRANSAC is the result of optimizingthe cost function 2 but without the fusion term ER.|
|305|Liuhao_Ge_Point-to-Point_Regression_PointNet_ECCV_2018_paper|The point-wise estimations are used to infer 3D joint loca-tions with weighted fusion.|
|||We infer 3D hand joint locations from the estimated heat-maps and unit vectorfields using weighted fusion.|
|306|Surveillance Video Parsing With Single Frame Supervision|The three components of SVP, namely frame pars-ing, optical flow estimation and temporal fusion are inte-grated in an end-to-end manner.|
|||Based on the mined correspondences and their con-fidences, the temporal fusion sub-network fuses the parsingresults of the each frame, and then outputs the final parsingresult.|
|||More-over, the feature learning, pixelwise classification, corre-spondence mining and the temporal fusion are updated in aunified optimization process and collaboratively contributeto the parsing results.|
|||Thetemporal fusion sub-network (Section 3.4) applies the ob-tained optical flow Ft,tl and Ft,ts to  Ptl and  Pts,producing Ptl,t and Pts,t.|
|||Temporal fusion: As shown in Figure 2, the estimatedparsing results  Ptl and  Pts are warped according to theoptical flow Ft,tl and Ft,ts via:Ptl,t = w(  Ptl, Ft,tl),Pts,t = w(  Pts, Ft,ts).|
|||They are fused with  Pt via a temporal fusion layerwith several 1  1 filters to produce the final Pt.|
|||(ii) we train the frame parsingsub-network and the temporal fusion sub-network togetherusing the optical flow estimated in step (i).|
|||The temporal fusion sub-network is initial-ized via standard Gaussian distribution (with zero mean andunit variance).|
|||(iv) keeping Conv1Conv5 layers fixed, wefine-tune the unique layers of frame parsing and temporalfusion sub-networks.|
|||The major reason of training the optical flow sub-network at the beginning is that, the temporal fusion sub-network depends on the optical flow results.|
|||Component AnalysisTemporal fusion weights: We visualize the learnedweights for the temporal fusion layers for R-arm and L-shoein Figure 3 in the Indoor dataset.|
|||The vertical axis illustrates the fusion weights.|
|||Thanks to the fusion fromPtl,t, the womens left shoe is labelled correctly in the fi-nal prediction Pt.|
|||But fortunately it does notaffect the fused prediction Pt, because the confidence of thisghost is very low in Ctl,t and hence it is filtered out duringthe fusion.|
|||Moreover, the fusionlayer contains several 1  1 convolutions and thus is notquite time-consuming.|
|307|Xin_Yuan_Towards_Optimal_Deep_ECCV_2018_paper|Liu, H., Ji, R., Wu, Y., Huang, F., Zhang, B.: Cross-modality binary codelearning via fusion similarity hashing.|
|308|Learning Multifunctional Binary Codes for Both Category and Attribute Oriented Retrieval Tasks|Follow-up works investigate the usage of attributecorrelation [29], fusion strategy [26, 22], relative attributes[25], natural language [7], and other techniques [10, 33] toimprove the retrieval performance.|
|||Multi-attribute spaces: Calibration for attribute fusion andsimilarity search.|
|309|Fabio_Tosi_Beyond_local_reasoning_ECCV_2018_paper|In [18] was proposed a methodto improve random forest-based approaches for confidence fusion [15,16,4] byusing a CNN.|
|||In addition to these approaches, acting inside stereo al-gorithms to improve their final output, other applications concern sensor fusion[23] and disparity map fusion [24].|
|||Marin, G., Zanuttigh, P., Mattoccia, S.: Reliable fusion of tof and stereo depthdriven by confidence measures.|
|||Poggi, M., Mattoccia, S.: Deep stereo fusion: combining multiple disparity hy-potheses with deep-learning.|
|310|cvpr18-Geometry Guided Convolutional Neural Networks for Self-Supervised Video Representation Learning|The video-level score is obtained by av-erage fusion over the 25 frames sampled from the video tobe classified.|
|||We investigate theefficacy of the progressive training strategy by comparing itto a few alternatives: Early fusion: we mix the FlyingChairs and 3D moviedata together and then train a single network.|
||| Late fusion: we train two models respectively usingFlyingChairs and 3D movies.|
|||First, the progressive training, which is equippedwith the learning without forgetting regularization, is veryeffective in capturing the two distinct geometry cues, com-pared with the naive early fusion and late fusion.|
|||To answer this question, we conduct late fusion of the clas-sification scores of the two types of networks.|
|||From Table 5,we can see that the late fusion leads to 2.8% performancegain on UCF101 dataset and 2.2% gain on HMDB51 datasetover the single ImageNet pre-trained model.|
|||By analyzing the action classes, we find that 48classes out of the 101 classes of UCF101 and 26 out of the51 classes of HMDB51 benefit from the fusion.|
|||Multimodal keyless attention fusion for video clas-sification.|
|311|Kai_Xu_LAPCSRA_Deep_Laplacian_ECCV_2018_paper|Keywords: Compressive sensing  Reconstruction  Laplacian pyramid Reconstructive adversarial network  Feature fusion.|
|||Taking RecGen2 in the 2nd RNN stage as anLAPRAN: A Laplacian Pyramid Reconstructive Adversarial Network7316deconv26488conv24644fc1conv1i216c1y2u2C283864166416316161616fc2conv3deconv1resblk1~3conv4o2r2Feature extraction Feature fusionResidual generationFig.|
|||To guarantee an equal contri-bution to the feature after fusion, the contextual latent vector c1 has the samedimension as the CS measurement y2.|
|||c1 is fused with the CSmeasurement y2 through concatenation (referred to as early fusion in [34]) ina feature space.|
|||The fusion of the context and CSmeasurements hence improve both convergence speed and recovery accuracy.|
|||The results without measurement fusion canbe regarded as the performance of an SR approach.|
|||The injected CS measure-ments at each pyramid level are the key for CS reconstruction, which distin-Epochs051015MSE00.020.040.06Epochs051015MSE00.020.040.06Epochs051015MSE00.020.040.06Epochs051015MSE00.020.040.06w/ fusionw/o fusionw/o fusionw/ fusionw/o fusionw/ fusionw/o fusionw/ fusion(a) stage 1(b) stage 2(c) stage 3(d) stage 410K. XU et al.|
|||To illustrate this point, we compare LAPRAN with a variantthat has no fusion mechanism implemented at each stage (an SR counterpart).|
|||: Early versus late fusion in se-mantic video analysis.|
|312|cvpr18-PointFusion  Deep Sensor Fusion for 3D Bounding Box Estimation|The resulting outputs are then com-bined by a novel fusion network, which predicts multiple3D box hypotheses and their confidences, using the input3D points as spatial anchors.|
|||In this paper, we show that our simple and genericsensor fusion method is able to handle datasets with distinctiveenvironments and sensor types and perform better or on-par withstate-of-the-art methods on the respective datasets.|
|||stage pipeline, which preprocesses each sensor modalityseparately and then performs a late fusion or decision-levelfusion step using an expert-designed tracking system suchas a Kalman filter [4, 7].|
|||Inspired by the successes of deep learn-ing for handling diverse raw sensory input, we propose anearly fusion model for 3D box estimation, which directlylearns to combine image and depth information optimally.|
|||Unlike the above approaches, the fusionarchitecture we propose is designed to be domain-agnosticand agnostic to the placement, type, and number of 3D sen-sors.|
|||Our deep network for 3D object box regression from im-ages and sparse point clouds has three main components:an off-the-shelf CNN [13] that extracts appearance and ge-ometry features from input RGB image crops, a variant ofPointNet [23] that processes the raw 3D point cloud, and afusion sub-network that combines the two outputs to predict3D bounding boxes.|
|||Our fusion sub-network features a novel dense 3D boxprediction architecture, in which for each input 3D point,the network predicts the corner locations of a 3D box rela-tive to the point.|
|||Unlike these245(A)3D point cloud (n x 3)(B)RGB image (RoI-cropped)point-wise featuren x 64global feature1 x 1024PointNet [512->128->128](C)fusion feature[, , ]n x 3136MLP3D box corner offsetsn x 8 x 3scoren x 1point-wise offsets to each cornerdzdx dyResNetblock-41 x 2048Dense Fusion (final model)argmax(score)[512->128->128](D)fusion feature[, ]1 x 3072MLP3D box corner locations1 x 8 x 3Global Fusion (baseline model)(E)Predicted 3D bounding boxFigure 2.|
|||We present two fusion network formulations: avanilla global architecture that directly regresses the box corner locations (D), and a novel dense architecture that predicts the spatial offsetof each of the 8 corners relative to an input point, as illustrated in (C): for each input point, the network predicts the spatial offset (whitearrows) from a corner (red dot) to the input point (blue), and selects the prediction with the highest score as the final prediction (E).|
|||2D-3D fusion Our paper is most related to recent methodsthat fuse image and lidar data.|
|||In addition, the current setup allows us to plug in any state-of-the-art detector without modifying the fusion network.|
|||2B), and a fusion network that combines both and out-puts a 3D bounding box for the object in the crop.|
|||We de-scribe two variants of the fusion network: a vanilla globalarchitecture (Fig.|
|||2C) and a novel dense fusion network(Fig.|
|||Below, we go intothe details of our point cloud and fusion sub-components.|
|||Fusion NetworkThe fusion network takes as input an image feature ex-tracted using a standard CNN and the corresponding pointcloud feature produced by the PointNet sub-network.|
|||Below we propose two fusionnetwork formulations, a vanilla global fusion network, anda novel dense fusion network.|
|||Global fusion network As shown in Fig.|
|||2C, the globalfusion network processes the image and point cloud fea-tures and directly regresses the 3D locations of the eightcorners of the target bounding box.|
|||We experimented witha number of fusion functions and found that a concatenationof the two vectors, followed by applying a number of fullyconnected layers, results in optimal performance.|
|||The lossfunction with the global fusion network is then:L = XismoothL1(xi , xi) + Lstn,(1)where xi are the ground-truth box corners, xi are the pre-dicted corner locations and Lstn is the spatial transforma-tion regularization loss introduced in [23] to enforce theorthogonality of the learned spatial transform matrix.|
|||A major drawback of the global fusion network is that thevariance of the regression target xi is directly dependent onthe particular scenario.|
|||These ideas motivate ourdense fusion network, which is described below.|
|||Dense fusion network The main idea behind this modelis to use the input 3D points as dense spatial anchors.|
|||The dense fusion networkprocesses this input using several layers and outputs a 3Dbounding box prediction along with a score for each point.|
|||Concretely, the loss func-tion of the dense fusion network is:L =1N XismoothL1(xi offset, xioffset) + Lscore + Lstn,(2)i where N is the number of the input points, xoffset is theoffset between the ground truth box corner locations and thei-th input point, and xoffset contains the predicted offsets.|
|||Effect of fusion Both car-only and all-category evaluationresults show that fusing lidar and image information al-ways yields significant gains over lidar-only architectures,but the gains vary across classes.|
|||We observe that the fusion model is better atestimating the dimension and orientation of objects than thelidar-only model.|
|||Second, weintroduce a novel dense fusion network, which combinesthe image and point cloud representations.|
|||Amulti-sensor fusion system for moving object detection andtracking in urban driving environments.|
|313|cvpr18-Left-Right Comparative Recurrent Model for Stereo Matching|Real-time probabilistic fusionof sparse 3d lidar and dense stereo.|
|314|Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields|Although there is con-fusion between left and right body parts and limbs in early stages,the estimates are increasingly refined through global inference inlater stages, as shown in the highlighted areas.|
|||Most of the false positives come from impreciselocalization, other than background confusion.|
|315|cvpr18-Learning Facial Action Units From Web Images With Scalable Weakly Supervised Clustering|Temporal modeled as the Long Short-Term Memory (LSTM) is aggregated to CNN architectureto construct fusion models (e.g., [8, 19]).|
|316|Yinlong_Liu_Efficient_Global_Point_ECCV_2018_paper|T. Whelan, M. Kaess, H. Johannsson, M. Fallon, J. J. Leonard, and J. McDonald,Real-time large-scale dense RGB-D SLAM with volumetric fusion, INTERNA-TIONAL JOURNAL OF ROBOTICS RESEARCH, vol.|
|317|cvpr18-Dual Attention Matching Network for Context-Aware Feature Sequence Based Person Re-Identification|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|318|RefineNet_ Multi-Path Refinement Networks for High-Resolution Semantic Segmentation|Multi-resolution fusion.|
|||All path inputs are then fused intoa high-resolution feature map by the multi-resolution fusionblock, depicted in Fig.|
|||The input adaptation in this blockalso helps to re-scale the feature values appropriately alongdifferent paths, which is important for the subsequent sum-fusion.|
|||In one pooling block, each pooling operation is followed byconvolutions which serve as a weighting layer for the sum-mation fusion.|
|||The fusion block fuses the information of multiple short-cut paths, which can be considered as performing summa-tion fusion of multiple residual connections with necessarydimension or resolution adaptation.|
|||In this aspect, the roleof the multi-resolution fusion block here is analogous tothe role of the summation fusion in a conventional resid-ual convolution unit in ResNet.|
|||There are certain layers inRefineNet, and in particular within the fusion block, thatperform linear feature transformation operations, like linearfeature dimension reduction or bilinear upsampling.|
|319|Multi-View 3D Object Detection Network for Autonomous Driving|We propose Multi-View 3Dnetworks (MV3D), a sensory-fusion framework that takesboth LIDAR point cloud and RGB images as input and pre-dicts oriented 3D bounding boxes.|
|||The network is composed of two subnetworks: one for 3Dobject proposal generation and another for multi-view fe-ature fusion.|
|||We design a deep fusion schemeto combine region-wise features from multiple views andenable interactions between intermediate layers of differentpaths.|
|||The fusionof LIDAR point cloud and RGB images should be able toachieve higher performance and safty to self-driving cars.|
|||detection by employing early or late fusion schemes.|
|||The main idea for utilizing multimodal information is toperform region-based feature fusion.|
|||The multi-view fusion network extracts region-wise features by projecting 3D proposals to the feature mapsfrom mulitple views.|
|||We design a deep fusion approachto enable interactions of intermediate layers from differentviews.|
|||Combined with drop-path training [14] and auxili-ary loss, our approach shows superior performance over theearly/late fusion scheme.|
|||A deep fusion network is used to combine region-wise features obtained via ROI pooling for each view.|
|||Related WorkWe briefly review existing work on 3D object detectionfrom point cloud and images, multimodal fusion methodsand 3D object proposals.|
|||In thiswork, we encode 3D point cloud with multi-view featuremaps, enabling region-based representation for multimodalfusion.|
|||In thispaper, we design a deep fusion approach inspired by Frac-talNet [14] and Deeply-Fused Net [26].|
|||Each 3D box1909CC(a) Early Fusion(b) Late FusionMMMMInputIntermediate layers Output(c) Deep FusionCMConcatenationElement-wise MeanFigure 3: Architectures of different fusion schemes: Weinstantiate the join nodes in early/late fusion with concate-nation operation, and deep fusion with element-wise meanoperation.|
|||Region(cid:173)based Fusion NetworkWe design a region-based fusion network to effectivelycombine features from multiple views and jointly classifyobject proposals and do oriented 3D box regression.|
|||To combine information from different fe-atures, prior work usually use early fusion [1] or late fu-sion [22, 12].|
|||Inspired by [14, 26], we employ a deep fusionapproach, which fuses multi-view features hierarchically.|
|||Acomparison of the architectures of our deep fusion networkand early/late fusion networks are shown in Fig.|
|||For anetwork that has L layers, early fusion combines features{fv} from multiple views in the input stage:fL = HL(HL1(   H1(fBV  fF V  fRGB)))(4){Hl, l = 1,    , L} are feature transformation functionsand  is a join operation (e.g., concatenation, summation).|
|||In contrast, late fusion uses seperate subnetworks to learnfeature transformation independently and combines theiroutputs in the prediction stage:1L (   HBVL (   HF VfL =(HBV(HF V(HRGB1L(fBV )))(fF V )))(5)(   HRGB1(fRGB)))To enable more interactions among features of the inter-mediate layers from different views, we design the follo-wing deep fusion process:f0 =fBV  fF V  fRGBfl =HBV(fl1)  HF Vll(fl1)  HRGBl(fl1),(6)l = 1,    , LWe use element-wise mean for the join operation for deepfusion since it is more flexible when combined with drop-path training [14].|
|||In particular,Oriented 3D Box Regression Given the fusion featu-res of the multi-view network, we regress to oriented3D boxes from 3D proposals.|
|||Network Regularization We employ two approaches toregularize the region-based fusion network: drop-path trai-ning [14] and auxiliary losses.|
||| In the muti-view fusion network, we add an extra fullyconnected layer f c8 in addition to the original f c6 andf c7 layer.|
|||lossAP3D (IoU=0.5)APloc (IoU=0.5)AP2D (IoU=0.7)Easy Moderate93.9293.5394.2196.0287.6087.7088.2989.05Hard87.2386.8887.2188.38Easy Moderate94.3193.8494.5796.3488.1588.1288.7589.39Hard87.6187.2088.0288.67Easy Moderate87.2987.4788.6495.0185.7685.3685.7487.59Hard78.7778.6679.0679.90Table 3: Comparison of different fusion approaches: Peformance are evaluated on KITTI validation set.|
|||We first compare our deep fusion net-work with early/late fusion approaches.|
|||As commonly usedin literature, the join operation is instantiated with conca-tenation in the early/late fusion schemes.|
|||As shown in Ta-ble 3, early and late fusion approaches have very similarperformance.|
|||Without using auxiliary loss, the deep fusionmethod achieves 0.5% improvement over early and latefusion approaches.|
|||Adding auxiliary loss further improvesdeep fusion network by around 1%.|
|||ConclusionWe have proposed a multi-view sensory-fusion modelfor 3D object detection in the road scene.|
|||A region-based fusion network is presented to deeply fusemulti-view information and do oriented 3D box regression.|
|320|Guorun_Yang_SegStereo_Exploiting_Semantic_ECCV_2018_paper|This redesignedmodel is also pretrained on the fusion set of CityScapes and FlyingThings3D,followed by fine-tuning on KITTI Stereo dataset.|
|321|Mang_YE_Robust_Anchor_Embedding_ECCV_2018_paper|Lan, X., Ma, A.J., Yuen, P.C., Chellappa, R.: Joint sparse representation androbust feature-level fusion for multi-cue visual tracking.|
|322|Shifeng_Zhang_Occlusion-aware_R-CNN_Detecting_ECCV_2018_paper|[4] exploit weakly annotated boxes via a segmentation infusionnetwork to achieve considerable performance gains.|
|||: Fused DNN: A deep neural networkfusion approach to fast and robust pedestrian detection.|
|323|Mohammad_Tavakolian_Deep_Discriminative_Model_ECCV_2018_paper|Peng, X., Wang, L., Wang, X., Qiao, Y.: Bag of visual words and fusion methodsfor action recognition: Comprehensive study and good practice.|
|324|Multi-View Supervision for Single-View Reconstruction via Differentiable Ray Consistency|While our approach and the alternativeway of depth fusion are comparable in the case of perfectdepth information, our approach is much more robust tonoisy training signal.|
|||This is because of the use of a raypotential where the noisy signal only adds a small penaltyto the true shape unlike in the case of depth fusion wherethe noisy signal is used to compute independent unary terms(see appendix [1] for detailed discussion).|
|||We observe that our approach is fairly robust tonoise unlike the fusion approach.|
|||Oct-netfusion: Learning depth fusion from data.|
|325|cvpr18-PoTion  Pose MoTion Representation for Action Recognition|Convolutionaltwo-stream network fusion for video action recognition.|
|326|cvpr18-Person Transfer GAN to Bridge Domain Gap for Person Re-Identification|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|327|cvpr18-A Deeper Look at Power Normalizations|Inparticular, Affine-Invariant Riemannian Metric [45, 4], KL-Divergence Metric (KLDM) [59], Jensen-Bregman LogDetDivergence (JBLD) [10] and Log-Euclidean (LogE) [2]have been used in the context of diffusion imaging and theRCD-based methods.|
|||Approach [40] applies a fusion of two CNN streams viaouter product in the context of the fine-grained image recog-nition.|
|||Optimal two-stream fusionNeural act.|
|||Log-euclidean metrics for fast and simple calculus on diffusiontensors.|
|||Non-euclideanstatistics for covariance matrices, with applications to dif-fusion tensor imaging.|
|||Attribute-enhanced face recognition with neural tensor fusion net-works.|
|328|cvpr18-Adversarially Occluded Samples for Person Re-Identification|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|329|Adrien_Kaiser_Proxy_Clouds_for_ECCV_2018_paper|Point-basedfusion [31] is also used to accumulate points without the need of a full volumetricrepresentation.|
|||Online scene reconstruction methods using vol-umetric fusion were pioneered by KinectFusion [15], then made more efficientwith VoxelHashing [38], and more accurate with BundleFusion [39].|
|||5. detect new proxy planes in Xt \ inliers(Pt):5.1 RANSAC-based plane detection [5];5.2 post-detection plane fusion;5.3 compute the local frame;5.4 initialize the new proxy with Xt.|
|||Usingthe global scene axes to compute the local frame leads to a fixed resolutionand spatial consistency for the grid of all proxies and allows efficient recoveryand fusion (step 5.2).|
|||Newcombe, R.A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A.J.,Kohi, P., Shotton, J., Hodges, S., Fitzgibbon, A.: Kinectfusion: Real-time densesurface mapping and tracking.|
|||Keller, M., Lefloch, D., Lambers, M., Izadi, S., Weyrich, T., Kolb, A.: Real-timeIn: International3d reconstruction in dynamic scenes using point-based fusion.|
|||Dai, A., Niener, M., Zollh ofer, M., Izadi, S., Theobalt, C.: Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration.|
|330|Huayi_Zeng_Neural_Procedural_Reconstruction_ECCV_2018_paper|The confusion matrix of the foun-dation classification in the right figurealso illustrates that U-shapes are one ofthe challenging cases together with III.|
|||The matrix shows confusion between IIand L or III and U, where these casesare not clearly distinguishable even tohuman eyes.|
|||We also observe the con-fusion between III and C, because some III-shaped houses are quite complex.|
|331|Steffen_Wolf_The_Mutex_Watershed_ECCV_2018_paper|: An efficient fusion move algo-rithm for the minimum cost lifted multicut problem.|
|||Cai, J., Lu, L., Zhang, Z., Xing, F., Yang, L., Yin, Q.: Pancreas segmentation inMRI using graph-based decision fusion on convolutional neural networks.|
|332|cvpr18-Tips and Tricks for Visual Question Answering  Learnings From the 2017 Challenge|MU-TAN: multimodal tucker fusion for visual question answer-ing.|
|333|Not All Pixels Are Equal_ Difficulty-Aware Semantic Segmentation via Deep Layer Cascade|The hard cases with high confusion will bepropagated and handled by the subsequent expert networks.|
|||To ensure a fair comparison, we evaluate DeepLab-v2 and SegNet without any pre- and post-processing, e.g.,training with extra data, multi-scale fusion, or smoothingwith conditional random fields (CRF).|
|334|cvpr18-Facial Expression Recognition by De-Expression Residue Learning|5 is the confusion matrix of our method,where fear expression shows the lowest recognition ratewith 90%.|
|||Confusion matrix on CK+is labeled as one of the six basic expressions.|
|||The confusionmatrix is shown in Fig.|
|||Confusion matrix on Oulu-CASIAquences from 31 subjects.|
|||Confusion matrix on MMIFigure 8.|
|||Confusion matrix on BU-3DFEmethod shows improvement of 1.7%.|
|||As shown from theconfusion matrix in Fig.|
|||However, our single modality(image-based) DeRL method achieves close performancecompared to the multi-modal fusion approach.|
|||As we cansee from the confusion matrix Fig.|
|||Confusion matrix of recognizing four expressions onBP4D+training, and BP4D+ for testing.|
|335|Simultaneous Geometric and Radiometric Calibration of a Projector-Camera Pair|This fusion,together with an innovative projection pattern design (dis-cussed in section 3), facilitates the geometric calibration andallows for simultaneous radiometric calibration.|
|336|Shape Completion Using 3D-Encoder-Predictor CNNs and Shape Synthesis|All views are integrated into a shared volumetric grid usingthe volumetric fusion approach by Curless and Levoy [5],where the voxel grids extent is defined by the model bound-ing box.|
|||3D Encoder-Predictor Network (3D-EPN)for Shape CompletionWe propose a 3D deep network that consumes a partialscan obtain from volumetric fusion [5], and predicts the dis-tance field values for the missing voxels.|
|||Bundlefusion: Real-time globally consistent 3d reconstruc-tion using on-the-fly surface re-integration.|
|||Kinectfusion: Real-time dense surface map-ping and tracking.|
|||Elasticfusion: Dense slam without a posegraph.|
|337|NIKOLAOS_ZIOULIS_OmniDepth_Dense_Depth_ECCV_2018_paper|Similarly,in [41] and [42] depth values were discretized in bins and densely classified, tobe afterwards refined either via a hierarchical fusion scheme or through the useof a CRF respectively.|
|||Li, B., Dai, Y., He, M.: Monocular depth estimation with hierarchical fusion ofdilated cnns and soft-weighted-sum inference.|
|338|cvpr18-Cross-Dataset Adaptation for Visual Question Answering|Mutan:Multimodal tucker fusion for visual question answering.|
|339|Songtao_Liu_Receptive_Field_Block_ECCV_2018_paper|For ASPP, dilated convolution varies the sam-pling distance from the center, but the features have a uniform resolution fromthe previous convolution layers of the same kernel size, which treats the cluesat all the positions equally, probably leading to confusion between object andcontext.|
|||Dilated convolutional layer: In early experiments, we choose dilated pool-ing layers for RFB to avoid incurring additional parameters, but these stationarypooling strategies limit feature fusion of RFs of multiple sizes.|
|340|cvpr18-What Have We Learned From Deep Representations for Action Recognition |First, cross-stream fusion enables the learning of true spa-tiotemporal features rather than simply separate appear-ance and motion features.|
|||Studying a single filter at layer conv5 fusion: (a) and(b) show what maximizes the unit at the input: multiple colouredblobs in the appearance input (a) and moving circular objects atthe motion input (b).|
||| is used for weighting the degree of spatiotem-poral variation and  is an explicit slowness parameter that37846ConvolutionalFeature Mapsconv1_1conv1_2conv2_1conv2_2conv2_3conv3_1conv3_2conv3_3conv4_1conv4_2conv4_3conv5_1conv5_2conv5_3widthdepthheightInputAppearanceMotionconv1_1conv1_2conv2_1conv2_2conv2_3conv3_1conv3_2conv3_3conv4_1conv4_2conv4_3conv5_1conv5_2conv5_3FusionMaximizechannelccLossconv5_fusionfc6fc7clsaccounts for the regularization strength on the temporal fre-quency.|
|||ExperimentsFor sake of space, we focus all our experimental stud-ies on a VGG-16 two-stream fusion model [8] that is illus-trated in Fig.|
|||Emergence of spatiotemporal featuresWe first study the conv5 fusion layer (i.e.|
|||2 for the overall architecture), which takes infeatures from the appearance and motion streams and learnsa local fusion representation for subsequent fully-connectedlayers with global receptive fields.|
|||At conv5 fusion we see the emergence of bothclass specific and class agnostic units (i.e.|
|||This factempirically verifies that the fusion unit also expects specificappearance when confronted with particular motion signals.|
|||We now consider unit f004 at conv5 fusion in Fig.|
|||Studying the Billiards unit at layer conv5 fusion fromFig.|
|||Specific unit at conv5 fusion.|
|||To begin,we consider filters f006 and f009 at the conv5 fusion layerthat fuses from the motion into the appearance stream, asshown in Fig.|
|||6, we similarly show general feature examples for theconv5 fusion layer that seem to capture general spatiotem-poral patterns for recognizing classes corresponding to mul-57848appearance  = 0flow  = 10flow  = 5flow  = 1flow  = 0appearanceflow  = 10flow  = 5flow  = 1flow  = 0700fnoisuf5vnoc900fnoisuf5vnocYoYo 1flowNunchucks 1flowNunchucks 2flowFigure 5.|
|||Two general units at the convolutional fusion layer.|
|||General units at the convolutional fusion layer that couldbe useful for representing ball sports.|
|||Visualization of fusion layers.|
|||We now briefly re-examinethe convolutional fusion layer (as in the previous Sect.|
|||8, we show the filters at the conv5 fusion layer,which fuses from the motion into the appearance stream,while varying the temporal regularization and keeping thespatial regularization constant.|
|||The visualizations reveal thatthese first 3 fusion filters at this last convolutional layershow reasonable combinations of appearance and motioninformation, a qualitative proof that the fusion model in [8]performs as desired.|
|||For example, the receptive field centreof conv5 fusion f002 seems matched to lip like appearancewith a juxtaposed elongated horizontal structure, while themotion is matched to slight up and down motions of theelongation (e.g.|
|||Visualization of 3 filters of the conv5 fusion layer.|
|||fully-connected layersthat operate on top of the convolutional fusion layer illus-trated above.|
|||In UCF101 the major confusions are between the classes77850appearance = 10 = 5 = 1 = 0ArcheryBabyCrawlingPlayingFluteCleanAndJerkBenchPressFigure 11.|
|||Explaining confusion for PlayingCello and PlayingVio-lin.|
|||We see that the learned representation focuses on the horizon-tal (Cello) and vertical (Violin) alignment of the instrument, whichcould explain confusions for videos where this is less distinct.|
|||This insight not only explains the confusion, but alsocan motivate remediation, e.g.|
|||Two classes, ApplyEye-Makeup and ApplyLipstick are, even though being visuallyvery similar, easily classified in the test set of UCF101 withclassification rates above 90% (except for some obviousconfusions with BrushingTeeth).|
|||Our visual explanations arehighly intuitive and indicate the efficacy of processing ap-pearance and motion in parallel pathways, as well as cross-stream fusion, for analysis of spatiotemporal information.|
|||Explaining confusion for BrushingTeeth and Shaving-Beard.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|341|Temporal Attention-Gated Model for Robust Sequence Classification|To modelthis neighborhood influence, we infer the attention score atin Equation 1 using a bi-directional RNN:at = (m(h t;h t) + b)(4)Herein, m is the weight vector of our fusion layer whichintegrates both directional layers of our bi-directional RNNand b is the bias term.|
|||We compare our model with the base-line method [15] on this dataset, which performs classifica-tion separately with Support Vector Machine (SVM) mod-els trained on the bag-of-words representations for severalpopular features separately and then combines the resultsusing late fusion.|
|||ModelTraining strategyFeaturemAPBOW+SVMSeparatelySIFTSTIP+late average fusion(one-vs-all)SIFT+STIPPlain-RNNGRULSTMTAGMJointlyJointlyJointlyJointlyCNNCNNCNNCNNCNN0.520.450.550.670.450.560.550.63capture the relevant action, object and scene to the event,e.g., the action of riding bike for the event biking, cake forthe event birthday and baseball playground for the eventbaseball.|
|342|cvpr18-Trapping Light for Time of Flight|Multi-view image and tof sensor fusion fordense 3d reconstruction.|
|343|Combining Bottom-Up, Top-Down, and Smoothness Cues for Weakly Supervised Image Segmentation|The fusion of (i)-(iii) isrealized via a conditional random field as recurrent networkaimed at generating a smooth and boundary-preserving seg-mentation.|
|||Finally, the bottom-up and top-down cues are fused anditeratively refined in the CRF-RNN for improving localization of object boundaries and spatial smoothness of the final segmentation (black links for fusionand refinement computation).|
|344|Andrew_Owens_Audio-Visual_Scene_Analysis_ECCV_2018_paper|We hypothesize that early fusion of audio and visual streamsis important for modeling actions that produce a signal in both modalities.|
|||We thereforepropose to solve our task using a 3D multisensory convolutional network (CNN) withan early-fusion design (Figure 2).|
|||We train an early-fusion, multisensory network to predictwhether video frames and audio are temporally aligned.|
|||Before fusion, we apply a small number of 3D convolution and pooling operationsto the video stream, reducing its temporal sampling rate by a factor of 4.|
|||We split the training/test to have disjoint speaker identities (72%,Audio-Visual Scene Analysis with Self-Supervised Multisensory Features11MethodAllMixed sexSame sexGRID transferOn/off + PITFull on/offMonoSingle frameNo early fusionScratchI3D + Kineticsu-net PIT [36]Deep Sep. [67]7.67.06.95.07.05.86.67.31.38.88.48.47.28.47.68.28.81.9On/off SDR SIR SAR On/off SDR On/off SDR On/off SDR7.811.27.311.47.311.414.85.76.911.66.312.96.612.38.12.210.29.89.810.310.19.49.710.38.712.111.511.47.811.09.710.711.43.010.610.710.813.211.011.811.611.811.911.916.212.113.912.913.013.113.117.813.515.214.46.55.75.73.15.74.25.15.90.8Table 2: Source separation results on speech mixtures from the VoxCeleb (broken down by genderof speakers in mixture) and transfer to the simple GRID dataset.|
|||One might also ask whether early audio-visual fusion is helpful  the network,after all, fuses the modalities in the spectrogram encoder-decoder as well.|
|||: Learning joint statistical models foraudio-visual fusion and segregation.|
|345|Xuecheng_Nie_Mutual_Learning_to_ECCV_2018_paper|We can also observe directfusion of representations from both models as VGG16-FCN-Add/Multi/Concatcannot sufficiently utilize guidance information, resulting in very limited perfor-mance improvement.|
|||In contrast to these naive fusion strategies, VGG16-FCN-MuLA can learn more powerful representations via dynamically adapting pa-rameters.|
|346|cvpr18-Motion Segmentation by Exploiting Complementary Geometric Models|By doing so, we make sure that our findingsare not an artifact of a particular fusion scheme.|
|||Wethen propose using affinity matrix fusion as a means of deal-ing with real-world effects that are often difficult to modelwith a pure homography or fundamental matrix.|
|||3.3.3 Subset Constrained Multi-View Spectral Clus-teringThe above two multi-view spectral clustering schemes aregeneric fusion methods that do not exploit any relation thatmight exist between the different views.|
|||Usu-ally, the fusion can produce the best of all performance re-gardless of the fusion scheme used.|
|||In both (a) and (b), the fusion schemes manage to correctthese errors.|
|||The geometrical exactness of the fundamen-tal matrix approach is theoretically appealing; we show howits potential can be harnessed in a multi-view spectral clus-tering fusion scheme.|
|348|cvpr18-CartoonGAN  Generative Adversarial Networks for Photo Cartoonization|Li and Wand [20] obtained style transfer by local match-ing of CNN feature maps and using a Markov Random Fieldfor fusion (CNNMRF).|
|349|cvpr18-Deep Hashing via Discrepancy Minimization|Cross-modalitybinary code learning via fusion similarity hashing.|
|350|Kang_Pairwise_Relational_Networks_ECCV_2018_paper|Comparison of the number of images, the number of networks, the dimen-sionality of feature, and the accuracy of the proposed method with the state-of-the-artmethods on the LFWMethodImagesNetworksDimensionAccuracy (%)DeepFace [34]DeepID [30]DeepID2+ [32]DeepID3 [41]FaceNet [28]Learning from Scratch [40]CenterFace [36]PIMNetTL-Joint Bayesian [17]PIMNetfusion [17]SphereFace [23]ArcFace [10]model A (baseline, only f g)PRNPRN+model B (f g + P RN )model C (f g + P RN +)4M202, 599300, 000300, 000200M494, 4140.7M198, 018198, 018494, 4143.1M2.8M2.8M2.8M2.8M2.8M912025501214411111114, 096  4150  120150  120300  100128160  25121, 02461, 0245122, 0481, 0001, 0001, 0241, 02497.2597.4599.4799.5299.6397.7399.2898.3399.0899.4299.7899.699.6199.6999.6599.76(the base CNN model, just uses f g) and P RN + outperforms model B whichis jointly combined both f g with P RN .|
|||Kang, B.N., Kim, Y., Kim, D.: Deep convolutional neural network using tripletsof faces, deep ensemble, and score-level fusion for face recognition.|
|351|AdaScan_ Adaptive Scan Pooling in Deep Convolutional Neural Networks for Human Action Recognition in Videos|For complimentaryfeatures we compute results with improved dense trajecto-ries (iDT) [40] and 3D convolutional (C3D) features [37]and report performance using weighted late fusion.|
|||UCF101 HMDB51XXXXXXXXXXXXXXXXXXXXXXXXX3D convolutional filtersXXshallowshallowlate fusionlate fusion88.088.088.288.690.377.089.289.192.493.194.283.484.388.589.491.393.259.459.463.241.356.465.262.063.369.453.958.463.854.961.066.9Table 3: Comparison with existing methods (Attn.|
|||Bag of visualwords and fusion methods for action recognition: Compre-hensive study and good practice.|
|352|cvpr18-PieAPP  Perceptual Image-Error Assessment Through Pairwise Preference|Perceptual image qualityassessment using block-based multi-metric fusion (BMMF).|
|||A multi-metric fusion approachto visual quality assessment.|
|||Image quality assessment usingmulti-method fusion.|
|353|T_M_Feroz_Ali_Maximum_Margin_Metric_ECCV_2018_paper|Zhao, H., Tian, M., Sun, S., Shao, J., Yan, J., Yi, S., Wang, X., Tang, X.: Spindle net: Personre-identification with human body region guided feature decomposition and fusion.|
|||Zheng, L., Wang, S., Tian, L., He, F., Liu, Z., Tian, Q.: Query-adaptive late fusion for imagesearch and person reidentification.|
|354|cvpr18-End-to-End Flow Correlation Tracking With Spatial-Temporal Attention|A naive feature fusion may even deteriorate the per-formance because of misalignment.|
|355|Eddy_Ilg_Occlusions_Motion_and_ECCV_2018_paper|We still keep the former fusionnetwork as it also performs smoothing and sharpening (see Figure 1(a)).|
|356|Human Shape From Silhouettes Using Generative HKS Descriptors and Cross-Modal Neural Networks|Various shape descriptors have been proposed, with mostrecent approaches being diffusion based methods [57, 9,49].|
|||Givensuch a graph constructed by connecting pairs of vertices ona surface with weighted edges, the heat kernel Ht(x, y) isdefined as the amount of heat that is transferred from thevertex x to vertex y at time t, given a unit heat source atx [57]:Ht(x, y) = Xieiti(x)i(y),(3)where Ht denotes the heat kernel, t is the diffusion time, iand i represent the ith eigenvalue and the correspondingeigenvector of the Laplace-Beltrami operator, respectively,and x and y denote two vertices.|
|||Anisotropic diffusion descriptors.|
|||A gromov-hausdorff frameworkwith diffusion geometry for topologically-robust non-rigidshape matching.|
|||Au-diovisual synchronization and fusion using canonical corre-lation analysis.|
|357|cvpr18-Structure Preserving Video Prediction|The upper part is proposed temporal-adaptive convolution kernel,Tem-K, while the lower part is the temporal fusion scheme,Fus-4.|
|||As shown in Figure3, the hidden state of the last 4 time-steps are first passedthrough a fusion sub-module, then feed into the next time-step for prediction.|
|358|cvpr18-Multi-Level Fusion Based 3D Object Detection From Monocular Images|With thehelp of a stand-alone module to estimate the disparity andcompute the 3D point cloud, we introduce the multi-levelfusion scheme.|
|||In addi-tion, another module is introduced to estimate the disparityinformation and adopt multi-level fusion method for accu-rate 3D localization, constituting our 3D object detectionprocedure.|
|||Our first contribution is an efficient multi-level fusionbased method for 3D object detection with a stand-alonemodule for estimating the disparity information.|
|||In particular, we adopt multi-level fusionmethods for accurate 3D localization and system enhance-ment, constructing the robust detection pipeline.|
|||Thislayer is regarded as RoI max pooling to prevent confusionin this paper.|
|||The joint estimation for 3D location can be seen as a latefusion between estimations from Sconv and Spc.|
|||The late fusion ensures the accurate 3D localization inthe network, which is the most important part of the whole3D object detection framework.|
|||This2349can be regarded as an early fusion or a pre-processing stepfor enhancing the input.|
|||In the esti-mation of 3D location, another fusion for different featuremaps is proposed.|
|||In total, there are three levels of fusion in the network.|
|||The earliest fusion is the concatenation between front viewfeature maps and the corresponding RGB image.|
|||The last fusionis the joint estimation from two different types of data forthe final 3D localization.|
|||Generally, the last fusion is nec-essary for the framework, while the other two can improvethe whole performance to a certain extent.|
|||We also measure the effect of fusion methodsin our framework.|
|||As we can see, both input fusion(FV)and feature fusion(FF) can improve the detection results.|
|||All experiments are done with estimation fusion for 3Dlocalization, which is also the core part for the 3D detec-tion pipeline.|
|||Typically, more fusion requires more param-eters.|
|||In particular, changing 3-channel RGB input to 6-channel RGB+FV input only introduces 0.009% additionalweights, and adding fusion of XYZ map only increases0.79% weights.|
|||FV indicates the fusion between the front view feature maps and the RGB image.|
|||FF indicates the fusionbetween Fmax and Fmean.|
|||FV indicates the fusion between the front view feature maps and the RGB image.|
|||FF means the fusion between Fmax andFmean.|
|||The main innovation in the framework is the multi-levelfusion scheme.|
|359|Dapeng_Chen_Improving_Deep_Visual_ECCV_2018_paper|Zhao, H., Tian, M., Sun, S., Shao, J., Yan, J., Yi, S., Wang, X., Tang, X.: Spindlenet: Person re-identification with human body region guided feature decompositionand fusion.|
|360|cvpr18-Zero-Shot Sketch-Image Hashing|The third network mitigatesthe sketch-image heterogeneity and enhances the semanticrelations among data by utilizing the Kronecker fusion layerand graph convolution, respectively.|
||| We propose an end-to-end three-network structure fordeep generative hashing, handling the train-test cat-egory exclusion and search efficiency with attentionmodel, Kronecker fusion and graph convolution.|
|||find the Kronecker product fusion layer suitable for ourmodel, which is discussed in Sec.|
|||Fusing sketch and image with Kronecker layerSketch-image feature fusion plays an important role inour task as is addressed in problem (b) of Sec.|
|||To this end, we uti-lize the recent advances in Kronecker-product-based featurelearning [22] as the fusion network.|
|||Denoting the attentionmodel outputs of a sketch-image pair {y, x} from the samecategory as h(sk)  R256 and h(im)  R256, a non-lineardata fusion operation can be derived asW  1h(sk)  3h(im).|
|||(1)Here W is a third-order tensor of fusion parameters and denotes tensor dot product.|
|||(2)Kronecker layer [22] is supposed to be a better choicein feature fusion for ZSIH than many conventional methodssuch as layer concatenation or factorized model [71].|
|||layer  MFB [71]Stochastic neuron  bit regularizationDecoder  classifierWithout GCNsGCNs  word vector fusiont = 1 for GCNst = 106 for GCNsZSIH (full model)0.2280.2360.1870.1620.2330.2190.0620.2410.2540.2070.2110.1580.1330.1710.1760.0550.2020.220art cross-modal hashing works are introduced includingCMSSH [4], CMFH [13], SCM [72], CVH [33], SePH [36]and DSH [38], where DSH [38] can also be subjected to anSBIR model and thus is closely related to our work.|
|||Todemonstrate the effectiveness of the Kronecker layer fordata fusion, we introduce two baselines by replacing theKronecker layer [22] with the conventional feature con-catenation and the multi-modal factorized bilinear pooling(MFB) layer [71].|
|||Data fusion through cross-modality metric learning us-ing similarity-sensitive hashing.|
|||Attribute-enhanced face recognition with neural tensor fusion net-works.|
|361|End-To-End 3D Face Reconstruction With Deep Neural Networks|Moreover, weintegrate in the DNN architecture two components, namelya multi-task loss function and a fusion convolutional neuralnetwork (CNN) to improve facial expression reconstruction.|
|||With thefusion-CNN, features from different intermediate layers arefused and transformed for predicting the 3D expressive fa-cial shape.|
|||Specifically, we add two key components, a sub convo-lutional neural network (fusion-CNN) that fuses featuresfrom intermediate layers of VGG-Face for regressing theexpression parameters and a multi-task learning loss func-tion for both the identity parameters prediction and the ex-pression parameters prediction.|
|||The second type of neural layers includes the three convolu-tional layers in the fusion-CNN and the following fully con-nected layers.|
|||Another difference isthat we employ a multi-task learning loss and a fusion-CNNfor fusing intermediate features.|
|||We also compare our method with the UH-E2FARModalgorithm, a modification of our UH-E2FAR algorithm byremoving the fusion convolutional neural networks (fusion-CNN) to demonstrate the advantage of our algorithm in re-constructing expressive 3D faces.|
|||We first fine-tune only the fully connected layers and thefusion-CNN for 40,000 iterations.|
|||We also compare our method with UH-E2FARMod to demonstrate the benefit of the fusion-CNNwe proposed in reconstructing expressive 3D face.|
|||We also introduce two key components to ourframework, namely a fusion-CNN and a multi-task learn-ing loss.|
|362|Junho_Jeon_Reconstruction-based_Pairwise_Depth_ECCV_2018_paper|As the quality of concurrently captured RGB image is relatively bet-ter than the depth image, exploiting the correlation between color and geometryinformation, called sensor fusion, was investigated, mainly with local filter-basedmethods [37, 38, 47].|
|||For each level of theimage pyramid, a long skip connection directly passes the extracted features tothe later corresponding part of the network to enable a fusion of the featuresextracted in different scales (red arrows).|
|||Dai, A., Niener, M., Zollhofer, M., Izadi, S., Theobalt, C.: Bundlefusion: Real-time globally consistent 3d reconstruction using on-the-fly surface reintegration.|
|||Newcombe, R.A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A.J.,Kohi, P., Shotton, J., Hodges, S., Fitzgibbon, A.: Kinectfusion: Real-time densesurface mapping and tracking.|
|363|cvpr18-Deep Mutual Learning|Spindle net: Person re-identification with human body region guided feature de-composition and fusion.|
|364|cvpr18-Self-Supervised Adversarial Hashing Networks for Cross-Modal Retrieval|To solve the problem, we have de-signed a multi-scale fusion model, which consists of multi-ple average pooling layers and a 1  1 convolutional layer.|
|||Generative Network for Text: We built TxtNet using athree-layer feed-forward neural network and a multi-scale(MS) fusion model (TMS4096512N).|
|||Data fusion through cross-modality metric learning us-ing similarity-sensitive hashing.|
|||Cross-modalitybinary code learning via fusion similarity hashing.|
|365|cvpr18-Visual Feature Attribution Using Wasserstein GANs|An a con-trario approach for the detection of patient-specific brain per-fusion abnormalities with arterial spin labelling.|
|||Patient-specific detection of perfusion abnormalities com-bining within-subject and between-subject variances in Ar-terial Spin Labeling.|
|366|cvpr18-First-Person Hand Action Benchmark With RGB-D Videos and 3D Hand Pose Annotations|5 (a) we show the recog-nition accuracies per category on a subset actions and theaction confusion matrix is shown in Fig.|
|||(b) Hand action confusion matrix for our LSTM baseline.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|367|cvpr18-Thoracic Disease Identification and Localization With Limited Supervision|first trained a CNN on image patchesand then an image-level decision fusion model by patch-level prediction histograms to generate the image-level la-bels [14].|
|||Since we areij and 1  pk8293AtelectasisCardiomegaly ConsolidationDiseasebaseline0.70ours0.80  0.00DiseasebaselineHernia0.870.810.87  0.01Infiltration0.660.70Mass0.69Edema0.81Effusion0.760.80  0.010.88  0.010.87  0.00NodulePleural ThickeningEmphysema0.83Fibrosis0.790.91  0.01Pneumonia0.78  0.02Pneumothorax0.670.680.660.80ours0.77  0.030.70  0.010.83  0.010.75  0.010.79  0.010.67  0.010.87  0.01Table 1.|
|||8295AUC scoreAUC score   Unannotated80%60%40%20%0%AnnotatedAtelectasis               Cardiomegaly          Consolidation                  Edema                     Effusion        Emphysema                 FibrosisHernia                  Infiltration                     Mass                          Nodule             Pleural Thickening         Pneumonia             PneumothoraxAnnotated0.79580.78250.87410.84940.79540.78530.88210.85050.86680.86340.90500.89300.78370.75970.78390.77370.85430.81060.79600.76640.87070.84150.86370.85960.89450.88510.76570.70480.76860.74230.84610.76010.78300.72360.85760.74630.85200.84420.89100.82690.73340.60300.73370.71510.80860.72750.71990.63920.76770.64640.82610.82360.82620.78290.66310.57620.67610.86850.66740.73870.76110.72500.52500.50.60.70.80.91.080%0%80%0%80%0%80%0%80%0%80%0%80%0%0.76630.67820.70170.65960.82770.80700.76730.72310.76490.75390.67320.66210.88290.85270.73510.60700.69870.65380.83310.78190.74980.70590.74640.71770.65590.65340.86640.83080.65610.54620.67310.64150.80830.74930.71200.66910.75900.69660.62090.59020.81990.80700.60590.51550.64660.63420.75390.71170.66540.63350.70820.65910.54780.52250.78500.76800.54950.61800.64210.57310.59750.60660.70720.50.60.70.80.91.080%0%80%0%80%0%80%0%80%0%80%0%80%0%Figure 5.|
|||Thus for some disease types like Pneumonia, when the8296AnnotatedUnannotated0.62990.88800.78310.90680.69630.29170.30570.43550.71750.93480.85820.92900.71830.43300.46560.53020.74320.92190.89930.92680.76930.48210.52550.62900.76480.97390.89650.93870.77800.49560.58690.66440.81340.98610.91670.97840.78490.48780.63730.66870.00.20.40.60.81.0AtelectasisCardiomegalyEffusionInfiltrationMassNodulePneumoniaPneumothorax0% : 100%20% : 100%40% : 100%60% : 100%80% : 100%AnnotatedUnannotated0.52790.99970.75310.87550.45240.11140.78580.47340.72380.99130.87440.92080.67360.27130.64360.62410.72280.99480.89160.94820.73150.36380.65050.64520.76720.99240.90030.95410.76110.46400.61720.61110.75680.98710.89600.94980.70030.54460.55810.63200.81340.98610.91670.97840.78490.48780.63730.66870.00.20.40.60.81.0AtelectasisCardiomegalyEffusionInfiltrationMassNodulePneumoniaPneumothorax80%: 0%80% : 20%80%: 40%80%: 60%80%: 80%80% : 100%T(IoU) Model AtelectasisCardiomegalyEffusionInfiltration0.690.940.660.710.10.20.30.40.50.60.7ref.|
|||MassEffusionEdemaCardiomegalyFibrosisConsolidationInfiltrationAtelectasisPneumothoraxFigure 7.|
|||Edema always appears inan area that is full of small liquid effusions as the exampleshows.|
|368|Isma_Hadji_A_New_Large_ECCV_2018_paper|Separate classification is performed by eachpathway, with late fusion used to achieve the final result.|
|||A close inspection of the confusion matrices (Fig.|
|||These two categories were specifically constructed to havethis potential source of appearance-based confusion to investigate an algorithmsability to abstract from appearance to model dynamics; see Fig.|
|||The confusions experienced by C3D and the Flow stream indicate thatthose approaches have poor ability to learn the appropriate abstractions.|
|||Confusion matrices of all the compared ConvNet architectures on the dynamicsbased organization of the new DTDBC3DRGB StreamFlow Stream MSOE StreamSOE-NetFig.|
|||Confusion matrices of all compared ConvNet architectures on the appearancebased organization of the new DTDBThese points are underlined by noting that MSOE stream has the best per-formance compared to the other individual streams, with increased performancemargin ranging from 4-8%.|
|||Here inspection of the confusion matrices (Fig.|
|||4), reveals that C3D andthe RGB stream tend to make similar confusions, which confirms the tendencyof C3D to capitalize on appearance.|
|||Notably, MSOE streams incursless confusions, which demonstrates the ability of MSOE filters to better capturefine grained differences.|
|||Second, closer inspec-tion of the confusion matrices show that optical flow fails on most categorieswhere the sequences break the fundamental optical flow assumptions of bright-ness constancy and local smoothness (e.g.|
|369|David_Schubert_Direct_Sparse_Odometry_ECCV_2018_paper|Lovegrove, S., Patron-Perez, A., Sibley, G.: Spline fusion: A continuous-time repre-sentation for visual-inertial fusion with application to rolling shutter cameras.|
|370|cvpr18-Wing Loss for Robust Facial Landmark Localisation With Convolutional Neural Networks|Real-time 3d face fitting and texture fusion on in-the-wild videos.|
|371|A Low Power, Fully Event-Based Gesture Recognition System|Convolu-tional two-stream network fusion for video action recogni-tion.|
|372|Fast Multi-Frame Stereo Scene Flow With Motion Segmentation|This flow proposal is fused with the camera motion-basedflow proposal using fusion moves to obtain the final opti-cal flow and motion segmentation.|
|||Finally, this flow proposal is fused withthe camera motion-based flow proposal using fusion movesto obtain the final flow map and motion segmentation.|
|||Since the final optimization per-formed on each frame fuses rigid and non-rigid optical flowproposals (using MRF fusion moves) the resulting binarylabeling indicates which pixels belong to non-rigid objects.|
|||3940VisualodometryInial moonsegmentaonOpcal flow,FRigid flowSInit.seg.Epipolar stereo/,,,Flow fusionFNon-rigid flow,+ D+,D+Ego-moonDDisparity+SBinocularstereoInput (,)DInit.disparity/,,,,+F,FFFlowSSegmentaonSubsequently, the problem was studied in the binocularstereo setting [26, 19, 45].|
|||Optical flow and flow fusion.|
|||Thisfusion step also produces the final segmentation S. Theseinputs and outputs are illustrated in Figs.|
|||The fusion process is similar to the initial segmentation.|
|||The fusion step only infers sp for pixels labeled fore-ground in the initial segmentation  S, since the backgroundlabels are fixed.|
|||The graph cut optimization for fusion istypically very efficient, since the pixels labeled foregroundin  S is often a small fraction of all the pixels.|
|||Since NCC and flow-based costmaps C nccp used in the segmentation and fusionsteps are noisy, we smooth them by averaging the valueswithin superpixels.|
|||ces( emarf repemgnnnuRi  43210Flow fusionInial segmentaonOpcal flowEpipolar stereoVisual odometryBinocular stereoPrior flowInializaonFigure 8.|
|373|cvpr18-Pose Transferrable Person Re-Identification|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|374|cvpr18-Reconstruction Network for Video Captioning|De-In ACM MM,scribing videos using multi-modal fusion.|
|375|Shao-Hua_Sun_Multi-view_to_Novel_ECCV_2018_paper|In computer vision, theseapproaches are isolated and tackled separately, and the fusion of data is less wellunderstood.|
|376|Saining_Xie_Rethinking_Spatiotemporal_Feature_ECCV_2018_paper|Since then, many video classification methods fol-low the same multi-stream 2D CNN design, and have made improvements in terms ofnew representations [25, 26], different backbone architecture [2729, 17], fusion of thestreams [3033] and exploiting richer temporal structures [3436].|
|||Feichtenhofer, C., Pinz, A., Zisserman, A.: Convolutional two-stream network fusion forvideo action recognition.|
|377|cvpr18-SO-Net  Self-Organizing Network for Point Cloud Analysis|networks, early, middle or late fusion may exhibit differentperformance [9].|
|||With a series of experiments, we foundthat middle fusion with average pooling is most effectivecompared to other fusion methods.|
|||Locality-sensitive deconvolution networks with gated fusion for rgb-dindoor semantic segmentation.|
|378|Low-Rank Embedded Ensemble Semantic Dictionary for Zero-Shot Learning|In each con-fusion matrix, the column denotes the ground truth and therow represents the predicted results.|
|||While for AwA, we observe fromthe confusion matrix that our algorithm achieves over 80%accuracy for some animal classes, e.g., leopard (84.21%)and rat (83.08%).|
|||Confusion matrices of the classification accuracy on unobserved categories for our approach on (a) aP&aY and (b) AwA, wherediagonal position indicates the classification accuracy.|
|379|cvpr18-Optimizing Filter Size in Convolutional Neural Networks for Facial Action Unit Recognition|Deci-sion level fusion of domain specific regions for facial actionrecognition.|
|380|Csaba_Domokos_MRF_Optimization_with_ECCV_2018_paper|1.1 Related WorkPartially ordered label sets are very common in several computer vision appli-cations like optical flow estimation, image registration, stereo exposure fusion,etc., where the label set L is the Cartesian product of totally ordered sets.|
|381|cvpr18-PAD-Net  Multi-Tasks Guided Prediction-and-Distillation Network for Simultaneous Depth Estimation and Scene Parsing|(ii) Second, we design and investigate three different multi-modal distillation modules for deep multi-modal data fu-sion, which we believe can be also applied in other sce-narios such as multi-scale deep feature fusion.|
|||The distillation module C is an attention-guided message passing mechanism for information fusion.|
|||A common way indeep networks for information fusion is to perform a naiveconcatenation of the feature maps or the score maps fromdifferent semantic layers of the network.|
|||Monocular depth estimation withhierarchical fusion of dilated cnns and soft-weighted-sum in-ference.|
|382|Ji_Zhu_Online_Multi-Object_Tracking_ECCV_2018_paper|Kutschbach, T., Bochinski, E., Eiselein, V., Sikora, T.: Sequential sensor fusioncombining probability hypothesis density and kernelized correlation filters formulti-object tracking in video data.|
|383|Lele_Chen_Lip_Movements_Generation_ECCV_2018_paper|Audio-Identity fusion network fuses features from twomodalities.|
|||Our fusion method is based on du-plication and concatenation.|
|||3: Audio-Identity fusion.|
|384|cvpr18-Multi-Level Factorisation Net for Person Re-Identification|The MLFN architecture is noteworthy in that:(i) Acompact FS is generated by concatenating the FSM out-put vectors from all blocks, and therefore multi-level fea-ture fusion is obtained without exploding dimensionality;(ii) Using the FSM output vectors to predict person identityvia skip connections and fusion provides deep supervision[21, 14] which ensures that the learned factors are identity-discriminative, but without introducing a large number ofparameters required for conventional deep supervision.|
|||More importantly, it ex-tends both in that it is the selection of which factor modulesor experts are active that provides a compact latent seman-tic feature, and enables the low-dimensional fusion acrosssemantic levels and deep supervision.|
|||Note that orthogo-nal to multi-level factorisation and fusion, multi-scale Re-ID has also been studied [28, 46] which focuses on fusingimage resolutions rather than semantic feature levels.|
|||Very few fusion architectures on specific tasks,e.g., edge detection [45], fuse features from all layers/levels.|
|||Using their fusion as a representation, we obtain state-of-the-art results on three large person Re-ID benchmarks.|
|||This sug-gests that our fusion architecture with deep supervision ismore effective than the handcrafted architectures with man-ual layer selection in [53, 29], which require extra effortbut may lead to suboptimal solutions.|
|||The FSM outputvectors Sn enable dynamic factorisation of an input imageinto distinctive latent attributes, and these are aggregatedover all blocks into a compact FS feature ( S) for fusion(Eq.|
|||MLFN-Fusion: MLFN using dynamic factor se-lection, but without fusion of the FS feature.|
|||Multilayer and mul-timodal fusion of deep neural networks for video classifica-tion.|
|||Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|385|cvpr18-Sketch-a-Classifier  Sketch-Based Photo Classifier Generation|(iii) Several extensions are presentedincluding a fine-grained variant and fusion of SBCS withthe standard ZSL paradigm.|
|||is the fusion function of the k different1Word-vector can equivalently replace image features, for descriptionsimplicity it will not be elaborated here.|
|||In this case the inputs of the regression network becomethe fusion of a prior coarse-grained photo SVM model wcgpand sketch image feature C(wcgp , ).|
|386|cvpr18-Semantic Visual Localization|For learning h, we generate training data using volumet-ric fusion.|
|||The incomplete in-put is completed using our encoder-decoder network h, while themulti-view fusion v is the ground-truth.|
|||We adapted the volumetric fusion approach by Hor-nung et al.|
|||[25] using (multi-view) stereo depth maps [24](KITTI) and sparse LIDAR measurements (NCLT) for ef-ficient large-scale semantic fusion.|
|||Note that this dataset is much more similar toKITTI as compared to NCLT, since it is an autonomousdriving dataset with a stereo camera pair that we use forvolumetric fusion.|
|||Note that our descriptor was trainedon a KITTI-like dataset using stereo for map fusion insteadof LIDAR used in NCLT.|
|387|Edgar_Margffoy-Tuay_Dynamic_Multimodal_Instance_ECCV_2018_paper|We review the state-of-the-art on the task of segmentation based on natural language expressions [3,4][5],highlighting the main contributions in the fusion of multimodal information, andthen compare them against our approach.|
|||9c is an interesting exampleof the networks confusion.|
|388|Fast Boosting Based Detection Using Scale Invariant Multimodal Multiresolution Filtered Features|Low-level fusion of color,texture and depth for robust road scene understanding.|
|389|Cheng_Wang_Mancs_A_Multi-task_ECCV_2018_paper|: Regularized diffusion process on bidirec-tional context for object retrieval.|
|||Zhao, H., Tian, M., Sun, S., Shao, J., Yan, J., Yi, S., Wang, X., Tang, X.: Spindlenet: Person re-identification with human body region guided feature decomposi-tion and fusion.|
|390|Visual Translation Embedding Network for Visual Relation Detection|We ablated VTransE into fourmethods in terms of using different features: 1) Classeme,2) Location, 3) Visual, and 4) All that uses classeme, lo-cations, visual features, and the fusion of the above with ascaling layer (cf.|
|391|cvpr18-In-Place Activated BatchNorm for Memory-Optimized Training of DNNs|In this way, the computation of z becomes slightlymore efficient than the one shown in Figure 2(b), for wesave the fusion operation.|
|||ConclusionsIn this work we have presented INPLACE-ABN, whichfusion of batchis a novel, computationally efficientnormalization and activation layers,targeting memory-optimization for modern deep neural networks during train-ing time.|
|392|Kemal_Oksuz_Localization_Recall_Precision_ECCV_2018_paper|2 Related WorkInformation Theoretic Performance Measures: Several performance mea-sures have been derived on the confusion matrix.|
|393|cvpr18-Dimensionality's Blessing  Clustering Images by Underlying Distribution|Query specific fusion for image retrieval.|
|394|Keizo_Kato_Compositional_Learning_of_ECCV_2018_paper|VP does not modelcontextuality between verbs and nouns, and thus can be considered as late fusion.|
|395|Mimicking Very Efficient Network for Object Detection|Tech-niques like multi-scale test [13], hierarchical feature fusion[3] and hole algorithms [4] are presented to improve de-tection performance for small objects but also bring largeincrease of time cost during the inference time.|
|396|cvpr18-Scalable and Effective Deep CCA via Soft Decorrelation|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|397|cvpr18-Recognize Actions by Disentangling Components of Dynamics|The comparison of results by fusion from differentbranches.|
|||Further in the fol-lowing discussion of representation fusion, we will showthat such representation provides complementary informa-tion and contributes to improved recognition accuracy.|
|||Convolu-tional two-stream network fusion for video action recogni-tion.|
|398|Zero-Shot Action Recognition With Error-Correcting Output Codes|Improving humanaction recognition using fusion of depth camera and inertialsensors.|
|399|cvpr18-Deformable Shape Completion With Graph Convolutional Autoencoders|For a quantitative analysis, we perform fusion on threepartial views from a static shape.|
|||The results show how reconstruc-tion accuracy changes according to the viewpoint, and con-sistently improves with latent space fusion.|
|||A qualitativeevaluation of the fusion problem is shown for the dynamicsetting in Figure 7.|
|||The latent space fusion of the completedshapes is shown in column 4.|
|||Dynamic fusion.|
|||Firstly, exploring a representation that disentanglesshape and pose would allow for more control in the comple-tion and likely improve dynamic fusion results.|
|||Kinectfusion: Real-time dense surface map-ping and tracking.|
|400|Deep Variation-Structured Reinforcement Learning for Visual Relationship and Attribute Detection|9600(d*history*phraseembeddingaction*spaceaction*space4096(dfusion2048(d1049(d*attribute*actions347(d*predicateactions1751(d*objectcategory* actionsVariation(structured*traversal*schemeFigure 3.|
|401|Yan-Pei_Cao_Learning_to_Reconstruct_ECCV_2018_paper|Al-though well studied, algorithms for volumetric fusion from multi-viewdepth scans are still prone to scanning noise and occlusions, making ithard to obtain high-fidelity 3D reconstructions.|
|||However, depth measurement acquired by con-sumer depth cameras contains a significant amount of noise, plus limited scan-ning angles lead to missing areas, making vanilla depth fusion suffer from blur-ring surface details and incomplete geometry.|
|||While [42]presents an OctNet-based [43] end-to-end deep learning framework for depthfusion, it refines the intermediate volumetric output globally, which makes itinfeasible for producing reconstruction results at higher resolutions even withmemory-efficient data structures.|
|||The truncation is performed to avoid surface interference,since in practice during scan fusion, the depth measurement is only locally reli-able due to surface occlusions.|
|||Similar to existingapproaches, we set up virtual cameras around the objects2 and render depthmaps, then simulate the volumetric fusion process [17] to generate ground-truthTSDFs.|
|||In essence, apart from shape completion, learning volumetric depth fusion isto seek a function g({D1, .|
|||Dai, A., Niener, M., Zollh ofer, M., Izadi, S., Theobalt, C.: Bundlefusion: Real-timeglobally consistent 3d reconstruction using on-the-fly surface reintegration.|
|||Keller, M., Lefloch, D., Lambers, M., Izadi, S., Weyrich, T., Kolb, A.: Real-time3d reconstruction in dynamic scenes using point-based fusion.|
|||Newcombe, R.A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A.J.,Kohi, P., Shotton, J., Hodges, S., Fitzgibbon, A.: Kinectfusion: Real-time densesurface mapping and tracking.|
|||Riegler, G., Ulusoy, A.O., Bischof, H., Geiger, A.: Octnetfusion: Learning depthfusion from data.|
|||: Elas-ticfusion: Dense slam without a pose graph.|
|||Robotics: Science and Systems (2015)A.J.,lightof Robotics Researchhttps://doi.org/10.1177/0278364916669237,Leutenegger,source35(14),https://doi.org/10.1177/0278364916669237Davison,slam andSalas-Moreno,Elasticfusion:estimation.|
|402|Deep Semantic Feature Matching|For solving the optimization problem we use the discretegraphical model library OpenGM [2] and use the fusion al-gorithm from Kappes et al.|
|||[26] for inference, where wechoose Loopy Belief Propagation [17] as proposal generatorand Lazy Flipping of search depth 2 [1] as fusion operator.|
|403|Tae_Hyun_Kim_Spatio-temporal_Transformer_Network_ECCV_2018_paper|: Optical flow via locally adaptive fusion of com-plementary data costs.|
|404|cvpr18-Visual Question Reasoning on General Dependency Tree|Secondly, for those child nodes withclausal predicate relation, our residual composition moduleintegrates the hidden representations weighted by attentionmaps of its child nodes using bilinear fusion.|
|||Mutan:Multimodal tucker fusion for visual question answering.|
|405|Gul_Varol_BodyNet_Volumetric_Inference_ECCV_2018_paper|Riegler, G., Ulusoy, A.O., Bischof, H., Geiger, A.: OctNetFusion: Learning depthfusion from data.|
|406|Deep View Morphing|Similarly to [4], we can consider two possibleways of such mechanisms: (i) early fusion by channel-wiseconcatenation of raw input images and (ii) late fusion bychannel-wise concatenation of CNN features of input im-ages.|
|||We chose to use the early fusion for the rectificationnetwork and late fusion for the encoder-decoder network(see Appendix A of the arXiv version of the paper [17] forFigure 2.|
|||The CNN features from the two encoders areconcatenated channel-wise by the late fusion and fed intothe correspondence decoder and visibility decoder.|
|407|Tz-Ying_Wu_Liquid_Pouring_Monitoring_ECCV_2018_paper|Given many success and failure demon-strations of liquid pouring, we train a hierarchical LSTM with late fusionfor monitoring.|
|||Given many success and failure demonstrations of liquid pouring, we train ahierarchical LSTM [8] with late fusion to incorporate rich sensories inputs with-out significantly increasing the model parameters as compared to early fusionmodels.|
|||Liquid Pouring Monitoring via Rich Sensory Inputs5Success / FailureSuccess / Failure(cid:2207)(cid:2778)M(cid:2186)(cid:2778)DTrajectory(cid:2779)G(cid:2190)(cid:2778)(cid:2168)(cid:2175)(cid:2176)(cid:2169)(cid:2190)(cid:2191)(cid:2187)(cid:2175)(cid:2778)(cid:2778)(cid:2207)(cid:2779)M(cid:2186)(cid:2779)DTrajectory(cid:2780)G(cid:2190)(cid:2779)(cid:2168)(cid:2175)(cid:2176)(cid:2169)(cid:2190)(cid:2191)(cid:2187)(cid:2175)(cid:2779)(cid:2779)Success / FailureM(cid:2207)(cid:2176)(cid:2778)(cid:2186)(cid:2176)(cid:2778)DSoftmaxMonitoringAuxiliary taskForecasting3D trajectory(Manipulation)Auxiliary taskInitial state classification(cid:2208)Multimodal data fusiontTrajectory(cid:2176)G(cid:2190)(cid:2176)(cid:2778)(cid:2168)(cid:2175)(cid:2176)(cid:2169)(cid:2190)(cid:2191)(cid:2187)(cid:2176)(cid:2778)(cid:2175)(cid:2176)(cid:2778)Fig.|
|||Our hierarchi-cal LSTM encoder LST Mhierconsistsof 3 LSTM cells (LST Mimg, LST Mpos,LST Mrot) at the first level and a LSTMfusion layer to fuse these hidden encodingsat the second level, fusing multimodal in-puts containing image feature Ft, hand po-sition feature At and hand rotation featureBt computed from IMU sensorstate classification (IOSC) and next-step hand 3D trajectory forecasting (TF).|
|||Vanilla RNN: Our fusion RNN without auxiliary tasks.|
|||RNN w/ IOSC: Our fusion RNN with an auxiliary task, initial object stateclassification (IOSC).|
|||RNN w/ TF: Our fusion RNN with an auxiliary task, trajectory forecasting(TF).|
|||: Our fusion RNN with two proposed auxiliary tasks, initialobject state classification and trajectory forecasting.|
|||Ours: Our fusion RNN with two proposed auxiliary tasks, initial object stateclassification and trajectory forecasting.|
|||The latter one is an early fusion method that data from differentmodalities is directly concatenated together and fed into the 2-layer LSTM.|
|||Theresults in Table 4 show that the hierarchical LSTM with late fusion outperformsthe naive 2-layer LSTM in all tasks and this may be due to the capability of thehierarchical LSTM to handle scale difference and imbalanced dimension amongmultimodal inputs.|
|408|Samuel_Albanie_Semi-convolutional_Operators_for_ECCV_2018_paper|Then template matching and proposalfusion techniques are applied.|
|409|cvpr18-Texture Mapping for 3D Reconstruction With RGB-D Sensor|Our method utilize the sparse-sequence fusion (SSF)method [28], instead of KinectFusion [18, 22], to recon-struct the initial 3D model and extract high confidence col-or frames.|
|||Kinectfusion:real-time 3d reconstruction and interaction us-ing a moving depth camera.|
|||Kinectfusion: Real-time dense surface map-ping and tracking.|
|||Real-time large-scale dense rgb-d slam with volumetric fusion.|
|||Elasticfusion: Dense slam without a pose graph.|
|||Bundlefusion: real-timeglobally consistent 3d reconstruction using on-the-fly surfacere-integration.|
|410|Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper|Park, S.J., Hong, K.S., Lee, S.: Rdfnet: Rgb-d multi-level residual feature fusionfor indoor semantic segmentation.|
|411|cvpr18-Mean-Variance Loss for Deep Age Estimation From a Face|The first-place method [1] in theCLAP2016 competition reported a lower error than ourmethod, but they used a score-level fusion of multiple CNNmodels.|
|412|cvpr18-Pose-Guided Photorealistic Face Rotation|Inspired by the recent success of u-net architecture inimage-to-image translation [15, 38], our GG consists of adown-sampling encoder and an up-sampling decoder withskip connections for multi-scale feature fusion.|
|413|Guanying_Chen_PS-FCN_A_Flexible_ECCV_2018_paper|PS-FCN is composedof three components, namely a shared-weight feature extractor for extracting fea-ture representations from the input images, a fusion layer for aggregating fea-tures from multiple input images, and a normal regression network for inferringthe normal map (see Fig.|
|||4.1 Max-pooling for multi-feature fusionCNNs have been successfully applied to dense regression problems like depthestimation [33] and surface normal estimation [34], where the number of inputimages is fixed and identical during training and testing.|
|||Given a variable number of inputs, a shared-weightfeature extractor can be used to extract features from each of the inputs (e.g.,siamese networks), but an additional fusion layer is required to aggregate suchfeatures into a representation with a fixed number of channels.|
|||A convolutionallayer is applicable for multi-feature fusion only when the number of inputs isfixed.|
|||2: A toy example for max-pooling and average-pooling mechanisms on multi-feature fusion.|
|||4.2 Network architecturePS-FCN is a multi-branch siamese network [37] consisting of three components,namely a shared-weight feature extractor, a fusion layer, and a normal regressionnetwork (see Fig.|
|||Hence, the input to our model has a dimension of q  6  h  w. Weseparately feed the image-light pairs to the shared-weight feature extractor toextract a feature map from each of the inputs, and apply a max-pooling operationin the fusion layer to aggregate these feature maps.|
|||Thanks to the max-pooling operation in the fusionlayer, it possesses the order-agnostic property.|
|||In particular,we first validated the effectiveness of max-pooling in multi-feature fusion by4 In our experiment, for each object in the Light Stage Data Gallery, we only used the133 pairs with the front side of the object under illumination.|
|||1compared the performance of average-pooling and max-pooling for multi-featurefusion.|
|||Similarly, experiments with IDs 2, 5 & 6 showed that fusion by convolutionallayers on the concatenated features was sub-optimal.|
|||5: Visualization of the learned feature map after fusion.|
|414|RON_ Reverse Connection With Objectness Prior Networks for Object Detection|[23], multi-layer fusion [3][16], context information [14][9]and more effective training strategy [26].|
|||Firstly, a deconvolutional layeris applied to the reverse fusion map (annotated as rf-map)n + 1, and a convolutional layer is grafted on backbone lay-er n to guarantee the inputs have the same dimension.|
|||The reverse fusion map 7 is the convolutional out-put (with 512 channels by 33 kernels) of the backbonelayer 7.|
|||In total, there are four reverse fusionmaps with different scales.|
|||The Figures show thecumulative fraction of detections that are correct (Cor) or false positive due to poor localization (Loc), confusion with similar categories(Sim), with others (Oth), or with background (BG).|
|415|Hyperspectral Image Super-Resolution via Non-Local Sparse Tensor Factorization|[13] firstly introduce matrix factorizationinto the spatial-spectral fusion by decomposing the HR-HSIon the learned dictionary with a sparse prior.|
|||By imposing priors on the distribution of the image inten-sities, Bayesian approaches [33, 3] apply MAP inference toregularize the fusion problem.|
|||Inspired by the above works, a novel non-local sparsetensor factorization (NLSTF) based HSI super-resolutionapproach is proposed for the fusion of a LR-HSI and aHR-MSI.|
|||Comparison of pan-sharpening algorithms:Outcome of the 2006 GRSS data-fusion contest.|
|||Hyperspectral and multispectral image fusion based on a s-parse representation.|
|||Coupled non-negativematrix factorization unmixing for hyperspectraland multi-spectral data fusion.|
|416|Hengcan_Shi_Key-Word-Aware_Network_for_ECCV_2018_paper|Li, Z., Gan, Y., Liang, X., Yu, Y., Cheng, H., Lin, L.: Lstm-cf: Unifying contextmodeling and fusion with lstms for rgb-d scene labeling.|
|417|Locality-Sensitive Deconvolution Networks With Gated Fusion for RGB-D Indoor Semantic Segmentation|The other is about RGB-D fusion.|
|||Re-cent state-of-the-art methods generally fuse RGB and depthnetworks with equal-weight score fusion, regardless of thevarying contributions of the two modalities on delineatingdifferent categories in different scenes.|
|||Towards RGB-D fusion, weintroduce a gated fusion layer to effectively combine thetwo LS-DeconvNets.|
|||Thanks to recent consumer depthfridge(a) Imprecise boundaries due to the large context when labeling each pixel (see the fridge)box(b) Misclassified objects due to the improper fusion of RGB and depth (see the box)Figure 1.|
|||Here a two-stream DeconvNet is used to represent RGB and depth,followed by score fusion with equal-weight sum just like the FCNmodel [19].|
|||3029Towards RGB-D fusion, a simple sum fusion with equalweights is adopted by [19] to combine the predictions ofRGB and depth FCN models.|
|||We adaptDeconvNet to RGB-D indoor scene segmentation with thesame fusion way of FCN, which achieves large performancegain compared to FCN in our experiments.|
|||The otherone is about RGB-D fusion.|
|||Instead of the simplescore fusion with equal weights for the two modalitieslike [19], we devise a gated fusion layer to automaticallylearn the varying contributions of each modality for clas-sifying different categories in different scenes.|
|||The gatedfusion layer is implemented by a series of standard layerswith learnable parameters, which makes our whole system(RGB LS-DeconvNet + depth LS-DeconvNet + Gated Fu-sion, termed as LSD-GF) can be trained end-to-end viaefficient back propagation algorithms.|
|||An effective fusion of the two complementarymodalities can improve the performance of semantic seg-mentation.|
|||Very recently,3030Convolution layerMax pooling layerUnpooling layerDeconvolution layerAverage pooling layerRGB417x417209x209105x10553x5353x53HHA53x5353x53105x105209x209417x417#1#2#3#1#2#353x53105x105209x209Dot productScore map GTSumRGBHHAAffinity matrixLocality sensitiveWeighted gate arrayGGSum53x53105x105209x2091  GSum209x209Dot productfloorwallchairdoortableothersFully convolutional networksDeconvolution networksGated fusionFigure 2.|
|||Due to the computational cost, only two-layer deconvolution networks are used; 3) the final gated fusion layer.|
|||recurrent networks [16] are explored for RGB-D fusion.|
|||Towards the popular convolutional neural networks (CNN),three levels of fusion are often used: Couprie et al.|
|||[5]concatenate the RGB and depth image as four-channel inputfor the CNN model (early fusion); Gupta et al.|
|||[11] leveragetwo CNN models to extract features from RGB and depthimages independently, and then concatenate them to learnthe final semantic classifier (middle fusion); Long et al.|
|||[19]also learn two independent CNN models but directly predictthe score map of each modality, followed by score fusionwith equal-weight sum (late fusion).|
|||find the late fusion can be moreeffective to benefit from the complementarities of the twomodalities, compared to other fusion levels.|
|||This paperadopts the late fusion version, but embeds a gate fusionlayer to further adapt our model to the varying contributionsof the two modalities for recognition of different categoriesin different scenes.|
|||As shown in the experiments,theproposed fusion way can achieve performance gains forthose confused categories.|
|||LSD-GF is composed of threeparts:the frontend fully convolutional networks (FCN),the intermediate locality-sensitive deconvolution networks(LS-DeconvNet), and the final gated fusion layer.|
|||Finally, a gatedfusion layer is introduced to fuse the RGB and depth cueseffectively for accurate scene semantic segmentation.|
|||Towards the gated fusionlayer, we concatenate the prediction maps of RGB anddepth to learn a weighted gate array, which is able toweigh the contributions of each modality for accurate objectrecognition in the scene.|
|||Gated FusionThe gated fusion layer is proposed to effectively combineRGB and depth for semantic segmentation.|
|||Afterconcatenation, we obtain a fused probability map Pfusion R2chw.|
|||The output of the convolution layer is acoefficient matrix G  Rchw with the valueGk,i,j =2c(cid:2)k=1Pfusionk ,i,j  Wk,k,i,jk  [1, c], i  [1, h], j  [1, w].|
|||Finally, we generatethe gated fusion probability map as Pfusion =  Prgb +  Pdepth.|
|||(5)We predict the label map by  Pfusion and leverage the groundtruth label map to optimize the whole network via stochasticgradient descent.|
|||In the first stage,we train two independent locality-sensitive DeconvNets onRGB and depth for semantic segmentation without thegated fusion layer.|
|||In the secondstage, we add the gated fusion layer, and then finetune thewhole networks on the synchronized RGB and depth data.|
|||Note that theonly differences between DeconvNet and the proposed ap-proach are that we replace the conventional deconvolutionnetworks with simple sum fusion by the locality-sensitivedeconvolution networks with gated fusion.|
|||To further verify the particular advantagesof our locality-sensitive deconvolution networks with gatedfusion, we compare the results of ours to that of DeconvNet.|
|||Weowe the improvements to two factors: 1) the local visualand geometrical cues from raw data embedded into thedeconvolution networks can effectively alleviate the impre-cise boundary representation from the frontend FCN modelwith large context; 2) the gated fusion layer can effectivelycombine the two complementary modalities for accurateobject recognition.|
|||To discover the importance of the proposed locality-sensitive DeconvNet and the gated fusion of LSD-GF, weconduct an ablation study via removing or replacing eachcomponent independently or both together for semanticsegmentation on the NYU-Depth v2 dataset.|
|||For eachcomparison pair, the only difference is with and withoutlocality-sensitive module; 2) Gated fusion is superior tothe sum fusion, as well as some other popular equal-weightscore fusion like pixelwise production and Dempster-Shafer(DS) [26] (comparing e  h and i  l).|
|||We owe the im-provement to the accurate recognition of some hard objectsin the scene by gated fusion, such as box on the sofa andchair in the weak lights.|
|||These objects need to effectivelyweigh the contributions of RGB an depth for recognition;3) Cascading the locality-sensitive deconvolution networksand the gated fusion can achieve the best result, i.e., 45.9%mean IOU.|
|||Specifically, rows (1)(3) of thefigure show some examples to witness the effectivenessof the proposed gated fusion, e.g., it helps to correctlyrecognize the box on the sofa (emphasize appearance), thefaraway fridge against the cabinet (emphasize shape), and3035RGBHHAGTLSD-GFw/o gated fusion  w/o locality-sensitivew/o both(1)(2)(3)(4)(5)(6)(7)(8)floorwallbedcabinettablechairsofadoorbookshelf windowpictureblindscurtainshelvespillow floormatceilingclothesfridgepapershowertowelboardnightstandsinktoiletlampbagcounterdeskdressermirorbookstvboxpersonbathtubopropsofurnostuctbackgroundFigure 4.|
|||For the scene image in each row, we show:(column 1) the RGB image; (column 2) the HHA image; (column 3) the ground truth of semantic segmentation; (column 4) the result ofour LSD-GF approach, i.e., l in Table 3; (column 5) the result of LSD-GF whose gated fusion is replaced by sum fusion, i.e., i in Table 3;(column 6) the result of LSD-GF whose locality-sensitive module is removed, i.e., h in Table 3; (column 7) the result of LSD-GF whoselocality-sensitive is removed and the gated fusion is replaced by sum fusion, i.e., e in Table 3.|
|||LSD-GF is composed of two main components: 1) the locality-sensitive deconvolution networks, which are designed forsimultaneously upsamping the coarse fully convolutionalmaps and refining object boundaries; 2) gated fusion, whichcan adapt to the varying contributions of RGB and depth forbetter fusion of the two modalities for object recognition.|
|||Rgb-d scene labeling with long short-term memorized fusionmodel.|
|||Sensortheory [for context-awareIn Instrumentation and Measurement Technologyfusion using dempster-shaferhci].|
|418|cvpr18-Fast End-to-End Trainable Guided Filter|Locality-sensitive deconvolution networks with gatedfusion for rgb-d indoor semantic segmentation.|
|419|cvpr18-Learning Generative ConvNets via Multi-Grid Modeling and Sampling|Grade: Gibbs reaction and dif-fusion equitions.|
|420|cvpr18-High-Speed Tracking With Multi-Kernel Correlation Filters|Multi-cue visual tracking usingrobust feature-level fusion based on joint sparse represen-tation.|
|||A statistical framework for genomic data fusion.|
|||Multiplesource data fusion via sparse representation for robust visualtracking.|
|421|Sachin_Mehta_ESPNet_Efficient_Spatial_ECCV_2018_paper|The large effective receptive field of the ESP module introduces griddingartifacts, which are removed using hierarchical feature fusion (HFF).|
|||(b) Visualization of feature mapsof ESP modules with and without hierarchical feature fusion (HFF).|
|||Hierarchical feature fusion (HFF) for de-gridding: While concatenating the outputsof dilated convolutions give the ESP module a large effective receptive field, it intro-duces unwanted checkerboard or gridding artifacts, as shown in Fig.|
|||Here, ERF represents effective receptive field, denotes that strided ESP was used for down-sampling,  indicates that the input reinforcementmethod was replaced with input-aware fusion method [36], and  denotes the values are in mil-lion.|
|||The closest work to our input reinforcement method is the input-aware fusion methodof [36], which learns representations on the down-sampled input image and additivelycombines them with the convolutional unit.|
|||When the proposed input reinforcementmethod was replaced with the input-aware fusion in [36], no improvement in accuracywas observed, but the number of network parameters increased by about 10%.|
|422|cvpr18-DecideNet  Counting Varying Density Crowds Through Attention Guided Detection and Density Estimation|Specifically, the ensemble and fusion strategy is employedby the M-CNN [42], Switching-CNN [28], CP-CNN [31] in Ta-ble 2.|
|||Further, late fusionby averaging two classes of density maps (RegNet+DetNet (LateFusion)) exhibits improvements than RegNet only and Det-Net only on that SHB dataset.|
|||This indicates that direct late fusion is not robust enoughto obtain better results across all kinds of datasets.|
|||Compared to late fusion, it almost decreasesthe MAE by half on two datasets, revealing the power of the atten-tion mechanism.|
|||Directly applying the late fusion (the pur-ple curves) helps to a certain extent, while its predicted counts arenot stable along all images.|
|423|Yeong_Jun_Koh_Sequential_Clique_Optimization_ECCV_2018_paper|However, the fusion processes oftencause temporal inconsistency and may fail to segment out primary objects properlywhen either spatial or temporal results are inaccurate.|
|||Chen, C., Li, S., Wang, Y., Qin, H., Hao, A.: Video saliency detection via spatial-temporalfusion and low-rank coherency diffusion.|
|||Yang, J., Zhao, G., Yuan, J., Shen, X., Lin, Z., Price, B., Brandt, J.: Discovering primaryobjects in videos by saliency fusion and iterative appearance estimation.|
|424|Xuelin_Qian_Pose-Normalized_Image_Generation_ECCV_2018_paper|Once ResNet-50-A and ResNet-50-B are trained, during test-ing, for each gallery image, we feed it into ResNet-50-A to obtain one featurevector; as for synthesize eight images of the canonical poses, in consideration ofconfidence, we feed them into ResNet-50-B to obtain 8 pose-free features andone extra FC layer for the fusion of original feature and each pose feature.|
|||Zhao, H., Tian, M., Sun, S., Shao, J., Yan, J., Yi, S., Wang, X., Tang, X.: Spindlenet: Person re-identification with human body region guided feature decomposi-tion and fusion.|
|425|Heewon_Kim_Task-Aware_Image_Downscaling_ECCV_2018_paper|Zhang, L., Wu, X.: An edge-guided image interpolation algorithm via directionalfiltering and data fusion.|
|426|Modeling Sub-Event Dynamics in First-Person Action Recognition|Convolutionaltwo-stream network fusion for video action recognition.|
|427|cvpr18-Erase or Fill  Deep Joint Recurrent Rain Removal and Reconstruction in Videos|Thus, we detec-t the degradation type of rain frames explicitly, providinguseful side information for successive spatial and temporalredundancy fusion.|
|428|cvpr18-Fusing Crowd Density Maps and Visual Object Trackers for People Tracking in Crowd Scenes|To train the fusion CNN, we proposea two-stage strategy to gradually optimize the parameter-s. The first stage is to train a preliminary model in batchmode with image patches selected around the targets, andthe second stage is to fine-tune the preliminary model us-ing the real frame-by-frame tracking process.|
|||Our densityfusion framework can significantly improves people track-ing in crowd scenes, and can also be combined with othertrackers to improve the tracking performance.|
|||An example of people tracking in crowd scene using d-ifferent trackers: KCF [14], S-KCF (ours), long-term correlationtracker (LCT) [20], density-aware [25], and our proposed fusiontracker.|
|||Our density fusion framework combines the S-KCF response mapwith the crowd density map, and effectively suppresses the irrele-vant responses and detects the target accurately (top-right).|
|||1 presents an example for people tracking using dif-ferent trackers: KCF [14], S-KCF (ours), long-term corre-lation tracker (LCT) [20], density-aware [25], and our pro-posed fusion tracker.|
|||We propose a density fusion framework, based on a C-NN, that combines the S-KCF tracker response and theestimated crowd density map to improve people track-ing in crowd scenes.|
|||Our density fusion frameworkcan also be used with other appearance-based trackersto improve their accuracy in crowded scenes.|
|||To train the fusion CNN, we propose a two-stage train-ing strategy to gradually optimize the parameters in anend-to-end fashion.|
|||MethodologyOur density fusion framework has three main parts: visu-al tracking model (S-KCF), crowd density estimation, andfusion neural network.|
|||The response map, image patch and the correspond-ing density map are fused together using a fusion CNN,yielding a final fused response map.|
|||The proposed density fusion framework.|
|||The fusion CNN takes in the image patch, S-KCF response map, and crowd density map and producesa refined fused response map, whose maximum value indicates the location of the target.|
|||Fusion CNNThe fusion CNN combines the tracker response map, thecrowd density map, and the image patch to produce a re-fined (fused) response map, where the maximum value in-dicates the target position.|
|||Thestructure of our fusion CNN is shown in Fig.|
|||Our fusion network has 3convolutional layers (Conv1-Conv3).|
|||The structure of the fusion CNN.|
|||The fusion networkhas three input channels, and can effectively fuse the appearanceinformation (image patch), with the crowd density map, and thevisual tracker response map.|
|||cording to the fusion response maps (see Fig.|
|||Becauseof the interplay between the output of one frame with theinput in the next frame, we adopt a two-stage training pro-cedure to gradually optimize the fusion CNN.|
|||In the first stage, we train the fusion CNN in batchmode, where each frame is treated independently.|
|||In the second stage, we run the fusion CNN, and use thefusion response map to predict target position.|
|||This is iterated over frames, and the samplesused for fine-tune training the fusion CNN.|
|||Two(cid:173)stage Training Strategy for Fusion CNNTraining the fusion CNN requires the response mapsgenerated by KCF, but the KCF model is also updated ac-4.1.|
|||We randomly select 80% of the unique people for train-ing the fusion CNN, and the remaining 20% are held outfor testing.|
|||For the PET-S2009 dataset, the density map CNN is only trained on thePETS2009 data, while the fusion CNN is fine-tuned fromthe network learned from the UCSD dataset.|
|||To show the general effectiveness of using density map fu-sion, we train a separate fusion CNN for each tracker, usingthe two-stage training procedure (denoted as FusionCNN-v2), and evaluate the tracking performance of the fusedresponse map.|
|||To show the effectiveness of two-stage train-ing, we compare with a fusion CNN using only the firststage of training, denoted as FusionCNN-v1.|
|||We imple-ment the fusion CNN using the Caffe [16] framework.|
|||When combined with crowd density us-ing our density fusion framework (FusionCNN-v2), all thetrackers can be improved significantly (e.g., KCF improvesP@10 from 0.4235 to 0.5501, while S-KCF improves from0.4356 to 0.5999).|
|||Thedensity-aware method [25] for fusion does not perform aswell as our fusion method.|
|||We also compare our CNN fusion method with theTracking results P@10 on the PETS2009 dataset are5358Table 2.|
|||OnUCSD, the running times of fusion CNN with KCF, S-KCF,LCT and DSST are 19, 18, 10, 23 fps.|
|||The fusion modelcan improve visual trackers more than incorporating colorand intensity cues (e.g., for UCSD, DSST tracker improvesfrom 0.4058 to 0.4364 using HOG+A features, while it im-proves from 0.4085 to 0.5300 when fused with our fusionCNN).|
|||The last row of the table (Fusion / HOG+A) showsthe fusion results using trackers with HOG+A features.|
|||Cross(cid:173)crowd and Cross(cid:173)scene GeneralizationThe experiment in Section 4.3 trained a separate fusionmodel for each crowd level in PETS2009, since they haveuniquely different properties.|
|||Here, we report the track-ing results when using uniform fusion model trained on allcrowd levels in Table 5.|
|||Overall, the uniform fusion modelperforms a little worse than the separate model (the aver-age P@10 of 0.3245 vs 0.3326, and the average IoU = 0.5of 0.3381 vs 0.3559).|
|||However, the uniform fusion modelcan still significantly improve S-KCF tracker (the averageP@10 improves from 0.2447 to 0.3245, and the average IoU= 0.5 improves from 0.2807 to 0.3381).|
|||We also evaluate training and testing across crowd-levels, where the fusion model is trained only on either L1,L2 or L3.|
|||Finally, we evaluate the cross-scene generalization abili-ty of the fusion model.|
|||Tracking results on UCSD dataset for architecture varia-tions of our fusion CNN.|
|||Comparison of fusion CNN architecturesIn this subsection we compare different variations of thefusion CNN architecture.|
|||Wefuse the appearance-based tracker and crowd density maptogether with a three-layer fusion CNN to produce a refinedresponse map.|
|||Experimental results show that our fusionframework can improve the people tracking performance ofappearance-based trackers in crowd scenes.|
|429|Joint Sequence Learning and Cross-Modality Convolution for 3D Biomedical Segmentation|Our model jointly optimizesthe slice sequence learning and multi-modality fusion in anend-to-end manner.|
||| We leverage convolution LSTM to model the spa-tial and sequential correlations between slices, andjointly optimize the multi-modal fusion and convolu-tion LSTM in an end-to-end manner.|
|||Cai et[4] combine MRI images with diffusion tensor imag-al.|
|||Theoverall structure of cross-modality convolution (CMC) andmulti-resolution fusion are shown in Figure 3.|
|||System overview of our multi-resolution fusion strat-egy.|
|||Our encoder-decoder modelwith convolutional LSTM and multi-resolution fusion achieve thebest results.|
|||Experimental results show thatthe proposed cross-modality convolution can effectively aggregate the informa-tion between modalities and seamlessly work with multi-resolution fusion.|
|430|cvpr18-Unifying Identification and Context Learning for Person Recognition|We propose a Region Attention Network to get instance-dependent weights for visual context fusion anddevelop a unified formulation that join social context learning, including event-person relations and person-person relations, with personrecognition.|
|||To combine these features adaptively, we devise a RegionAttention Network (RANet) as shown in Figure 2 to com-pute the fusion weights.|
|||While our model uses only 4 CNNsand a fusion module whose computing cost is negligible.|
|431|cvpr18-Bidirectional Retrieval Made Simple|Mu-tan: Multimodal tucker fusion for visual question answering.|
|||Learning arecurrent residual fusion network for multimodal matching.|
|432|TGIF-QA_ Toward Spatio-Temporal Reasoning in Visual Question Answering|VQA-MCB, onthe other hand, uses multimodal compact bilinear pooling tohandle visual-textual fusion and spatial attention [10].|
|433|A Multi-View Stereo Benchmark With High-Resolution Images and Multi-Camera Videos|Massively parallelmultiview stereopsis by surface normal diffusion.|
|||Stereo depth map fusion for robot navigation.|
|434|Deep Crisp Boundaries|To get a better fusion of multi-layerfeatures, we introduce the backward-refining pathway withrefinement modules, similar to [28].|
|||There are two core compo-nents in this module, namely fusion and up-sampling.|
|||Fusion: A straightforward strategy of fusion is todirectly concatenate two feature maps.|
|||u + kUp-sampling: After fusion, our refinement modulewill also expand the resolution of feature maps.|
|||Moreover, we tested the multi-scale fusionstrategy for the evaluation.|
|435|Lei_Zhu_Bi-directional_Feature_Pyramid_ECCV_2018_paper|3: (a) The schematic illustration of the attention module in RAR; (b) Thedetails of attentional fusion for the final shadow detection map; see Sec.|
|||After that, we computethe weight (attention) map (cid:16)A(cid:0)Cat(F ui , Fj)(cid:1)(cid:17) by using a sigmoid function onthe feature maps (denoted as H) learned from three residual blocks:a(p, q, c) =1where a(p, q, c) is the weight at the spatial position (p, q) of the c-th channel ofthe learned weight map (cid:16)A(cid:0)Cat(F uat the spatial position (p, q) of the c-th channel of H.1 + exp(cid:0)  H(p, q, c)(cid:1) ,i , Fj)(cid:1)(cid:17), while H(p, q, c) is the feature value(2)conv,sigmoid+(b) Attentional fusion113311sigmoid11(a) Attention module in RAR113311113311Bidirectional FPN with Recurrent Attention Residual Modules73.2 Our NetworkNote that the original FPN [9] iteratively merges features in a top-down pathwayuntil reaching the last layer with the largest resolution.|
|||Then, we take the output of the attentionalfusion module (see Fig.|
|436|Chao-Yuan_Wu_Video_Compression_through_ECCV_2018_paper|Specifically, we perform the fusion before each Conv-LSTMlayer by concatenating the corresponding U-net features of the same spatialresolution.|
|437|Rameswar_Panda_Contemplating_Visual_Emotions_ECCV_2018_paper|The distinct diagonal in confusion matrix(Figure 2.a) shows that these datasets possesses an unique signature leadingto the presence of bias.|
|||Contemplating Visual Emotions5Confusion MatrixDeep SentimentDeep EmotionEmotion-6n ten ti mp  S eeeDno ti omp  EeeDn - 6o ti omE(a)(b)Fig.|
|||(a) Confusion matrix, (b) From top to bottom, depicted are examples of high confidentcorrect predictions from Deep Sentiment, Deep Emotion and Emotion-6 datasets respectively.|
|||Eleftheriadis, S., Rudovic, O., Pantic, M.: Joint facial action unit detection andfeature fusion: A multi-conditional learning approach.|
|439|cvpr18-Spline Error Weighting for Robust Visual-Inertial Fusion|Wedemonstrate the effectiveness of the prediction in a syntheticexperiment, and apply it to visual-inertial fusion on rollingshutter cameras.|
|||Visual-inertial fusion using splines has traditionally bal-anced the sensor modalities using inverse noise covarianceFigure 1.|
|||SEW makes visual-inertial fusion robust on realsequences, acquired with rolling shutter cameras.|
|||Related workVisual-inertial fusion on rolling shutter cameras has clas-sically been done using Extended Kalman-filters (EKF).|
|||[8] study visual-inertial fusion withpreintegration of IMU measurements between keyframes,with a global shutter camera model.|
|||Visual-inertial fusionWe will use the residual error prediction introduced insection 2 to balance visual-inertial fusion on rolling shut-ter cameras.|
|||Such learnedcharacteristic spectra could be useful as a priori informationwhen adapting spline error weighting to do on-line visual-inertial fusion.|
|||We plan to release our spline error weighting frameworkfor visual-inertial fusion under an open source license.|
|||Spline fusion:A continuous-time representation for visual-inertial fusionwith application to rolling shutter cameras.|
|||A spline-based trajectory representation for sensor fusion and rollingshutter cameras.|
|440|cvpr18-Focus Manipulation Detection via Photometric Histogram Analysis|In this paper, we presenta photo forensic method to distinguish images having a nat-urally shallow DoF from manipulated ones, by integratinga number of cues under a fusion of two deep convolutionnetworks with small receptive fields for histogram classifi-cation.|
|||Starting from iterative filtering [26], algorithmsin this vein have evolved to more sophisticated solution-s including pre-blurring [25, 14, 15], anisotropic diffusion[10, 23], separable Gaussian filters [25, 30], etc.|
|||The FMIHNetis a fusion of two relatively deep sub-networks: FMIHNet1with 20 CONV layers for VAR and CFA features, and FMI-HNet2 with 30 CONV layers for GRAD, ADQ and NOIfeatures.|
|||Real-time,accurate depth of field using anisotropic diffusion and pro-grammable graphics cards.|
|||Interactive depth of fieldusing simulated diffusion on a gpu.|
|||An algorithm for renderinggeneralized depth of field effects based on simulated heat d-iffusion.|
|441|Mohammed_Fathy_Hierarchical_Metric_Learning_ECCV_2018_paper| We experimentally validate our ideas by comparing against state-of-the-artgeometric matching approaches and feature fusion baselines, as well as performan ablative analysis of our proposed solution.|
|||We show gains in correspondenceestimation by using our approach over prior feature fusion methods, e.g.|
|||Our proposed hierarchical matching is implemented on CUDA and run on aP6000 GPU, requiring an average of 8.41 seconds to densely extract features andcompute correspondences for a pair of input images of size 1242  376.s(ps), forming a correspondence (ps, qs (ps) is closest to I4 ExperimentsIn this section, we first benchmark our proposed method for 2D correspondenceestimation against standard metric learning and matching approaches, featurefusion, as well as state-of-the-art learned and hand-crafted methods for extractingcorrespondences.|
|||The 1x1 maxpooling layer after conv1 in thehypercolumn-fusion baseline (b) isadded to down sample the conv1feature map for valid concatenationwith other feature maps.|
|||KCP( ycaruccA9080706050403020100 1 100 conv1netconv2netconv3netconv4netconv5nethypercolumnfusiontopdownfusionHiLM (conv2+conv3)HiLM (conv2+conv4)HiLM (conv2+conv5)HiLM (conv2+conv5) Sintel2345678910Threshold (pixel))KCP( ycaruccA9080706050403020 10conv1netconv2netconv3netconv4netconv5nethypercolumnfusiontopdownfusionHiLM (conv2+conv3)HiLM (conv2+conv4)HiLM (conv2+conv5)HiLM (conv2+conv5) Sintel2030405060708090100Threshold (pixel)(a) Accuracy over small thresholds(b) Accuracy over large thresholdsFig.|
|||One is hypercolumn-fusion  Figure 3 (b), where feature sets from all layers(first through fifth) are concatenated for every interest point and a set of 1x12 LIFT [61] is not designed for dense matching and hence not included in our experi-ments.|
|||Another approach we consider is topdown-fusion, where refinementmodules similar to [43] are used to refine the top-level conv5 features graduallydown the network by combining with lower-level features till conv2 (please seesupplementary material for details).|
|||hypercolumn-fusion with 69.41%versus conv5-net with 61.78% @ 5 pixels), they do not perform on par with thesimple conv2 -based features (e.g.|
|||As expected the Sintel model is subpar compared tothe same model trained on KITTI (72.37% vs. 79.11% @ 5 pixels), however itoutperforms both hypercolumn-fusion (69.41%) and topdown-fusion (63.14%)trained on KITTI, across all PCK thresholds.|
|||Our evaluation on the task of explicit keypoint matching outperformshand-crafted descriptors, a state-of-the-art descriptor learning approach [16], aswell as various ablative baselines including hypercolumn-fusion and topdown-fusion.|
|442|Junjie_Zhang_Goal-Oriented_Visual_Question_ECCV_2018_paper|Informativeness Reward When we human ask questions (especially in aguess what game), we expect an answer that can help us to eliminate the con-fusion and distinguish the candidate objects.|
|||For the informativeness reward, we evaluate theinformativeness of each generated question by asking human subjects to rate iton a scale of 1 to 5, if this question is useful for guessing the target object fromthe human perspective, i.e., it can eliminate the confusion and distinguish thecandidate objects for the human, the higher score will be given by the subject.|
|443|Can Walking and Measuring Along Chord Bunches Better Describe Leaf Shapes_|Online reranking via ordinal  informative  concepts  for  context  fusion  in  concept detection  and  video  search.|
|444|ActionVLAD_ Learning Spatio-Temporal Aggregation for Action Classification|Using latefusion or averaging is not the optimal solution since it re-quires frames belonging to same sub-action to be assignedto multiple classes.|
|||MethodRGBFlowMethodRGBFlow2-Streamconv4 3conv5 3fc747.145.051.243.355.253.558.453.12-StreamAvgMaxActionVLAD47.141.641.551.255.253.454.658.4Table 3: Comparison of (a) Different fusion techniques describedin Sec.|
|||We observe that late fusion performs best.|
|||In contrast, concatfusion limits the modelling power of the model as it usesthe same number of cells to capture a larger portion of thefeature space.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|||Bag of visualwords and fusion methods for action recognition: Compre-hensive study and good practice.|
|445|cvpr18-RayNet  Learning Volumetric 3D Reconstruction With Ray Potentials|This allows the CNN to specialize its represen-tation to the joint task while explicitly considering the 3Dfusion process.|
|||Oct-NetFusion: Learning depth fusion from data.|
|447|cvpr18-Learning Pixel-Level Semantic Affinity With Image-Level Supervision for Weakly Supervised Semantic Segmentation|This semantic dif-fusion revises CAMs significantly so that fine object shapesare recovered.|
|||Locality-sensitive deconvolution networks with gated fusion for rgb-dindoor semantic segmentation.|
|448|cvpr18-Real-Time Monocular Depth Estimation Using Synthetic Data With Domain Adaptation via Image Style Transfer|Conventional depth estimation methods have relied on nu-merous strategies such as stereo correspondence [67, 28],structure from motion [14, 9], depth from shading and lightdiffusion [73, 82, 1] and alike.|
|||Structure guidedfusion for depth map inpainting.|
|||Deep domain confusion: Maximizing for domain invariance.|
|449|Mingze_Xu_Joint_Person_Segmentation_ECCV_2018_paper|To jointly consider both spatial and temporal information, weuse early fusion to concatenate features at levels pool3, pool4, and pool5(middle of Fig.|
|450|Deep Multimodal Representation Learning From Temporal Data|Deep Multimodal Representation Learning from Temporal DataXitong Yang1, Palghat Ramesh2, Radha Chitta3, Sriganesh Madhvanath3,Edgar A. Bernal4 and Jiebo Luo51University of Maryland, College Park 2PARC 3Conduent Labs US4United Technologies Research Center5University of Rochester1xyang35@cs.umd.edu, 2Palghat.Ramesh@parc.com, 3{Radha.Chitta,Sriganesh.Madhvanath}@conduent.com, 4bernalea@utrc.utc.com, 5jluo@cs.rochester.eduAbstractIn recent years, Deep Learning has been successfullyapplied to multimodal learning problems, with the aim oflearning useful joint representations in data fusion applica-tions.|
|||When the available modalities consist of time seriesdata such as video, audio and sensor signals, it becomesimperative to consider their temporal structure during thefusion process.|
|||In this paper, we propose the CorrelationalRecurrent Neural Network (CorrRNN), a novel temporalfusion model for fusing multiple input modalities that areinherently temporal in nature.|
|||Related workIn this section, we briefly review some related work ondeep-learning-based multimodal learning and temporal datafusion.|
|||Generally speaking, and from the standpoint of dy-namicity, fusion frameworks can be classified based on thetype of data they support (e.g., temporal vs. non-temporaldata) and the type of model used to fuse the data (e.g., tem-poral vs. non-temporal model) as illustrated in Fig.|
|||Multimodal Deep LearningWithin the context of data fusion applications, deeplearning methods have been shown to be able to bridge thegap between different modalities and produce useful jointrepresentations [13, 21].|
|||Generally speaking, two mainapproaches have been used for deep-learning-based mul-timodal fusion.|
|||The multimodal inputs are first mapped to sepa-rate hidden layers before being fed to a common layer calledthe fusion layer.|
|||Specifically, the activations of the fusion layer inthe encoder at the last time step is output as the sequencefeature representation.|
|||ConclusionsIn this paper, we have proposed CorrRNN, a new modelfor multimodal fusion of temporal inputs such as audio,video and sensor data.|
|||We have demonstrated that the CorrRNNmodel achieves state-of-the-art accuracy in a variety of tem-poral fusion applications.|
|||Multimodal fusion using dynamic hybrid mod-els.|
|||Audiovi-sual fusion: Challenges and new approaches.|
|451|cvpr18-Robust Depth Estimation From Auto Bracketed Images|These photographic techniques for gathering morelight have recently attracted interest from the field of com-(a) Input: Exposure bracketed images(b) Camera pose & 3D points(c) Depth map result(d) Exposure fusion(e) Synthetic refocusingFigure 1: Given exposure bracketed images (a), we estimatecamera pose (b) and depth map (c).|
|||We compare exposure fusion results frominput images (L) and aligned images using our depth (R).|
|||The bracketed images are necessary to trulyachieve HDR or exposure fusion.|
|||We convert2948(a) Averaged images(b) Our depths(c) Reference images(d) Denoising(e) Exposure fusionFigure 4: Averaged image of input exposure bracketed images, our depths and example of photographic applications (denois-ing, exposure fusion) using aligned images.|
|||Thealigned images can be used for image quality enhancementapplications such as noise reduction and exposure fusionas shown in Fig.|
|||4dand exposure fusion algorithm [23] in Fig.|
|||We also found that our accurate depth can be addi-tionally useful for exposure fusion and depth-aware photo-graphic editing applications, such as digital refocusing andimage stylization in Fig.|
|||Exposure fusion assembles themulti-exposure sequence into a high quality image using aweighted blending of the input images [23].|
|||On the otherhand, exposure fusion with real bracketing can cover all ofthe areas of the input image, as shown in Fig.|
|||2952(a) Reference images(b) Exposure fusion(c) Photographic editing(d) Our depthsFigure 8: Depth-aware photographic editing applications to Synthetic refocusing (top), Image stylization (bottom) and ourdepths captured by Canon 1D Mark III(a) Microsoft selfie (iPhone)(b) Ours (iPhone)(c) Google camera (Nexus)(d) Ours (Nexus)Figure 9: Qualitative comparison with the state-of-the-art methods [19, 12].|
|||(b) Our noise-free exposure fusion results.|
|||(d) Ournoise-free exposure fusion results.|
|||Exposure fusion: Asimple and practical alternative to high dynamic range pho-tography.|
|452|Sameh_Khamis_StereoNet_Guided_Hierarchical_ECCV_2018_paper|Dou, M., Davidson, P., Fanello, S.R., Khamis, S., Kowdle, A., Rhemann, C.,Tankovich, V., Izadi, S.: Motion2fusion: Real-time volumetric performance cap-ture.|
|||Izadi, S., Kim, D., Hilliges, O., Molyneaux, D., Newcombe, R., Kohli, P., Shotton,J., Hodges, S., Freeman, D., Davison, A., Fitzgibbon, A.: Kinectfusion: Real-time3d reconstruction and interaction using a moving depth camera.|
|||Pradeep, V., Rhemann, C., Izadi, S., Zach, C., Bleyer, M., Bathiche, S.: Mono-fusion: Real-time 3d reconstruction of small scenes with a single web camera.|
|453|Bryan_Plummer_Conditional_Image-Text_Embedding_ECCV_2018_paper|The outputs of these layers, in the form of amatrix of size M  K, are fed into the embedding fusion layer, together witha K-dimensional concept weight vector U , which can be produced by severalmethods, as discussed in Section 2.3.|
|||The fusion layer simply performs a matrix-vector product, i.e., F = CU .|
|454|Scalable Surface Reconstruction From Point Clouds With Extreme Scale and Density Diversity|This brings us to the main con-tribution of our work, the fusion of these hypotheses.|
|||the irregularspace division via Delaunay tetrahedralization) also makesthe fusion of the surface hypotheses a non-trivial problem.|
|||As pre-processingstep, we apply scale sensitive point fusion.|
|||This step can be seen as the non-volumetricequivalent to the fusion of points on a fixed voxel grid.|
|||Massively par-Inallel multiview stereopsis by surface normal diffusion.|
|455|Zhenyu_Wu_Towards_Privacy-Preserving_Visual_ECCV_2018_paper|Semi-coupledtwo-stream fusion convnets for action recognition at extremely low resolutions.|
|456|Jingwei_Ji_End-to-End_Joint_Semantic_ECCV_2018_paper|[3] proposed the I3D architecture, which considers atwo-stream network configuration [22, 12] and performs late fusion of the outputsof individual networks trained on RGB and optical flow input, trained separately.|
|||As suggestedby [11] for 2D architecture, and corroborated by our own experiments, late fusionin standard action recognition approaches [22, 3, 7] does not work well whenconsidering the joint task of actor/action recognition and semantic segmentation.|
|457|cvpr18-Fully Convolutional Adaptation Networks for Semantic Segmentation|Deep Domain Confusion (DDC) [32]applies MMD as well as the regular classification loss on thesource to learn representations that are both discriminativeand domain invariant.|
|||We refer to this fusion version as AAN in the following e-valuations unless otherwise stated.|
|||Domain Confusion [30] (DC) aligns domains via domainconfusion loss, which is optimized to learn a uniform distri-bution across different domains.|
|||Deep domain confusion: Maximizing for domain invariance.|
|458|Deep Temporal Linear Encoding Networks|They exploit fusion techniques liketrajectory-constrained pooling [37], 3D pooling [8], and12329consensus pooling [38].|
|||The fusion methods of spatial andmotion information lie at the heart of the state-of-the-arttwo-stream ConvNets.|
|||Similarto [25] and [33] is Feichtenhofer et al.s [8] work, wherethey employ 3D Conv fusion and 3D pooling to fuse spatialand temporal networks using RGB images and a stack of 10optical flow frames as input.|
|||[38] use multipleclips sparsely sampled from the whole video as input forboth streams, and then combine the scores for all clips in alate fusion approach.|
|||Fi-nally, the scores for the two ConvNets are combined in alate fusion approach as averaging.|
|||Unlike IDTs,these techniques use ConvNets with late fusion to combinespatial and temporal cues, but they still fail to efficientlyencode all frames together.|
|||The prediction scores of thespatial and temporal ConvNets are combined in a late fusionapproach as averaging before softmax normalization.|
|||Theprediction scores of the spatial and temporal ConvNets arecombined in a late fusion approach via averaging.|
|||Similar to two-stream ConvNets, TLE:Bilinearoutperforms other methods, and achieves an accuracy of86.3% and 60.3% on UCF101 and HMDB51, respec-tively, which is 4/3.5%, and 0.4/3.1% better than the orig-inal C3D ConvNets [33] and iDT+FV [35] methods onMethodDT+MVSM [2]iDT+FV [35]Two Stream [25]VideoDarwin [9]C3D [33]Two Stream+LSTM [41]FST CV (SCI fusion) [30]TDD+FV [37]LTC [34]KVMF [44]TSN [38]3DConv+3DPool [8]TLE: FC-Pooling (ours)TLE: Bilinear+TS (ours)TLE: Bilinear (ours)UCF101 HMDB5183.585.988.082.388.688.190.391.793.194.093.592.295.195.655.957.259.463.756.859.163.264.863.368.569.268.870.671.1Table 4: Two-stream ConvNets.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|459|cvpr18-V2V-PoseNet  Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation From a Single Depth Map|Sensor fusion for 3dhuman body tracking with an articulated 3d body model.|
|460|Haitian_Zheng_CrossNet_An_End-to-end_ECCV_2018_paper|Our net-work contains image encoders, cross-scale warping layers, and fusion de-coder: the encoder serves to extract multi-scale features from both theLR and the reference images; the cross-scale warping layers spatiallyaligns the reference feature map with the LR feature map; the decoderfinally aggregates feature maps from both domains to synthesize the HRoutput.|
|||However, the oversimplified and down-sampled correspondenceestimation of [1] does not take advantage of the high frequency information formatching, while the synthesizing step does not utilize high resolution image priorfor better fusion.|
|||2) after the warping operations, a novelfusion scheme is proposed for image synthesis.|
|||Our fusion scheme is differentfrom the existing synthesizing practices that include image-domain early fusion(concatenation) [41, 37] and linearly combining images [42, 36].|
|||Our network, containsa LR image encoder which extracts multi-scale feature maps from the LRimage IL, a reference image encoder which extracts and aligns the referenceimage feature maps at multiple scales , and a decoder which perform multi-scalefeature fusion and synthesis using the U-Net[44] structure.|
|||After extracting the LR image feature and the warped reference imagefeature at different scales, a U-Net like decoder is proposed to perform fusionand SR synthesis.|
|||After generating the decoder feature at scale 0, three additional convolutionlayers with filter sizes 5  5 and filter number {64, 64, 3} are added to performpost-fusion and to generate the SR output,F1 = (WF2 = (WIp = (W1),1  F (0)D + b2),2  F1 + bp  F2 + bp).|
|||Zhang, L., Wu, X.: An edge-guided image interpolation algorithm via directionalIEEE transactions on Image Processing 15(8) (2006)filtering and data fusion.|
|461|Generalized Rank Pooling for Activity Recognition|Thus, various simplifications have been explored to makethe problem amenable, such as using 3D spatio-temporalconvolutions [42], recurrent models such as LSTMs orRNNs [7, 8], decoupling spatial and temporal action com-ponents via a two-stream model [38, 12], early or late fusionof predictions from a set of frames [25].|
|||Thus,in this paper, we focus on late fusion techniques on the CNNfeatures generated by a two-stream model, and refer to re-cent surveys for a review of alternative schemes [23].|
|||This difficulty can be circumvented via early-fusion of the frames as described in Bilen et al.|
|||Our results on these datasets are lower than the re-cent method in [12] that uses sophisticated residual deepmodels with intermediate stream fusion.|
|||Convolu-tional two-stream network fusion for video action recogni-tion.|
|462|Accurate Depth and Normal Maps From Occlusion-Aware Focal Stack Symmetry|Interna-tion and fusion of surface normal maps.|
|463|cvpr18-Im2Struct  Recovering 3D Shape Structure From a Single RGB Image|All the featurefusions get through a jump connections layer, which has a5  5 convolutional layer and a 2x or 4x up-sampling tomatch the 56  56 feature map size in the second scale; thejump connection from the fully connected layer is a sim-ple concatenation.|
|||Feature fusion.|
|464|Yangyu_Chen_Less_is_More_ECCV_2018_paper|: Attention-based multi-modal fusion for video description.|
|||Wang, J., Jiang, W., Ma, L., Liu, W., Xu, Y.: Bidirectional attentive fusion with context gatingfor dense video captioning.|
|465|Recurrent Convolutional Neural Networks for Continuous Sign Language Recognition by Staged Optimization|We observe from Table 2 that our proposed approachfor spatio-temporal representations in a later fusion man-ner [14] outperforms the recurrent 3D-CNN in this problemby a large margin.|
|466|Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper|Yao, Z., Yi, W.: License plate detection based on multistage information fusion.|
|467|Ting_Yao_Exploring_Visual_Relationship_ECCV_2018_paper|In the inference stage, we adopt a latefusion scheme to linearly fuse the results from two decoders.|
|||At the inference time, we adopt a late fusion scheme to connect the two visualgraphs in our designed GCN-LSTM architecture.|
|||In ad-dition, by utilizing both spatial and semantic graphs in a late fusion manner,our GCN-LSTM further boosts up the performances.|
|||One is to perform early fusion schemeby concatenating each pair of region features from graphs before attention mod-ule or the attended features from graphs after attention module.|
|||The other isour adopted late fusion scheme to linearly fuse the predicted word distributionsfrom two decoders.|
|||Figure 6 depicts the three fusion schemes.|
|||Different schemes for fusing spatial and semantic graphs in GCN-LSTM: (a)Early fusion before attention module, (b) Early fusion after attention module and (c)Late fusion.|
|||The fusion operator could be concatenation or summation.|
|||performances of our GCN-LSTM in the three fusion schemes (with cross-entropyloss).|
|||The results are 116.4%, 116.6% and 117.1% in CIDEr-D metric for earlyfusion before/after attention module and late fusion, respectively, which indicatethat the adopted late fusion scheme outperforms other two early fusion schemes.|
|468|Relja_Arandjelovic_Objects_that_Sound_ECCV_2018_paper|The input image and 1 second of audio (rep-resented as a log-spectrogram) are processed by vision and audio subnetworks(Figures 2a and 2b), respectively, followed by feature fusion whose goal is to de-termine whether the image and the audio correspond under the AVC task.|
|||features are inadequate for cross-modal retrieval (as will be shown in the resultsof Section 3.1) as they are not aligned in any way  the fusion is performed byconcatenating the features and the correspondence score is computed only afterthe fully connected layers.|
|||: Audio-visual fusion and trackingwith multilevel iterative decoding: Framework and experimental evaluation.|
|469|Kaiyue_Pang_Deep_Factorised_Inverse-Sketching_ECCV_2018_paper|The fusion is denoted as f(s)  f(sc), where  is the element-wiseaddition3.|
|||Our final objective for discriminativelytraining SBIR becomes:mintTLtri + decorrLdecorr(9)3 Other fusion strategies have been tried and found to be inferior.|
|470|Re-Ranking Person Re-Identification With k-Reciprocal Encoding|[20] propose a bidirectional rankingmethod to revise the initial ranking list with the new sim-ilarity computed as the fusion of both content and contex-tual similarity.|
|||Query-adaptive late fusion for image search and person re-identification.|
|471|cvpr18-Deep Group-Shuffling Random Walk for Person Re-Identification|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|472|cvpr18-Generate to Adapt  Aligning Domains Using Generative Adversarial Networks|Deep Domain Con-fusion (DDC) [33] jointly minimizes the classification lossand MMD loss of the last fully connected layer.|
|||Deep domain confusion: Maximizing fordomain invariance.|
|473|cvpr18-A Bi-Directional Message Passing Model for Salient Object Detection|Our fusionmodule takes the feature map h32i1 ])and the high-level prediction Si+1 as input.|
|||The fusion pro-cess is summarized as follows:i ; fi (resolution is [ Wi ) + U p(Si+1), i < 52i1 , HSi = (cid:26) Conv(h3Conv(h3i ; fi ), i = 5(7)where Conv(; f ) is the convolutional layer with kernelsize 1  1 for predicting saliency maps.|
|474|cvpr18-MiCT  Mixed 3D 2D Convolutional Tube for Human Action Recognition|We argue that the hightraining complexity of spatio-temporal fusion and the hugememory cost of 3D convolution hinder current 3D CNNs,which stack 3D convolutions layer by layer, by outputtingdeeper feature maps that are crucial for high-level tasks.|
|||The MiCT enablesthe feature map at each spatio-temporal level to be muchdeeper prior to the next spatio-temporal fusion, which inturn makes it possible for the network to achieve better per-formance with fewer spatio-temporal fusions, while reduc-ing the complexity of each round of spatio-temporal fusionby using the cross-domain residual connection.|
|||The performance of 3D CNNs is further improved byemploying more complex spatio-temporal fusion strategies[7].|
|||Frequent spatio-temporalfusions in the structure drastically increase the difficulty ofoptimizing the whole network and restrict the depth of fea-ture maps in instances of limited memory resources.|
|||It integrates 2D convolutionswith 3D convolutions to output much deeper feature mapsat each round of spatio-temporal fusion.|
|||We thuspropose enhancing M() by a deeper and capable alterna-tive G() to extract much deeper features during every roundof spatio-temporal fusion.|
|||Spatio-temporal fusion is achieved by both the 2D convolution block togenerate stationary features and 3D convolution to extract tempo-ral residual information.|
|||In other words, feature maps of a 3D inputV using G() are achieved by coupling the 3D convolutionwith the 2D convolution block serially in which the 3D con-volution enables spatio-temporal information fusion whilethe 2D convolution block deepens feature learning for each2D output of the 3D convolution.|
|||It introducesa 2D convolution between the input and output of the 3Dconvolution to further reduce spatio-temporal fusion com-plexity and facilitate the optimization of the whole network.|
|||Unlikethe residual connections in previous work [12, 9], the short-cut in our scheme is cross-domain, where spatio-temporalfusion is derived by both a 3D convolution mapping with re-spect to the full 3D inputs and a 2D convolution block map-ping with respect to the sampled 2D inputs.|
|||Regarding the baseline C3D architecture [30, 31], theMiCT-Net contains fewer 3D convolutions for spatio-temporal fusion while it produceing deeper feature mapsand limiting the complexity of the entire deep model.|
|||MethodUCF101 HMDB51MethodUCF101 HMDB51Slow fusion [15]C3D [30]LTC [31]Two-stream [25]Two-stream fusion [11]Two-stream+LSTM [40]Transformations [36]TSN [35]FST CN [28]ST-ResNet [9]Key-volume mining CNN [41]TLE(C3D CNN) [7]TLE(BN-Inception) [7]I3D [5]P3D ResNet [22]MiCT-Net65.4%44.0%159.9%73.0%82.6%82.6%81.9%85.7%71.3%82.2%84.5%86.3%86.9%84.5%88.6%88.9%-43.92%-40.5%47.1%47.1%44.1%54.6%342.0%43.4%-60.3%63.2%49.8%-63.8%Table 3.|
|||Even some of these referred works adopt ad-vanced spatio-temporal fusion methods to the feature mapsC3D + IDT [30]TDD + IDT [34]LTC [31]LTC + IDT [31]ST-ResNet + IDT [9]P3D ResNet + IDT [22]Two-stream+LSTM [40]Two-stream(conv.|
|||ConclusionIn this paper, we propose the Mixed 2D/3D Convo-lutional Tube (MiCT) which enables 3D CNNs to ex-tract deeper spatio-temporal features with fewer 3D spatio-temporal fusions and to reduce the complexity of the infor-mation that a 3D convolution needs to encode at each roundof spatio-temporal fusion.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|||Two-stream3d convnet fusion for action recognition in videos with ar-bitrary size and length.|
|475|cvpr18-An End-to-End TextSpotter With Explicit Alignment and Attention|Inspired by the success in semanticsegmentation [30], we exploit feature fusion by combiningconvolutional features of conv5, conv4, conv3 and conv2layers gradually, with the goal of maintaining both localdetailed features and high-level context information.|
|476|cvpr18-Now You Shake Me  Towards Automatic 4D Cinema|Finally, our7431#POV ModelExist AP Effect ACCUnariesCRF: U + sub-shot123 Camera CRF: U + shot45CRF: U + threadCRF: U + all pairwiseCam Unaries678 Tracks CRF: U + all pairwise+CRF: U + video pairwise55.456.351.453.857.329.427.928.843.644.450.444.652.445.348.048.8Figure 9: Our modified form of confusion matrix on trimmed ef-fect classification.|
|||9 presents the modified confusion matrix.|
|||Theconfusion between light - temperature, or splash - wind -weather seem genuine as these are difficult effects to dis-criminate.|
|477|cvpr18-A Closer Look at Spatiotemporal Convolutions for Action Recognition|[16]presented a thorough study on how to fuse temporal infor-mation in CNNs and proposed a slow fusion model thatextends the connectivity of all convolutional layers in timeand computes activations though temporal convolutions inaddition to spatial convolutions.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|478|Cognitive Mapping and Planning for Visual Navigation|Given the robots step-size isfairly large we consider a late fusion architecture and fusethe information extracted from ResNet-50.|
|479|Yi_Zhou_Semi-Dense_3D_Reconstruction_ECCV_2018_paper|To improve the density of the recon-struction and to reduce the uncertainty of the estimation, a probabilisticdepth-fusion strategy is also developed.|
|||Based on the derived uncertainty, a fusion strategy is developed and isincrementally applied as sparse reconstructions of new RVs are obtained.|
|||5: Depth map fusion strategy.|
|||Using the fusion from RV5 to RV3 as anexample, the fusion rules are illustrated in the dashed square, in which a part ofthe image plane is visualized.|
|||Using xij}41 as an example, the fusion is performed based on the following rules:a) to xi1.|
|||An illustration of the fusion strategy is given in Fig.|
|||Additionally, the depth fusion process is illustrated tohighlight how it improves the density of the reconstruction while reducing depthuncertainty.|
|||Semi-dense depthmaps (after fusion with several neighboring RVs) are given in the third column,pseudo-colored from red (close) to blue (far).|
|||To show how the fusion strategy improves the density of thereconstruction as well as reduces the uncertainty, we additionally perform anexperiment that visualizes the fusion process incrementally.|
|||8,the first column visualizes the uncertainty maps before the fusion.|
|||Semi-dense depth maps (after fusion withseveral neighboring RVs) are given in the third column, colored according todepth, from red (close) to blue (far).|
|||8: Illustration of how the fusion strategy increasingly improves the density ofthe reconstruction while reducing depth uncertainty.|
|||The first column shows theuncertainty maps  before the fusion.|
|480|Helge_Rhodin_Unsupervised_Geometry-Aware_Representation_ECCV_2018_paper|Our method (Left)extends a conventional auto encoder (Right) with a 3D latent space, rotation opera-tion, and background fusion module.|
|||The background fusion enables application to natural images.|
|481|Quo Vadis, Action Recognition_ A New Model and the Kinetics Dataset|Convolutionaltwo-stream network fusion for video action recognition.|
|482|SouYoung_Jin_Unsupervised_Hard-Negative_Mining_ECCV_2018_paper|Du, X., El-Khamy, M., Lee, J., Davis, L.: Fused dnn: A deep neural network fusionapproach to fast and robust pedestrian detection.|
|483|Borrowing Treasures From the Wealthy_ Deep Transfer Learning Through Selective Joint Fine-Tuning|NoteMethodmean Acc(%)MPP [47]Multi-model Feature Concat [1]MagNet [35]VGG-19 + GoogleNet + AlexNet [20]Training from scratch using target domain onlySelective joint training from scratchFine-tuning w/o source domainJoint fine-tuning with all source samplesSelective joint FT with random source samplesSelective joint FT w/o iterative NN retrievalSelective joint FT with Gabor filter bankSelective joint fine-tuningSelective joint FT with model fusionVGG-19 + Part Constellation Model [38]Selective joint FT with val set91.391.391.494.558.280.692.393.493.294.293.894.795.895.397.0Table 3.|
|484|Deep Multitask Architecture for Integrated 2D and 3D Human Sensing|The function ctB shares thesame structure as the first four convolutions in ctJ , but a clas-sifier in the form of a (1  1  NB) convolution is appliedafter the fusion with the current 2d pose belief maps J t, inorder to obtain semantic probability maps Bt.|
|485|cvpr18-Learning to Extract a Video Sequence From a Single Motion-Blurred Image|It consists of three parts, feature extraction,feature refinement and feature fusion.|
|||The featurefusion part works on color images to compensate misalign-ments from the three separately-generated color-refined fea-tures.|
|486|Jongbin_Ryu_DFT-based_Transformation_Invariant_ECCV_2018_paper|The SVM is used for the late fusion.|
|||fusion approach to combine the outputs of multiple middle layers.|
|||In the fusion layer, all probabilistic estimates fromthe middle layers and the final layer are vectorized and concatenated, and SVMon the vector determines the final decision.|
|||Each layergroup consists of more than one convolution layers of the same size, and dependingon the level of fusion, different numbers of groups are used in training and testing.|
|||5 shows theclassification accuracy of the individual middle layers by the DFT magnitude andaverage pooling layers before the late fusion.|
|487|cvpr18-Reconstructing Thin Structures of Manifold Surfaces by Integrating Spatial Curves|However, the volumetric model is initialized by the fusionof multiple depth maps computed from stereo matching,which still suffers from background changes problem asmentioned before.|
|||To avoidconfusion, we refer the term curve to 3D curves, andedge to 2D image edges in the whole context of the paper.|
|||2) For type (b) and (c),the fusion with MVS points is straightforward in the Delau-nay tetrahedra framework [25].|
|||Adaptive fusion.|
|||The Delaunay tetrahedra frameworkmakes fusion adaptive,including the fusion with MVSpoints and the self-intersections of curves (when two curvescross each other or one curve is reconstructed multipletimes).|
|||In prac-tice, most thin structures are reconstructed by the fusion ofdense points and multiple curves, and all of them may affectto the final thickness.|
|488|Spatio-Temporal Vector of Locally Max Pooled Features for Action Recognition in Videos|As the UCF101 is an extension ofthe UCF50 dataset, to avoid the risk of overfitting, for anyfurther fusion and for the comparison with the state-of-the-art, we excluded TCN features for the UCF50 dataset re-sults.|
|||For these four feature combinations we evaluate differentfusion strategies: Early, where after we individually buildthe final representation for each feature type and normalizeit accordingly, we concatenate all resulted representations ina final vector, we apply L2 normalization for making unitlength and then perform the classification part; sLate, wherewe make late fusion by making sum between the classifiersoutput from each representation; wLate, where we give dif-ferent weights for each feature representation classifier out-put, and then we perform the sum.|
|||The weight combinationsare tuned by taking values between 0 and 1 with the step0.05; sDouble, where besides summing the classifier out-put from the individual feature representations, we also addthe classifier output resulted from the early fusion; wDouble,where we tune the weight combinations for the sum, similarto wLate.|
|||Table 4 shows that early fusion performs better than latefusion.|
|||Double fusion combines the benefit of both, earlyand late fusion, and boosts further the accuracy.|
|||The best performanceresults are in bold for each fusion type over each feature representation combination.|
|||Bag of visualwords and fusion methods for action recognition: Com-prehensive study and good practice.|
|||Multilayer andmultimodal fusion of deep neural networks for videoclassification.|
|489|Daniel_Worrall_CubeNet_Equivariance_to_ECCV_2018_paper|: An efficient fusion move algorithmfor the minimum cost lifted multicut problem.|
|490|HOPE_ Hierarchical Object Prototype Encoding for Efficient Object Instance Search in Videos|Query-adaptive late fusion for image search and personre-identification.|
|491|cvpr18-Densely Connected Pyramid Dehazing Network|A possi-ble reason may be due to the gradient diffusion caused bydifferent tasks.|
|||Single image dehazing bymulti-scale fusion.|
|||Gated fusion network for single image dehazing.|
|492|cvpr18-SobolevFusion  3D Reconstruction of Scenes Undergoing Free Non-Rigid Motion|To sum up, we propose a variational non-rigid fusiontechnique, called SobolevFusion, which: is based on Sobolev gradient flow, allowing for a morestraightforward, faster to compute energy that pre-serves geometric details without over-smoothing; handles topological changes and large motion, thus re-quiring only a few views to build a model; can estimate voxel correspondences and colour the re-construction.|
|||Dynamic reconstruction Template-free methodsfornon-rigid fusion using a single depth sensor have been onthe rise since 2015 with the development of the offline bun-dle adjustment scheme of Dou et al.|
|||ConclusionWe have presented a method for non-rigid fusion ofscenes undergoing free motion, including fast movements,changing topology and interacting agents.|
|493|cvpr18-Estimation of Camera Locations in Highly Corrupted Scenarios  All About That Base, No Shape Trouble|Global fusion of relativemotions for robust, accurate and scalable structure from motion.|
|494|Learning Dynamic Guidance for Depth Image Enhancement|On learning opti-mized reaction diffusion processes for effective imagerestoration.|
|||Kinectfusion: real-time 3d recon-struction and interaction using a moving depth camer-[31] R. Rubinstein, T. Peleg, and M. Elad.|
|||Reliability fusion of time-of-flight depth and stereogeometry for high quality depth maps.|
|495|Kyungmin_Kim_Multimodal_Dual_Attention_ECCV_2018_paper|The key ideais to use a dual attention mechanism with late fusion.|
|||Multimodal fusion is performed after the dual attention processes (latefusion).|
|||We confirmthe best performance of the dual attention mechanism combined withlate fusion by ablation studies.|
|||3) At the multimodal fusion step, thequestion, caption, and frame information are fused using residual learning.|
|||During thewhole inference process, multimodal fusion occurs only once.|
|||Therefore, the use of multimodal fusion methodssuch as concatenation [15,8] or Multimodal Bilinear Pooling [3,16,11] along withtime axis might be prohibitively expensive and have the risk of over-fitting.|
|||After that, multimodal fusionoccurs only once during the entire QA process, using the multimodal residuallearning used in image QA [10].|
|||This learning pipeline consists of five submod-ules, preprocessing, self-attention, attention by question, multimodal fusion, andanswer selection, which is learned end-to-end, supervised by given annotations.|
|||The experimental results demonstrate two hypotheses of ourmodel that 1) maximize QA related information through the dual attention pro-cess considering high-level video contents, and 2) multimodal fusion should beapplied after high-level latent information is captured by our early process.|
|||The main contributions of this paper are as follow: 1) we propose a novelvideo story QA architecture with two hypotheses for video understanding; dualattention and late multimodal fusion, 2) we achieve the state-of-the-art resultson both PororoQA and MovieQA datasets, and our model is ranked at the firstentry in the MovieQA Challenge at the time of submission.|
|||(4) The fused representation o is calculated using residual learning fusion (Section3.4 and Fig.|
|||Note that ourmultimodal fusion is applied to the latent variables instead of the early fusionin this work for high-level reasoning process.|
|||We tackle this problem by introducing the two attentionlayers, which leverage the multi-head attention functions [22], followed by theresidual learning of multimodal fusion.|
|||4) These attentively refined frames and captions, and aquestion are fused using the residual function in the multimodal fusion module.|
|||A schematic diagram of the multimodal fusion module with the two deep resid-ual blocks.|
|||3.4 Multimodal FusionDuring the entire QA process, multimodal fusion occurs only once in this module.|
|||4 illustrates an example of our multimodal fusion module.|
|||3.5 Answer SelectionThis module learns to select the correct answer sentence using the basic element-wise calculation between the output of multimodal fusion module, o  R512,10K.M.|
|||1) MDAM-MulFusion:model using element-wise multiplication instead of the residual learning functionin the multimodal fusion module (self-attention is used).|
|||4) MDAM-EarlyFusion: model that moves the position of the multimodal fusionmodule forward in the QA pipeline; thus the information flow goes through thefollowing steps (i) preprocessing, (ii) multimodal fusion, (iii) self-attention, (iv)attention by question, (v) answer selection.|
|||The fusions of frames and captionsoccur N times by fusing M V and M C. 5) MDAM-NoSelfAttn: model withoutthe self-attention module.|
|||Lm denotes the depth of thelearning blocks in the multimodal fusion module.|
|||Dueto the small size of the MovieQA data set, the overall performance pattern showsa tendency to decrease as the depth of the attention layers Lattn and the depthof the learning blocks in the multimodal fusion module Lm increase.|
|||Comparingthe performance results by module, the models, in which multimodal fusionsoccur early in the QA pipeline (MDAM-EarlyFusion), shows little performancedifference with the models, which use only sub-part of the video input (MDAM-FrameOnly, MDAM-CaptOnly).|
|||In addition, even if multimodal fusion occurslate, the performance is degraded where a simple element-wise multiplication isused as the fusion method (MDAM-MulFusion).|
|||The self-attention module helps MDAM achieve betterperformance (48.9 % for MDAM vs. 47.3 % for MDAM-NoSelfAttn), and multi-modal fusion with high-level latent information by our module performs betterthan early fusion baseline (46.1 % for MDAM-EarlyFusion).|
|||The fundamental idea ofMDAM is to provide the dual attention structure that captures a high-level ab-straction of the full video content by learning the latent variables of the videoinput, i.e., frames and captions, then, late multimodal fusion is applied to get ajoint representation.|
|||Exploring various alternative models in our ablation studies, we con-jecture the following two points: 1) The position of multimodal fusion in ourQA pipeline is important to increase the performance.|
|||We learned that the earlyfusion models are easy to overfit, and the training loss fluctuates during a train-ing phase due to many fusions occurred on time domain.|
|||On the other hand,the late fusion model were faster in convergence, leading to better performanceresults.|
|496|Nuno_Garcia_Modality_Distillation_with_ECCV_2018_paper|3.1 Cross-stream multiplier networksTypically in two-stream architectures, the two streams are trained separatelyand the predictions are fused with a late fusion mechanism [25][5].|
|||Feichtenhofer, C., Pinz, A., Zisserman, A.: Convolutional two-stream network fusionfor video action recognition.|
|497|HSfM_ Hybrid Structure-from-Motion|Global fusion of generalizedcamera model for efficient large-scale structure from motion.|
|||Global fusion of rela-tive motions for robust, accurate and scalable structure frommotion.|
|498|Predictive-Corrective Networks for Action Detection|MethodMultiTHUMOS mAPSingle-frame RGB4-frame late fusionPredictive-corrective (our)25.125.326.9Table 1.|
|||Second, weconsider a model similar to the late fusion model of [19] (orthe late pooling model of [58]).|
|||While incorporating these cues provide asmall 0.2% boosts over the baseline (25.1% mAP for single-frame vs 25.3% mAP for late fusion), it does not match theperformance of our predictive-corrective model.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|499|cvpr18-Group Consistent Similarity Learning via Deep CRF for Person Re-Identification|Weevaluate the proposed weighted combination by comparingit with two different fusion methods: (i) basel.|
|||The presented results are not refined by anypost-processing technique such as re-ranking [53] or multi-query fusion [49].|
|500|cvpr18-Attend and Interact  Higher-Order Object Interactions for Video Understanding|3.1.3 Late fusion of coarse and fineFinally, the attended context information vc obtained fromthe image representation provides coarse-grained under-standing of the video, and the object interactions discoveredthrough the video sequences voi,T provide fine-grained un-derstanding of the video.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|501|cvpr18-A Pose-Sensitive Embedding for Person Re-Identification With Expanded Cross Neighborhood Re-Ranking|[15] extends this to revise the initial ranking list with a newsimilarity obtained from fusion of content and contextualsimilarity.|
|||Within the ECN frame-work just using the direct euclidean distances in Equation 3ECN (orig-dist) results in similar high performance gainsin the rank-1 scores, in fact better than the state-of-the-artk-reciprocal [48] method that uses the reciprocal list com-parisons with local query expansion and fusion of rank andeuclidean distances.|
|||Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|502|Tete_Xiao_Unified_Perceptual_Parsing_ECCV_2018_paper|However, thisis a necessary step to overcome the fusion of natural and synthetic data sources.|
|503|Vincent_Leroy_Shape_Reconstruction_Using_ECCV_2018_paper|Wesweep viewing rays with this volumetric receptive field, a process we coin volumesweeping, and embed the algorithm in a multi-view depth-map extraction andfusion pipeline followed by a geometric surface reconstruction.|
|||3 Method OverviewAs for many recent multi-view stereo reconstruction methods, ours estimatesper camera depth maps, followed by depth fusion, allowing therefore each cam-era to provide local details on the observed surface with local estimations.|
|||By considering the surface detection problem alone, and letting the subsequentstep of fusion integrate depth in a robust and consistent way, we simplify theproblem and require little spatial coherence, hence allowing for small grids.|
|||: Real-timevisibility-based fusion of depth maps.|
|||: Dynamicfusion: Reconstruction and trackingof non-rigid scenes in real-time.|
|504|Yao_Yao_MVSNet_Depth_Inference_ECCV_2018_paper|Depth Map Fusion Similar to other multi-view stereo methods [8, 32], weapply a depth map fusion step to integrate depth maps from different viewsto a unified point cloud representation.|
|||The visibility-based fusion algorithm[26] is used in our reconstruction, where depth occlusions and violations acrossdifferent viewpoints are minimized.|
|||Galliani, S., Lasinger, K., Schindler, K.: Massively parallel multiview stereopsisby surface normal diffusion.|
|||Merrell, P., Akbarzadeh, A., Wang, L., Mordohai, P., Frahm, J.M., Yang, R.,Nist er, D., Pollefeys, M.: Real-time visibility-based fusion of depth maps.|
|||Newcombe, R.A., Izadi, S., Hilliges, O., Molyneaux, D., Kim, D., Davison, A.J.,Kohi, P., Shotton, J., Hodges, S., Fitzgibbon, A.: Kinectfusion: Real-time densesurface mapping and tracking.|
|505|cvpr18-CleanNet  Transfer Learning for Scalable Image Classifier Training With Label Noise|Some studies for learning con-volutional neural networks (CNNs) with noise also rely onmanual labeling to estimate label confusion [20, 35].|
|||[20] rely on manual labeling toestimate label confusion for real-world label noise.|
|||[20] used the part of data in Clothing1Mthat has both noisy and correct class labels to estimate con-fusion among classes and modeled this information in lossfunction.|
|||Since we only compare the noisy class label to thecorrect class label for an image to verify whether the noisyclass label is correct, we lose the label confusion informa-tion, and thus these numbers are not directly comparable.|
|||Our proposed method achieves 79.90%, which iscomparable to the state of the art 80.38% reported in [20]which benefits from the extra label confusion information.|
|506|cvpr18-Unsupervised Cross-Dataset Person Re-Identification by Transfer Learning of Spatial-Temporal Patterns|Secondly, a Bayesian fusionmodel is proposed to combine the learned spatio-temporalpatterns with visual features to achieve a significantly im-proved classifier.|
|||Figure 1: The TFusion model consists of 4 steps: (1) Trainthe visual classifier C in the labeled source dataset (Section4.2); (2) Using C to learn the pedestrians spatio-temporalpatterns in the unlabeled target dataset (Section 4.3); (3) Con-struct the fusion model F (Section 4.4); (4) Incrementallyoptimize C by using the ranking results of F in the unlabeledtarget dataset (Section 4.6).|
|||Secondly,a Bayesian fusion model is proposed to combine the learnedspatio-temporal patterns with visual features to achieve asignificantly improved fusion classifier F for Re-ID in thetarget dataset.|
|||During the iterative optimization procedure, both of the vi-sual classifier C and the fusion classifier F are updated in amutual promotion way.|
||| We propose a Bayesian fusion model, which combinesthe spatio-temporal patterns learned and the visual fea-tures to achieve high performance of person Re-ID inthe unlabeled target datasets.|
||| We propose a learning-to-rank based mutual promotionprocedure, which uses the fusion classifier to teachthe weaker visual classifier by the ranking results onunlabeled dataset.|
|||Then we combine the patterns with the visualfeatures to build a more precise fusion classifier.|
|||ABayesian fusion model F is proposed to combine the vi-sual classifier C and the newly learned spatio-temporalpatterns for precise discrimination of pedestrian images.|
|||In this step, we leveragethe fusion model F to further optimize the visual clas-sifier C based on the learning-to-rank scheme.|
|||Firstly,given any surveillance image Si, the fusion model Fis applied to rank the images in the unlabeled targetdataset according to the similarity with Si.|
|||Inthis way, all of the visual classifier C, the fusion model F ,and the spatio-temporal patterns can achieve collaborativeoptimization.|
|||Bayesian Fusion modelAs represented in the last section, the spatio-temporalpattern P r(ij, ci, cj|(Si) = (Sj)), which is estimatedfrom the visual classifier C, provides a new perspective todiscriminate surveillance images besides the visual featuresused in C. This motivates us to propose a fusion model,which combines the visual features with the spatio-temporal7951Figure 3: Incremental optimization by the learning-to-rank scheme.|
|||Formally, thefusion model is based on the conditional probability:P r((Si) = (Sj )| ~vi, ~vj , ij , ci, cj ).|
|||(9), we can construct a fusion classifierF , which takes the visual features and spatio-temporal infor-mation of two images as input, and outputs their matchingprobability.|
|||Precision Analysis of the Fusion ModelIn this section, we will analyze the precision of the fusionmodel F .|
|||The following Theorem 1 shows theperformance of the fusion model:E n < Ep + En.|
|||Theorem 1 : If Ep + En < 1 and  +  < 1, we havep + E Theorem 1 means that the error rate of the fusion modelF may be lower than the original visual classifier C under theconditions of Ep + En < 1 and  +  < 1.|
|||1, the fusion model F is derived fromthe visual classifier C by integrating with spatio-temporalpatterns.|
|||Subsequently, the improvement ofC may also derive a better fusion model F .|
|||In the first step, given any query image Si, thefusion classifier F is applied to rank the other images in theunlabeled target dataset according to the matching probabil-ity defined in Eq.(11).|
|||As mentioned in section 4.4, thecapturing time of each image frame is required to build thefusion model.|
|||(11) in the fusion model,  and are two tunable parameters.|
|||1, learning the spatio-temporal patternsin the unlabeled target dataset is a key step of our fusionmodel.|
|||The following Fusion Model F  column shows that the per-formance of the fusion model, which integrates with thespatio-temporal patterns, gains significant improvementcompared with the original visual classifier C.The Incremental Optimization step in Table.|
|||This provesthe effectiveness of the learning-to-rank scheme to transferknowledge from the fusion model F to the visual classifier Cin the unlabeled target dataset.|
|||1 also shows that theperformance of the fusion model F achieves significantimprovement after the incremental learning.|
|||The fusion with pedestrians spatio-temporal pattern cansignificantly improve the Re-ID performance.|
|||This proves again theeffectiveness of the fusion with spatio-temporal information.|
|||(11),  and  are two tunable pa-rameters in the fusion model.|
|||Theorem 1 proves that when +  < 1, the fusion model F may have chance to performbetter than the original visual classifier C. Thus, we try differ-ent combinations of  and , which satisfy  +  < 1, andtest the performance of the fusion model.|
|||In each iteration,the fusion model F is used to train the visual classifier C, andsubsequently a more accurate C can derive a better F .|
|507|Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper|Furthermore, the outputs of each MSRB are used as the hierarchical fea-tures for global feature fusion.|
|||Secondly, the outputs of each MSRB are combined forglobal feature fusion.|
|||Besides,we introduce a convolution layer with 11 kernel as a bottleneck layer to ob-Multi-scale Residual Network for Image Super-Resolution3tain global feature fusion.|
|||Contributionsof this paper are as follows: Different from previous works, we propose a novel multi-scale residual block(MSRB), which can not only adaptively detect the image features, but alsoachieve feature fusion at different scales.|
||| We propose a simple architecture for hierarchical features fusion (HFFS) andimage reconstruction.|
|||In addition, a 11convolution layer at the end of the block can be used as a bottleneck layer, whichcontributes to feature fusion and reduces computation complexity.|
|||Thefeature extraction module is composed of two structures: multi-scale residualblock (MSRB) and hierarchical feature fusion structure (HFFS).|
|||3, our MSRB contains two parts: multi-scale featuresfusion and local residual learning.|
|||In this work, a simple hierar-chical feature fusion structure is utilized.|
|||The output of hierarchical feature fusion structure (HFFS) can beformulated as:FLR = w  [M0, M1, M2, ..., MN ] + b,(9)where M0 is the output of the first convolutional layer, Mi(i 6= 0) represents theoutput of the ith MSRB, and [M0, M1, M2, ..., MN ] denotes the concatenationoperation.|
|508|Hai_Ci_Video_Object_Segmentation_ECCV_2018_paper|4.3 Combining Foreground Predictions and EmbeddingsIn this section, we present a fusion module to combine the LSEs and foregroundpredictions.|
|||It is worth noting that, after introducingthis fusion module, the overall network can still be trained end-to-end by passingthe gradients directly to the two branches.|
|||It is worth noting that the fusion is trainedend-to-end without manually setting the model parameters.|
|509|Joel_Janai_Unsupervised_Learning_of_ECCV_2018_paper|: Optical flow with geometric occlusion estimation and fusion ofmultiple frames.|
|510|cvpr18-Learning Deep Sketch Abstraction|The four query sketches are then fed to the trainedFG-SBIR model and the final result is obtained by score-level fusion over the four sketches.|
|511|Gaze Embeddings for Zero-Shot Image Classification|Second,we concatenate per-class gaze embedding of each participantthrough early fusion, i.e.|
|||Third, we learn a model for each participant separately andthen we average their classification scores before making thefinal prediction decision in the late fusion setting, i.e.|
|||We evaluate 5 participants separately as well as theirvarious combinations: Averaging each participants gazeembeddings (AVG), Combining them through early fusion(EARLY) and through late fusion (LATE).|
|||We first consider thegaze embeddings of our 5 participants separately and thencombine the gaze embeddings of each participant by aver-aging them (AVG), concatenating them through early fusion(EARLY), and combining the classification scores obtainedby each participants gaze data through late fusion (LATE).|
|||Compar-ing gaze and bag-of-words results shows that gaze neverconfuses cats and dogs whereas such confusion occurs forbag-of-words.|
|512|From Red Wine to Red Tomato_ Composition With Context|using late fusion, compose concepts but fail to model con-textuality.|
|||It can be thought of as late fusion.|
|513|cvpr18-Pix3D  Dataset and Methods for Single-Image 3D Shape Modeling|For eachobject, we take a short video and fuse the depth data to get its3D mesh by using fusion algorithm provided by Occipital, Inc.We also take 1020 images for each scanned object in frontof various backgrounds from different viewpoints, makingsure the object is neither cropped nor occluded.|
|||Kinectfusion: real-time 3d reconstructionand interaction using a moving depth camera.|
|515|cvpr18-Making Convolutional Networks Recurrent for Visual Sequence Learning|Convolutionaltwo-stream network fusion for video action recognition.|
|||Multilayer and mul-timodal fusion of deep neural networks for video classifica-tion.|
|517|Spindle Net_ Person Re-Identification With Human Body Region Guided Feature Decomposition and Fusion|In this study, we propose a novelConvolutional Neural Network (CNN), called Spindle Net,based on human body region guided multi-stage feature de-composition and tree-structured competitive feature fusion.|
|||In order to make better use of theregion features, a tree-structured feature fusion strategy isadopted in our approach instead of directly concatenatingthe region features together.|
|||Moreover, a competitive strategy is alsoused in the feature fusion process.|
|||Then the regionsfeatures of different semantic levels are merged by a tree-structured fusion network with a competitive strategy.|
|||A fusionunit is proposed for the feature fusion process, which takestwo or more feature vectors of the same size as input andoutputs one merged feature vector.|
|||3, and each fusion unit isrepresented by one green block.|
|||Illustration of feature fusion.|
|||(b-d) Three input featurevectors of the body fusion unit.|
|||The fusion unit has two main processes.|
|||2) The feature transformation process is conductedby a inner product layer, so that the transformed result canbe used for later fusion units.|
|||A tree-structured fusionstrategy is proposed and features representing micro bodysub-regions are merged in early stages and some macro fea-tures are merged in later stages.|
|||3, in the first stage, the features of thetwo leg regions, and the features of the two arm regions, aremerged by two fusion units, separately.|
|||Afterwards, the twofusion results of the first stage are further merged with lowerbody features and upper body features, separately.|
|||Then,a fusion unit takes the two fusion results of the previousstage, together with the feature vector of the head-shoulderregion as input and compute the merged feature vector ofthe whole body.|
|||5 to demonstrate the featurecompetition and fusion strategy based on the element-wisemax operation.|
|||In this example, we focus on the fusion unitwhich takes three feature vectors, i.e.|
|||Moreover, thefeature selection and fusion strategy also helps obtain goodcompact features.|
|||Investigations on FFNThere are two key factors of the proposed FFN, i.e., thetree fusion structure and the feature competition strategy.|
|||For the tree fusion structure, the results of using only oneregion feature are evaluated and listed in Table 5.|
|||Thus the tree-structured fusion technology isadopted and better features are merged in later stages.|
|||Onthe other hand, such fusion structure is also consistent withTable 6.|
|||Comparison results of different fusion structures and com-petition strategies on Market-1501 [33] datasets.|
|||The proposed fusion structure (Tree) is compared withsome other possible ones, including the Linear structure(features are merged one by one) and the inverse tree (i-Tree) structure (macro features are merged first).|
|||The performance ofdifferent fusion structures and competition strategies are re-ported in Table 6.|
|||Features of different body regions are separated bya multi-stage ROI pooling network and merged by a tree-structured fusion network.|
|||Strong capacity of the proposed feature com-petition and fusion network is also verified.|
|518|Fast 3D Reconstruction of Faces With Glasses|For the fusion of multiple depth maps into a joint 3Dmodel, various approaches have been proposed, with themost prominent ones being volumetric approaches fusingoccupancies [23, 25, 27, 28, 55], signed-distance functions[14,33,52,56], or mesh-based techniques [15,18,20].|
|||Kinectfusion: Real-time dense surface map-ping and tracking.|
|||Mobilefusion: Real-timevolumetric surface reconstruction and dense tracking on mo-bile phones.|
|||Fast and high quality fusion of depth maps.|
|519|cvpr18-Fast Spectral Ranking for Similarity Search|Iscen1We avoid the term diffusion [11, 24] in this work.|
|||Along with the diffusion kernel [32, 31], it has been studiedin connection to regularization [58, 57].|
|||Following [24], general-ized max-pooling [35, 23] is used to pool regional diffusionscores per image.|
|||The approx-imation quality is arbitrarily close to the optimal one at aMethodm  d INSTRE Oxf5k Oxf105k Par6k Par106kGlobal descriptors - Euclidean searchR-MAC [45]R-MAC [15]5122,04847.762.677.783.970.180.884.193.876.889.9Global descriptors - Manifold searchDiffusion [24]FSR.RANK-rDiffusion [24]FSR.RANK-r5125122,0482,04870.370.380.580.585.785.887.187.582.785.087.487.994.193.896.596.492.592.495.495.3Regional descriptors - Euclidean search21512R-match [46]R-match [46] 212,04855.571.081.588.176.585.786.194.979.991.3Regional descriptors - Manifold search5512Diffusion [24]5512FSR.APPROX21512Diffusion [24]21512FSR.APPROXDiffusion [24] 52,048FSR.APPROX 52,048Diffusion [24] 212,048FSR.APPROX 212,04877.578.480.080.488.488.589.689.291.591.693.293.095.095.195.895.884.786.590.3-90.093.094.2-95.695.696.596.596.496.596.997.093.092.492.6-95.895.295.3-Table 1: Performance comparison to the baseline methodsand to the state of the art on manifold search [24].|
|||Diffusion kernels.|
|||Diffusion kernels on graphs and otherdiscrete structures.|
|||Diffusion processes for retrieval revis-ited.|
|||Efficient dif-fusion on region manifolds: Recovering small objects with compactcnn representations.|
|520|Zhongzheng_Ren_Learning_to_Anonymize_ECCV_2018_paper|Chen, J., Wu, J., Konrad, J., Ishwar, P.: Semi-coupled two-stream fusion convnetsfor action recognition at extremely low resolutions.|
|521|cvpr18-HATS  Histograms of Averaged Time Surfaces for Robust Event-Based Object Classification|Real-time classification and sensor fusion with a spikingdeep belief network.|
|522|A Non-Local Low-Rank Framework for Ultrasound Speckle Reduction|Among these methods, the mostsuccessful ones are those based on anisotropic diffusion(e.g., [20, 8, 32]) and the bilateral filter (e.g., [2]).|
|||ExperimentsWe evaluate the performance of our method on a num-ber of synthetic and clinical ultrasound images by compar-ing with the following state-of-the-art despeckling filters:(1) speckle reducing anisotropic diffusion (SRAD) [32], (2)squeeze box filter (SBF) [22], (3) optimized Bayesian non-local means (OBNLM) [5], (4) anisotropic diffusion guidedby Log-Gabor filters (ADLG) [8], and (5) non-local meanfilter combined with local statistics (NLMLS) [30].|
|||Edge-preserving speckle texture removal by interference-basedspeckle filtering followed by anisotropic diffusion.|
|||Breast ultrasound despeckling using anisotropic dif-fusion guided by texture descriptors.|
|||Anisotropic diffusion filter with memory based on specklestatistics for ultrasound images.|
|||Ultrasound speckleInreduction via super resolution and nonlinear diffusion.|
|||Ultrasound speckle reductionby a susan-controlled anisotropic diffusion method.|
|523|Exploiting 2D Floorplan for Building-Scale Panorama RGBD Alignment|Kinectfusion: real-time 3d reconstruction and inter-action using a moving depth camera.|
|524|Look Closer to See Better_ Recurrent Attention Convolutional Neural Network for Fine-Grained Image Recognition|To leverage the benefit of feature ensemble, we first nor-malize each descriptor independently, and concatenate themtogether into a fully-connected fusion layer with softmaxfunction for the final classification.|
|525|cvpr18-Density-Aware Single Image De-Raining Using a Multi-Stream Dense Network|To estimate the residual component (r) from the observa-tion y, a multi-stream dense-net (without the label fusionpart) using the new dataset with heavy-density is trained.|
|||In the second ablation study, we demonstrate the effec-tiveness of different modules in our method by conductingthe following experiments: Single: A single-stream densely connected network(Dense2) without the procedure of label fusion.|
||| Yang-Multi [36] 6 : Multi-stream network trainedwithout the procedure of label fusion.|
||| Multi-no-label: Multi-stream densely connected net-work trained without the procedure of label fusion.|
||| DID-MDN (our): Multi-stream Densely-connectednetwork trained with the procedure of estimated labelfusion.|
|||In contrast, the proposed multi-streamnetwork with label fusion approach is capable of removingrain streaks while preserving the background details.|
|526|Jiahui_Zhang_Efficient_Semantic_Scene_ECCV_2018_paper|This design restricts its application in shape completion, RGB-D fusion and etc.,which aim to predict unknown structures.|
|||Riegler, G., Ulusoy, A.O., Bischof, H., Geiger, A.: Octnetfusion: Learning depthfusion from data.|
|527|cvpr18-MovieGraphs  Towards Understanding Human-Centric Situations From Videos|To take into account both video anddialog, we perform late fusion of video and dialog scores(see Sec.|
|||We combine video and dialog cues usinglate fusion of the scores from the models used in rows 8 and12/14, and see a large increase in performance in both pr-prand gt-gt settings (rows 15, 16).|
|528|cvpr18-PoseFlow  A Deep Motion Representation for Understanding Human Behaviors in Videos|Convolutionaltwo-stream network fusion for video action recognition.|
|529|cvpr18-Learning and Using the Arrow of Time|Forthe temporal feature fusion stage (Figure 2a), we first mod-ify the VGG-16 network to accept a number of frames (e.g.|
|||To understandthe effectiveness of the AoT features from the different lay-ers, we fine-tune three sets of layers separately:the lastlayer only, all layers after temporal fusion, and all layers.|
|||[24], we redo the random and68057InitializationFine-tuneLast layer After fusion All layersRandomImageNetAoT(ours)[24]--T-CAM 38.0%53.1%[24]-T-CAM 47.9%58.6%UCF10157.2%55.3%KineticsFlickr-68.3%81.2%79.2%74.3 %81.7%79.3%85.7%84.1%86.3%84.1%79.4%Table 5: Action classification on UCF101 split-1 with flowinput for different pre-training and fine-tuning methods.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|530|cvpr18-Hallucinated-IQA  No-Reference Image Quality Assessment via Adversarial Learning|(2) Anovel high-level semantic fusion mechanism is introducedto further reduce the instability of the quality regression net-work caused by the hallucination model.|
|||733 Since the result of hallucinated reference is crucial forfinal prediction, an IQA-Discriminator and an implicitranking relationship fusion scheme are introduced tobetter guide the learning of generator and suppress thenegative scene hallucination influence to quality re-gression in a low-level to high-level manner.|
|||The incorporated discrepancyinformation together with high-level semantic fusion from the generative network can supply the regression network with rich informationand greatly guide the network learning.|
|||To further stabilizing the optimiza-tion procedure of R, a high-level semantic fusion scheme isproposed.|
|||We fuse the ones afterthe last encoder residual block of second stack with the fea-ture maps after the last block of R, then we have the fusionterm:mn (Id)}CmnF = f (H5,2(Id))  (R1(Id, Imap))(11)where f is a linear projection to ensure the dimensions ofH and R1 are equal, R1 denotes the feature extraction be-fore the fully connected layers (R2) of R, and  denotes736the concatenation operation.|
|||Thus, the loss of R could beformulated as:LR =1TkR2(f (H5,2(Id))  (R1(Id, Imap)))  stkl1TXt=1(12)The form of the loss LR allows the high-level semantic in-formation of G participating in the optimization procedureof R. As we discussed in the introduction, the fusion termF explores implicit ranking relationship5 within G to as aguidance to help R adjusting the quality prediction in anadaptive manner.|
|||Multi-level semantic fusion.|
|||We also show the im-provements brought by the multi-level semantic fusionmechanism.|
|531|cvpr18-Learning Latent Super-Events to Detect Multiple Activities in Videos|Convolutionaltwo-stream network fusion for video action recognition.|
|532|Anirudh_Som_Perturbation_Robust_Representations_ECCV_2018_paper|In this paper we present theoreticallywell-grounded approaches to develop novel perturbation robust topolog-ical representations, with the long-term view of making them amenableto fusion with contemporary learning architectures.|
|||The SSM-based descriptors are com-puted using the histogram of gradi-ents (HOG), optical flow (OF) andfusion of HOG, OF features.|
|||Future directions include fusion with contemporary deep-learning architectures to exploit the complementarity of both paradigms.|
|||Sun, J., Ovsjanikov, M., Guibas, L.: A concise and provably informative multi-scale signature based on heat diffusion.|
|533|Hongyu_Xu_Deep_Regionlets_for_ECCV_2018_paper|Deep Regionlets for Object DetectionHongyu Xu1, Xutao Lv2, Xiaoyu Wang2, Zhou Ren3,Navaneeth Bodla1 and Rama Chellappa11University of Maryland, College Park, Maryland, USA2Intellifusion3Snap Inc.1{hyxu,nbodla,rama}@umiacs.umd.edu2{lvxutao,fanghuaxue}@gmail.com 3zhou.ren@snap.comAbstract.|
|||Bodla, N., Zheng, J., Xu, H., Chen, J., Castillo, C.D., Chellappa, R.: Deep het-erogeneous feature fusion for template-based face recognition.|
|535|cvpr18-Guided Proofreading of Automatic Segmentations for Connectomics|[32] consider whole EM volumes ratherthan a per-section approach, then solve a fusion problem witha global context.|
|||[17] propose a random forestclassifier coupled with an anisotropic smoothing prior in aconditional random field framework with 3D segment fusion.|
|||Segmentation fusion for connec-tomics.|
|536|Generalized Semantic Preserving Hashing for N-Label Cross-Modal Retrieval|Data fusion through cross-modality metric learning us-ing similarity-sensitive hashing.|
|537|Person Re-Identification in the Wild|Query-adaptive late fusion for image search and person re-identification.|
|538|cvpr18-Video Representation Learning Using Discriminative Pooling|Convolutionaltwo-stream network fusion for video action recognition.|
|539|Dongang_Wang_Dividing_and_Aggregating_ECCV_2018_paper|Finally, we introduce a new fusion approach by using thepredicted view probabilities as the weights for fusing the classification resultsfrom multiple view-specific classifiers to output the final prediction score foraction classification.|
|||3) A new view-prediction-guided fusion method for combining action classi-fication scores from multiple branches is proposed.|
|||(3) In theview-prediction-guided fusion module, we design several view-specific action clas-sifiers for each branch.|
|||.Final action class score  YView prediction scoreShared CNNCNN branch(V)CNN branch(u)CNN branch(1)1,vC,1uC,uvC1,1Cmessage passingmessage passingView classifierRefined view-specific feature(1)Refined view-specific feature(u)Refined view-specific feature(V)View-specific classifier (1,1)View-specific classifier (1, v)View-specific classifier (u, 1)View-specific classifier (u, v)Score fusion.|
|||The details for (a) inter-view message passing module discussed in Section 3.3,and (b) view-prediction-guided fusion module described in Section 3.4.|
|||Specifically, for the video xi, the fused(a) Message passing module(b) View-prediction-guided fusion module............1,1C1,Cv1,CV,1Cu,Cuv,CuV.|
|||The cross-view multi-branch module with view-prediction-guided fusion mod-ule forms our Dividing and Aggregating Network (DA-Net).|
|||We firsttrain the network based on the basic multi-branch module to learn the basicInception 5a output1x1 convolutions1x1 convolutions1x1 convolutions1x1 convolutions3x3 convolutions3x3 convolutions3x3 convolutionsInception 5b outputpoolingShared CNNCNN Branch10D. Wang, W. Ouyang, W. Li and D. Xufeatures of each branch and then fine-tune the learnt network by additionallyadding the message passing module and view-prediction-guided fusion module.|
|||The scoresindicate the similarity between the videos from the target view and those fromthe source views, based on which we can still obtain the weighted fusion scoresthat can be used for classifying videos from the target view.|
|||fusion scheme using view prediction probabilities as the weight also contributesto performance improvement.|
|||In particular, in the firstvariant, we remove the view-prediction-guided fusion module, and only keep thebasic multi-branch module and message passing module, which is referred to asDA-Net (w/o fus.).|
|||Similarly in the second variant, we remove the message pass-ing module, and only keep the basic multi-branch module and view-prediction-guided fusion module, which is referred to as DA-Net (w/o msg.).|
|||), since the fusion part is ablated, we only train one classifier for eachbranch, and we equally fuse the prediction scores from all branches for obtainingthe action recognition results.|
|||outperformsthe Ensemble TSN method for both modalities and after two-stream fusion,which indicates that learning common features (i.e.|
|||), which demonstrates the effectiveness of our view-prediction-guided fusionmodule.|
|||In the view-prediction-guided fusion mod-ule, all the view-specific classifiers integrate the total V  V types of cross-viewinformation.|
|540|Chaojian_Yu_Hierarchical_Bilinear_Pooling_ECCV_2018_paper|2 Related WorkIn the following, we briefly review previous works from the two viewpoints ofinterest due to their relevance to our work, including fine-grained feature learningand feature fusion in CNNs.|
|||We also compare our cross-layer integration with hypercolumn [3] based fea-ture fusion.|
|541|cvpr18-Recognizing Human Actions as the Evolution of Pose Estimation Maps| With CNNs and late fusion scheme, our methodachieves state-of-the-art performances on NTU RG-B+D, UTD-MHAD and PennAction datasets.|
|||e) Deep features are extracted from both types of imagesand the late fusion result predicts action label.|
|||While, accuracies drop by only 1.86% from 82.38% to80.52% for cross subject setting and 0.90% from 86.65%116446 47 48 49 50 51 52 53 54 55(cid:20)(cid:27)(cid:22)(cid:20)(cid:27)(cid:22)(cid:20)(cid:24)(cid:20)(cid:24)(cid:21)(cid:19)(cid:23)(cid:21)(cid:19)(cid:23)(cid:21)(cid:23)(cid:21)(cid:23)(cid:20)(cid:27)(cid:20)(cid:27)(cid:20)(cid:20)(cid:20)(cid:20)(cid:20)(cid:28)(cid:23)(cid:20)(cid:28)(cid:23)(cid:22)(cid:22)(cid:21)(cid:21)(cid:20)(cid:20)(cid:20)(cid:20)(cid:23)(cid:23)(cid:22)(cid:22)(cid:21)(cid:21)(cid:27)(cid:27)(cid:21)(cid:21)(cid:25)(cid:25)(cid:25)(cid:25)(cid:21)(cid:21)(cid:22)(cid:22)(cid:19)(cid:19)(cid:27)(cid:27)(cid:21)(cid:21)(cid:23)(cid:23)(cid:20)(cid:19)(cid:20)(cid:19)(cid:21)(cid:21)(cid:25)(cid:25)(cid:20)(cid:27)(cid:19)(cid:20)(cid:27)(cid:19)(cid:21)(cid:26)(cid:21)(cid:26)(cid:28)(cid:28)(cid:21)(cid:21)(cid:20)(cid:20)(cid:20)(cid:28)(cid:20)(cid:28)(cid:22)(cid:22)(cid:19)(cid:19)(cid:21)(cid:21)(cid:21)(cid:21)(cid:21)(cid:21)(cid:19)(cid:19)(cid:24)(cid:24)(cid:19)(cid:19)(cid:20)(cid:21)(cid:20)(cid:21)(cid:19)(cid:19)(cid:27)(cid:27)(cid:21)(cid:21)(cid:20)(cid:20)(cid:20)(cid:20)(cid:20)(cid:19)(cid:20)(cid:19)(cid:23)(cid:23)(cid:21)(cid:23)(cid:26)(cid:21)(cid:23)(cid:26)(cid:21)(cid:21)(cid:21)(cid:21)(cid:22)(cid:22)(cid:19)(cid:19)(cid:21)(cid:27)(cid:21)(cid:27)(cid:22)(cid:21)(cid:22)(cid:21)(cid:21)(cid:21)(cid:20)(cid:26)(cid:20)(cid:26)(cid:24)(cid:24)(cid:20)(cid:20)(cid:26)(cid:26)(cid:25)(cid:25)(cid:19)(cid:19)(cid:28)(cid:28)(cid:24)(cid:24)(cid:28)(cid:28)(cid:21)(cid:21)(cid:20)(cid:20)(cid:20)(cid:20)(cid:21)(cid:21)(cid:27)(cid:27)(cid:21)(cid:21)(cid:24)(cid:24)(cid:26)(cid:26)(cid:24)(cid:24)(cid:21)(cid:21)(cid:22)(cid:19)(cid:22)(cid:19)(cid:20)(cid:19)(cid:20)(cid:19)(cid:19)(cid:19)(cid:22)(cid:22)(cid:26)(cid:26)(cid:26)(cid:26)(cid:21)(cid:25)(cid:21)(cid:25)(cid:28)(cid:28)(cid:21)(cid:19)(cid:27)(cid:21)(cid:19)(cid:27)(cid:20)(cid:20)(cid:26)(cid:26)(cid:20)(cid:20)(cid:19)(cid:19)(cid:19)(cid:19)(cid:20)(cid:20)(cid:19)(cid:19)(cid:20)(cid:19)(cid:20)(cid:19)(cid:19)(cid:19)(cid:21)(cid:21)(cid:20)(cid:21)(cid:21)(cid:20)(cid:20)(cid:21)(cid:20)(cid:21)(cid:21)(cid:24)(cid:21)(cid:24)(cid:22)(cid:19)(cid:22)(cid:19)(cid:19)(cid:19)(cid:20)(cid:27)(cid:26)(cid:20)(cid:27)(cid:26)(cid:25)(cid:25)(cid:21)(cid:21)(cid:19)(cid:19)(cid:21)(cid:25)(cid:27)(cid:21)(cid:25)(cid:27)46 47 48 49 50 51 52 53 54 55(cid:21)(cid:22)(cid:21)(cid:21)(cid:22)(cid:21)(cid:20)(cid:23)(cid:20)(cid:23)(cid:21)(cid:28)(cid:21)(cid:28)(cid:20)(cid:24)(cid:20)(cid:24)(cid:20)(cid:20)(cid:20)(cid:22)(cid:20)(cid:22)(cid:28)(cid:28)(cid:21)(cid:21)(cid:22)(cid:22)(cid:22)(cid:22)(cid:19)(cid:19)(cid:21)(cid:21)(cid:19)(cid:21)(cid:21)(cid:19)(cid:25)(cid:25)(cid:20)(cid:20)(cid:21)(cid:21)(cid:23)(cid:23)(cid:24)(cid:24)(cid:24)(cid:24)(cid:22)(cid:22)(cid:20)(cid:20)(cid:28)(cid:28)(cid:20)(cid:20)(cid:20)(cid:20)(cid:21)(cid:23)(cid:21)(cid:21)(cid:23)(cid:21)(cid:23)(cid:23)(cid:19)(cid:19)(cid:20)(cid:20)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:21)(cid:20)(cid:22)(cid:21)(cid:20)(cid:22)(cid:20)(cid:28)(cid:20)(cid:28)(cid:20)(cid:25)(cid:20)(cid:25)(cid:21)(cid:22)(cid:19)(cid:21)(cid:22)(cid:19)(cid:20)(cid:20)(cid:22)(cid:22)(cid:21)(cid:21)(cid:24)(cid:24)(cid:19)(cid:19)(cid:19)(cid:19)(cid:23)(cid:23)(cid:20)(cid:20)(cid:27)(cid:27)(cid:19)(cid:19)(cid:23)(cid:23)(cid:20)(cid:20)(cid:23)(cid:23)(cid:27)(cid:27)(cid:20)(cid:20)(cid:21)(cid:24)(cid:24)(cid:21)(cid:24)(cid:24)(cid:19)(cid:19)(cid:21)(cid:21)(cid:21)(cid:21)(cid:19)(cid:19)(cid:23)(cid:23)(cid:26)(cid:26)(cid:21)(cid:21)(cid:26)(cid:26)(cid:24)(cid:24)(cid:23)(cid:23)(cid:20)(cid:20)(cid:20)(cid:20)(cid:20)(cid:21)(cid:20)(cid:21)(cid:23)(cid:23)(cid:19)(cid:19)(cid:20)(cid:20)(cid:19)(cid:19)(cid:21)(cid:21)(cid:19)(cid:21)(cid:21)(cid:19)(cid:20)(cid:28)(cid:20)(cid:28)(cid:21)(cid:23)(cid:21)(cid:23)(cid:20)(cid:25)(cid:20)(cid:25)(cid:19)(cid:19)(cid:21)(cid:21)(cid:20)(cid:21)(cid:21)(cid:20)(cid:24)(cid:24)(cid:20)(cid:20)(cid:20)(cid:20)(cid:19)(cid:19)(cid:19)(cid:19)(cid:26)(cid:26)(cid:26)(cid:26)(cid:21)(cid:21)(cid:21)(cid:20)(cid:21)(cid:20)(cid:26)(cid:26)(cid:21)(cid:22)(cid:22)(cid:21)(cid:22)(cid:22)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:20)(cid:20)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:21)(cid:26)(cid:22)(cid:21)(cid:26)(cid:22)46 47 48 49 50 51 52 53 54 55(cid:21)(cid:22)(cid:26)(cid:21)(cid:22)(cid:26)(cid:20)(cid:20)(cid:20)(cid:20)(cid:21)(cid:23)(cid:21)(cid:23)(cid:20)(cid:23)(cid:20)(cid:23)(cid:22)(cid:22)(cid:20)(cid:20)(cid:20)(cid:20)(cid:22)(cid:22)(cid:20)(cid:20)(cid:21)(cid:21)(cid:20)(cid:20)(cid:19)(cid:19)(cid:21)(cid:22)(cid:28)(cid:21)(cid:22)(cid:28)(cid:25)(cid:25)(cid:19)(cid:19)(cid:21)(cid:21)(cid:21)(cid:21)(cid:25)(cid:25)(cid:26)(cid:26)(cid:23)(cid:23)(cid:20)(cid:20)(cid:27)(cid:27)(cid:25)(cid:25)(cid:21)(cid:23)(cid:26)(cid:21)(cid:23)(cid:26)(cid:23)(cid:23)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:20)(cid:20)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:21)(cid:21)(cid:25)(cid:21)(cid:21)(cid:25)(cid:20)(cid:20)(cid:20)(cid:20)(cid:23)(cid:23)(cid:22)(cid:22)(cid:20)(cid:20)(cid:23)(cid:23)(cid:23)(cid:23)(cid:19)(cid:19)(cid:21)(cid:23)(cid:25)(cid:21)(cid:23)(cid:25)(cid:19)(cid:19)(cid:23)(cid:23)(cid:19)(cid:19)(cid:24)(cid:24)(cid:19)(cid:19)(cid:23)(cid:23)(cid:20)(cid:20)(cid:23)(cid:23)(cid:27)(cid:27)(cid:19)(cid:19)(cid:21)(cid:25)(cid:26)(cid:21)(cid:25)(cid:26)(cid:21)(cid:21)(cid:20)(cid:20)(cid:21)(cid:21)(cid:19)(cid:19)(cid:23)(cid:23)(cid:22)(cid:22)(cid:19)(cid:19)(cid:20)(cid:20)(cid:28)(cid:28)(cid:19)(cid:19)(cid:25)(cid:25)(cid:21)(cid:21)(cid:23)(cid:23)(cid:21)(cid:20)(cid:21)(cid:20)(cid:19)(cid:19)(cid:20)(cid:20)(cid:21)(cid:22)(cid:23)(cid:21)(cid:22)(cid:23)(cid:20)(cid:19)(cid:20)(cid:19)(cid:20)(cid:22)(cid:20)(cid:22)(cid:20)(cid:25)(cid:20)(cid:25)(cid:19)(cid:19)(cid:21)(cid:22)(cid:26)(cid:21)(cid:22)(cid:26)(cid:24)(cid:24)(cid:20)(cid:20)(cid:20)(cid:20)(cid:20)(cid:20)(cid:19)(cid:19)(cid:23)(cid:23)(cid:23)(cid:23)(cid:19)(cid:19)(cid:20)(cid:27)(cid:20)(cid:27)(cid:25)(cid:25)(cid:21)(cid:22)(cid:28)(cid:21)(cid:22)(cid:28)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:19)(cid:21)(cid:26)(cid:22)(cid:21)(cid:26)(cid:22)Figure 8: Confusion matrices of body pose evolution image-basedmethod (first row), body shape evolution image-based method(second row), body pose and body shape evolution images-basedmethod (third row) on NTU RGB+D dataset using cross subjectprotocol.|
|||Confusion matrices of ten actions are enlarged.|
|||8, we ana-lyze the confusion matrices among 10 types of actions.|
|||Our method jointly learns 2D poses andheatmaps, and the complementary property between thembaseball pitch   (cid:25)(cid:22)baseball swing  bench press  bowling  (cid:20)clean and jerk  golf swing  jump rope  jumping jacks  pullup  pushup  situp  squat  strum guitar  tennis forehand  tennis serve  (cid:20)---hctip llabesab(cid:24)(cid:27)(cid:20)(cid:20)(cid:20)i---gnws llabesab(cid:25)(cid:27)(cid:20)--- sserphcneb(cid:27)(cid:20)(cid:23)(cid:24)(cid:26)(cid:25)(cid:23)(cid:20)(cid:22)(cid:24)(cid:25)(cid:20)(cid:19)(cid:20)(cid:20)(cid:19)(cid:25)(cid:24)(cid:19)(cid:19)---puhsup(cid:20)(cid:20)(cid:25)---putis---tauqs(cid:23)(cid:27)(cid:21)---ratiugmurts (cid:20)---gnilwob---krej dnanaelc i---gnws flog---epor pmuj---skcaj ignpmuj---pullup(cid:20)(cid:23)(cid:26)(cid:19)---evres sinnet(cid:20)(cid:20)(cid:26)(cid:19)---dnaherof sinnetFigure 10: Confusion matrix of our method on PennAction datasetalleviate the effect of noisy 2D poses.|
|||The confusion matrix ofour method is shown in Fig.|
|543|Multi-Scale Continuous CRFs as Sequential Deep Networks for Monocular Depth Estimation|Our model is composed of two main components: a front-end CNN and a fusionmodule.|
|||The fusion module uses continuous CRFs to integrate multiple side output maps of the front-end CNN.|
|||The second component of our model is a fusion block.|
|||The main idea behind the proposed fusion block is touse CRFs to effectively integrate the side output maps of ourfront-end CNN for robust depth prediction.|
|||Specifically, weintroduce and compare two different multi-scale models,both based on CRFs, and corresponding to two differentversion of the fusion block.|
|||Multi(cid:173)scale models as sequential deep networksIn this section, we describe how the two proposed CRFs-based models can be implemented as sequential deep net-works, enabling end-to-end training of our whole networkmodel (front-end CNN and fusion module).|
|||Comparison of different multi-scale fusion schemes.|
|||Experimental ResultsAnalysis of different multi-scale fusion methods.|
|||Specifically, we consider: (i)the HED method in [33], where the sum of multiple sideoutput losses is jointly minimized with a fusion loss (we usethe square loss, rather than the cross-entropy, as our prob-lem involves continuous variables), (ii) Hypercolumn [10],where multiple score maps are concatenated and (iii) a CRFapplied on the prediction of the front-end network (lastlayer) a posteriori (no end-to-end training).|
|||Itis evident that with our CRFs-based models more accuratedepth maps can be obtained, confirming our idea that in-tegrating complementary information derived from CNNside output maps within a graphical model framework ismore effective than traditional fusion schemes.|
|||As discussed above, the proposed multi-scale fusion mod-els are general and different deep neural architectures canbe employed in the front end network.|
|||The extensive exper-iments confirmed the validity of the proposed multi-scalefusion approach.|
|544|cvpr18-Revisiting Salient Object Detection  Simultaneous Detection, Ranking, and Subitizing of Multiple Salient Objects|This is followed by a de-tailed description of the stage-wise refinement network, andmulti-stage saliency map fusion in sections 3.2 and sec-tion 3.3 respectively.|
|||A fusion layer combines predictions from all stages to generate thefinal saliency map (S Tm) of each refinement stage.|
|||Inspired by the success of refinement basedapproaches [25, 11, 12], we propose a multi-stage fusionbased refinement network to recover lost contextual infor-mation in the decoding stage by combining an initial coarserepresentation with finer features represented at earlier layers.|
|||To facilitateinteraction, we add a fusion layer at the end of network thatconcatenates the predicted saliency maps of different stages,fm. Then, we apply a 1  1resulting in a fused feature map Sconvolution layer  to produce the final predicted saliencymap S Tm of our network.|
|||RSDNet-R: RSDNetwith stage-wise rank-aware refinement units + multi-stagesaliency map fusion.|
|545|Yalong_Bai_Deep_Attention_Neural_ECCV_2018_paper|Recently,many studies have explored the multi-modal feature fusion of image representa-tion learned from deep convolutional neural network and question representationlearned from time sequential model.|
|||Nearly all of these previous works train aclassifier based on the fusion of image and question feature to predict an answer,and the relationship of image-question-answer triplets is ignored.|
|||As a result, theimage and question feature fusion strategies become the key factor for improv-ing the performance of VQA.|
|||After that, Hedi et al.proposed Multimodal TuckerFusion (MUTAN) [6] which is also a multimodal fusion scheme based on bilinearinteractions between modalities but relying on a low-rank Tucker tensor-baseddecomposition to explicitly constrain the interaction rank.|
|||The structurein the red box is the base model used to generate question representation vq and thefusion of image and question feature vector vqI .|
|||Ben-younes, H., Cadene, R., Cord, M., Thome, N.: Mutan: Multimodal tuckerfusion for visual question answering.|
|547|cvpr18-Depth-Aware Stereo Video Retargeting|Otherwise, incoher-ent scaling among frames often result in incorrect motiondirection (e.g., confusion between moving into or out-ofthe screen of non-salient objects).|
|||Depth-preserving stereo image retargeting based on pixel fusion.|
|548|cvpr18-Recurrent Saliency Transformation Network  Incorporating Multi-Stage Visual Cues for Small Organ Segmentation|Both multi-slice segmentation (3 neighboring slices are combined as abasic unit in training and testing) and multi-axis fusion (ma-jority voting over three axes) are performed to incorporatepseudo-3D information into segmentation.|
|||68285NIH Case #031axial view (-axis)slice #1fusion @ R1, DSC=(cid:889)(cid:887).|
|||(cid:890)0%slice #1slice #1slice #2slice #2slice #2slice #3slice #3slice #3Coarse SegmentationDSC=(cid:887)(cid:889).0(cid:885)%weights @ R1input @ R1P-map @ R1fusion @ R8, DSC=(cid:890)0.|
|549|Zeng_Huang_Deep_Volumetric_Video_ECCV_2018_paper|Esteban, C.H., Schmitt, F.: Silhouette and stereo fusion for 3d object modeling.|
|||Ahmed, N., Theobalt, C., Dobrev, P., Seidel, H.P., Thrun, S.: Robust fusion ofdynamic shape and normal capture for high-quality reconstruction of time-varyinggeometry.|
|||Du, R., Chuang, M., Chang, W., Hoppe, H., Varshney, A.: Montage4d: Interactiveseamless fusion of multiview video textures.|
|551|cvpr18-Frustum PointNets for 3D Object Detection From RGB-D Data|While out of the scope for this work, we expect thatsensor fusion (esp.|
|||Our method, without sensor fusion or multi-view aggregation, outperforms those methods by large marginson all categories and data subsets.|
|552|Improved Stereo Matching With Constant Highway Networks and Reflective Confidence Learning|Theerror is measured for pixel disparity predictions that differfrom the ground truth by more than two pixels, and is al-4646MethodPCBP[25]1 Ours23 Displets v2[10]4 MC-CNN-acrt[36]5cfusion[25]SetNOC ALL runtime2.272.362.372.43MV 2.4648s68s265s67s70s3.403.453.093.632.69STable 2: The highest ranking methods on KITTI 2012 due toNovember 2016, ordered by the error rate for non occludedpixels.|
|||Stereo simi-larity metric fusion using stereo confidence confidence mea-sures.|
|||Confidence driven tgv fusion.|
|||Stereo matching with nonlin-ear diffusion.|
|||In Stereo Matching with Nonlinear Diffusion,1998.|
|554|gao_peng_Question-Guided_Hybrid_Convolution_ECCV_2018_paper|The proposed approach is also complementary to ex-isting bilinear pooling fusion and attention based VQA methods.|
|||State-of-the-art feature fusion methods, such as Multimodal Compact Bilinearpooling (MCB) [10], utilize bilinear pooling to learn multi-model features.|
|||The multi-modalfeatures are fused in the latter model stage and the spatial information fromvisual features gets lost before feature fusion.|
|||To solve these problems, we propose a feature fusion scheme that generatesmulti-modal features by applying question-guided convolutions on the visual fea-tures (see Figure 1).|
|||Our model tightlycouples the multi-modal features in an early stage to better capture the spatialinformation before feature fusion.|
|||Our experiments on VQA datasets validate the effectiveness of our approachand show advantages of the proposed feature fusion over the state-of-the-arts.|
|||1) We propose a novelmulti-modal feature fusion method based on question-guided convolution ker-nels.|
|||Earlymethods utilize feature concatenation [9] for multi-modal feature fusion [15,27, 34].|
|||The model generates question attention andspatial attention masks so that salient words and regions could be jointly selectedfor more effective feature fusion.|
|||Instead of fusing the textual and visual information in highlevel layers, such as feature concatenation in the last layer, we propose a novelmulti-modal feature fusion method, named Question-guided Hybrid Convolution(QGHC).|
|||It learns question-guided convolutionkernels and reserves the visual spatial information before feature fusion, and thusachieves accurate results.|
|||Conventional ImageQA systems focus on designing robust feature fusion func-tions to generate multi-modal image-question features for answer prediction.|
|||Most state-of-the-art feature fusion methods fuse 1-d visual and language fea-ture vectors in a symmetric way to generate the multi-modal representations.|
|||3.2 Question-guided Hybrid Convolution (QGHC) for multi-modalfeature fusionTo fully utilize the spatial information of the input image, we propose Language-guided Hybrid Convolution for feature fusion.|
|||3.5 QGHC network with bilinear pooling and attentionOur proposed QGHC network is also complementary with the existing bilinearpooling fusion methods and the attention mechanism.|
|||To combine with the MLB fusion scheme [11], the multi-modal features ex-tracted from the global average pooling layer could be fused with the RNNquestion features again using a MLB.|
|||The second stage fusion of textual and visual featuresbrings a further improvement on the answering accuracy in our experiments.|
|||The output feature maps of our QGHC module utilizethe textual information to guide the learning of visual features and outperformstate-of-the-art feature fusion methods.|
|||Our feature fusion is performed before the spatialpooling and can better capture the spatial information than previous methods.|
|||Stacked Attention (SA) [18] adopts multiple attentionmodels to refine the fusion results and utilizes linear transformations to obtainthe attention maps.|
|||The proposed approach is complementary with existing featurefusion methods and attention mechanisms.|
|||Ben-younes, H., Cadene, R., Cord, M., Thome, N.: Mutan: Multimodal tuckerfusion for visual question answering.|
|555|Co-Occurrence Filter|The BF is just one of a large number of edge-preservingfilters that include Anisotropic Diffusion [17], guided im-age filter [9], or the domain transform filter [7] to name afew.|
|||The diffusion distance between two pointsequals the difference between the probabilities of randomwalkers to start at both points and end up in the same point.|
|||To approximate this, [5] uses the dominant eigenvectors ofthe affinity matrix, dubbed diffusion maps.|
|||Diffusion mapscan be efficiently calculated using the Nayst om method.|
|||The last row againstWLS with diffusion distances [5].|
|||The forth row compares CoF to WLS enhanced with Dif-fusion distance [5].|
|||Non-linear gaussian filters perform-ing edge preserving diffusion.|
|||Diffusion maps foredge-aware image editing.|
|||Scale-space and edge detection us-ing anisotropic diffusion.|
|556|Jiajun_Wu_Learning_3D_Shape_ECCV_2018_paper|: Kinectfusion: real-time 3d reconstruction and interaction using a moving depth camera.|
|||Riegler, G., Ulusoy, A.O., Bischof, H., Geiger, A.: Octnetfusion: Learning depthfusion from data.|
|557|Sunghun_Kang_Pivot_Correlational_Neural_ECCV_2018_paper|[19] trained a deep CNN on large video datasetwhile investigating the effectiveness of various temporal fusion.|
|||3.3 Adaptive AggregationWe propose a soft-attention based late fusion algorithm referred as adaptive ag-gregation.|
|||The adaptive aggregation is an extension of the attention mechanismin the late fusion framework based on the confidence between modal-specificpredictions and modal-agnostic pivot prediction.|
|||The multimodal attention weights are obtained using a neuralnetwork analogous to the soft-attention mechanism:whereagg,m =exp(sm)i=1 exp(si)PM, m = 1,    , M,sm = Ws [hm; hpivot] + bs, m = 1,    , M.Unlike widely used late fusion algorithm such as mean aggregation, the adap-tive aggregation can regulate the ratio of each modality on final prediction.|
|||Ben-Younes, H., Cadene, R., Cord, M., Thome, N.: Mutan: Multimodal tuckerfusion for visual question answering.|
|||: Fast semanticdiffusion for large-scale context-based image and video annotation.|
|559|A Practical Method for Fully Automatic Intrinsic Camera Calibration Using Directionally Encoded Light|However, since thedevice is meant to be mounted closely in front of the lens,which is likely focused at a farther distance, the projectionof the holes will amount to circular disks on the sensor in-stead of points, known as circles of confusion (CoC) [16].|
|||Hole pixel locations: To compute the projected locationof the holes on the image sensor the background display isturned on completely, yielding homogeneously filled circlesof confusion on the sensor.|
|||Tosolve this problem first it is assumed that the circle of con-fusion for the holes do not overlap with each other.|
|560|Medhini_Gulganjalli_Narasimhan_Straight_to_the_ECCV_2018_paper|Wefound a late fusion of the visual concepts to results in a better model as the factsexplicitly contain these terms.|
|||Ben-younes, H., Cadene, R., Cord, M., Thome, N.: Mutan: Multimodal tuckerfusion for visual question answering.|
|561|cvpr18-Deep Cross-Media Knowledge Transfer|It includes 200 distinct semantic categories basedon wordNet hierarchy to avoid semantic confusion, includ-ing 47 animal species like dog and 153 artifact specieslike airplane.|
|||CCL: cross-modalcorrelation learning with multi-grained fusion by hierarchi-cal network.|
|562|Sequential Person Recognition in Photo Albums With a Recurrent Network|We provide in-depth analysisof some feature fusion methods in the experiments section(Sec.|
|||We also observe that Acc single has stayedstable between the Appearance-only and Our-relation1343Test split Addition Element-wise maxOriginalAlbumTimeDay81.5173.2162.9743.1581.7574.2163.7342.75Test splitOriginalAlbumTimeDay81.7574.2163.7342.75Single regionHeadMultiple region fusionAvgUpper79.9270.7858.8034.6184.9378.2566.4343.73Max84.0775.8865.6343.55Concat82.8674.6663.6241.56Table 3.|
|||3, both embedding formulationshave shown very close performance in the four settings,with the Max fusion slightly bypassing the Addition fu-sion in the three out of four settings.|
|||Therefore, we opt fortheMax fusion method to report results in the following.|
|||Classification accuracy (%) using a single body region(columns 2-3), and their different fusions (columns 3-5).|
|||Multiple region fusion.|
|||Clearly, the Avg fusion method showsthe largest improvement over the performance of using asingle head or upper body region.|
|||Therefore, we use theperformance of Avg fusion method to compare with state-of-the-art approaches in the following.|
|||[14] and ours under the four different settings on the PIPA test set, using head region, upperbody region and their fusion.|
|||More specifically, we compare with [14] on head region,upper body region and their fusion (see Table.|
|563|Ian_Cherabier_Learning_Priors_for_ECCV_2018_paper|Recently, deep learning based approaches have beenproposed for depth map fusion [15], 3D object recognition [16, 24], or 3D shapecompletion [6, 8, 9, 36, 38, 40] using dense voxel grids as input.|
|||As intraditional TSDF fusion, we trace rays from every pixel in each depth map todetermine which voxels are occupied or empty.|
|||For ScanNet we re-integrate the provided depthmaps and semantic segmentations using TSDF fusion based on the providedcamera poses to establish voxelized ground truth.|
|||Riegler, G., Ulusoy, A.O., Bischof, H., Geiger, A.: Octnetfusion: Learning depthfusion from data.|
|564|IRINA_ Iris Recognition (Even) in Inaccurately Segmented Data|[21] propose an information fusion framework wherethree distinct feature extraction and matching schemes arefused to handle the significant variability in the input ocularimages.|
|565|cvpr18-Non-Linear Temporal Subspace Representations for Activity Recognition|Along theselines, the popular two-stream CNN model [43] for actionrecognition has been extended using more powerful CNNarchitectures incorporating intermediate feature fusion in[14, 12, 46, 53], however typically the features are pooledindependently of their temporal order during the final se-quence classification.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|566|cvpr18-Convolutional Neural Networks With Alternately Updated Clique|Multi-column [6] nets and Deeply-Fused Nets [38] also usefusion strategy and have a wide network structure.|
|567|cvpr18-Progressively Complementarity-Aware Fusion Network for RGB-D Salient Object Detection|To this end, we design a novel complementarity-aware fusion (CA-Fuse)  module  when  adopting  the  Convolutional Neural  Network  (CNN).|
|||The  proposed RGB-D  fusion  network  disambiguates  both  cross-modal and  cross-level fusion  processes  and  enables  more sufficient fusion results.|
|||(a)  Early  fusion  scheme adopted in [13] and (b) late fusion scheme adopted in [14].|
|||Although  encouraging performance has been achieved by these networks, there is  Figure  2:  The  architecture  of  the  proposed  progressively  complementarity-aware  fusion  network  for  RGB-D  salient  object  detection.|
|||Most  of  previous  RGB-D  fusion  networks explore the cross-modal complementarity by a two-stream architecture  shown  in  Fig.|
|||Most  of RGB-D  fusion  networks  [19,  25,  27]  combine  RGB  and depth  modalities  by  only  fusing  their  deep  CNN  features (i.e.,  late  fusion),  while  we  believe  that  the  cross-modal complement  for  saliency  detection  exists  across  multiple levels, which are not well-explored by previous works.|
|||supply  more features spatial In  our  view,  addressing  these  problems  will  enable  the multi-modal  fusion  network  to  capture  cross-modal  and cross-level  complement  more  sufficiently.|
|||To  this  end,  in this progressively complementarity-aware fusion network (shown in Fig.|
|||In propose work, we a the network, complementarity-aware this fusion (CA-Fuse) module (see Fig.|
|||Compared to directly concatenating multi-modal  features,  the  proposed  CA-Fuse  module formulates the cross-modal complementarity explicitly, thus allowing more efficient multi-modal fusion.|
|||Hence,  the  multi-modal  fusion process  will  be  complementarity-aware  in  terms  of  both cross-modal  and  cross-level  views,  resulting  in  sufficient multi-modal  multi-level  fusion.|
|||the In  summary, the  proposed  RGB-D  salient  object detection network enjoys several distinguished benefits: 1)  The  cross-modal  complementarity  can  be  explicitly 3052 Figure 3: The architectures of different multi-modal fusion modules.|
|||Related work    Previous  RGB-D  salient  object  detection  models [31-41]  fuse  RGB  and  depth  information  by  three  main modes:  serializing  RGB  and  depth  as  undifferentiated 4-channel  input  (input  fusion),  combining  handcrafted RGB  and  depth  saliency  features  (feature  fusion),  or performing unimodal predictions separately and then make joint decisions (result fusion).|
|||Result  fusion  methods  include summation [35, 42], multiplication [31] and designed rules [33].|
|||However, due to the lack of cross-modal interactions in the feature-extraction  stage,  the  result  fusion  scheme  is insufficient to leverage underlying cooperative information during the unimodal prediction course.|
|||[19]  use  a two-stream  late  fusion  architecture  to  fuse  RGB-D  deep features.|
|||Nonetheless,  in  its multi-modal  fusion  stage,  it  still  follows  the  paradigm  of direct feature  concatenation  without  any  explicit formulation on the cross-modal complementarity.|
|||Draw inspiration from unimodal  networks  [44]  and  [45], in  which  deep supervisions  are  introduced  to  facilitate  convergence  and generate  hierarchical  representations,  we  consider  that  an effective solution is to introduce intermediate supervisions on  top  of  each  multi-modal  fusion  level  (Fig.|
|||The added  intermediate  supervision  can  act  as  instruction  to encourage  multi-modal  fusion  in  each  level  timely,  thus 3054Module CA-Fuse 6   Adaptation layers 1 - 2 - Transition layer   384, 11 CA-Fuse 5 384, 11 384, 11 384, 11 CA-Fuse 4 384, 33 384, 33 256, 11 CA-Fuse 3 192, 33 192, 33 128, 11 CA-Fuse 2 128, 33 12833 - Table  1:  Illustration  of  the  parameters  of  the  intra-level adaptation  layers  inside  the  CA-Fuse  module  and  the transition layer between two neighboring CA-Fuse modules.|
|||reducing the multi-level fusion uncertainty.|
|||Although  this  strategy  is  able  to  ease  the  multi-level multi-modal  fusion  process  effectively,  the  multi-modal fusion component in each level still does not go beyond the traditional direct concatenation scheme, which in our view, is  unlikely the  cross-modal complementary  information.|
|||To  address  this  problem,  we further  tailor  a  complementarity-aware  fusion  (CA-Fuse) module  (Fig.|
|||By  this  way,  the RP ,   mDP  and RDPmFigure 4: Visual comparison of using different multi-modal fusion modules shown in Fig.|
|||Then  the  enhanced will  be  selected  by  a  transition  layer  (detailed mRFmDFand  along  with  the  selected  features  from the m+1 layer are concatenated and fused by one convolutional layer to learn cooperative  and make integrated predictions  features + mm RDF11 1,mRDFrepresentations   PmRD=mRD(F F F,,mDmR+1m,m RD), (1)  where  mRD denotes  the  parameters  of  the  fusion  layer  and 3055 Side- out NLPR NJUD STEREO Fig.|
|||3(c) 2 3 4 5 6 0.836  0.850  0.845  0.862  0.864  0.872 0.839  0.851  0.843  0.860  0.864  0.871 0.838  0.846  0.837  0.854  0.863  0.869 0.813  0.821  0.809  0.833  0.848  0.856 0.808  0.817  0.813  0.829  0.846  0.855 Table  2:  F-measure  scores  on  three  datasets  with  adopting different multi-modal fusion modules in Fig.|
|||3(b) - BPDC in Fig.4), the multi-modal fusion network is basically able to learn level-specific predictions.|
|||readily  and  the  optimization  objectives  for  shallow  layers are  degenerated  into  learning  complementary  low-level features only, thus easing the learning process and affording more  cooperative  multi-level  fusion.|
|||Nonetheless,  owing  to  that  the  multi-modal feature  fusion  component  is  still  implemented  by  direct concatenation, the Fig.|
|||This visualization reveals the contribution of each layer clearly and verifies the effectiveness of the proposed cross-level fusion strategy.|
|||pre-training stage However,  benefit fusion  of multi-modal  and  multi-level  features,  our  method  still achieves much better performance than the DF and CTMF models.|
|||In  these challenging  cases,  most  of  other  methods  are  unlikely  to locate  the  salient  object  due  to  the  lack  of  high-level contextual reasoning or robust multi-modal fusion strategy.|
|||Although the CTMF method is able to obtain more correct and uniform saliency maps  than  others,  the  fine  details  of the salient objects are lost severely due to the deficiency of cross-level fusion.|
|||The introduced  cross-modal/level  connections  and  modal/ level-wise supervisions explicitly encourage the capture of complementary  information  from  the  counterpart,  thus reducing fusion ambiguity and increasing fusion sufficiency.|
|||Comprehensive experiments demonstrate the effectiveness of  the  proposed  multi-modal  multi-level  fusion  strategies, which  may  also  benefit  other  RGB-D  systems  and  even other multi-model fusion problems.|
|||Saliency  detection  for  stereoscopic  images  based  on  depth confidence  analysis  and  multiple  cues  fusion.|
|568|Learning Deep Context-Aware Features Over Body and Latent Parts for Person Re-Identification|The proposed model consists three components:the global body-based featurelearning with MSCAN, the latent pedestrian parts localization with spatial transformer networks and local part-based feature embedding,the fusion of full body and body parts for multi-class person identification tasks.|
|||In thissection, we introduce our model from four aspects: a multi-scale context-aware network for efficient feature learning(Section 3.1), the latent parts learning and localization forbetter local part-based feature representation (Section 3.2),the fusion of global full-body and local body-part featuresfor person ReID (Section 3.3), and our final objective func-tion in Section 3.4.|
|||As shown in Table 2, the fusion model of full body andbody parts improves Rank-1 identification rate by more than4.00% compared with the body and parts-based models sep-arately in single query.|
|||Compared with metric learning methods, such as thestate-of-the-art approach DNS, the proposed fusion mod-el improves the Rank-1 identification rate by 11.66% and13.29% on the labeled and detected datasets respectively.|
|||Compared with the similar multi-class person identificationnetwork DGD, the Rank-1 identification rate improves by1.63% using our fusion model on the labeled dataset.|
|||Ourfusion model improves Rank-1 identification rate and mAPby 6.47% and by 8.45% in single query.|
|||Our fusion-based model obtains better Rank-1 identifica-tion rate than existing deep models, e.g.|
|||For better understanding the learned pedestrian parts, wevisualize the localized latent parts in Figure 4 using ourfusion model.|
|||Instead, we conductcross-dataset evaluation from the pretrained model on theIn this work, we have studied the problem of person ReI-D in three levels: 1) a multi-scale context-aware network tocapture the context knowledge for pedestrian feature learn-ing, 2) three novel constraints on STN for effective laten-t parts localization and body-part feature representation, 3)the fusion of full-body and body-part identity discriminativefeatures for powerful pedestrian representation.|
|569|Efficient Global Point Cloud Alignment Using Bayesian Nonparametric Mixtures|Kinectfusion: Real-time dense surfacemapping and tracking.|
|||Real-time large scale dense RGB-DSLAM with volumetric fusion.|
|570|Long_Zhao_Learning_to_Forecast_ECCV_2018_paper|Compared with multi-scale feature fusion in [14]where feature maps are only concatenated to the last layer of the network, ourdense connections upsample and concatenate feature maps with different scalesto all intermediate layers.|
|571|cvpr18-Towards Universal Representation for Unseen Action Recognition|Zero-shotlearning using synthesised unseen visual data with diffusionregularisation.|
|572|Minho_Shim_Teaching_Machines_to_ECCV_2018_paper|In a baseball game, these situations may occur simultaneously with other situ-ations leading to the confusion.|
|||Confusion matrix for CNN+GRU result.|
|||BallStrikeFoulSwing and a missFly outGround outOne-base hitStrike outHome inBase on ballsTouch outTwo-base hitHomerunFoul fly outDouble playTag outStealing baseInfield hitLine-drive outErrorDeadballBunt foulWild pitchSacrifice bunt outCaught stealingThree-base hitBunt hitBunt outPassed ballPickoff outPredicted labelBallStrikeFoulSwing and a missFly outGround outOne-base hitStrike outHome inBase on ballsTouch outTwo-base hitHomerunFoul fly outDouble playTag outStealing baseInfield hitLine-drive outErrorDeadballBunt foulWild pitchSacrifice bunt outCaught stealingThree-base hitBunt hitBunt outPassed ballPickoff outTrue labelNormalized confusion matrix0.00.20.40.60.81.0Large-Scale Baseball Video Database for Multiple Video Understanding Tasks15References1.|
|||Feichtenhofer, C., Pinz, A., Zisserman, A.: Convolutional two-stream network fusion forvideo action recognition.|
|573|cvpr18-End-to-End Learning of Motion Representation for Video Understanding|Convolutionaltwo-stream network fusion for video action recognition.|
|574|Keisuke_Tateno_Distortion-Aware_Convolutional_Filters_ECCV_2018_paper|3D shape recovery from single 360 image Approaches to recover 3D shape and se-mantic from a single equirectangular image by geometrical fusion have been exploredin [27][26].|
|575|cvpr18-Who's Better  Who's Best  Pairwise Deep Ranking for Skill Determination|To fuse the spatial and tempo-ral networks for all snippets we take the weighted averageof the outputs,f (pi) =1Xk=1fs(pki ) + (1  )ft(pki )(9)where  is the fusion weighting between spatial and tempo-ral information, and  is the number of testing snippets.|
|||We assess the sensitivity of our resultsto the late fusion weighting  in Equation 9.|
|||These two results are then combined withlate fusion of  = 0.4.|
|||The results from each networkare then fused using late fusion with  = 0.4.|
|||Further work involves exploringmid-level fusion between the two streams of the network, aswell as testing on additional and across datasets and tasks.|
|576|Kuan-Chuan_Peng_Zero-Shot_Deep_Domain_ECCV_2018_paper|We also extend ZDDA toperform sensor fusion in the SUN RGB-D scene classification task by sim-ulating task-relevant target-domain representations with task-relevantsource-domain data.|
|||To the best of our knowledge, ZDDA is the firstdomain adaptation and sensor fusion method which requires no task-relevant target-domain data.|
|||Keywords: zero-shot  domain adaptation  sensor fusion1 IntroductionThe useful information to solve practical tasks often exists in different domainscaptured by various sensors, where a domain can be either a modality or adataset.|
|||We propose zero-shot deep domain adaptation (ZDDA) for domain adapta-tion and sensor fusion.|
|||Such impractical assumption is also assumed true in the existingworks of sensor fusion such as [31, 48], where the goal is to obtain a dual-domain(source and target) TOI solution which is robust to noise in either domain.|
|||Thisunsolved issue motivates us to propose zero-shot deep domain adaptation (ZD-DA), a DA and sensor fusion approach which learns from the task-irrelevantdual-domain training pairs without using the task-relevant target-domain train-ing data, where we use the term task-irrelevant data to refer to the data which isnot task-relevant.|
|||(2)Given no task-relevant target-domain training data, we show that ZDDAcan perform sensor fusion and that ZDDA is more robust to noisy testing data ineither source or target or both domains compared with a naive fusion approachin the scene classification task from the SUN RGB-D [36] dataset.|
|||Although different strategies such as the domain adversarial loss [40] and thedomain confusion loss [39] are proposed to improve the performance in the DAtasks, most of the existing methods need the T-R target-domain training data,which can be unavailable in reality.|
|||In terms of sensor fusion, Ngiam et al.|
|||[31] define the three components formultimodal learning (multimodal fusion, cross modality learning, and sharedrepresentation learning) based on the modality used for feature learning, su-pervised training, and testing, and experiment on audio-video data with theirproposed deep belief network and autoencoder based method.|
|||Although certain progress about sensor fusion is achieved inthe previous works [31, 48], we are unaware of any existing sensor fusion methodwhich overcomes the issue of lacking T-R target-domain training data, which isthe issue that ZDDA is designed to solve.|
|||2) Sensor fusion: Given the previousassumption, derive the solution of TOI when the testing data in both Ds and Dtis available.|
|||ZDDA simulates the target-domain representationusing the source-domain data, builds a joint network with the supervision from thesource domain, and trains a sensor fusion network.|
|||2, where we simulate the RGB representation using the depth image, builda joint network with the supervision of the TOI in depth images, and train asensor fusion network in step 1, step 2, and step 3 respectively.|
|||t can also be trainable in step 2, but given our limited6K.-C. Peng, Z. Wu, and J. Ernst(a) testing domain adaptation(b) testing sensor fusionFig.|
|||To perform sensor fusion, we propose step 3, where we train a joint classifierfor RGB-D input using only the T-R depth training data.|
|||For sensor fusion, we experiment on the SUNRGB-D [36] dataset.|
|||Forthe baseline of sensor fusion, we compare ZDDA3 with a naive fusion method bypredicting the label with the highest probability from CRGB and CD in Sec.|
|||4b) outperforms the naive fusionmethod (Fig.|
|||Traditionally, training a fusion model requires theT-R training data in both modalities.|
|||However, we show that without the T-Rtraining data in the RGB domain, we can still train an RGB-D fusion model, andthat the performance degrades smoothly when the noise increases.|
|||In additionto using black images as the noise model, we evaluate the same trained jointclassifier in ZDDA3 using another noise model (adding a black rectangle with arandom location and size to the clean image) at testing time, and the result alsosupports that ZDDA3 outperforms the naive fusion method.|
|||Although we only14K.-C. Peng, Z. Wu, and J. Ernst(a) naive fusion(b) ZDDA3(c) accuracy diff.|
|||Performance comparison between the two sensor fusion methods with blackimages as the noisy images.|
|||We compare the classification accuracy (%) of (a) naivefusion and (b) ZDDA3 under different noise levels in both RGB and depth testing data.|
|||(c) shows that ZDDA3 outperforms the naive fusion under most conditionsuse black images as the noise model for ZDDA3 at training time, we expect thatadding different noise models can improve the robustness of ZDDA3.|
|||6 Conclusion and Future WorkWe propose zero-shot deep domain adaptation (ZDDA), a novel approach toperform domain adaptation (DA) and sensor fusion with no need of the task-relevant target-domain training data which can be inaccessible in reality.|
|||Experimenting on theMNIST [27], Fashion-MNIST [46], NIST [18], EMNIST [9], and SUN RGB-D [36]datasets, we show that ZDDA outperforms the baselines in DA and sensor fusioneven without the task-relevant target-domain training data.|
|577|Matthew_Trumble_Deep_Autoencoder_for_ECCV_2018_paper|Data-drivenvolumetric SR has been explored using multiple image fusion across the depthof field in [15] and across multiple spectral channels in [6].|
|578|cvpr18-Bilateral Ordinal Relevance Multi-Instance Regression for Facial Action Unit Intensity Estimation|Facialaction units intensity estimation by the fusion of featureswith multi-kernel support vector machine.|
|579|Learning to Predict Stereo Reliability Enforcing Local Consistency of Confidence Maps|For instance, to improve stereo accuracy [6, 23, 22, 25, 30]or for depth sensor fusion [15, 18].|
|||Confidence measures can be used for several purposes;for instance to detect uncertain disparity assignments [23,27] and occlusions [10, 19], improve accuracy near depthdiscontinuities [6], improve overall disparity map accu-racy [12, 20, 8, 22, 25] and for sensor fusion [15, 18].|
|||Pixel weighted average strategy for depth sensordata fusion.|
|||Reliable fusion oftof and stereo depth driven by confidence measures.|
|||Real-time visibility-based fusion of depth maps.|
|||Deep stereo fusion: combiningIn Pro-multiple disparity hypotheses with deep-learning.|
|581|Spatio-Temporal Self-Organizing Map Deep Network for Dynamic Object Detection From Videos|The variation of whole frame and the variation of sin-gle pixel at different frames will simultaneously contributeto the threshold of each pixel, so the threshold of each pixelis obtained by fusion of spatial and temporal thresholds.|
|582|Ciprian_Corneanu_Deep_Structure_Inference_ECCV_2018_paper|In [12], different CNNsare trained on different parts of the face merging features in an early fusionfashion with fully connected layers.|
|||(b) Each fusion unit is a stack of 2 FC layers.|
|||Finally, py(y, p, ) is definedas the log probability of P (y|p, ) which is modeled by a set of independentfunctions, so called fusion functions {j(sj; j)}Nj=1, where sj  p correspondsto the set of j-th AU predictions from all patches and j is function parameters.|
|||During training we use supervision on the patch prediction p, the fusionf and the structure inference outputs y.|
|||On the fusion and structure inference outputs we apply a binary cross-entropyloss (denoted by L(f, y) and L(y, y)).|
|||i=1, y}i=1Training data: {{I}PModel parameters: patch prediction: {i}Pinference {i}NStep 0: random initialization around 0: , ,   N (0, 2)Step 1: train patch prediction: i  min(L (i(Ii; i)), y), i  {1, ..., P }Step 2: freeze patch prediction; train fusion:   min L((; ), y)Step 3: train patch prediction and fusion jointly:i=1, fusion {i}Ni=1, structure,   min,(L ((I; )), y) + L((; ), y))Step 4: freeze patch prediction and fusion; train structure inference:  min L((; ), y)Step 5. train all:, ,   min,,(w1L ((I; )), y) + w2L((; ), y) + w3L((; ), y))Output: optimized parameter: opt, opt, opta regularization on the correction factors (denoted by  in Eq.|
|||F stands for the fusion and DSIN is the final model.|
|||F stands for the fusion.|
|||2 and 3 show results of AU-wise fusion for BP4D andDISFA (PP+F).|
|||On both, patch learning through fusion is beneficial, but onDISFA benefits are higher.|
|||Overall on BP4Dthe fusion improves results on almost all AUs compared to face prediction.|
|||However, the fusion is not capable to replicate the result of the mouth predictionon AU14.|
|||On DISFA, in almost every case fusion gets close or higher to thebest patch prediction.|
|||In both cases, fusion has greater problems in improvingindividual patches in cases where input predictions are already very noisy.|
|||Adding the structureinference brings more than 5% improvement over the fusion.|
|||When we add patch prediction fusion (PP+F) we get just0.5% lower than ROI while the addition of the structure inference and thresholdtuning improves ROI performance.|
|||In the first 3 column examples, AU06 andAU07 are not correctly classified by the fusion model (middle row).|
|||8: (a) Examples of AU predictions: ground-truth (top), fusion module (middle)and structure inference (bottom) prediction (: true positive, : false positive).|
|583|Missing Modalities Imputation via Cascaded Residual Autoencoder|Most prior work on information fusion of multi-modaldata assumes that all modalities are available for every train-ing data point [10, 21].|
|||Recovered face cubes are fused into 2D face images usingthe spatiospectral fusion method [32], and the recognitionis performed by collaborative representation [36].|
|||Hyperspectral and LiDAR data fusion:Outcome of the 2013 GRSS data fusion contest.|
|||Ob-ject level HSI-LiDAR data fusion for automated detection ofdifficult targets.|
|||Hyperspectralface recognition with spatiospectral information fusion andPLS regression.|
|584|Yifei_Shi_PlaneMatch_Patch_Coplanarity_ECCV_2018_paper|Keller, M., Lefloch, D., Lambers, M., Izadi, S., Weyrich, T., Kolb, A.: Real-time 3dreconstruction in dynamic scenes using point-based fusion.|
|585|Binary Coding for Partial Action Analysis With Limited Observation Ratios|Data fusion through cross-modality metric learning us-ing similarity-sensitive hashing.|
|||Improving humanaction recognition using fusion of depth camera and inertialsensors.|
|586|Face Normals _In-The-Wild_ Using Fully Convolutional Networks|As in [32] we keep the task-specificmemory and computation budget low by applying linear op-erations within these skip layers, and fuse skip-layer resultsthrough additive fusion with learnt weights.|
|||The outputs of the different res-olutions are combined through an additional fusion schemethat delivers the final normal estimates.|
|587|Dinesh_Jayaraman_ShapeCodes_Self-Supervised_Feature_ECCV_2018_paper|Our network architecture naturally splits into four modular sub-networks with different functions: elevation sensor, image sensor, fusion, andfinally, a decoder.|
|||Together, the elevation sensor, image sensor, and fusion mod-ules process the observation and proprioceptive camera elevation information toproduce a single feature vector that encodes the full object model.|
|||The outputs of the image and elevation sensor modules are concatenatedand passed through a fusion module which jointly processes their information to4 omitting object indices throughout to simplify notation.|
|||6D. Jayaraman, R. Gao, and K. Grauman13232image sensor3215764256input view5532551555725633max-poolReLU(3x3, stride2)avg-poolReLU(3x3, stride2)ReLUavg-pool(3x3)fc1 (fullyconnected)    ReLUfusion25625616fc2fc3ReLU ReLU16elevation sensordecoderoutput viewgridedoCepahS256128(M azimuths)x(N elevations)81632644481632Leaky ReLULeaky ReLULeaky ReLULeaky ReLUShapeCode feature extractorViewgrid decoder (training only)Fig.|
|||A single view of an object (top left) and the corre-sponding camera elevation (bottom left) are processed independently in image sensorand elevation sensor neural net modules, before fusion to produce the ShapeCode rep-resentation of the input, which embeds the 3D object shape aligned to the observedview.|
|||Then, to apply our net-work to novel examples, the representation of interest is that same latent spaceoutput by the fusion module of the encoderthe ShapeCode.|
|||Specifically, for each new8D. Jayaraman, R. Gao, and K. Graumanclass-labeled image, we directly represent it in the feature space represented byan intermediate fusion layer in the network trained for reconstruction.|
|||Recall that the output of the fusion module in Fig 2, which is the fc3 featurevector, is trained to encode 3D shape.|
|588|Hengshuang_Zhao_PSANet_Point-wise_Spatial_ECCV_2018_paper|UNet [33] concatenated output from low-levellayers with higher ones for information fusion.|
|||We concatenate the new representations Zc and Zd and apply a convolutionallayer with batch normalization and activation layers for dimension reductionand feature fusion.|
|589|cvpr18-Resource Aware Person Re-Identification Across Multiple Resolutions|Furthermore, later layers in CNNs8043RGB 256x12864x32...Conv Block 1xStage 164Global Avg PoolingDown SamplingDown Sampling32x16...128Global Avg PoolingDown Sampling16x8...256Global Avg Pooling8x4...Conv Block 4512Global Avg PoolingStage 4Conv Block 3Stage 3Conv Block 2Stage 2`fusionWeghit edSumLinearsLinearsLinearsLinearsfusion(x)1(x)2(x)3(x)4(x)`1`2`3`4Figure 2.|
|||Different parts are trained jointly with loss lall = 4ls + lfusion.|
|||However, such fusion of multiple features will only beuseful if each individual feature vector is discriminativeenough for the task at hand.|
|||Given an image x, denote by s(x) the embedding pro-duced at stage s. We fuse these embeddings using a simpleweighted sum:fusion(x) =4s=1wss(x),(1)where the weights ws are learnable parameters.|
|||Loss functionThe loss function we use to train our network is the sumof per-stage loss functions ls operating on the embeddings(x) from every stage s and a loss function on the finalfused embedding fusion(x): lall =4For each loss function, we use the the triplet loss.|
|||s=1 ls + lfusion.|
|||There are two possible factors behindthis improvement: a) the fusion of information from mul-tiple layers, and b) deep supervision.|
|||The impact of fusion: Figure 3 shows the performanceof the different stages of DaRe, trained with random eras-ing and evaluated without re-ranking, on the Market-1501dataset.|
|||As expected, the error rate decreases as one goesdeeper into the network and the fusion of features from dif-ferent stages actually achieves the lowest error rate.|
|||Qualitative resultsTo gain a better understanding of how the features fromvarious stages differ in identifying people, and how thefusion helps, we visualize the retrieved images from fourcases in Figure 7 for which the fused representation clas-sifies the images correctly.|
|||Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|590|Chunze_Lin_Graininess-Aware_Deep_Feature_ECCV_2018_paper|The confusions, such as box-in-box detection, are suppressed with our scale-aware attention masks.|
|||Du, X., El-Khamy, M., Lee, J., Davis, L.: Fused dnn: A deep neural network fusionapproach to fast and robust pedestrian detection.|
|591|Samuel_Albanie_Learnable_PINs_Cross-Modal_ECCV_2018_paper|Cross-modal learning with faces and voices: In biometrics, an active re-search area is the development of multimodal recognition systems which seekto make use of the complementary signal components of facial images andspeech [7, 24], in order to achieve better performance than systems using a singlemodality, typically through the use of feature fusion.|
|||However,because face and voice representations are usually not aligned, in prior workthe query face cannot be directly compared to the audio track, necessitatingthe use of complex fusion systems to combine information from both modalites.|
|||We report results for 3 cases, retrieval using visual embeddingsalone, retrieval using audio embeddings alone, and a simple fusion method wherewe take the maximum score out of the two (i.e.|
|||Such a fusionmethod is useful for cases when one modality is a far stronger cue, e.g.|
|||Note how FV fusion allows more profile faces to be retrieved  row 2,second and fourth frames, and row 4, third ranked frame.|
|||We note that a superior fusion strategy could be applied in orderto better exploit this complementary information from both modalities (e.g.|
|592|Fabien_Baradel_Object_Level_Visual_ECCV_2018_paper|Our modeldetects a cell phone and a person but fails to detect hand-cell-phone contact; b) con-fusion between semantically similar objects (on the right).|
|593|Peng_Tang_Weakly_Supervised_Region_ECCV_2018_paper|Results fromleft to right are original images, response from the first to the fifth layers, and thefusion of responses from the second layer to the fourth layerThe first stage of our method is motivated by the intuition that CNNs trainedfor object recognition contain latent object location information.|
|594|Yue_Cao_Cross-Modal_Hamming_Hashing_ECCV_2018_paper|Bronstein, M., Bronstein, A., Michel, F., Paragios, N.: Data fusion through cross-In: CVPR, IEEEmodality metric learning using similarity-sensitive hashing.|
|595|cvpr18-Gated Fusion Network for Single Image Dehazing|The constructed network adopts a novel fusion-based strat-egy which derives three inputs from an original hazy im-age by applying White Balance (WB), Contrast Enhancing(CE), and Gamma Correction (GC).|
|||We exploit a gated fusion net-work for single image deblurring.|
|||The proposed neuralnetwork is built on a fusion strategy which aims to seam-lessly blend several input images by preserving only thespecific features of the composite output image.|
|||Another line of research tries to make use of a fusionprinciple to restore hazy images in [1, 5].|
|||In contrast, we introduce a gatedfusion based single image dehazing technique that blendsonly the derived three input images.|
|||Different from these CNNs based approaches, our pro-posed network is built on the principle of image fusion, andis learned to produce the sharp image directly without esti-mating transmission and atmospheric light.|
|||The main ideaof image fusion is to combine several images into a singleone, retaining only the most significant features.|
|||Gated Fusion NetworkThis section presents the details of our gated fusion net-work that employs an original hazy image and three derivedimages as inputs.|
|||By learning the con-fidence map for each input, we demonstrate that our fusionbased method is able to dehaze images effectively.|
|||We perform an early fusion by concatenating the originalhazy image and three derived inputs in the input layer.|
|||Figure 4 shows the proposed multi-scale fusion network,in which the coarsest level network is shown in Figure 2.|
|||(a) Hazy inputs(b) Without gating(c) Without fusion(d) GFNFigure 8.|
|||Effectiveness of the gated fusion network.|
|||Effectiveness of Gating StrategyImage fusion is a method to blend several images into asingle one by retaining only the most useful features.|
|||Consequently, in our gated fusion net-work, the derived inputs are gated by three pixel-wise confi-dence maps that aim to preserve the regions with good visi-bility.|
|||Our fusion network has two advantages: the first oneis that it can reduce patch-based artifacts (e.g.|
|||To show the effectiveness of fusion network, we alsotrain an end-to-end network without fusion process.|
|||In addition, we also conduct a ex-periment based on equivalent fusion strategy, i.e., all thethree derived inputs are weighted equally using 1/3.|
|||In these examples, the ap-proach without gating generates very dark images in Fig-ure 8(b), and the method without fusion strategy generatesresults with color distortion and dark regions as shown in(a) Hazy input(b) DCP [9](b) DehazeNet [4](d) GFNFigure 9.|
|||ConclusionsIn this paper, we addressed the single image dehazingproblem via a multi-scale gated fusion network (GFN),a fusion based encoder-decoder architecture, by learningconfidence maps for derived inputs.|
|||Single image dehazing by multi-scale fusion.|
|597|Ali_Diba_Spatio-Temporal_Channel_Correlation_ECCV_2018_paper|Feichtenhofer, C., Pinz, A., Zisserman, A.: Convolutional two-stream network fusion forvideo action recognition.|
|598|Zhenli_Zhang_ExFuse_Enhancing_Feature_ECCV_2018_paper|In this paper, we first point out that asimple fusion of low-level and high-level features could be less effectivebecause of the gap in semantic levels and spatial resolution.|
|||We findthat introducing semantic information into low-level features and high-resolution details into high-level features is more effective for the laterfusion.|
|||b) Introducing semantic information into low-level features or spatial information into high-level features benefits the feature fusion.|
|||Intuitively, the fusion of high-level features with such pure low-level featureshelps little, because low-level features are too noisy to provide sufficient high-resolution semantic guidance.|
|||In contrast, if low-level features include more se-mantic information, for example, encode relatively clearer semantic boundaries,then the fusion becomes easy  fine segmentation results could be obtained byaligning high-level feature maps to the boundary.|
|||Empirically, the semantic andresolution overlap between low-level and high-level features plays an importantrole in the effectiveness of feature fusion.|
|||In other words, feature fusion could beenhanced by introducing more semantic concepts into low-level features or byembedding more spatial information into high-level features.|
|||Motivated by the above observation, we propose to boost the feature fusionby bridging the semantic and resolution gap between low-level and high-levelfeature maps.|
|||bridging the semantic and resolution gap between low-level and high-levelfeatures by more effective feature fusion.|
|||Significant improvements are obtainedfrom the enhanced feature fusion.|
|||2 Related WorkFeature fusion in semantic segmentation.|
|||Feature fusion is frequently employedin semantic segmentation for different purposes and concepts.|
|||3 ApproachIn this work we mainly focus on the feature fusion problem in U-Net segmenta-tion frameworks [12, 2, 28, 26, 25, 22].|
|||A common way of feature fusion[27, 12, 14, 2, 28, 26, 22] is to formulate as a residual form:yl = U psample(yl+1) + F(xl)(1)where yl is the fused feature at l-th level; xl stands for the l-th feature generatedby the encoder.|
|||In Sec 1 we argue that feature fusion could become less effective if thereis a large semantic or resolution gap between low-level and high-level features.|
|||To examinethe effectiveness of feature fusion, we select several subsets of feature levels anduse them to retrain the whole system.|
|||It is clearthat even though the segmentation quality increases with the fusion of morefeature levels, the performance tends to saturate quickly.|
|||Especially, the lowesttwo feature levels (1 and 2) only contribute marginal improvements (0.24% forResNet 50 and 0.05% for ResNeXt 101), which implies the fusion of low-leveland high-level features is rather ineffective in this framework.|
|||To generate seman-tic outputs in the auxiliary branches, low-level features are forced to encode moresemantic concepts, which is expected to be helpful for later feature fusion.|
|||To address the drawback, we generalize the fusion as follows:yl = U psample (yl+1) + F(xl, xl+1, .|
|||Our insight is to involve more semanticinformation from high-level features to guide the resolution fusion.|
|||At the beginning of Sec 3 we demonstratethat feature fusion in our baseline architecture (GCN [26]) is ineffective.|
|||Despite the improved performance, a question raises: is feature fusion in theframework really improved?|
|||The comparison implies our insights and methodologyenhance the feature fusion indeed.|
|||Empiricallywe conclude that boosting high-level features not only benefits feature fusion,but also contributes directly to the segmentation performance.|
|||5 ConclusionsIn this work, we first point out the ineffective feature fusion problem in cur-rent U-Net structure.|
|||Eventually, better feature fusion is demonstrated by theperformance boost when fusing with original low-level features and the overallsegmentation performance is improved by a large margin.|
|599|Christos_Sakaridis_Semantic_Scene_Understanding_ECCV_2018_paper|: Single image defogging by multiscale depth fusion.|
|600|Optical Flow Requires Multiple Strategies (but Only One Network)|Optical flow with geometric oc-clusion estimation and fusion of multiple frames.|
|601|cvpr18-Monocular Relative Depth Perception With Web Stereo Data Supervision|Two alternatives can effectively obtain a finer predic-tion, one is dilated convolution [46] (or atrous convolution),U24x24Residual Conv+F48x48F96x96F192x192AResidual Conv2x upUpsamplingFeature fusion(b) Feature fusionResidual ConvULeRvnoC3x3ULeRvnoC3x3+(c) Residual ConvAdaptive output3x3Conv1283x3Conv12x up(d) Adaptive output384x384(a) Proposed networkFigure 4.|
|||(b) shows the process of multi-scale featurefusion, and (c) is a Residual Convolution module.|
|||and the other is multi-scale feature fusion [40, 47].|
|||We choose the output of the last layers ofindividual building blocks as one input to our multi-scalefeature fusion modules.|
|||Multi-scale feature fusion modulestake two groups of feature maps as input.|
|||For each feature fusion module, we first use a resid-ual convolution block to transfer feature maps from specificlayers of pre-trained ResNet for our task, and then mergewith fused feature maps that produced by last feature fusionmodule via summation.|
|602|Ying_Zhang_Deep_Cross-Modal_Projection_ECCV_2018_paper|: Learning a recurrent residual fusionnetwork for multimodal matching.|
|604|cvpr18-Generative Image Inpainting With Contextual Attention|The first group represents traditionaldiffusion-based or patch-based methods with low-level fea-tures.|
|||Traditional diffusion or patch-based approaches suchas [2, 4, 9, 10] typically use variational algorithms or patchsimilarity to propagate information from the background re-gions to the holes.|
|||Attention propagation We further encourage coherencyof attention by propagation (fusion).|
|605|Polarimetric Multi-View Stereo|For fair compari-son, we show the results after depth fusion for all the meth-ods.|
|||Massively parallelmultiview stereopsis by surface normal diffusion.|
|606|cvpr18-Deep Depth Completion of a Single RGB-D Image|Guided depth enhancement viaanisotropic diffusion.|
|||When can we use kinectfusion for groundtruth acquisition.|
|607|Using Ranking-CNN for Age Estimation|The training strategy can alsobe extended to ensemble learning with other decisionfusion methods.|
|608|FlowNet 2.0_ Evolution of Optical Flow Estimation With Deep Networks|Finally we apply a small fusion network to provide the final estimate.|
|||In some cases, this refinement can be approx-imated by neural networks: Chen & Pock [9] formulatetheir reaction diffusion model as a CNN and apply it to im-age denoising, deblocking and superresolution.|
|||Thefusion network receives the flows, the flow magnitudes andthe errors in brightness after warping as input.|
|||Trainable nonlinear reaction diffusion:A flexible framework for fast and effective image restora-tion.|
|609|CATS_ A Color and Thermal Stereo Benchmark|Night-time pedestrian detection byvisual-infrared video fusion.|
|||Background-subtraction usingcontour-based fusion of thermal and visible imagery.|
|||An iterative in-tegrated framework for thermalvisible image registration,sensor fusion, and people tracking for video surveillanceapplications.|
|||Orientation-based face recognition using mul-tispectral imagery and score fusion.|
|||A new sensorfusion framework to deal with false detections for low-costservice robot localization.|
|||Thermal-visible videofusion for moving target tracking and pedestrian classifica-tion.|
|610|cvpr18-Learning Deep Structured Active Contours End-to-End|However, even the latestevolutions struggle to precisely delineating borders, whichoften leads to geometric distortions and inadvertent fusionof adjacent building instances.|
|611|cvpr18-LiDAR-Video Driving Dataset  Learning Driving Policies Effectively|For the point clouds ob-tained by our fusion algorithm are stored in PCD format, weemployed a standard software to transform data into LASformat, which is an industry standard for LiDAR data.|
|||This hidden layer is fully connected to fusionnetwork, which outputs final driving behavior prediction.|
|||Different from DNN-only, we replace fusionnetwork with stacked LSTM nets in DNN-LSTM frame-work.|
|612|Reflection Removal Using Low-Rank Matrix Completion|Double lowrank matrix recovery for saliency fusion.|
|613|Weiyue_Wang_Depth-aware_CNN_for_ECCV_2018_paper|[5] propose alocality-sensitive deconvolution network with gated fusion.|
|||Cheng, Y., Cai, R., Li, Z., Zhao, X., Huang, K.: Locality-sensitive deconvolutionnetworks with gated fusion for rgb-d indoor semantic segmentation.|
|||Hazirbas, C., Ma, L., Domokos, C., Cremers, D.: Fusenet: incorporating depth intosemantic segmentation via fusion-based cnn architecture.|
|||Li, Z., Gan, Y., Liang, X., Yu, Y., Cheng, H., Lin, L.: Lstm-cf: Unifying contextmodeling and fusion with lstms for rgb-d scene labeling.|
|||Park, S.J., Hong, K.S., Lee, S.: Rdfnet: Rgb-d multi-level residual feature fusionfor indoor semantic segmentation.|
|614|cvpr18-Textbook Question Answering Under Instructor Guidance With Memory Networks|[7], we propose new categories of contradictions that un-derline the fusion of small facts among different parts of theessays and images.|
|615|Yantao_Shen_Person_Re-identification_with_ECCV_2018_paper|Different from conventional GNN approaches, SGGNN learnsthe edge weights with rich labels of gallery instance pairs directly, whichprovides relation fusion more precise information.|
|||Unlike most previousGNNs designs, in SGGNN, the weights for feature fusion are determined by sim-ilarity scores by gallery image pairs, which are directly supervised by traininglabels.|
|||With these similarity guided feature fusion weights, SGGNN will fullyexploit the valuable label information to generate discriminative person imagefeatures and obtain robust similarity estimations for probe-gallery image pairs.|
|||(2) Different from mostGraph Neural Network (GNN) approaches, SGGNN exploits the training labelsupervision for learning more accurate feature fusion weights for updating thenodes features.|
|||This similarity guided manner ensures the feature fusion weightsto be more precise and conduct more reasonable feature fusion.|
|||Different from most existing GNN approaches, our proposed approach exploitsthe training data label supervision for generating more accurate feature fusionweights in the graph message passing.|
|||With gallery-gallery similarity scores,the probe-gallery relation feature fusion could be deduced as a message passing andfeature fusion schemes, which is defined as Eq.|
|||The node features are then updated as a weighted addition fusion of all8Y. Shen, H. Li, S. Yi, D. Chen and X. Wanginput messages and the nodes original features.|
|||This relation feature fusioncould be deduced as a message passing and feature fusion scheme.|
|||After obtaining the edge weights Wij and deep message ti from each node,the updating scheme of node relation feature di could be formulated asd(1)i = (1  )d(0)i + Wij t(0)jNXj=1for i = 1, 2, ..., N,(4)denotes the i-th refined relation feature, d(0)iiwhere d(1)relation feature and t(0)weighting parameter that balances fusion feature and original feature.|
|||denotes the i-th inputj denotes the deep message from node j.  represents theNoted that such relation feature weighted fusion could be performed itera-tively as follows,d(t)i = (1  )d(t1)i+ NXj=1Wijt(t1)jfor i = 1, 2, ..., N,(5)where t is the iteration number.|
|||(4) as our relation feature fusion in both training and testingstages.|
|||Person Re-ID with Deep Similarity-Guided Graph Neural Network93.3 Relations to Conventional GNNIn our proposed SGGNN model, the similarities among gallery images are servedas fusion weights on the graph for nodes feature fusion and updating.|
|||In conventionalGNN [66, 45] models, the feature fusion weights are usually modeled as a non-linear function h(di, dj) that measures compatibility between two nodes di anddj.|
|||To overcome such limitation,we propose to use similarity scores S(gi, gj) between gallery images gi and gj withdirectly training label supervision to serve as the node feature fusion weights inEq.|
|||(6), these direct andrich supervisions of gallery-gallery similarity could provide feature fusion withmore accurate information.|
|||HydraPlus-Net [39] is proposed for better exploiting the global and local con-tents with multi-level feature fusion of a person image.|
|||We also validate the importance of learning visual fea-ture fusion weight with gallery-gallery similarities guidance.|
|||We therefore remove thedirectly gallery-gallery supervisions and train the model with weight fusion ap-proach in Eq.|
|||For conventional Graph Neural Networksetting, the rich gallery-gallery similarity labels are ignored while our approachutilized all valuable labels to ensure the weighted deep message fusion is moreeffective.|
|||Spindlenet: Person re-identification with human body region guided feature decomposi-tion and fusion.|
|616|cvpr18-Learning 3D Shape Completion From Laser Scan Data With Weak Supervision|Oct-NetFusion: Learning depth fusion from data.|
|617|cvpr18-Attention Clusters  Purely Attention Based Local Feature Integration for Video Classification|[14] studied multiple fusionmethods based on pooling local spatio-temporal features ex-tracted by 2D CNNs from RGB frames.|
|||On UCF101 and HMDB, our approach obtains robustimprovements over the two-stream fusion results for CNNs.|
|||The im-plemented three stream fusion methods also act as a strongbaseline.|
|||Wealso find that our results can beat other fusion methods us-ing the same local features.|
|||We alsoimplement a series of three stream fusion methods using theReferences[1] S. Abu-El-Haija, N. Kothari, J. Lee, P. Natsev, G. Toderici,B. Varadarajan, and S. Vijayanarasimhan.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|||Bag of visualwords and fusion methods for action recognition: Compre-hensive study and good practice.|
|618|Robust Joint and Individual Variance Explained|Dynamic prob-abilistic cca for analysis of affective behavior and fusion ofcontinuous annotations.|
|619|cvpr18-DoubleFusion  Real-Time Capture of Human Performances With Inner Body Shapes From a Single Depth Sensor|Therefore, we propose a pipeline that executes joint motiontracking, geometric fusion and volumetric shape-pose op-timization sequentially (Fig.|
|||Geometric fusion Similar to previous work [28], we non-rigidly integrate depth observation of multiple frames in areference volume (Sec.|
|||4), geometric fusion (Sec.|
|||Geometric FusionSimilar to the previous non-rigid fusion works [28, 15,14], we integrate the depth information into a reference vol-ume.|
|||We follow the work [14] tocope with collided voxels in live frame to prevent erroneousfusion results caused by collisions.|
|||ResultsAfter the non-rigid fusion, we have an updated surface inthe canonical volume with more complete geometry.|
|||The geometric fusion takes 6 ms7292Figure 5: Example results reconstructed by our system.|
|||The lack of semantic information re-sults in wrong connections (connection between two legs)and erroneous fusion results as shown in Fig.|
|||Only using all the energy terms we can get accuratepose and fusion results as shown in Fig.|
|||Dynamicfusion:Reconstruction and tracking of non-rigid scenes in real-time.|
|||Bodyfusion: Real-time capture of human motionand surface geometry using a single depth camera.|
|620|Daniel_Maurer_Structure-from-Motion-Aware_PatchMatch_for_ECCV_2018_paper|Among the most popular ap-proaches that are considered useful as initialization are EpicFlow [30], Coarse-to-fine PatchMatch [15] and DiscreteFlow [25]  approaches that rely on theinterpolation or fusion of feature matches.|
|||Galliani, S., Lasinger, K., Schindler, K.: Massively parallel multiview stereopsis bysurface normal diffusion.|
|621|cvpr18-Dynamic Scene Deblurring Using Spatially Variant Recurrent Neural Networks|Leaky ReLU with negative slope 0.1 is also added afterevery convolution layer in the feature extraction network,RNN fusion and image reconstruction network, except forthe last convolution layer in the whole network.|
|622|cvpr18-Augmented Skeleton Space Transfer for Depth-Based Hand Pose Estimation|The 3D pose estimates are then constructedby applying 2D convolutional neural networks (CNNs) to eachviews followed by multiple view scene fusion.|
|623|Unsupervised Learning of Long-Term Motion Dynamics for Videos|The confusion matrix for action recognition on MSR-DailyActivity3D dataset [14].|
|||Table 3 presents classi-fication accuracy on the MSRDailyActivity3D dataset [14]and Figure 7 its confusion matrix.|
|||We combine the softmax score from our model withthe semantic softmax score by late fusion.|
|624|cvpr18-Video Captioning via Hierarchical Reinforcement Learning|De-scribing videos using multi-modal fusion.|
|625|cvpr18-Scale-Transferrable Object Detection|Although DSSD improves accuracy comparedto SSD, the speed of the object detector has been greatlydamaged due to the extremely deep base network and inef-ficient feature fusion.|
|||This is mainly dueto the fact that the base network of DSSD is too deep and itsfeature fusion method is inefficient.|
|626|Spatiotemporal Multiplier Networks for Video Action Recognition|The most closely related work to ours is the two-streamConvNet architecture [28], which initially processes colourand optical flow information in parallel for subsequent latefusion of their separate classification scores.|
|||Extensionsto that work that investigated convolutional fusion [9] andresidual connections [8] are of particular relevance for thecurrent work, as they serve as points of departure.|
|||Each stream performs video recognition on its ownand prediction layer outputs are combined by late fusion forfinal classification.|
|||Connecting the two streamsThe original two-stream architecture only allowed thetwo processing paths to interact via late fusion of their re-spective softmax predictions [28].|
|||We conjecture that the decrease in performance isdue to the large change of the input distribution that the lay-ers in one network stream undergo after injecting a fusionsignal from the other stream.|
|||3.2.3 DiscussionInclusion of the multiplicative interaction increases theorder of the network fusion from first to second order[10].|
|||Dur-ing backpropagation, instead of the fusion gradient flow-ing through the appearance, (3), and motion, (4), streamsbeing distributed uniformly due to additive forward inter-action (2), it now is multiplicatively scaled by the oppos-ing streams current inputs, f (xml in equations (6)and (7), respectively.|
|||Inprevious work, different fusion functions have been dis-cussed [9] where it has been shown that additive fusionperformed better than maxout or concatenation of featurechannels from the two paths.|
|||Additive fusion of ResNetshas been used in [8], but was not compared to alternatives.|
|||Table 2lists the type of connection (direct or into residual units), thefusion function (additive +(cid:13) or multiplicativeJ), the direc-tion (from the motion into the appearance stream , con-versely  or bidirectional ).|
|||This overly aggressive change is induced in two ways:via the forwarded signal as it passes through the deep lay-ers; via the backpropagated signal in all preceding layersthat emit the fusion signal.|
|||This variation again leads to inferior results, bothfor additive and multiplicative residual fusion.|
|||Late fusion is implemented by averaging the pre-diction layer outputs.|
|||As a final experiment, we are interested if there isstill something to gain from a fusion with hand-craftedIDT features [37].|
|||These results indicate that thedegree of complementary between hand-crafted representa-tions and our end-to-end learned ConvNet approach is van-ishing for UCF101, given the fact that other representationssee much larger gains by fusion with IDT.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|627|Deeply Supervised Salient Object Detection With Short Connections|Besides, a weighted-fusion layer is added to better capture the advantage of eachside output.|
|||The fusion loss at the fusion layer can be ex-pressed asLfuse(W, w, f ) = (cid:0)Z, h(MXm=1fmA(m)side )(cid:1),(3)where f = (f1, .|
|||, fM ) is the fusion weight, A(m)side areactivations of the mth side output, h() denotes the sig-moid function, and (, ) denotes the distance between the(4)Lfinal(cid:0)W, w, f ) = Lfuse(cid:0)W, w, f ) + Lside(cid:0)W, w).|
|||Similar to [49], we add a weighted-fusion layer to con-nect each side activation.|
|||The loss function at the fusionlayer in our case can be represented by), where A(m)jjLfuse(W, w, f ) = (cid:0)Z,PMm=1 fm A(m)side(cid:1),(6)3205(b)(d)Hidden LayerLoss Layer(a)(c)88pool5conv5_31616conv4_33232conv3_36464conv2_2128128conv1_2256256side outputs256256Fusion loss646411 conv6464CE loss2upCross entropy (CE) loss12812811 conv12812811 convCE lossFusion weightShort connectionFigure 4: Illustration of short connections in Fig.|
|||The new sideiloss function and fusion loss function can be respectivelyrepresented by Lside(W,  w, r) =MXm=1m l(m)side(cid:0)W,  w(m), r(cid:1)(8) Lfuse(W,  w, f , r) = (cid:0)Z,PMm=1 fm  R(m)side(cid:1),(9)i }, i > m. Note that this time  l(m)where r = {rmside representsthe standard cross-entropy loss which we have defined inEqn.|
|||Therefore, the fusion output mapand the final output map can be computed by4 Zfuse = h(cid:0)Xm=2fm  R(m)side(cid:1), Zfinal = Mean(  Zfuse,  Z2, Z3, Z4).|
|||Our fusion layer weights are allinitialized with 0.1667 in the training phase.|
|||Albeit the fusion predictionmap gets denser, some non-salient pixels are wrongly pre-dicted as salient ones even though the CRF is used there-after.|
|||Fast saliency-awaremulti-modality image fusion.|
|628|Multi-Level Attention Networks for Visual Question Answering|incorpo-rate a powerful feature fusion method into visual attention,and achieve impressive results in VQA task.|
|||However, theyhave to keep a much higher dimension after fusion at costof more computation and storage.|
|||The two-level atten-tion is combined by fusion of their attended representation.|
|||tion that element-wise multiplication is a better multimodalfusion approach than addition in visual question answer-ing task.|
|||However, their method usesa much higher dimension fusion method (16,000 dim v.s.|
|||Future work includesfurther exploring on spatial encoding with context informa-tion, attention on sentence-level representation and betterfusion methods to join different level attention.|
|629|cvpr18-SketchMate  Deep Hashing for Million-Scale Human Sketch Retrieval|Two-branch Late-fusion As illustrated in Figure 2, ourtwo-branch encoder consists of three sub-modules: (1) aCNN encoder takes in a raster pixel sketch and translatesinto a high-dimensional space; (2) a RNN encoder takes in avector sketch and outputs its final time-step state; (3) branchinteraction via a late-fusion layer by concatenation.|
|||Quantization Encoding layer After the final fusionlayer, we have to encode that deep feature into the low-dimensional real-valued hashing feature fn (one fully con-nected layer with sigmoid activation), which will be furthertransformed to the hashing code, bn.|
|||Limited by this, wetrain a smaller model and use 256d deep feature (extractedfrom the fusion layer) as inputs.|
|||Generalization to Sketch RecognitionTo validate the generality of our sketch-specific design,we apply our two-branch CNN-RNN network to sketchrecognition task, by directly adding a 2048d fully con-nected layer after joint fusion layer and before the 345-way classification layer.|
|630|Jie_Zhang_Geometric_Constrained_Joint_ECCV_2018_paper|: Real-time traversable surface detec-tion by colour space fusion and temporal analysis.|
|631|POSEidon_ Face-From-Depth for Driver Pose Estimation|The coreof the proposal is a regressive neural network, called PO-SEidon, which is composed of three independent convolu-tional nets followed by a fusion layer, specially conceivedfor understanding the pose by depth.|
|||POSEidonThe POSEidon network is mainly obtained as a fusionof three CNNs and has been developed to perform a regres-sion on the 3D pose angles.|
|||A fusion step combines the contributionsof the three above described networks: in this case, the lastfully connected layer of each component is removed.|
|||Dif-ferent fusion approaches that have been proposed [42] areinvestigated.|
|||After the fusion step, three fully connectedlayers composed of 128, 84 and 3 activations respectivelyand two dropout regularization ( = 0.5) complete the ar-chitecture.|
|||last row highlights the best performance reached using convfusion of couples of input types, followed by the concatstep.|
|||Even if the choice of the fusion method has a limitedeffect (as deeply investigated in [42, 22]), the most signif-icant improvement of the system is reached exploiting thethree input types together.|
|||Convolu-tional two-stream network fusion for video action recogni-tion.|
|632|MDNet_ A Semantically and Visually Interpretable Medical Image Diagnosis Network|Multimodal mapping for knowledge fusion Image featuredescriptions in diagnostic reports contain strong underlyingsupports for diagnostic conclusion inference.|
|633|cvpr18-GVCNN  Group-View Convolutional Neural Networks for 3D Shape Recognition|As mentioned above, the group module not only decideswhich group each view belongs to, but also determines theweight of each group when conducting group fusion.|
|||Thus we define the weight of group Gj as:(Gj) =Ceil((Ik)  |Gj|)|Gj|Ik  Gj(2)In this way, we can have both the grouping scheme (withgroup information) and the grouping weights, which can beused for the following intra-group view pooling and groupfusion procedures.|
|||Therefore, weconduct a weighted fusion process using all group descrip-tors according to Eq.2 to get the final 3D shape descriptorD(S)D(S) =Mj=1(Gj)D(Gj)Mj=1(Gj).|
|||In GVCNN, the shape descriptor comesfrom the output of group fusion module, which is more rep-resentative than the view descriptor extracted from singleview.|
|||And the positionof its view pooling layer is the same as the fusion module of GVCNN.|
|||Therefore,the weighted fusion leads to better performance comparedto direct pooling on all views.|
|634|Richer Convolutional Features for Edge Detection|At last, a cross-entropy loss / sigmoid layer isfollowed to get the fusion loss / output.|
|||We show3002imagestage 133-64 conv11-21 conv33-64 conv11-21 conv22 poolstage 233-128 conv11-21 conv33-128 conv11-21 conv22 poolstage 333-256 conv11-21 conv33-256 conv11-21 conv33-256 conv11-21 conv22 poolstage 433-512 conv11-21 conv33-512 conv11-21 conv33-512 conv11-21 conv22 poolstage 533-512 conv11-21 conv33-512 conv11-21 conv33-512 conv11-21 conv11-1 convdeconvloss/sigmoid11-1 convdeconvloss/sigmoid11-1 convdeconvloss/sigmoidfusionconcat11-1 convloss/sigmoidFigure 2: Our RCF network architecture.|
|||Therefore, our im-proved loss function can be formulated asL(W ) =|I|Xi=1(cid:16)KXk=1l(X (k)i; W ) + l(X f usei; W )(cid:17),(3)iwhere X (k)is the activation value from stage k whileX f useis from fusion layer.|
|||The weights of 11 conv layer in fusion stage are initializedto 0.2 and the biases are initialized to 0.|
|635|Fully Convolutional Instance-Aware Semantic Segmentation|The FCNs are extended with globalcontext [28], multi-scale feature fusion [4], and deconvo-lution [31].|
|636|Self-Supervised Video Representation Learning With Odd-One-Out Networks|As odd-one-out task requires a comparisonamong (N+1) elements of the given question and cannotbe solved by only looking at individual elements, we in-troduce a fusion layer which merges the information from(N+1) branches after the first fully connected layer.|
|||Thesefusion layers help the network to perform reasoning aboutelements in the question to find the odd one.|
|||We experiment with two fusion models, the Concatenationmodel and sum of difference model leading to two differentnetwork architectures as shown in Fig.|
|||The sum of difference model architecture (see sec-tion 3) is our default activation fusion method.|
|||Secondly, we use sum of difference (SOD) asthe fusion method instead of simply concatenating (CON)the activations in our multi-branch network architecture.|
|||First, we evaluatethe impact of using 128 dimensional activations comparedto 4096 using sum of difference model as the fusion method.|
|||differences (SOD) fusion.|
|637|ScanNet_ Richly-Annotated 3D Reconstructions of Indoor Scenes|We use volumetric fusion [11]to perform the dense reconstruction, since this approachis widely used in the context of commodity RGB-D data.|
|||Semanticfusion: Dense 3d semantic mapping with convo-lutional neural networks.|
|638|Temporal Action Localization by Structured Maximal Sums|Ablation Studyoverlap threshold 0.10.210.228.832.642.245.045.20.34.523.227.833.036.236.518.540.542.548.050.751.0Separate networks46.247.646.051.040.344.043.245.231.535.632.836.5Baselinew/o cls + smew/o clsw/o smew/o priorOurs (full)spatialmotionlate fusionOurs (full)0.41.816.419.624.827.427.823.225.824.027.80.50.213.215.716.217.517.816.016.914.517.8Table 2: Ablation experiments for the structured objective(top) and the two-stream architecture (bottom).|
|||We evaluateusing each stream individually (spatial and motion), as wellas simply averaging the two streams rather than fine-tuningjointly (late fusion).|
|||We find that the full model outperformslate fusion, suggesting the importance of joint training ofthe two streams.|
|||The late fusion does not outperform sepa-rate networks, suggesting an incompatibility of confidencescores from the two networks.|
|639|Chao_Li_ArticulatedFusion_Real-time_Reconstruction_ECCV_2018_paper|Compared to previous fusion-based dynamic scene reconstruc-tion methods, our experiments show robust and improved reconstructionresults for tangential and occluded motions.|
|||Recently, volumetric depth fu-sion methods for dynamic scene reconstruction, such as DynamicFusion [17],VolumeDeform [10], Fusion4D [5] and albedo based fusion [8] open a new gatefor people in this field.|
|||Among all these works, fusion methods by a single depthcamera [17, 10] are more promising for popularization, because of their low costand easy setup.|
|||In this paper, we propose to add articulated motion priorinto the depth fusion system.|
|||In-tegrating the articulated motion prior into the reconstruction framework assistsin the non-rigid surface registration and geometry fusion, while surface registra-tion results improve the quality of segmentation and its reconstructed motion.|
|||We present ArticulatedFusion, a system that involves registration, segmen-tation, and fusion, and enables real-time reconstruction of motion, geometry,and segmentation for dynamic scenes of human and non-human subjects.|
|||The orange box represents our two-level node motionoptimization, and the blue box represents fusion of depth and node graph segmentation.|
|||For newly added nodes after depthfusion, their cluster belongings are determined by their closest existing neighbornodes.|
|||Pons-Moll, G., Baak, A., Helten, T., M uller, M., Seidel, H.P., Rosenhahn, B.:Multisensor-fusion for 3D full-body human motion capture.|
|||Yu, T., Guo, K., Xu, F., Dong, Y., Su, Z., Zhao, J., Li, J., Dai, Q., Liu, Y.:Bodyfusion: Real-time capture of human motion and surface geometry using asingle depth camera.|
|||Yu, T., Zheng, Z., Guo, K., Zhao, J., Dai, Q., Li, H., Pons-Moll, G., Liu, Y.:Doublefusion: Real-time capture of human performances with inner body shapesfrom a single depth sensor.|
|640|cvpr18-Improved Fusion of Visual and Language Representations by Dense Symmetric Co-Attention for Visual Question Answering|This progress has beenmainly brought about by two lines of research, the devel-opment of better attention mechanisms and the improve-ment in fusion of features extracted from an input imageand question.|
|||Meanwhile, researchers have proposed several methods forfeature fusion [5, 14, 32], where the aim is to obtain bet-ter fused representation of image and question pairs.|
|||This is particularly the casewith the studies of feature fusion methods, where attentionis considered to be optional, even though the best perfor-mance is achieved with it.|
|||Motivated by this, we propose a novel co-attentionmechanism for improved fusion of visual and language rep-resentations.|
|||Related WorkIn this section, we briefly review previous studies ofVQA with a special focus on the developments of attentionmechanisms and fusion methods.|
|||[32] combined themechanism with a novel multi-modal feature fusion of im-age and question.|
|||In early stud-ies, researchers employed simple fusion methods such asthe concatenation, summation, and element-wise product ofthe visual and language features, which are fed to fully con-nected layers to predict answers.|
|||[5] that a more com-plicated fusion method does improve prediction accuracy;they introduced the bilinear (pooling) method that uses anouter product of two vectors of visual and language fea-tures for their fusion.|
|||The attention mechanisms can also be considered featurefusion methods, regardless of whether it is explicitly men-tioned, since they are designed to obtain a better represen-tation of image-question pairs based on their interactions.|
|||We adopt a similar approach that uses multipleattention maps here, but we use average instead of concate-nation for fusion of the multiple attended features, becausewe found it works better in our case.|
|||Vl = softmax  A(i)A(i)(8)(9)and also normalize Al in column-wise to derive attentionmaps on image regions conditioned by each question wordasAs we employ multiplicative (or dot-product) attention asexplained below, average fusion of multiple attended fea-tures is equivalent to averaging our attention maps asAVl = softmax(Al ).|
|||, QT and QTable 4: Model sizes of DCNs and several bilinear fusionmethods.|
|||The coreof the network is the dense co-attention layer, which is de-signed to enable improved fusion of visual and languagerepresentations by considering dense symmetric interac-tions between the input image and question.|
|641|WSISA_ Making Survival Prediction From Whole Slide Histopathological Images|In classifying cancer subtype with WSIs, one pioneeringwork [8] was proposed to use a patch-level convolutionalneural network (CNN) and train a decision fusion model asa two-level model for tumor classification.|
|644|cvpr18-Deep Spatial Feature Reconstruction for Partial Person Re-Identification  Alignment-Free Approach| Besides, we further replace the pixel-level reconstruc-tion with a block-level one, and develop a multi-scale(different block sizes) fusion model to enhance the per-formance.|
|||Partial-iLIDSr = 3r = 1Partial REIDr = 3Methodr = 1Resizing modelSWM [32]AMC [32]AMC+SWM [32]DSR (single-scale)DSR (multi-scale)19.3324.3333.3336.0039.3343.0032.6745.0046.0051.0055.6760.3321.8533.6146.7849.5851.0654.5836.9747.0664.7563.3461.6664.503 different fusion ways are adopted: 1  1 blocks, 1  1blocks combined with 2  2 and 1  1 blocks, 2  2 blockscombined with 3  3 blocks.|
|||Spindle net: Person re-identification with human body region guided featuredecomposition and fusion.|
|645|Zhiwen_Fan_A_Segmentation-aware_Deep_ECCV_2018_paper|In thispaper, we proposed a segmentation-aware deep fusion network calledSADFN for compressed sensing MRI.|
|||Then, the aggregated fea-ture maps containing semantic information are provided to each layer inthe reconstruction network with a feature fusion strategy.|
|||We prove theutility of the cross-layer and cross-task information fusion strategy bycomparative study.|
|||In this paper, we propose a segmentation-aware deep fusion network (SADFN)architecture for compressed sensing MRI to fuse the semantic supervision infor-mation in the different depth from the segmentation label and propagate thesemantic features to each layer in the reconstruction network.|
||| The semantic information from the segmentation network is provided toreconstruction network using a feature fusion strategy, helping the recon-4W. Fan et al.|
|||3 The Proposed ArchitectureTo incorporate the information from segmentation label into the MRI recon-struction, we proposed the segmentation-aware deep fusion network (SADFN).|
|||Then a segmentation-aware featureextraction module is designed to provide features with rich segmentation infor-mation to reconstruction network using a feature fusion strategy.|
|||3.3 Deep Fusion NetworkWith the well-trained Pre-RecNet and Pre-SegNet, we can construct the segmentation-aware deep fusion network with N blocks (SADFNN ) by integrating the features8W. Fan et al.|
|||from the Pre-RecNet and Pre-SegNet, which involving a cross-layer multilayerfeature aggregation strategy and a cross-task feature fusion strategy.|
|||Also, in the Figure2, the feature fusion strategy is also utilized in each block of the Pre-RecNet.|
|||The Fine-tuning Strategy With the well-constructed deep fusion network,we further fine-tune the resulting architecture.|
|||Then the MR image is sent to the Pre-SegNetto extract the segmentation features, which are then utilized for the multilayerfeature aggregation in Pre-SegNet and feature fusion.|
|||Meanwhile, the zero-filledMR image is also input to the deep fusion network.|
|||During the optimization, theparameters in the Pre-RecNetN and Pre-SegNet are kept fixed, while we onlyadjust the parameters in the deep fusion network.|
|||The selected feature maps from the feature tensors produced by the featurefusion in the deep fusion network.|
|||Again, we notethat during the fine-tuning of the SADFN model, compressed feature tensoris yielded by multilayer features aggregation (MLFA) and the feature tensoris propagated to the Pre-RecNet before the feature fusion in each block.|
|||6 ConclusionIn this paper, we proposed a segmentation-aware deep fusion network (SADFN)for compressed sensing MRI.|
|||The multilayer feature aggregation is adopted to fuse cross-layer information in the MRI segmentation network and the feature fusion strat-egy is utilized to fuse cross-task information in the MRI reconstruction network.|
|646|cvpr18-M3  Multimodal Memory Modelling for Video Captioning|The performance comparison with the other five state-of-the-art methods using multiple visual feature fusion on MSVD.|
|||4.5.1 Experimental Results on MSVDFor comprehensive experiments, we evaluate and comparewith the state-of-the-art methods using single visual fea-ture and multiple visual feature fusion, respectively.|
|||When using multiple visual feature fusion, we com-pare our model with the other five state-of-the-artapproaches([37], [30], [21], [38], [2]).|
|||Similarly, we perform experiments with thesetwo methods using single visual feature and multiple visu-al feature fusion simultaneously.|
|647|cvpr18-Dense Decoder Shortcut Connections for Single-Pass Semantic Segmentation|The decoder features a novel architecture,consisting of blocks, that (i) capture context information,(ii) generate semantic features, and (iii) enable fusion be-tween different output resolutions.|
|||The densedecoder connections allow for effective information prop-agation from one decoder block to another, as well as formulti-level feature fusion that significantly improves the ac-curacy.|
|||The decoder fea-tures a novel architecture consisting of blocks, that capturecontext information, generate semantic features and enablefusion between different resolution levels.|
|||The dense decoder short-cut connections allow for effective information propagationfrom one block of a decoder to another, and for multi-levelfeature fusion that significantly improves the accuracy.|
|||More-over, the above methods use dense connections to createmore efficient building blocks, whereas we use them tostrengthen feature propagation for multi-level semantic fea-ture fusion.|
|||The encoder extracts appearance information on var-65971cne2cne3cne4cneAEAEencoder adaptationAEfusionsemantic feature generationdec 4dec 3dec 2dec 1Figure 1: Overview of our architecture for single-pass semantic segmentation: cascaded architecture with our context-aggregating decoder, feature-level long-range skip connections, and dense decoder shortcut connections for multi-level fea-ture fusion.|
|||Moreover, the blocks dec1,dec2, and dec3 have a fusion stage in between them.|
|||We callthese 3 blocks the fusion blocks, as they fuse the output ofthe previous decoder with the output from the encoder adap-tation.|
|||The block dec4 has only one input and therefore, itdoes not perform any fusion.|
|||purpose is to prepare features from an encoder encx for thefusion stage (for x = 1, .|
|||Finally, if a decoders block performs a fusion (i.e.|
|||The fusion is the second stage of the blocks dec1, dec2and dec3.|
|||decoderA-d indecoderB-d in    33, DUPSCALEencoderC-d in33, D+D-d outFigure 3: Decoders fusion stage.|
|||If we generate semantic featuresfor further processing, we then apply a 3  3 convolution,as explained in the fusion stage.|
|||Moreover, we do not use the ReLU right af-ter the last convolution of the encoder adaptation stage andsemantic feature generation stage, where features are beingadopted for the fusion stage or final prediction, and beforethe pooling layer in the semantic feature generation stage.|
|||Moreover, we have proposed a novel de-coders architecture, consisting of blocks, each capturingcontext information, generating semantic features, and en-abling fusion between different output resolutions.|
|||The dense decoder shortcut connections al-low for effective information propagation from one decoderblock to another, and for multi-level feature fusion that sig-nificantly improves the accuracy.|
|648|Xiaohan_Fei_Visual-Inertial_Object_Detection_ECCV_2018_paper|We leverage recent developments in visual-inertial sensor fusion, and its use forsemantic mapping, an early instance of which was given in [2], where objectswere represented by bounding boxes in 3D.|
|||We achieve this by employing some tools from the literature, namely visual-inertial fusion, and crafting a novel likelihood model for objects and their pose,leveraging recent developments in deep learning-based object detection.|
|||= {z}Np(Zt, Xt|yt)  p(Zt|Xt, yt)p(Xt|yt)(1)where p(Xt|yt) is typically approximated as a Gaussian distribution whose den-sity is estimated recursively with an EKF [3] in the visual-inertial sensor fusionliterature [4,5].|
|||Given the prior distribution p({k, g}t1|I t1),Visual-Inertial Object Detection and Mapping5a hypothesis set {k, g}t can be constructed by a diffusion process around the prior{k, g}t1.|
|||During top-down phase, proposals needed by Fast R-CNN are generated by firstsampling from the prior distribution p(z|yt1) followed by a diffusion and thenmapping each sample to a bounding box b and a class label c. Fig.|
|||This work is related to visual-inerital sensor fusion [4] and vision-only monoc-ular SLAM [36] in a broader sense.|
|||Tsotsos, K., Chiuso, A., Soatto, S.: Robust inference for visual-inertial sensorfusion.|
|||: Elas-ticfusion: Dense slam without a pose graph.|
|||: Incremental dense semantic stereo fusion for large-scale semanticscene reconstruction.|
|||McCormac, J., Handa, A., Davison, A., Leutenegger, S.: Semanticfusion: Dense 3dsemantic mapping with convolutional neural networks.|
|649|Deep Video Deblurring for Hand-Held Cameras|Related WorkThere existtwo main approachesto deblurring:deconvolution-based methods that solve inverse problems,and those that rely on multi-image aggregation and fusion.|
|||[20] show that3D reconstruction can be used to project pixels into a sin-gle reference coordinate system for pixel fusion.|
|||We perform an early fusion of neigh-boring frames that is similar to the FlowNetSimple modelin [9], by concatenating all images in the input layer.|
|||We tested with different fusionstrategies, for example late fusion, i.e.|
|650|Goutam_Bhat_Unveiling_the_Power_ECCV_2018_paper|Furthermore, we propose a noveladaptive fusion approach that leverages the complementary propertiesof deep and shallow features to improve both robustness and accuracy.|
|||As our sec-ond contribution, we propose a novel fusion strategy to combine the deep andshallow predictions in order to exploit their complementary characteristics.|
|||We propose a novel adaptive fusion approach that aims at fully exploiting theircomplementary nature, based on a quality measure described in section 4.1.|
|||4.2 Target PredictionWe present a fusion approach based on the quality measure (1), that combinesthe deep and shallow model predictions to find the optimal state.|
|||4: An illustration of our fusion approach, based on solving the optimization prob-lem (7).|
|||Forthe fusion method presented in section 4, the regularization parameter  in (6)is set to 0.15.|
|||We plot theperformance of our approach using sum-fusion with fixed weights (red) for a range ofdifferent shallow weights s.|
|||These results are also compared with the baseline ECO(orange) and our adaptive fusion (blue).|
|||For a wide range of s values, our sum-fusionapproach outperforms the baseline ECO in robustness on both datasets.|
|||Our adaptivefusion achieves the best performance both in terms of accuracy and robustness.|
|||We observe that ourtracker with a fixed sum-fusion outperforms the baseline ECO for a wide range ofweights s.|
|||Figure 5 also shows the results of our proposedadaptive fusion approach (section 4), where the model weights  are dynamicallycomputed in each frame.|
|||Compared to using a sum-fusion with fixed weights, ouradaptive approach achieves improved accuracy without sacrificing robustness.|
|||Figure 6 shows a qualitative example of our adaptive fusion approach.|
|||6: Qualitative example of our fusion approach.|
|||Later,when encountered with occlusions, clutter and out-of-plane rotations (b,d), our fusionemphasizes the deep model due to its superior robustness.|
|||In (c), where the targetundergoes scale changes, our fusion exploits the shallow model for better accuracy.|
|||These results show that our analysis in section 3 and the fusion approachproposed in section 4 generalizes across different network architectures.|
|||We further propose a novel fusionstrategy to combine the deep and shallow appearance models leveraging theircomplementary characteristics.|
|651|Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper|Zhang, Z., Zhang, X., Peng, C., Cheng, D., Sun, J.: Exfuse: Enhancing featurefusion for semantic segmentation.|
|652|cvpr18-Revisiting Dilated Convolution  A Simple Approach for Weakly- and Semi-Supervised Semantic Segmentation|We then propose a simpleyet effective anti-noise fusion strategy to address this is-sue.|
|||To address this is-sue, we propose a simple anti-noise fusion strategy to sup-press object-irrelevant regions and fuse the generated local-ization maps into an integral one where the object region-s are sharply highlighted.|
|||Examples of the localization maps produced by different dilated blocks as well as the dense localization maps with the anti-noisefusion strategy.|
|||Comparison of mIoU scores using different localization maps on PASCAL VOC 2012.settingsbkg plane bike bird boat bottle buscarcatchair cow table dog horse motor person plant sheep sofa traintv mIoUResults on the validation set:87.0 76.1 31.4 67.7 54.9 58.0 24.9 55.1 73.7 2.6 62.6 0.3 70.3 61.8d=187.2 75.8 31.7 66.9 54.0 58.1 33.6 57.9 73.4 5.2 61.9 1.7 70.0 62.3d=387.8 77.0 32.3 67.1 55.6 59.5 48.0 62.6 73.6 9.5 62.5 6.3 69.4 60.4d=687.9 76.5 32.1 68.0 56.1 59.2 51.3 62.9 73.0 9.3 63.7 6.2 68.0 60.7d=9fusion88.5 77.9 32.5 68.3 56.7 59.9 64.2 70.6 73.2 17.0 63.7 12.2 69.8 62.7fusion (CRF) 89.5 85.6 34.6 75.8 61.9 65.8 67.1 73.3 80.2 15.1 69.9 8.1 75.0 68.465.065.766.066.067.570.967.567.366.165.068.571.515.8 68.2 15.1 68.0 29.6 50.318.5 68.2 16.9 68.8 32.9 51.328.6 68.2 21.2 69.7 41.8 54.031.0 69.3 22.9 69.3 44.1 54.432.9 68.1 24.8 70.3 49.5 57.132.6 74.9 24.8 73.2 50.8 60.4Results on the test set:fusion (CRF) 89.8 78.4 36.2 82.1 52.4 61.7 64.2 73.5 78.4 14.7 70.3 11.9 75.3 74.281.072.638.8 76.7 24.6 70.7 50.3 60.854.4%) by enlarging the dilation rate of the convolution-al kernel, which can further validate the effectiveness ofusing dilated convolutional blocks for object localization.|
|||Furthermore, the mIoU score can be further improved to57.1% based on the dense localization maps produced bythe proposed anti-noise fusion strategy, which can furtherdemonstrate the effectiveness of this strategy for highlight-ing object and removing noise.|
|||The mIoU score drops almost 1% compared with using thecurrent fusion strategy.|
|653|cvpr18-Single View Stereo Matching|ily studied in the literature and is mainly tackled with twotypes of technical methodologies namely active stereo vi-sion such as structured light [33], time-of-flight [40], andpassive stereo vision including stereo matching[17, 25],structure from motion [35], photometric stereo [5] anddepth cue fusion [31], etc.|
|||Reliabilityfusion of time-of-flight depth and stereo geometry for highquality depth maps.|
|654|Yonggen_Ling_Modeling_Varying_Camera-IMU_ECCV_2018_paper|Combining cameras and inertial measurement units (IMUs)has been proven effective in motion tracking, as these two sensing modal-ities offer complementary characteristics that are suitable for fusion.|
|||Ling, Y., Kuse, M., Shen, S.: Edge alignment-based visual-inertial fusion for track-ing of aggressive motions.|
|||Ling, Y., Shen, S.: Aggressive quadrotor flight using dense visual-inertial fusion.|
|||Lynen, S., Achtelik, M., Weiss, S., Chli, M., Siegwart, R.: A robust and mod-ular multi-sensor fusion approach applied to MAV navigation.|
|||Shen, S., Michael, N., Kumar, V.: Tightly-coupled monocular visual-inertial fusionfor autonomous flight of rotorcraft MAVs.|
|||Steven, L., Alonso, P.P., Gabe, S.: Spline fusion: A continuous-time representationfor visual-inertial fusion with application to rolling shutter cameras.|
|||Yang, Z., Shen, S.: Monocular visual-inertial fusion with online initialization andcamera-IMU calibration.|
|||Yang, Z., Shen, S.: Tightly-coupled visual-inertial sensor fusion based on IMU pre-integration.|
|655|cvpr18-Reinforcement Cutting-Agent Learning for Video Object Segmentation|Supervision by fusion: to-wards unsupervised learning of deep salient object detector.|
|656|Kuang-Jui_Hsu_Unsupervised_CNN-based_co-saliency_ECCV_2018_paper|Its worth mentioning that ourmethod also outperforms the unsupervised CNN-based single-saliency method,SVFSal [42] that requires saliency proposal fusion for generating high-qualitypseudo ground-truth as training data.|
|||Jerripothula, K., Cai, J., Yuan, J.: Image co-segmentation via saliency co-fusion.|
|||: Segmentation guided local proposal fusion forco-saliency detection.|
|||: Image co-saliency detection via locally adaptivesaliency map fusion.|
|||Jiang, P., Vasconcelos, N., Peng, J.: Generic promotion of diffusion-based salientobject detection.|
|||Zhang, D., Han, J., Zhang, Y.: Supervision by fusion: Towards unsupervised learn-ing of deep salient object detector.|
|657|cvpr18-Human Semantic Parsing for Person Re-Identification|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|658|Tao_Song_Small-scale_Pedestrian_Detection_ECCV_2018_paper|Du, X., El-Khamy, M., Lee, J., Davis, L.: Fused dnn: A deep neural network fusionapproach to fast and robust pedestrian detection.|
|659|cvpr18-Correlation Tracking via Joint Discrimination and Reliability Learning|When the number oftraining samples exceeds the pre-defined value Tmax, wefollow the ECO method and use the Gaussian Mixture Mod-el (GMM) for sample fusion.|
|660|cvpr18-Self-Supervised Multi-Level Face Model Learning for Monocular Reconstruction at Over 250 Hz|[31] P. Huber, P. Kopp, M. R atsch, W. Christmas, and J. Kit-3D face tracking and texture fusion in the wild.|
|661|cvpr18-End-to-End Convolutional Semantic Embeddings|Meanwhile, recurrentresidual fusion (RRF) model [23], which consists of sev-eral recurrent units with residual blocks and a fusion model,showed improved performance on both Flickr30K and MS-COCO benchmarking datasets.|
|||However,to keep it concise, we drop the subindex i if no confusion arises.|
||| Context In this setting, we use the image context vec-tor cv to retrieve sentences represented by global se-mantics s. Early fusion For sentence retrieval, we use (cv +v)/2to represent images and use global s to represent sen-tences.|
||| Late fusion Assuming f i and f icvare the similaritiesof sentence si with global semantics of v, and cv re-spectively, we choose F i = (f i + f icv )/2 as the finalsimilarities and then compute the ranks.|
|||Meanwhile, both late fusion and early fusionhave almost the same performance with the global seman-tics.|
|||Learning arecurrent residual fusion network for multimodal matching.|
|663|cvpr18-Person Re-Identification With Cascaded Pairwise Convolutions|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|664|Anti-Glare_ Tightly Constrained Optimization for Eyeglass Reflection Removal|During theday, indoor glass reflections are overwhelmed by the pro-fusion of light refracted from the outdoor.|
|665|Jointly Learning Energy Expenditures and Activities Using Egocentric Multimodal Signals|We providea publicly available dataset of egocentric video augmentedwith heart rate and accelerometer signals, and we investi-gate the fusion of these signals for activity detection andenergy expenditure estimation.|
|||Multimodal fusion We adopt an early fusion scheme.|
|||showstheFigure 4(a)confusion matrix ofanacceleration-based baseline.|
|||Significant ambiguities can beseen, especially for light activities (right hand side in thematrix), whereas Figure 4(c) shows that adding visual fea-tures make it possible to resolve confusion between manyclasses such as meeting, sitting tasks, and standing in line,which are all fairly sedentary.|
|||Confusion matrices of activity detection: (a) acceleration-based baseline, (b) Inception network, and (c) EnergyRCN (V+A).|
|||Multisensor data fusion for physical activity as-sessment.|
|666|Semantic Multi-View Stereo_ Jointly Estimating Objects and Voxels|Volumetric Reconstruction from Images: While there isa large body of literature on volumetric fusion from rangeimages [10,25], in this paper we focus on reconstruction di-rectly from RGB images.|
|||Kinectfusion: Real-time dense surface map-ping and tracking.|
|667|Yin_Li_In_the_Eye_ECCV_2018_paper|The fusion is done using element-wise summation as sug-gested by [8].|
|||NetworksI3D RGBI3D FlowI3D FusionI3D JointAction AccAction Acc(Clip)43.6932.08N/A46.42(Video)47.2638.3148.8449.79(a) Backbone Network: We compareRGB, Flow, late fusion and joint train-ing of I3D for action recognition.|
|||Concretely, we testedRGB and flow streams of I3D [4], the late fusion of two streams, and the jointtraining of two streams [8].|
|||Moreover, EgoConv+I3D only slightly improves the I3D late fusionresults (+0.1%).|
|||Specifically, we compare the confusion matrixof our model with backbone I3D Joint network and second best I3D+Gaze inFig 4.|
|||Confusion matrix for action recognition.|
|||Feichtenhofer, C., Pinz, A., Zisserman, A.: Convolutional two-stream networkfusion for video action recognition.|
|668|cvpr18-Learning Rich Features for Image Manipulation Detection|Among various fusionmethods, we apply bilinear pooling on features from bothstreams.|
||| Late Fusion: Direct fusion combining all detected bound-ing boxes for both RGB Net and noise Net.|
|||For all datasets, late fusion performs worse than RGB-N,which shows the effectiveness of our fusion approach.|
|||Not surprisingly, the fusionof the two streams leads to improved performance.|
|||Imageforgery localization through the fusion of camera-based,feature-based and pixel-based techniques.|
|669|Kaiyue_Lu_Deep_Texture_and_ECCV_2018_paper|: Fast and effective l0 gradient minimization by regionfusion.|
|670|cvpr18-Toward Driving Scene Understanding  A Dataset for Learning Driver Behavior and Causal Reasoning|Fourth, a multimodalfusion for driver behavior detection can be studied.|
|||The visual data anddata from sensors are complementary to each other in thisrespect and thus their fusion gives the best results, as shownin the last row of the table.|
|||7705right turnleft turnintersection passingrailroad passingleft lane branchright lane changeleft lane changeright lane branchcrosswalk passingmergeu-turn0.80.60.40.20.0nrut thgirnrut tfel gnissapdaorliarhcnarb enal tfelegnahc enal thgiregnahc enal tfelhcnarb enal thgirgnissap klawssorc gnissapnoitcesretniegremnrut-uFigure 7: Confusion matrices for Goal-oriented driver behavior classes using CNN conv model (left) and CNN+Sensors (right).|
|||The confusion of behaviorclasses with a background class remains the most frequentsource of errors for all layers.|
|671|Yilei_Xiong_Move_Forward_and_ECCV_2018_paper|: Attention-based multimodal fusion for video description.|
|672|Chang_Chen_Deep_Boosting_for_ECCV_2018_paper|Furthermore, we propose a path-widening fusion scheme cooperated with the dilated convolution to de-rive a lightweight yet efficient convolutional network as the boostingunit, named Dilated Dense Fusion Network (DDFN).|
|||Lastbut not least, we further propose a path-widening fusion scheme cooperated withthe dilated convolution to make the boosting unit more efficient.|
|||Cooperating with the dilated convolution, we propose apath-widening fusion scheme to expand the capacity of each boosting unit.|
|||[11] proposed a stage-wise model (i.e., TNRD)which introduced the well-designed convolutional layers into the non-linear dif-fusion model to derive a flexible framework.|
|||3.3 Relationship to TNRDThe TNRD model proposed in [11] is also a stage-wise model trained jointly,which can be formulated asxn  xn1 = D(xn1)  R(xn1, y),(12)where D() stands for the diffusion term which is implemented using a CNN withtwo layers and R() denotes the reaction term as R(xn1, y) = (xn1y), where is a factor which denotes the strength of the reaction term.|
|||From the plain structure to the dilated dense fusion: the evolution of structurefor the boosting unit.|
|||And the symbol / denotes the path-wideningfusion.|
|||(12), we then havexn = xn1  D(xn1)  R(xn1, y)= (x + u)  D(xn1)  (u  v)= x  D(xn1) + v.(14)(15)The target of TNRD is to let xn  x, i.e., D(xn1)  v. Thus, the diffusionterm is actually trained for fitting the white Gaussian noise v. In contrast, ourproposed DBF is trained for directly restoring the original signal x, leveragingon the availability of denoised images and the growth of SNR.|
|||4.3 Path-widening FusionWe further propose a path-widening fusion scheme to make the boosting unitmore efficient.|
|||The proposed path-widening fusion exploits the poten-tial of these two orders at the same time, and thus promotes the possibility tolearn better representations.|
|||Note that, we restrict the parameter number of DDFN not greater than DDN(i.e., about 4  104) to eliminate the influence of additional parameters due topath-widening fusion, and thus the efficiency of DDFN is also justified.|
|||5.2 Ablation Experiments of DDFNThe proposed DDFN integrates three concepts: dense connection, dilated con-volution, and path-widening fusion, deriving a fundamentally different structurecompared with existing models.|
|||Path-widening fusion (DDFN).|
|||4, we further pro-pose the path-widening fusion which aggregates the concatenated features of pre-ceding layers using a 1  1 convolution in the dense block, as shown in Fig.|
|||This fusion can further promote the denoising performance, i.e., DDFN as shownin Fig.|
|||Based on the densely connected structure, we further proposethe path-widening fusion cooperated with the dilated convolution to optimizethe DDFN for efficiency.|
|||Also, the idea of path-widening fusion is demonstrated tobe useful in the task of spectral reconstruction from RGB images [37].|
|||Chen, Y., Pock, T.: Trainable nonlinear reaction diffusion: A flexible frameworkfor fast and effective image restoration.|
|674|Quanlong_Zheng_Task-driven_Webpage_Saliency_ECCV_2018_paper|We also tried other fusion operationse.g., multiplication, but found addition performs better.|
|675|Depth From Defocus in the Wild|However, the noise, sparsity and am-biguities inherent in DFD require fusion over far greaterimage distances and irregularly-shaped regions, and require)13.|
|||Thisformulation can be viewed as an extension of semi-densemethods [30, 33, 34] that handles far sparser depth/flowdata, accounts for spatially-varying data uncertainty, anddoes not rely on square-shaped regions for fusion [7].|
|||Shapefrom defocus via diffusion.|
|676|Lipeng_Ke_Multi-Scale_Structure-Aware_Network_ECCV_2018_paper|For the case multiple poseestimation test trials are performed, only the results with scores higher than athreshold s are selected for the fusion of the pose output.|
|||Our method does not require repeated runs and fusion of different scales aspost-processing.|
|677|Yunpeng_Chen_Fast_Multi-fiber_Network_ECCV_2018_paper|Feichtenhofer, C., Pinz, A., Zisserman, A.: Convolutional two-stream networkIEEE Conference on Computer Vision andfusion for video action recognition.|
|678|Deep Joint Rain Detection and Removal From a Single Image|En-hancing underwater images and videos by fusion.|
|679|Tomas_Hodan_PESTO_6D_Object_ECCV_2018_paper|Buch, A.G., Petersen, H.G., Kr uger, N.: Local shape feature fusion for improvedmatching, pose estimation and 3D object recognition.|
|680|cvpr18-AdaDepth  Unsupervised Content Congruent Adaptation for Depth Estimation|Another line of work uses adversarial loss in conjunc-tion with classification loss, with an objective to diminishdomain confusion [44, 8, 9, 45].|
|||On simi-lar lines, we implement an additional skip multi-layer CNNblock with additive feature fusion to model M such thatMt = Ms + M (Figure 4a).|
|||Deep domain confusion: Maximizing for domain invariance.|
|681|Lifting From the Deep_ Convolutional 3D Pose Estimation From a Single Image|The overall architecture is fully differentiable  including the new projected-posebelief maps and 2D-fusion layers  and can be trained end-to-end using back-propagation.|
|||[22] incorporate model jointdependencies in the CNN via a max-margin formalism, oth-ers [48] impose kinematic constraints by embedding a dif-2501STAGE 12D joint prediction3D lifting &projectionFusionFeature extraction 2D LossSTAGE 22D joint prediction3D lifting &projectionFusionFeature extraction2D LossSTAGE 62D joint prediction3D lifting &projectionFusionFeature extraction2D LossProbabilistic 3Dpose model3D pose3D/2Dprojectionpredictedbelief mapsprojected posebelief maps999999991111predictedbelief mapspredictedbelief mapsprojected posebelief maps2Dfusionfusedbelief mapsInput imageProbabilistic 3Dpose modelOutput 2D poseFinal 3D poseferentiable kinematic model into the deep learning architec-ture.|
|||From an implementation point of view thisis done by introducing two distinct layers, the probabilistic3D pose layer and the fusion layer (see Figure 1).|
|||The Objective and TrainingFollowing [44], the objective or cost function ct min-imized at each stage is the the squared distance betweenthe generated fusion maps of the layer f pt , and ground-truthbelief maps bp generated by Gaussian blurring the sparseground-truth locations of each landmark pct =L+1Xp=1XzZ||f pt  bp||22(8)For end-to-end training the total loss is the sum over alllayers Pt6 ct.|
|683|Detecting Visual Relationships With Deep Relational Networks|Given s and o, thenthe posterior distribution of r is given byrelations between them via the links in the inference units,rather than combining them using a fusion layer.|
|684|Qiang_Qiu_ForestHash_Semantic_Hashing_ECCV_2018_paper|Bronstein, M., Bronstein, A., Michel, F., Paragios, N.: Data fusion through cross-modalitymetric learning using similarity-sensitive hashing.|
|685|cvpr18-One-Shot Action Localization by Learning Sequence Matching Network|Although our encoder is not the best performing one, it isAccuracy(%)# parameterstwo-stream network[37]dynamic image [2]two-stream fusion [11]my encoder (mini-VGG16)82.976.985.281.327.5M36.1M97.3M3.6MTable 5.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|686|cvpr18-Optical Flow Guided Feature  A Fast and Robust Motion Representation for Video Action Recognition|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|687|Differential Angular Imaging for Material Recognition|In our goal of combining spatialand angular image information to account for texture and re-flectance, we are particularly motivated by the two-streamfusion framework [15, 37] which achieves state-of-art re-sults in UCF101 [38] action recognition dataset.|
|||Convolu-tional two-stream network fusion for video action recogni-tion.|
|688|Dense Captioning With Joint Inference and Visual Context|We proposea new model pipeline based on two novel ideas, joint infer-ence and context fusion, to alleviate these two challenges.|
|||The second component is contextfusion, where pooled features from regions of interest arecombined with context features to predict better region de-scriptions.|
|||To reiterate, the contributions of this work are two-fold: We design network structures that incorporate twonovel ideas, joint inference and context fusion, to ad-dress the challenges we identified in dense captioning.|
|||Context fusion for accurate descriptionVisual context is important for understanding a local re-gion in an image, where it has already shown to bene-fit tasks such as object detection and semantic segmenta-tion [3] [7] [33].|
|||6 and termed as early-fusion and late-fusion.|
|||Early-fusion (Fig.|
|||6(a)) directly combines the region feature andcontext feature together before feeding into the LSTM,while late-fusion (Fig.6(b)) uses an extra LSTM to generatea recurrent representation of the context feature, and thencombines it with the local feature.|
|||The context feature rep-resentation is combined with the region feature represen-2196next	word	next	word	fusion operator Table 1: Comparison of our final model with previous bestresult on Visual Genome V1.0 and V1.2.|
|||LSTM	LSTM	LSTM	fusion operator region	features	context	feature	region	features	context	feature	(a) (b) Figure 6: Model structures for region description assistedby context features.|
|||(a) Early-fusion.|
|||(b) Late-fusion.|
|||Thefusion operator denoted by the red dot can be concatenation,summation, multiplication, etc.|
|||(a) The inte-grated model of T-LSTM and the late-fusion context model.|
|||tation via a fusion operator for both variants.|
|||Such fusion designs can be easily integratedwith any of the models in Fig.|
|||Integrated modelThe aforementioned model structures of joint inferenceand context fusion can be easily plugged together to pro-duce an integrated model.|
|||For example, the integration ofT-LSTM and the late-fusion context model can be viewedin Fig.|
|||[20]-5.39baseline5.266.85S-LSTM SC-LSTM T-LSTM5.156.475.576.835.648.03Table 3: The mAP performance of integrated models withcombinations of joint inference models and context fusionstructures on Visual Genome V1.0.|
|||modelearly-fusionlate-fusion[, ][, ]S-LSTM SC-LSTM T-LSTM6.746.546.697.507.197.577.187.297.047.727.477.648.248.168.198.498.538.60Table 4: The mAP performance of different dense caption-ing models on Visual Genome V1.2.|
|||modelno contextlate-fusionbaseline6.98S-LSTM T-LSTM6.447.767.067.638.169.038.718.52[, ]batch size of 1 to train the whole network.|
|||We found that training models with contextfusion from scratch tends not to converge well, so we fine-tune these models from their non-context counterparts, witha total of 600K training iterations.|
|||Integrated modelsWe evaluate the integrated models with different de-signs for both joint inference and context fusion in this sec-tion.|
|||For context fu-sion, we compare the different settings proposed in Sec-tion 3.3, where we evaluate early-fusion and late-fusionwith different fusion operators: concatenation, summation,and multiplication.|
|||For early-fusion with concatenation, weplug in a fully-connected layer after the concatenated fea-ture to reduce it to the same input dimension as the LSTM.|
|||Effectiveness of context fusion.|
|||In all models, contextinformation helps to improve mAP ranging from 0.07 (S-LSTM, early-fusion, summation) to 1.10 (S-LSTM, late-fusion, multiplication).|
|||The three types of fusion meth-ods all yield improvements in mAP for different models.|
|||With T-LSTM and late-fusion with multiplication, we obtain thebest mAP performance 8.60 in this set of experiments.|
|||9 shows example predictions for compar-ison of T-LSTM without context fusion and T-LSTM-mult.|
|||Late-fusion is better than early-fusion.|
|||Comparingearly-fusion and late-fusion of context information, we findthat late-fusion is better than early-fusion for all pairs ofcorresponding models.|
|||Also, early fusion only outperformsTable 5: The chosen hyper-parameters and the performanceon Visual Genome V1.0 and V1.2 respectively.|
|||One disad-vantage of early-fusion is that it directly combines the lo-cal and context features that have quite differing visual ele-ments, making it unlikely able to decorrelate the visual ele-ment into the local region or the context region in the laterstages of the model.|
|||Here, we see similar results ason V1.0, which further verifies the advantage of T-LSTMover S-LSTM (mAP 8.16 vs 6.44 for no-context), and thatcontext fusion greatly improves performance for both mod-els.|
|||For context fusion, we can see that the T-LSTM modelwith late concatenation achieves the best result with mAP9.03.|
|||ConclusionsIn this work, we have proposed a novel model structurewhich incorporates two ideas, joint inference and contextfusion, to address specific challenges in dense captioning.|
|689|Lip Reading Sentences in the Wild|Convolutionaltwo-stream network fusion for video action recognition.|
|690|cvpr18-Low-Shot Learning With Large-Scale Diffusion|Low-shot learning with large-scale diffusionMatthijs Douze, Arthur Szlam, Bharath Hariharan, Herv e J egouFacebook AI Research*Cornell UniversityAbstractThis paper considers the problem of inferring image la-bels from images when only a few annotated examples areavailable at training time.|
|||The diffusion setup.|
|||The arrows indicate the directionof diffusion.|
|||There is no diffusion performed from the test im-ages.|
|||In more de-tail, we make the following contributions: We carry out a large-scale evaluation for diffusionmethods for semi-supervised learning and compare13349seed images (labeled)background images (unlabeled)test images(labels witheld)it to recent low-shot learning papers.|
||| We show that our approach is efficient and that the dif-fusion process scales up to hundreds of millions of im-ages, which is order(s) of magnitude larger than whatwe are aware in the literature on image-based diffu-sion [19, 18].|
||| We evaluate several variants and hypotheses involvedin diffusion methods, such as using class frequencypriors [38].|
|||Diffusion methods We refer the reader to [3, 12] for areview of diffusion processes and matrix normalization op-tions.|
|||Since the eigenvalues are ob-tained via Lanczos iterations [15, Chapter 10], the basic op-eration is similar to a diffusion process.|
|||Efficient kNN-graph construction The diffusion meth-ods use a matrix as input containing the similarity betweenall images of the dataset.|
|||Therefore diffusion methods are usu-ally implemented with sparse matrices.|
|||Several ap-proximate algorithms [10, 23, 1, 17] have been proposed toefficiently produce the kNN graph used as input of itera-tive/diffusion methods, since this operation is of quadraticcomplexity in the number of images.|
|||Label propagationThis section describes the initial stage of our proposal,which estimates the class of the unlabelled images with adiffusion process.|
|||It includes an image description step, theconstruction of a kNN graph connecting similar images, anda label diffusion algorithm.|
|||Affinity matrix: approximate kNN graphAs discussed in the related work, most diffusion pro-cesses use as input the kNN graph representing the N  Nsparse similarity matrix, denoted by W, which connectsthe N images of the collection.|
|||In our preliminary experi-ments, the approximation in the knn-graph does not induceany sub-optimality, possibly because the diffusion processcompensates the artifacts induced by the approximation.|
|||We now give details about the diffusion process itself,which is summarized in Figure 1.|
|||The set offorward label propagation algorithm ofimages on which we perform diffusion is composed of nLlabelled seed images and nB unlabelled background images(N = nL +nB).|
|||Early stopping per-forms better in both cases, so we cross-validate the numberof diffusion iterations.|
|||[37], we have also optimized a loss balancing the fit-ting constraint with the diffusion smoothing term.|
|||Howeverwe found that a simple late fusion (weighted mean of log-probabilities, parametrized by a single cross-validated co-efficient) of the scores produced by diffusion and logisticregression achieves better results.|
|||The Markov Clustering (MCL) [13]is another diffusion algorithm with nonlinear updates orig-inally proposed for clustering.|
|||In the on-line stage, we receive train-ing and test images from novel classes, (i) compute featuresfor them, (ii) complement the knn-graph matrix to includethe training and test images, and (iii) perform the diffusioniterations.|
|||Thenumber of iterations and batch size are Ilogreg and B.Diffusion the complexity is decomposed into: comput-ing the matrices WLL, WLB and WBL, which in-volves O(d  nL  nB) multiply-adds using brute-force distance computations; and performing Idif it-erations of sparse-dense matrix multiplications, whichincurs O(k N C Idif ) multiply-adds (note, sparsematrix operations are more limited by irregular mem-ory access patterns than arithmetic operations).|
|||There-fore the diffusion complexity is linear in the numberof background images nB.|
|||For the diffusion, we PCA-reduce the fea-ture vector to 256 dimensions and L2-normalize it, whichis standard in prior works on unsupervised image matchingwith pre-learned image representations [2, 34].|
|||Background images for diffusion We consider the fol-lowing sets of background images:1.|
|||None: the diffusion is directly from the seed images tothe test images;2.|
|||This corresponds to a more challenging settingwhere we have no prior knowledge about the imageused in the diffusion.|
|||Parameters of diffusionWe compare a few settings of the diffusion algorithm asdiscussed in section 3.4.|
|||Large(cid:173)scale diffusionFigure 2 reports experiments by varying the number ofbackground images nB and the number k of neighbors, forn = 2.|
|||An additional num-ber: before starting the diffusion iterations, with k=1000and no background images (the best setting) we obtain anaccuracy of 60.5%.|
|||Comparison with low(cid:173)shot classifiersWe compare the performance of diffusion against the lo-gistic baseline classifiers and a recent method of the state ofthe art [16], using the same features.|
|||For low-shot learning (n  5), thein-domain diffusion outperforms the other methods by alarge margin, see Table 2.|
|||[16]63.671.580.083.385.2logisticregression60.40.7868.80.8279.10.3583.40.1686.00.15in-domaindiffusion69.70.8675.40.6479.90.1782.10.1483.60.12diffusion+ logistic69.760.8875.600.6981.350.2284.560.1286.720.09n1251020Table 2.|
|||In-domain diffusion on Imagenet: We compare againstlogistic regression and a recent low-shot learning technique [16]on this benchmark.|
|||Results are reported with k = 30 for diffusion.|
|||Out-of-domain diffusion.|
|||Table 3 shows that the perfor-mance of diffusion is competitive only when 1 or 2 imagesare available per class.|
|||As stated in Section 3.2, we do notinclude the test points in the diffusion, which is standardfor a classification setting.|
|||However, if we allow this, asin a fully transductive setting, we obtain a top-5 accuracyof 69.6%0.68 with n = 2 with diffusion over F1M, i.e., onpar with diffusion over F100M.|
|||We experimented with a verysimple late fusion: to combine the scores of the two clas-sifiers, we simply take a weighted average of their predic-tions (log-probabilities), and cross validate the weight fac-tor.|
|||This shows that the logistic re-gression classifier and the diffusion classifier access dif-ferent aspects of image collection.|
|||With the in-domain diffusion, we notice that our method outperformsthe state-of-the-art result of [16] and which, itself, outper-forms or is closely competitive with [35, 36] in this setting.|
|||In contrast, our diffusionprocedure is generic and has only two parameters (nB andk).|
|||Note that the out-of-domain setting is comparable withthe standard low-shot setting, because the unlabeled imagesfrom F100M are generic, and have nothing to do with Ima-genet; and because the neighbor construction and diffusionare efficient enough to be run on a single workstation.|
|||Complexity: Runtime and memoryWe measured the run-times of the different steps in-volved in diffusion process and report them in Table 4.|
|||The3354out-of-domain diffusionn1251020noneF1MF10MF100M58.50.5263.60.6069.00.4673.90.1578.00.1561.40.6166.80.7172.50.2776.20.1979.10.2362.70.7668.40.7474.00.3577.40.3180.00.2763.60.6169.50.6075.20.4078.50.3480.80.18logisticregression60.40.7868.80.8279.10.3583.40.1686.00.15+F10M63.30.7370.60.8079.40.3483.60.1386.20.12diffusion+logisticHariharan+ F100M et al.|
|||Out-of-domain diffusion: Comparison of classifiers for different values of n, with k = 30 for the diffusion results.|
|||The nonecolumn indicates that the diffusion solely relies on the labelled images.|
|||backgroundoptimal iterationtiming: graph completiontiming: diffusionnone2F1M F10M F100M52m57s 8m36s 40m41s 4h08m54m3m44s4.4s19s34Table 4.|
|||This is themain drawback of using diffusion.|
|||However Table 3 showsthat restricting the diffusion to 10 million images alreadyprovides most of the gain, while dividing by an order ofmagnitude memory and computational complexity.|
|||Analysis of the diffusion processWe discuss how fast L fills up (it is dense after a fewiterations).|
|||We consider the rate of nodes reached by thediffusion process: we consider very large graphs, few seedsand a relatively small graph degree.|
|||Figure 3 measures thesparsity of the matrix L (on one run of validation), which in-dicates the rate of (label, image) tuples that have not been at-tained by the diffusion process at each diffusion step.|
|||Qualitative resultsFigure 4 shows paths between a seed image and test im-ages, which gives a partial view of the diffusion.|
|||Images visited during the diffusion process from a seed (left) to the test image (right).|
|||Unsurprisingly, we have found that per-forming diffusion over images from the same domain worksmuch better than images from a different domain.|
|||Furthermore, labeled im-ages should be included in the diffusion process and not justused as sources, i.e., not enforced to keep their label.|
|||The main outcome of our study is to show that diffusionover a large image set is superior to state-of-the-art methodsfor low-shot learning when very few labels are available.|
|||Interestingly, late-fusion with a standard classifiers result iseffective.|
|||In these cases diffusion combined withlogistic regression is the best method.|
|||Diffusion maps, spectral clustering and reaction coordinatesof dynamical systems.|
|||Diffusion processes for retrievalrevisited.|
|||Efficient diffusion on region manifolds: Recovering smallobjects with compact CNN representations.|
|691|Exploiting Saliency for Object Segmentation From Image Level Labels|Our fusion strategy uses five simple ideas.|
|692|Deep Affordance-Grounded Sensorimotor Object Recognition|Inparticular, object perception is based on the fusion of sen-sory (object appearance) and motor (human-object interac-tion) information.|
|||However, current systems have been designed basedon rather simple classification, fusion, and experimentalframeworks, failing to fully exploit the potential of the af-fordance stream.|
||| Extensive quantitative evaluation of the proposed fu-sion methods and comparison with traditional proba-bilistic fusion approaches.|
|||Related WorkMost sensorimotor object recognition works have so farrelied on simple fusion schemes (e.g using simple Bayesianmodels or the product rule), hard assumptions (e.g.|
|||Eventually, appearance and affordance informationare combined to yield improved object recognition, follow-ing various fusion strategies.|
|||The visual front-end module (left) processes the captured data, providing three information streams (middle)that are then fed into a single-stream or fusion DL model (right).|
|||Fusion architecturesPrior to the detailed description of the evaluated sensori-motor information fusion principles, it needs to be notedthat these are implemented within two general NN ar-chitectures, namely the Generalized Template-Matching(GTM) and the Generalized Spatio-Temporal (GST) one.|
|||4.4.1 Late fusionLate fusion refers to the combination of information at theend of the processing pipeline of each stream.|
|||The FC layer fusion is performed by concatenat-ing the FC features of both streams.|
|||It was experimentallyshown that fusion after the RL6 layer was advantageous,compared to concatenating at the output of the FC6 layerFigure 3.|
|||Detailed topology of the GTM architecture for: a) late fusion at FC layer, b) late fusion at last CONV layer, c) slow fusion, andd) multi-level slow fusion.|
|||After fusion, a single process-ing stream is formed (Fig.|
|||Regarding fusion at the lastCONV layer, the RL53 activations of both appearance andaffordance CNNs are stacked.|
|||For the GST architecture, the late fusion scheme con-siders only the concatenation of the features of the last FClayers of the appearance CNN and the affordance LSTMmodel, as depicted in Fig.|
|||In this context, an asynchronous late fusion approach is alsoinvestigated for the GST architecture.|
|||Specifically, the GSTlate fusion scheme (Fig.|
|||the in-ternal state vector h(t) of the last LSTM layer] is providedwith a time-delay factor, denoted by  > 0, compared tothe FC features of the appearance stream; in other words,the features of the affordance stream at time t   are com-bined with the appearance features at time t.4.4.2 Slow fusionSlow fusion for the GTM architecture corresponds to thecase of combining the CONV feature maps of the appear-ance and affordance CNNs in an intermediate layer (i.e.|
|||For realizingthis, two scenarios are considered, which correspond to thefusion of information from the two aforementioned CNNsat different levels of granularity: a) combining the featuremaps of the appearance and the affordance CNN from thesame layer level; and b) combining the feature maps of theappearance and the affordance CNN from different layerlevels.|
|||The actual fusion operator is materialized by sim-ple stacking of the two feature maps.|
|||For the GST architecture, theslow fusion scheme considers only the concatenation of thefeatures of the RL7 layer of the appearance and the affor-dance CNNs models, followed by an LSTM model, as canbe seen in Fig.|
|||In order to simulate the complex information exchangeroutes at different levels of granularity between the twostreams, a multi-level slow fusion scheme is also examined.|
|||The particularNN topology that implements this multi-level slow fusionscheme for the GTM architecture is illustrated in Fig.|
|||GTM and GST architectures evaluationIn Table 4, evaluation results from the application of dif-ferent GTM-based fusion schemes (Section 4.4) are given.|
|||From the presented results, it can be seen that for the caseof late fusion combination of CONV features (i.e.|
|||fusion atthe RL53 layer) is generally advantageous, since the spa-tial correspondence between the appearance and the affor-dance stream is maintained.|
|||Concerning single-level slowfusion models, different models are evaluated.|
|||However,single-level slow fusion tends to exhibit lower recognitionperformance than late fusion.|
|||Building on the evaluationoutcomes of the single-level slow and late fusion schemes,multi-level slow fusion architectures are also evaluated.|
|||This is mainly due tothe preservation of the spatial correspondence (initial fusionat the CONV level), coupled with the additional correlationslearned by the fusion at the FC level.|
|||, RL5af f33Experimental results from the application of the GST-based fusion schemes (Section 4.4) are reported in Table5.|
|||Forthe case of the synchronous late fusion, it can be seen thatthe averaging of the predictions from all frames is advan-tageous.|
|||It can be observed that asyn-chronous fusion leads to decreased performance, comparedto the synchronous case, while increasing values of the de-lay parameter  lead to a drop in the recognition rate.|
|||More-over, the slow fusion approach results to a significant de-crease of the object recognition performance.|
|||For providing a better insight, the objectrecognition confusion matrices obtained from the applica-6173Figure 6.|
|||Detailed topology of the GST architecture for: a) latefusion and b) slow fusion.|
|||architectures: GATF T (param), where the Generalized Ar-chitecture Type, GAT  {GTM, GST}, and the FusionType, FT  {LS, LA, SSL, SM L}  {Late Synchronous,Late Asynchronous, Slow Single Level, Slow Multi Level}and param indicates the specific parameters for each par-ticular fusion scheme (as detailed above).|
|||At this point, itneeds to be highlighted that any further information pro-cessing performed in the affordance stream after the fusionstep does not contribute to the object recognition process;hence, it is omitted from the descriptions in this work.|
|||From the resultspresented in Table 3 (only overall classification accuracy is1http://torch.ch/MethodAppearance CNNAffordance CNNaffordance recognitionAffordance CNN-LSTM affordance recognitionobject recognitionTask85.1281.9269.27Accuracy (%)GST-based fusion architecture [after fusion] Accuracy (%)Table 3.|
|||GTM-based fusion architecture [after fusion] Accuracy (%)GTMLS(FC6)GTMLS(RL53) [1 CONV, 1 FC]GTMLS(RL53) [1 CONV, 2 FC]GTMLS(RL53) [2 CONV, 1 FC]GTMLS(RL53) [2 CONV, 2 FC]GTMSSL(RL3appGTMSSL(RL4appGTMSSL(RL4appGTMSSL(RL5appGTMSM L(RL5appGTMSM L(RL5app, RL3af f), RL4af f), RL4af f), RL5af f), RL5af f, RL5af f331133331333, RL6), RL6)87.4087.6588.2487.6486.4078.7487.2085.8288.1388.2389.43Table 4.|
|||Fusion architectureAppearance CNNProduct RuleSVM [15, 4]Bayes [13]GTMSM LFusion LayerAccuracy (%)no fusionSoftmaxRL7RL7, RL5af f3, RL6RL5app385.1273.4583.4375.8689.43Table 6.|
|||Object recognition confusion matrices of appearanceCNN (left) and GTMSM L(RL5app, RL6) architecture(right).|
|||Comparison with probabilistic fusion6.|
|||Conclusions33, RL5af fThe GTMSM L(RL5app, RL6) architecture isalso comparatively evaluated, apart from the appearanceCNN model, with the following typical probabilistic fusionapproaches of the literature: a) the product rule for fusingthe appearance and the affordance CNN output probabil-ities, b) concatenation of appearance and affordance CNNfeatures and usage of a SVM classifier (RBF kernel) [4, 15],and c) concatenation of appearance and affordance CNNfeatures and usage of a naive Bayes classifier [13].|
|||Twogeneralized neuro-biologically and neuro-physiologicallygrounded neural network architectures, implementing mul-tiple fusion schemes for sensorimotor object recognitionwere presented and evaluated.|
|||The proposed sensorimo-tor multi-level slow fusion approach was experimentallyshown to outperform similar probabilistic fusion methodsof the literature.|
|693|cvpr18-Deep Regression Forests for Age Estimation|Facial age estimationthrough the fusion of texture and local appearance descrip-tors.|
|695|cvpr18-SurfConv  Bridging 3D and 2D Convolution for RGBD Images|Lstm-cf: Unifying context modeling and fusion with lstms for rgb-d scene labeling.|
|||Multimodal information fusion for urban scene understand-ing.|
|696|cvpr18-Look at Boundary  A Boundary-Aware Face Alignment Algorithm|Boundary heatmap fusion scheme is introduced to incorporate boundaryinformation into the feature learning of regressor.|
|||Mi(x, y) =(exp( Di(x,y)2220,),if Di(x, y) < 3otherwise(1)In order to fully utilise the rich information containedin boundary heatmaps, we propose a multi-stage boundaryheatmap fusion scheme.|
|||Boundary heatmap fusion is conducted at the input and ev-ery stage of the network.|
|||Boundary HeatmapsConcatenationM(N+13)*32*32SigmoidS13*64*64Down Sampling13*32*32TN*32*32Feature Map FusionFN*32*32Input Feature MapsConcatenationN*32*32HElementwise Dot Product(N+N)*32*32Refined Feature MapsFigure 4: An illustration of the feature map fusion scheme.|
|||have shown that the more fusion we conducted to the base-line network, the better performance we can get.|
|||Input image fusion.|
|||Feature map fusion.|
|||Details of feature map fusionsubnet are illustrated in Fig.|
|||In order to verify the capacity of handling cross-datasetface alignment of our method, we use boundary heatmapsestimator trained on 300W Fullset which has no overlapwith COFW and AFLW dataset and compare the perfor-mance with and without using boundary information fusionnoitroporP segamI10.90.80.70.60.50.40.30.20.100HPM(HELEN,LFPW), Error: 6.72%, Failure: 6.71%SAPM(HELEN), Error: 6.64%, Failure: 5.72%RCPR(HELEN,LFPW), Error: 8.76%, Failure: 20.12%TCDCN(HELEN,LFPW,AFW,MAFL), Error: 7.66%, Failure: 16.17%CFSS(HELEN,LFPW,AFW), Error: 6.28%, Failure: 9.07%LAB(HELEN,LFPW,AFW), Error: 4.62%, Failure: 2.17%0.010.020.030.040.050.060.070.080.090.1Normalized Point-to-Point ErrorFigure 7: CED for COFW-68 testset (68 landmarks).|
|||Ablation studyOur framework consists of several pivotal components,i.e., boundary information fusion, message passing and ad-versarial learning.|
|||Boundary information fusion is one of the key steps inour algorithm.|
|||To evaluate the relationship between the quantity of bound-ary information fusion and the final prediction accuracy, wevary the number of fusion levels from 1 to 4 and reportthe mean error results in Table 6.|
|||MethodMean ErrorBL7.12BL+L1 BL+L1&2 BL+L1&2&3 BL+L1&2&3&46.566.326.196.13Table 6: Mean error (%) on 300W Challenging Subset for vari-ous fusion levels.|
|||MethodMean ErrorBL7.12BL+HG/B BL+CL BL+HG6.956.246.13Table 7: Mean error (%) on 300W Challenging Set for differentsettings of boundary fusion scheme.|
|||To verify the effectiveness of the fusion scheme shown inFig.|
|||The comparison between BL+HG and BL+HG/B in-dicates the effectiveness of boundary information fusionBLBL+HBLBL+HBL+MPBL+HBL+MP+GANBLBL+HBLBL+HBL+MPBL+HBL+MP+GAN121110987654321.|
|697|cvpr18-Rethinking the Faster R-CNN Architecture for Temporal Action Localization|TAL-Net addresses threekey shortcomings of existing approaches: (1) we improvereceptive field alignment using a multi-scale architecturethat can accommodate extreme variation in action dura-tions; (2) we better exploit the temporal context of actionsfor both proposal generation and action classification byappropriately extending receptive fields; and (3) we explic-itly consider multi-stream feature fusion and demonstratethat fusing motion late is important.|
|||How-ever, there has been limited work in exploring such fea-ture fusion for Faster R-CNN.|
|||We propose a late fusionscheme and empirically demonstrate its edge over thecommon early fusion scheme.|
|||We propose to extend the input extent of SoIProposal Logits AveragingProposal Logits (RGB)Proposal Logits (Flow)SegmentProposalNetwork1D Feature Map (RGB)1D Feature Map (Flow)Proposal GenerationFigure 6: The late fusion scheme for the two-stream Faster R-CNN framework.|
|||We hypothe-size such two-stream input and feature fusion may also playan important role in temporal action localization.|
|||There-fore we propose a late fusion scheme for the two-streamFaster R-CNN framework.|
|||Conceptually, this is equivalentto performing the conventional late fusion in both the pro-posal generation and action classification stage (Fig.|
|||Note that a more straightforward way to fuse two fea-tures is through an early fusion scheme: we concatenate thetwo 1D feature maps in the feature dimension, and applythe same pipeline as before (Sec.|
|||We showby experiments that the aforementioned late fusion schemeoutperforms the early fusion scheme.|
|||used two-stream features, but either did not performfusion [15] or only tried the early fusion scheme [8, 14].|
|||[54]66.0 59.4 51.9 41.0 29.8Ours59.8 57.1 53.2 48.5 42.8 33.8 20.8Table 4: Results for late feature fusion in mAP (%).|
|||4 reports the action localiza-tion results of the two single-stream networks and the earlyand late fusion schemes.|
|||Finally, the late fusion scheme outperforms theearly fusion scheme except at tIoU threshold 0.1, validatingour proposed design.|
|||TAL-Net features threenovel architectural changes that address three key shortcom-ings of existing approaches: (1) receptive field alignment;(2) context feature extraction; and (3) late feature fusion.|
|698|cvpr18-Matryoshka Networks  Predicting 3D Geometry via Nested Shape Layers|In gen-eral, view-based methods are able to generate shapes at highresolutions, but occasionally suffer from noisy estimates,which need to be addressed in the fusion step.|
|||Our proposed method addresses the fusion step and han-dling of occlusions in a simple, but efficient formulation.|
|||Through careful align-ment of the depth maps and an appropriate loss function, weavoid noisy estimates, a costly fusion via optimization, andminimize the dimensionality of the final network layer.|
|||Shape fusion.|
|||This fusion process and the placement of the three orthogo-nal views vi is motivated by our observation that depth mappredictions are often less accurate near the silhouette of anobject.|
|||Let  : D  S be the fusionof a shape from the set of depth maps as defined in Eq.|
|||To that end, let T1:L  S be the (true) target shape and : S  (D) be the projection from an arbitrary shapeto the space of shapes that can be represented by the depthmap fusion process  from Eq.|
|||Weleave this and learning the shape fusion [21] for future work.|
|||Oct-netfusion: Learning depth fusion from data.|
|699|Switching Convolutional Neural Network for Crowd Counting|Current state-of-the artapproaches tackle these factors by using multi-scale CNNarchitectures, recurrent networks and late fusion of featuresfrom multi-column CNN with different receptive fields.|
|||Multi-column CNN used by [2, 19] perform late fusionof features from different CNN columns to regress the den-sity map for a crowd scene.|
|||Traditional convolutional architectures havebeen modified to model the extreme variations in scale in-duced in dense crowds by using multi-column CNN ar-chitectures with feature fusion techniques to regress crowddensity.|
|700|Rex_Yue_Wu_BusterNet_Detecting_Copy-Move_ECCV_2018_paper|It features a two-branch architecture followed bya fusion module.|
|||Finally, complex cloning entails a more complicated relationship between D andT , often with extra diffusion estimation, edge blending, color change or othermore sophisticated image processing steps.|
|||Instead of training BusterNet all modules together, we adopt athree-stage training strategy  (i) train each branch with its auxiliary task4 https://github.com/fbessho/PyPoi.git5 http://ifc.recod.ic.unicamp.br/BusterNet for Copy-Move Forgery Detection9independently, (ii) freeze both branches and train fusion module, and (iii)unfreeze entire network and fine tune BusterNet end-to-end.|
|||For main task, we also Adam optimizer with categorical crossentropy loss,but use initial learning rate of 1e-2 for fusion training while 1e-5 for fine-tuning.|
|701|Yiran_Zhong_Stereo_Computation_for_ECCV_2018_paper|Li, B., Dai, Y., He, M.: Monocular depth estimation with hierarchical fusion ofdilated cnns and soft-weighted-sum inference.|
|702|cvpr18-Learning to Adapt Structured Output Space for Semantic Segmentation|However, we do not usethe multi-scale fusion strategy [2] due to the memory issue.|
|703|Spatiotemporal Pyramid Network for Video Action Recognition|From the architecture perspective, our network con-stitutes hierarchical fusion strategies which can be trainedas a whole using a unified spatiotemporal loss.|
|||A series ofablation experiments support the importance of each fusionstrategy.|
|||This operator enables efficient training of bilinearfusion operations which can capture full interactions be-tween the spatial and temporal features.|
|||An overview of our spatiotemporal pyramid network,which constitutes a multi-level fusion pyramid of spatial features,long-term temporal features and spatiotemporal attended features.|
|||To learn more global video features,we use multi-path temporal subnetworks to sample opticalflow frames in a longer sequence, and explore several fusionstrategies to combine the temporal information effectively.|
|||Webring in the compact bilinear fusion strategy, which cap-tures full interactions across spatial and temporal features,while significantly reduces the number of parameters of tra-ditional bilinear fusion methods from millions to just sev-eral thousands.|
|||[12] compare mul-tiple CNN connectivity methods in time, including late fu-sion, early fusion and slow fusion.|
|||They propose a spatiotemporal fusion method andclaim that the two-stream networks should be fused at thelast convolutional layer.|
|||First and foremost, we proposea multi-layer pyramid fusion architecture, replacing a 3Dconvolutional layer and a pooling layer in [6], to combinethe spatial and temporal features at different abstraction lev-els.|
|||In contrast, our fusion network is trained end-to-end withone single spatiotemporal loss function.|
|||Spatiotemporal Pyramid NetworkThe spatiotemporal pyramid network supports long-termtemporal fusion and a visual attention mechanism.|
|||Also, wepropose a new spatiotemporal compact bilinear operator toenable a unified modeling of various fusion strategies.|
|||Spatiotemporal Compact Bilinear FusionThe fusion of spatial and temporal features in com-pact representations proves to be the key to learning high-quality spatiotemporal features for video recognition.|
|||Agood fusion strategy should maximally preserve the spatialand temporal information while maximize their interaction.|
|||Typical fusion methods including element-wise sum, con-catenation, and bilinear fusion have been extensively evalu-ated in the convolutional two-stream fusion framework [6].|
|||Bilin-ear fusion allows all spatial and temporal features in differ-ent dimensions to interact with each other in a multiplicativeway.|
|||Since our spatiotemporal pyramid constitutes spatialfeatures, temporal features, and their hierarchy, the bilinearfusion is the only appropriate strategy for our approach.|
|||Specifically, denote by x and y the spatial and temporalfeature vectors respectively, the bilinear fusion is defined asz = vec(xy), where  denotes the outer product xyT, andvec denotes the vectorization of a vector.|
|||Bilinear fusionleads to high dimensional representations with million ofparameters, which will make network training infeasible.|
|||To circumvent the curse of dimensionality, we proposea Spatiotemporal Compact Bilinear (STCB) operator to en-able various fusion strategies.|
|||See Algorithm 1 for the details, where m is thenumber of feature pathways for compact bilinear fusion.|
|||We invoke the algorithm with m pathways of spatialand/or temporal features that need to be fused, which en-ables spatiotemporal fusion into compact representations.|
|||Forthe fusion method, we exploit STCB and make it support ascalable number of input feature maps.|
|||We show that STCBis effective not only for spatiotemporal fusion, but also fortemporal combination.|
|||Another difference betweenour method and [6] is that their temporal fusion includesfusing the features of multiple RGB frames as well, whilewe only combine optical flow representations.|
|||fusion stage (attention) in our architecture.|
|||We observe thatcompact bilinear fusion can preserve the temporal cues tosupervise the spatiotemporal attention module.|
|||Spatiotemporal AttentionThe second level of our spatiotemporal fusion pyramidis a variant of the attention model, which is originally pro-posed in multi-modal tasks [36, 35, 18].|
|||We design our architecture by inject-ing the proposed fusion layers between the convolutionaland the fully connected layers.|
|||These features are then fed intothe next fusion level, the spatiotemporal attention subnet-1532work (red layers), where we use another STCB to fuse thespatial feature maps with the corresponding motion repre-sentations, and offer the attention cues of salient activities.|
|||At the top of the fusion pyramid, all the three previous out-comes are used: the original spatial and temporal featuresthrough average pooling, as well as the resulting attendedfeatures through the attention module.|
|||All models but the VGGnet one followthe same architecture, that the fusion layer is put betweenthe last convolutional layer (i.e.|
|||Ourexperiments show that such a late fusion architecture out-performs its alternatives in which the fusion layer is movedforward.|
|||As shown in Table 3, spatiotemporal compact bilinear1533fusion results in the highest accuracy and improves the per-formance by around 1.5 points.|
|||Table 3 also reveals that the outputdimension makes a difference on the performance of spa-tiotemporal compact bilinear fusion.|
|||Accuracy of various fusion methods on UCF101 (Split 1).|
|||First, amongall these fusion strategies, spatiotemporal compact bilinearfusion presents the best performance.|
|||It is the first time thatcompact bilinear fusion is demonstrated effective for merg-ing multi-path optical flow representations.|
|||The columns in Table 4denotes the number of pathways before the fusion layer.|
|||Among all these models, a 3-path network with spatiotem-poral compact bilinear fusion outperforms the others.|
|||Moreover, this set of experiments testify the value ofcompact bilinear fusion again.|
|||We then try to merge tem-poral and spatial features in advance, while in this scenariothe compact bilinear fusion performs surprisingly well.|
|||We feed the attention module with representa-tions generated by various fusion methods.|
|||Ablation ResultsTo testify the individual effect of fusion approaches wediscuss above, we stack them one by one and test the overallperformance.|
|||From Table 6, we observe that our spatiotemporal fusionmethod improves the average accuracy by 1.5 points.|
|||Fur-thermore, the proposed multi-path temporal fusion methodresults in another 0.4 points performance gain.|
|||ST Fusion denotestwo-stream spatiotemporal compact bilinear fusion.|
|||Multi-T Fu-sion denotes multi-path temporal fusion.|
|||Bothbased on VGG-16, our result (93.2%) is still competitiveto the original two-stream fusion [6] (92.5%).|
|||Thanks to the multi-path temporalfusion, it produces more global features over longer videosequences and can easily differentiate actions that look sim-ilar in short-term snippets but may vary substantially in along-term.|
|||On the contrary, it offers the fusion pyramid some usefuland additional cues for accurate predictions.|
|||This showsthat one component may amend the error of others in thefusion pyramid.|
|||pact bilinear fusion at the top of the pyramid can furtherincrease the discriminative performance.|
|||From the architecture perspective, ournetwork is hierarchical, consisting of multiple fusion strate-gies at different abstraction levels.|
|||These fusion modulesare trained as a whole to maximally complementing eachother.|
|||A series of ablation studies validate the importance ofeach fusion technique.|
|||Weextensively show its benefit over other fusion methods, suchas concatenation and element-wise sum.|
|||Convolutionaltwo-stream network fusion for video action recognition.|
|704|cvpr18-VITAL  VIsual Tracking via Adversarial Learning|Joint sparserepresentation and robust feature-level fusion for multi-cuevisual tracking.|
|705|cvpr18-Visual Question Generation as Dual Task of Visual Question Answering|Visual Question Generation as Dual Task of Visual Question AnsweringYikang Li1, Nan Duan2, Bolei Zhou3, Xiao Chu1, Wanli Ouyang4, Xiaogang Wang1, Ming Zhou21The Chinese University of Hong Kong, Hong Kong, China2Microsoft Research Asia, China3Massachusetts Institute of Technology, USA4University of Sydney, AustraliaAbstractencoderfusiondecoderVisual question answering (VQA) and visual questiongeneration (VQG) are two trending topics in the computervision, but they are usually explored separately despite theirintrinsic complementary relationship.|
|||Problem solving schemes of VQA (top) and VQG (bot-tom), both of which utilize the hencoder-fusion-decoderi pipelinewith Q and A in inverse order.|
|||sharing visual input and tak-ing encoder-fusion-decoder pipeline with inverse input andoutput.|
|||Sowe formulate the dual training of VQA and VQG as learn-ing an invertible cross-modality fusion model that can inferQ or A when given the counterpart conditioned on the givenimage.|
|||From this perspective, we derive an invertible fusionmodule, Dual MUTAN, based on a popular VQA modelMUTAN [3].|
|||Apart from proposing new frameworks, some focus on de-signing effective multimodal fusion schemes [5, 13].|
|||Therefore, VQG can be mod-eled as a multi-modal fusion problem like VQA.|
|||Differentfrom one-to-one translation problems, where there existslarge quantities of available unpaired data, visual questionanswering is a multimodal fusion problem, which is hard tomodel as an unsupervised learning problem.|
|||With attention and MUTAN fusion module, predictedfeatures are obtained.|
|||Then another MUTAN fusion module is usedfor obtaining the answer features a  Rda by fusing vqand q.|
|||We willbriefly review the core part, MUTAN fusion module, whichtakes an image feature vq and a question feature q as input,and predicts the answer feature a.|
|||3.1.1 Review on MUTAN fusion moduleSince language and visual representations are in differentmodalities, merging visual and linguistic features is crucialin VQA.|
|||Bilinear models are recently used in the multi-modal fusion problem, which encodes bilinear interactionsbetween q and vq as follows:a = (T 1 q) 2 vq(1)where the tensor T  Rdq dv da denotes the fully-parametrized operator for answer feature inference, and idenotes the mode-i product between a tensor X and a matrix6118M:Di(X i M) [d1, ...di1, j, di+1...dN ] =X [d1...dN ]M[di, j]Xdi=1(2)To reduce the complexity of the full tensor T , Tuckerdecomposition [3] is introduced as an effective way to fac-torize T as a tensor product between factor matrices Wq,Wv and Wa, and a core tensor Tc:T = ((Tc 1 Wq) 2 Wv) 3 Wa(3)with Wq  Rtq dq , Wv  Rtv dv and Wa  Rtada ,and Tc  Rtq tv ta .|
|||MUTAN is also utilized for visual at-tention module and visual & answer representations fusionat VQG.|
|||Dual MUTANTo leverage the duality of questions and answers, wederive a Dual MUTAN from the original MUTAN to fin-6119Dual FormPrimal Formish the primal (question-to-answer) and its dual (answer-to-question) inference on the feature level with one fusionkernel.|
|||Duality RegularizerWith Dual MUTAN, we have reformulated the featurefusion part of VQA and VQG ( and ) as the inverseprocess to each other.|
|||Cor-respondingly, we implement a dual VQG model with iden-tical feature concatenation fusion.|
|||Since there is no param-eter for fusion part, Dual Training only requires decoder& encoder weight sharing and duality regularizers.|
|||Mu-tan: Multimodal tucker fusion for visual question answering.|
|706|cvpr18-Efficient and Deep Person Re-Identification Using Multi-Level Similarity|Spindle net: Person re-identification with hu-man body region guided feature decomposition and fusion.|
|707|cvpr18-Direction-Aware Spatial Context Features for Shadow Detection|7456concat& 11 convMLIFDeep Supervision (weighted cross entropy loss)DSC moduleDSC moduleDSC moduleDSC moduleDSC moduleDSC moduleDirection-awareSpatial Context Moduleoutputfusion input11 conv& up-samplingFeature Extraction Networkconcat11 convscore maps1stround in spatial RNN 2ndround in spatial RNN (a)input feature map(after 1*1 conv)(c) output map(b) intermediatefeature map3x3 convReLU3x3 convReLU1x1 convDirection-aware Attention Mechanism (shared)(shared)recurrent translation at four directionselement-wise multiplication1x1 convFeaturesr(cid:2919)(cid:2917)(cid:2918)t(cid:3031)(cid:3042)(cid:3041)(cid:3032)(cid:3033)(cid:3047)1x1 conv(cid:3048)(cid:3043)concatrecurrent translation at four directionselement-wise multiplication(cid:3048)(cid:3043)1x1 convReLUconcatDSCr(cid:2919)(cid:2917)(cid:2918)t(cid:3031)(cid:3042)(cid:3041)(cid:3032)(cid:3033)(cid:3047)Context FeaturesAttention WeightsContext FeaturesAttention WeightsFigure 4: The schematic illustration of the direction-aware spatial context module (DSC module).|
|||Theweighted cross entropy loss L equals L1 + L2:L1 = (NnNp + Nn)y log(p)(NpNp + Nn)(1y) log(1p) ,(3)the fusion layer, with a supervision signal added to eachlayer.|
|||After that, we compute the mean of the score mapsover the MLIF layer and the fusion layer to produce the finalprediction map.|
|||Hence,the overall loss function Loverall is a summation of the indi-vidual loss on all the predicted score maps:Loverall = XiwiLi + wmLm + wf Lf ,(5)where wi and Li denote the weight and loss of the i-th layer(level) in the overall network, respectively; wm and Lm arethe weight and loss of the MLIF layer; and wf and Lf arethe weight and loss of the fusion layer, which is the lastlayer in the overall network to produce the final shadow de-tection result; see Figure 2.|
|708|cvpr18-Bootstrapping the Performance of Webly Supervised Semantic Segmentation|Comparedwith the raw prediction on left bottom, the confusion is re-moved in the refined prediction shown on right bottom.|
|||In thisimageground truthraw predictionrefined predictionFigure 5: Illustration of removing confusions of the initialmasks by using image-level labels.|
|||The fusion strategy is as follows:M (f )i =iM (t)M (t)iM (w)i6= backgroundii = backgroundk = M (t)(M (w)iif M (w)if M (w)and Pkotherwisewhere o is a small number.|
|709|cvpr18-Coding Kendall's Shape Trajectories for 3D Action Recognition|Similarly, these latter were fed into different RNNsand their outputs fusion form the global body representa-tion.|
|||Segmentation of high angular resolution diffusion mri usingsparse riemannian manifold clustering.|
|710|cvpr18-Path Aggregation Network for Instance Segmentation|(e) Fully-connected fusion.|
|||But this method extractedfeature maps on input with different scales and then con-ducted feature fusion (with the max operation) to improvefeature selection from the input image pyramid.|
|||Then a fusionoperation (element-wise max or sum) is utilized to fuse fea-ture grids from different levels.|
|||In following sub-networks, pooled feature grids gothrough one parameter layer independently, which is fol-lowed by the fusion operation, to enable network to adaptfeatures.|
|||We apply the fusion operation after the firstlayer.|
|||Mask prediction branch with fully-connected fusion.|
|||Besides bottom-up pathaugmentation, adaptive feature pooling and fully-connectedfusion, we also analyze multi-scale training, multi-GPUsynchronized batch normalization [67, 28] and heavierhead.|
|||Based on our re-implemented baseline (RBL), we gradually add multi-scale training(MST), multi-GPU synchronized batch normalization (MBN), bottom-up path augmentation (BPA), adaptive feature pooling (AFP), fully-connected fusion (FF) and heavier head (HHD) for ablation study.|
|||Fully-connected fusion pre-dicts masks with better quality.|
|||Ablation Studies on Adaptive Feature Pooling Abla-tion studies on adaptive feature pooling are to verify fusionoperation type and location.|
|||As shown in Table 4, adaptive feature pooling is not sen-sitive to the fusion operation type.|
|||Ablation study on fully-connected fusion on val-2017 interms of mask AP.|
|||In our final system, we use maxfusion operation behind the first parameter layer.|
|||max,sum and product operations are used for fusion.|
|||They clearlyshow that staring from conv3 and taking sum for fusion pro-duce the best results.|
