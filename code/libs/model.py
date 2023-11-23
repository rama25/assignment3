import math
import torch
import torchvision

from torchvision.models import resnet18,resnet, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        ###...my code....
        output = [] # Create a list and append all the classification logits in it
        for features in x:
          # Pass the features corresponding to a depth to the entire head
          logits = self.conv(features)
          output.append(self.cls_logits(logits)) # shape will be N x C x H x W with C being no of classes (20 in our case)
          # Rearrangement of outputs is done in compute loss function
        return output
        #return x


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            #conv.append(nn.ReLU(inplace=True))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            #nn.ReLU(inplace=True)
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        ###..my code ....
        # Now we have 2 lists, one for regression outputs and one for centerness outputs
        out_regress = []
        out_centerness = []

        for feature in x:
          # Pass every feature from every depth (feature pyramid) to the regression head to get logits
          logits = self.conv(feature)
          # These logits are passed to bbox conv to get 4 numbered output and bbox centerness to get 1 number output
          out_regress.append(self.bbox_reg(logits)) # shape will be N x 4 x H x W
          out_centerness.append(self.bbox_ctrness(logits)) # shape will be N x 1 x H x W
          # Rearrangement of outputs is done in compute loss function

        return out_regress, out_centerness



class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )
        #my code
        # backbone network (resnet18)
        #self.backbone = create_feature_extractor(
        #    resnet18(weights=ResNet18_Weights.DEFAULT), return_nodes=return_nodes
        #)

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode) 
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function changes depending on if the model is
    in training or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        #my code
        full_ground_truth_boxes_targets = []
        full_ground_truth_classes_targets = []
        
        full_ground_truth_regress_out = []
        full_ground_truth_centerness_targets = []

        for target_id,target in enumerate(targets):
          # Get the coordinates of ground truth boxes for an example target_id (it can have M objects)  
          ground_truth_boxes = target['boxes']  # Mx4

          # Compute center of a box; which looks like - (x0, y0, x1, y1)
          ground_truth_centers = (ground_truth_boxes[:, :2] + ground_truth_boxes[:, 2:]) / 2  # Mx2 (M boxes)
          
          every_stride_ground_truth_classes_targets = []
          every_stride_ground_truth_boxes_targets = []
          every_stride_ground_truth_regress_outputs = []
          every_stride_ground_truth_centerness_targets = []
        
          # Looping over all pyramid levels
          for level,stride in enumerate(strides):
            
            # For a particular level, get the predicted points/anchors using the point generator
            predictions = points[level].view(-1,2)   # HWx2,  or Nx2 (N anchors)

            # Get the distance between the predicted points and the ground truth center
            pairwise_predictions_matching = predictions[:,None,:] - ground_truth_centers[None,:,:] # NxMx2

            # Take only the points which are close to a particular threshold i.e the sampling_radius*stride
            pairwise_predictions_matching = pairwise_predictions_matching.abs_().max(dim=2).values < (self.center_sampling_radius*stride) # NxM

            # Get the x, y for the predicted anchors. Unbind removes the tensor dimension and returns as numbers
            x, y = predictions.unsqueeze(dim=2).unbind(dim=1)  # Nx1,Nx1
            # Top-left and bottom-right corners
            x0, y0, x1, y1 = ground_truth_boxes.unsqueeze(dim=0).unbind(dim=2)  # 1xM each

            # Concatenates a sequence of tensors along a new dimension using torch stack
            pairwise_distance = torch.stack([x - x0, y - y0, x1 - x, y1 - y], dim=2)

            pairwise_predictions_matching &= pairwise_distance.min(dim=2).values > 0   # NxM (Inside the Ground truth box)
             
            target_distance = pairwise_distance.abs().max(dim=2).values
            # Get the lower and upper values from regression range
            lower, upper = reg_range[level][0], reg_range[level][1]
            
            # The target distance should be in the range given by the particular level in regression range. 
            # So, for every level, the lower and upper bounds are fixed
            pairwise_predictions_matching &= (target_distance > lower) & (target_distance < upper)  # N,M

            # match the Ground truth box with minimum area, if there are multiple Ground truth matches
            # Area is nothing but the l*b where l and b can be computed by the top-left and bottom-right coordinates
            ground_truth_areas = (ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]) * (ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1])  # M
            pairwise_predictions_matching = pairwise_predictions_matching.to(torch.float32) * (1e8 - ground_truth_areas[None, :])
            min_vals, matched_idx = pairwise_predictions_matching.max(dim=1)  # R, per-anchor match
            matched_idx[min_vals < 1e-5] = -1  # unmatched anchors are assigned -1, (N,)

            # Clip the ground truth class targets and boxes targets with minimum value as 0
            ground_truth_class_targets = target["labels"][matched_idx.clip(min=0)]   # (N,)
            ground_truth_boxes_targets = target["boxes"][matched_idx.clip(min=0)]      # (N,4)
            ground_truth_class_targets[matched_idx < 0] = -1

            top_left_pred = predictions - ground_truth_boxes_targets[:,:2]         #(N,2), calculating l* and t* for N points
            bottom_right_pred = ground_truth_boxes_targets[:,2:] - predictions         #(N,2) i.e calculating r* and b* for N points
            target_pred = torch.cat([top_left_pred,bottom_right_pred],dim=-1)/stride     #(N,4)
            
            # Rearranging to get the lr and tb, will be useful while calculating the centerness metric
            left_right = target_pred[:, [0, 2]]
            top_bottom = target_pred[:, [1, 3]]

            # Centerness target formula from paper
            ground_truth_centerness_targets = torch.sqrt(
                (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0])
                * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
            )          # (N,)

            # For the particular level and image, get the regression outputs and reshape it
            regression_output = (reg_outputs[level][target_id].view(4,-1)).permute(1,0)*stride     # (N(HW),4)
            ground_truth_regress_output = torch.cat([predictions-regression_output[:,:2],predictions+regression_output[:,2:]],dim=-1)   # HWx4,  or Nx4 (N anchors)
            
            # Append all the ground truth targets related to one stride (feature level)
            every_stride_ground_truth_classes_targets.append(ground_truth_class_targets)
            every_stride_ground_truth_boxes_targets.append(ground_truth_boxes_targets)
            every_stride_ground_truth_regress_outputs.append(ground_truth_regress_output)
            every_stride_ground_truth_centerness_targets.append(ground_truth_centerness_targets)

          # Append all the ground truth targets for all stride levels (feature pyramid) to a bigger list
          full_ground_truth_classes_targets.append(torch.cat(every_stride_ground_truth_classes_targets,dim=0))  # List, batchsz. (A,1) 
          full_ground_truth_boxes_targets.append(torch.cat(every_stride_ground_truth_boxes_targets,dim=0))          # List, (A,4) 
          full_ground_truth_regress_out.append(torch.cat(every_stride_ground_truth_regress_outputs,dim=0))     # list ((A,4) : A = (HW)_1+(HW)_2+(HW)_3
          full_ground_truth_centerness_targets.append(torch.cat(every_stride_ground_truth_centerness_targets,dim=0))


        # Reshaping the classification and centerness logits
        cls_logits = [t.view(t.shape[0],t.shape[1],-1) for t in cls_logits]
        #reg_outputs = [t.view(t.shape[0],t.shape[1],-1) for t in reg_outputs]
        ctr_logits = [t.view(t.shape[0],t.shape[1],-1) for t in ctr_logits]
        
        # Contiguous memory allocation for all the predicted classification and centerness logits.
        cls_logits,ctr_logits = (
                      torch.cat(cls_logits,dim=2).permute(0,2,1).contiguous(), # (bs,A,C)
                      torch.cat(ctr_logits,dim=2).permute(0,2,1).contiguous()) # (bs,A,1)
        
        full_ground_truth_boxes_targets, full_ground_truth_classes_targets,full_ground_truth_regress_out,full_ground_truth_centerness_targets = (
            torch.stack(full_ground_truth_boxes_targets),
            torch.stack(full_ground_truth_classes_targets),
            torch.stack(full_ground_truth_regress_out),
            torch.stack(full_ground_truth_centerness_targets)
        )      # [bs,A,4], [bs,A,1] , [bs,A,4], [bs,A,1]

        # compute foregroud. When the target labels are >=0, it's forground (note the =)
        foregroud_mask = full_ground_truth_classes_targets >= 0
        num_foreground = foregroud_mask.sum().item()

        # classification loss - Sigmoid focal loss
        ground_truth_class_targets = torch.zeros_like(cls_logits)
        # Similar to One-hot encoding, just give 1 to the index where the class object idx is present
        ground_truth_class_targets[foregroud_mask, full_ground_truth_classes_targets[foregroud_mask]] = 1.0
        cls_loss = sigmoid_focal_loss(cls_logits, ground_truth_class_targets, reduction="sum")

        # regression loss - GIoU loss
        # It takes the ground truth regression labels and the predicted regression labels, but only for the foreground points
        reg_loss = giou_loss(full_ground_truth_regress_out[foregroud_mask],full_ground_truth_boxes_targets[foregroud_mask],reduction='sum')
        
        # centerness loss - Binary Cross Entropy
        # It also takes the ground truth centerness logits and the predicted centerness values only for the foreground points
        ctr_logits = ctr_logits.squeeze(dim=-1)
        ctr_loss = nn.functional.binary_cross_entropy_with_logits(
            ctr_logits[foregroud_mask], full_ground_truth_centerness_targets[foregroud_mask], reduction="sum"
        )

        # Normalize the losses for only foreground points and create a dictionary
        losses = {}
        losses['cls_loss'] = cls_loss / max(1,num_foreground)
        losses['reg_loss'] = reg_loss / max(1,num_foreground)
        losses['ctr_loss'] = ctr_loss / max(1,num_foreground)
        final_loss = cls_loss + reg_loss + ctr_loss
        losses['final_loss'] = final_loss
        return losses
        
    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """
    

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        #my code
        detections = []

        cls_logits = [t.view(t.shape[0],t.shape[1],-1).permute(0,2,1) for t in cls_logits]  # List. (N,HW,C)
        reg_outputs = [t.view(t.shape[0],t.shape[1],-1).permute(0,2,1) for t in reg_outputs]  # List. (N,HW,4)
        ctr_logits = [t.view(t.shape[0],t.shape[1],-1).permute(0,2,1) for t in ctr_logits]  # List. (N,HW,1)

        # looping over every image
        for image_id in range(len(image_shapes)):
            image_shape = image_shapes[image_id]

            # For every image, we have 3 lists which stores the box coordinates, the score/confidcence of prediction
            # and the label information of the prediction
            image_boxes = []
            image_scores = []
            image_labels = []

            # loop over all the feature pyramid levels
            for level, stride in enumerate(strides):
                # Find the classification, regression and centerness score for every image at every level 
                # HW --> N where the understanding is that the number of anchors in a feature space at a particular level
                # will be equal to the dimensions of the feature space because every point in a particular level in feature space
                # is a potential prediction and is mapped back to image space to understand whether it has some information or not
                cls_logits_level = cls_logits[level][image_id]   # (HW,C)
                ctr_logits_level = ctr_logits[level][image_id]   # (HW,1)
                reg_outputs_level = reg_outputs[level][image_id] # (HW,4)

                num_classes = cls_logits_level.shape[-1]
                # compute scores, formula from paper
                scores_level = torch.sqrt(torch.sigmoid(cls_logits_level) * torch.sigmoid(ctr_logits_level)).flatten()
                # (HW,C) -->(HW*C) i.e for C classes, we get scores

                # threshold scores. Take the boxes which satisfies a particular threshold
                keep_ids = scores_level > self.score_thresh
                scores_level_thresholded = scores_level[keep_ids]
                topk_idxs = torch.where(keep_ids)[0]
                # Getting the number of boxes satisfying the threhold criteria
                num_ids = min(len(topk_idxs),self.topk_candidates)

                # keep only top K candidates
                scores_level_thresholded_top_k, top_k_candidate_indices = scores_level_thresholded.topk(k = num_ids, dim = 0)
                topk_idxs = topk_idxs[top_k_candidate_indices]

                box_ids = torch.div(topk_idxs,num_classes,rounding_mode='floor')

                labels_per_level = topk_idxs % num_classes

                predictions = points[level].view(-1,2)
                reg_out = reg_outputs_level*stride     # (N(HW),4)
                boxes_pred = torch.cat([predictions-reg_out[:,:2],predictions+reg_out[:,2:]],dim=-1)  
                # HWx4,  or Nx4 (N anchors) as each pixel is considered as a potential anchor in the feature space.

                boxes_pred = boxes_pred[box_ids]

                # Get all the predicted boxes, x is indexed at 0 and y at 1
                boxes_pred_x = boxes_pred[...,0::2]
                boxes_pred_y = boxes_pred[...,1::2]
                # clip boxes so that it stays within image. Sometimes, the predictions can be outside the image dimensions!
                boxes_pred_x = boxes_pred_x.clamp(min = 0, max = image_shape[1])
                boxes_pred_y = boxes_pred_y.clamp(min = 0, max = image_shape[0])
                
                boxes_level_clipped = torch.stack([boxes_pred_x[:,0],boxes_pred_y[:,0],boxes_pred_x[:,1],boxes_pred_y[:,1]],dim=-1)

                image_boxes.append(boxes_level_clipped)
                image_scores.append(scores_level_thresholded_top_k)
                image_labels.append(labels_per_level + 1)    
            
            # Stack the predicted boxes, scores and labels so it can be passed to nms as a batch
            image_boxes = torch.cat(image_boxes, dim = 0)
            image_scores = torch.cat(image_scores, dim = 0)
            image_labels = torch.cat(image_labels, dim = 0)

            # non-maximum suppression. Remove the boxes for which the predicted threshold is less than the nms_threshold
            # Incase it has a lots of predictions in an image, restrict it by how many detections we have per image 
            # (from detections_per_images)
            boxes_top_keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)[ : self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[boxes_top_keep],
                    "scores": image_scores[boxes_top_keep],
                    "labels": image_labels[boxes_top_keep],
                }
            )

        return detections
