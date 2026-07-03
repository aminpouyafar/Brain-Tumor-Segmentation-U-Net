# Experiment: U-Net with ResNet34 Encoder for Medical Image Segmentation

# Model Configuration

* **Architecture:** U-Net
* **Encoder:** ResNet34 (Pretrained on ImageNet)
* **Input Channels:** 3
* **Output Classes:** 1 (Binary Segmentation)

The segmentation model is based on the U-Net architecture with a pretrained ResNet34 encoder.
Transfer learning was employed by initializing the encoder with ImageNet pretrained weights, allowing the encoder to leverage rich low-level and high-level visual representations learned from a large-scale natural image dataset.

This significantly accelerated convergence and improved feature extraction for brain tumor segmentation.

To reduce model complexity and maintain stable optimization, only the three deepest skip connections were utilized. The shallow skip connection from the initial convolutional layer was intentionally omitted in the baseline architecture.

An additional experiment was conducted by incorporating this shallow skip connection into the decoder. Although the modified architecture achieved competitive Dice scores, training became noticeably less stable, exhibiting larger fluctuations in validation performance. Furthermore, the final segmentation accuracy was slightly lower than that of the baseline model. Consequently, the original architecture without the shallow skip connection was selected as the final model.

---

# Data Preprocessing and Augmentation

Prior to training, all images and corresponding masks were resized to a fixed spatial resolution.

The input images were normalized before being fed into the network.

To improve the generalization capability of the model and reduce overfitting, several data augmentation techniques were applied during training, including:

* Random horizontal flipping
* Shift, scale, and rotation augmentation
* Random brightness and contrast adjustment
* Gaussian noise injection
---
### Extended Data Augmentation Experiment

An additional experiment was conducted using a more aggressive augmentation strategy to investigate whether increasing data diversity could further improve segmentation performance.

The additional augmentations included:

* Elastic Transformation
* Gaussian Blur

These augmentations were applied in addition to the baseline augmentation pipeline.

Although the model converged normally and achieved stable optimization, the best validation Dice score reached **0.8684**, which was lower than the baseline ResNet34 model (**0.8777**). This suggests that the original augmentation strategy already provided sufficient variability for the current dataset, while stronger geometric and appearance transformations introduced unnecessary variations that slightly reduced segmentation accuracy.
---

# Training Configuration

## Optimizer

* Adam Optimizer

### Hyperparameters

* Initial Learning Rate: **1 × 10⁻⁴**
* Weight Decay: **1 × 10⁻⁵**
* Number of Epochs: **30**
* Early Stopping Patience: **8 epochs**

---

# Learning Rate Scheduling

A **ReduceLROnPlateau** scheduler was employed to automatically reduce the learning rate whenever the validation Dice score stopped improving.

### Scheduler Parameters

* mode = `"max"`
* factor = `0.5`
* patience = `2`

The scheduler gradually reduced the learning rate throughout training, enabling more stable optimization during later epochs and helping the model converge to a better local optimum.

---

# Evaluation Metric

Model performance was evaluated using the **Dice Similarity Coefficient (DSC)**, which is one of the most widely used evaluation metrics for medical image segmentation because it directly measures the overlap between predicted masks and ground-truth annotations.

---

# Baseline U-Net Experiments

Before introducing transfer learning, a **custom implementation of the original U-Net architecture** was developed and trained entirely from scratch to establish a baseline for comparison. The network was implemented without using any pretrained encoder or external pretrained weights, and all parameters were randomly initialized prior to training.

The initial implementation achieved a best validation Dice score of **0.5791**, indicating limited segmentation capability.

To improve the baseline model, several architectural and training refinements were introduced, including:

* Batch Normalization
* Dropout regularization
* Extended data augmentation
* Weight decay (L2 regularization)
* Learning rate scheduling
* Early stopping
* Model checkpointing

These modifications substantially improved optimization stability and reduced overfitting, resulting in a significant increase in segmentation performance.

The enhanced baseline model achieved:

* **Best Validation Dice Score:** **0.8190**
* **Best Epoch:** **29**

This represents an improvement of approximately **24 percentage points** over the original implementation (0.5791 → 0.8190), demonstrating the considerable impact of proper regularization and training strategies.

Although this improved baseline produced competitive results, it remained inferior to the U-Net utilizing a pretrained ResNet34 encoder, highlighting the effectiveness of transfer learning for medical image segmentation.

---

# Loss Function Experiments

To determine the most effective optimization objective, several loss functions were investigated throughout the experiments.

The evaluated configurations included:

* BCE Loss + Dice Loss
* BCE Loss + Tversky Loss
* Dice Loss + Focal Loss

Each loss function was trained using identical optimization settings to enable a fair comparison.

---

# Comparative Experiments

Multiple architectural and optimization experiments were performed throughout this project.

These experiments investigated:

* Different encoder architectures
* Different skip-connection configurations
* Different segmentation loss functions

The following models were evaluated:

* Baseline U-Net (trained from scratch)
* Improved Baseline U-Net
* U-Net + ResNet34
* U-Net + ResNet34 with additional shallow skip connection
* U-Net + ResNet50
* U-Net + ResNet34 using BCE + Tversky Loss
* U-Net + ResNet34 using Dice + Focal Loss

---

# Training Results

The U-Net with the pretrained ResNet34 encoder demonstrated rapid convergence during the early stages of training.

The validation Dice score increased steadily throughout training and reached its highest value during the later epochs.

### Best Performance

* **Best Validation Dice Score:** **0.8777**
* **Best Validation Loss:** **0.0935**
* **Best Epoch:** **29**

---

# Observations

## Baseline (BCE + Dice)

The BCE and Dice Loss combination achieved the highest segmentation performance among all evaluated configurations. Training remained stable and consistently converged toward better Dice scores.

---

## Skip Connection Experiment

Introducing the shallow skip connection did not improve segmentation quality.

Instead:

* Validation Dice fluctuated more.
* Training stability decreased.
* Final Dice score slightly dropped.

This indicates that additional low-level features introduced unnecessary noise for the current dataset.

---

## ResNet50 Experiment

Replacing ResNet34 with ResNet50 increased computational complexity while providing no improvement in segmentation accuracy.

The deeper encoder required considerably more computation and GPU memory, yet produced a lower Dice score than ResNet34.

---

## BCE + Tversky Loss

Tversky Loss was evaluated because it is specifically designed to alleviate class imbalance.

Although convergence occurred rapidly, the model saturated earlier and achieved a lower final Dice score compared to the baseline BCE + Dice configuration.

---

## Dice + Focal Loss

Focal Loss was combined with Dice Loss to further address foreground-background imbalance.

This configuration achieved competitive segmentation performance and stable optimization, but it still remained slightly below the baseline BCE + Dice model.

---
## Extended Data Augmentation

A more aggressive augmentation pipeline incorporating Elastic Transformation and Gaussian Blur was also evaluated.

Despite increasing the diversity of the training samples, the resulting model achieved a best validation Dice score of **0.8684**, which remained below the baseline performance.

These findings indicate that stronger augmentation did not improve generalization for the current dataset and may have introduced unrealistic image variations that slightly degraded segmentation performance.
---
# Discussion

The conducted experiments demonstrate that increasing architectural complexity does not necessarily improve segmentation performance.

For the current dataset:

* A deeper encoder (ResNet50) did not outperform ResNet34.
* Adding an additional shallow skip connection slightly degraded performance.
* More sophisticated loss functions (Tversky and Focal Loss) produced competitive results but failed to surpass the standard BCE + Dice combination.
* Transfer learning proved to be more beneficial than increasing encoder complexity for this dataset.

These results indicate that transfer learning had a greater impact on segmentation performance than simply increasing encoder depth or adopting more complex loss functions.

---

# Conclusion

The U-Net model with a pretrained ResNet34 encoder achieved the best overall segmentation performance among all evaluated configurations.

The highest validation Dice score of **0.8777** was obtained using the combination of **Binary Cross Entropy Loss** and **Dice Loss**.

Experimental comparisons further revealed that:

* ResNet34 outperformed the deeper ResNet50 encoder.
* Removing the shallow skip connection improved optimization stability.
* BCE + Dice Loss remained superior to both BCE + Tversky and Dice + Focal Loss for this dataset.

Furthermore, the experiments demonstrate that **transfer learning contributes more to segmentation performance than increasing encoder depth or employing more sophisticated loss functions** for the current dataset.

Overall, the U-Net with a pretrained ResNet34 encoder and BCE + Dice Loss achieved the best balance between segmentation accuracy, training stability, and computational efficiency, making it an effective solution for brain tumor segmentation on the LGG MRI dataset.

Future work may include:

1. Attention U-Net
2. UNet++
3. Transformer-based segmentation models
4. Deep supervision
5. Advanced data augmentation techniques

---

# Key Findings

* Transfer learning with ResNet34 significantly improved segmentation performance compared to training U-Net from scratch.
* Proper regularization techniques (Batch Normalization, Dropout, Data Augmentation, Weight Decay, Early Stopping, and Learning Rate Scheduling) substantially improved the baseline U-Net performance from **0.5791** to **0.8190**.
* Additional shallow skip connections did not improve segmentation accuracy and reduced training stability.
* Increasing encoder depth to ResNet50 increased computational cost without improving Dice score.
* Among all evaluated loss functions, **BCE + Dice Loss consistently achieved the highest segmentation accuracy**.
* The U-Net + ResNet34 model achieved the best overall balance between segmentation accuracy, optimization stability, and computational efficiency.

---

# Experimental Comparison

| Model | Encoder | Loss Function | Best Dice | Best Epoch | Remarks |
|-------|----------|---------------|----------:|-----------:|---------|
| Baseline U-Net (Initial) | None | BCE + Dice | 0.5791 | 12 | Initial implementation |
| Baseline U-Net (Improved) | None | BCE + Dice | 0.8190 | 29 | BatchNorm, Dropout, Data Augmentation, Weight Decay, Learning Rate Scheduling and Early Stopping |
| **U-Net + ResNet34** | **ResNet34** | **BCE + Dice** | **0.8777** | **29** | **Best overall performance** |
| U-Net + ResNet34 (Shallow Skip Added) | ResNet34 | BCE + Dice | 0.8737 | 29 | Lower accuracy and reduced training stability |
| U-Net + ResNet50 | ResNet50 | BCE + Dice | 0.8686 | 24 | Higher computational cost without accuracy improvement |
| U-Net + ResNet34 | ResNet34 | BCE + Tversky | 0.8621 | 12 | Faster convergence but lower final Dice score |
| U-Net + ResNet34 | ResNet34 | Dice + Focal | 0.8719 | 30 | Competitive performance for class imbalance |
| U-Net + ResNet34 (Extended Augmentation) | ResNet34 | BCE + Dice | 0.8684 | 28 | ElasticTransform + GaussianBlur |

---

# Reproducibility

The experiments can be reproduced using the notebooks provided in this repository.

The trained model weights are not included because of GitHub file size limitations. After downloading the corresponding `.pth` files, place them inside the `models/` directory before running `predict.py`.

The project structure, notebooks, and scripts are organized to facilitate reproducibility and future experimentation.