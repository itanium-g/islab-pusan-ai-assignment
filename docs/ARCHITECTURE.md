# 🛸 DroneNet-FPN-Attention: Architecture & Theoretical Foundations

This document provides a comprehensive technical breakdown of the deep neural network architectures, custom mathematical loss formulations, coordinate attention mechanisms, and sequence-aware data pipelines designed for detecting small Unmanned Aerial Vehicles (UAVs) under severe atmospheric degradation.

---

## 1. Physical Domain Challenges & Empirical Motivations

Small UAV object detection presents extreme physical and mathematical challenges compared to standard general-domain object detection datasets (e.g., COCO, Pascal VOC):

1. **Extreme Small-Target Distribution (< 32px)**:
   - Analysis of the 2,400 raw frames (4,800 drone instances) reveals that **95.42% of all ground-truth bounding boxes are smaller than 32 x 32 pixels**, with **51.52% smaller than 16 x 16 pixels**.
   - Standard object detectors with deep downsampling (stride 32 or 64) completely obliterate sub-16px targets in high-level feature maps.
2. **Atmospheric Scattering & Specular Glare**:
   - The dataset contains synthetic AirSim captures spanning 6 adverse environmental conditions: dense fog, hazy urban, high-contrast sunny glare, rainy overcast, low-light sunset, and motion-blurred drone maneuvers.
   - Atmospheric fog induces uniform contrast reduction (Koschmieder's Law: $I(x) = J(x)e^{-\beta d(x)} + A(1 - e^{-\beta d(x)})$), obscuring drone silhouettes and merging rotor edges with building textures.
3. **Severe Class Imbalance**:
   - Across a 640 x 640 spatial grid, the drone occupies less than 0.05% of the total pixels, creating a > 10,000 : 1 ratio of background anchors to foreground targets.

---

## 2. Comparative Architecture Overview

We developed three progressively sophisticated neural architectures strictly **from scratch** without any pretrained backbones:

| Architectural Component | Model 1: Vanilla Base CNN | Model 2: DroneNet-FPN | Model 3: DroneNet-FPN-Attention (Best) |
| :--- | :--- | :--- | :--- |
| **Backbone** | 4-Stage ConvNet | 4-Stage Res-ConvNet | 4-Stage Residual Backbone |
| **Multi-Scale FPN** | ❌ (Single P3 only) | ✅ (P2, P3, P4 Top-Down) | ✅ High-Res FPN (P2, P3, P4) |
| **Spatial Strides** | Stride 8 | Strides 4, 8, 16 | Strides 4, 8, 16 (160 / 80 / 40 px) |
| **Attention Module** | ❌ None | ❌ None | ✅ Coordinate Attention (CA) |
| **Multi-Scale Context**| ❌ Standard Conv | ❌ Standard Conv | ✅ Receptive Field Block (RFB) |
| **Decoupled Heads** | Coupled ConvHead | Shared FPN Heads | Decoupled Cls / Reg Heads |
| **Anchor Scales** | 3 anchors (P3) | 9 anchors (3 per scale) | 9 calibrated multi-scale anchors |
| **Loss Formulation** | Smooth L1 + BCE | CIoU + Focal Loss | Focal + CIoU + Label-Smoothed CE |
| **Parameters** | 1.17M | 3.87M | **4.12M** |
| **FLOPs (640x640)** | 12.8 GFLOPs | 21.6 GFLOPs | **24.8 GFLOPs** |
| **Inference (FPS)** | 186.6 FPS | 80.1 FPS | **74.6 FPS** |

---

## 3. Detailed Component Deep-Dive: Model 3 (DroneNet-FPN-Attention)

### 3.1 High-Resolution Feature Pyramid Network (HR-FPN)
Unlike standard FPNs that operate on strides {8, 16, 32} (P3, P4, P5), our **High-Resolution FPN** shifts receptive attention to strides {4, 8, 16} (P2, P3, P4):

- **P2 (Stride 4, Resolution 160 x 160)**: Dedicated to tiny drones (4px - 24px). Preserves microscopic spatial features such as rotor tips and landing skids.
- **P3 (Stride 8, Resolution 80 x 80)**: Dedicated to medium-scale drones (24px - 64px). Balances semantic context and localization precision.
- **P4 (Stride 16, Resolution 40 x 40)**: Dedicated to larger, close-range UAVs (> 64px). Captures structural fuselage context.

### 3.2 Directional Coordinate Attention (CA)
Standard Squeeze-and-Excitation (SE) attention performs 2D global average pooling, which destroys precise spatial coordinate localization vital for pin-pointing tiny drones.

**Coordinate Attention** replaces 2D pooling with two 1D spatial pooling operations along the horizontal ($X$) and vertical ($Y$) axes:

$$
\mathbf{z}_c^h(h) = \frac{1}{W} \sum_{i=0}^{W-1} x_c(h, i), \quad \mathbf{z}_c^w(w) = \frac{1}{H} \sum_{j=0}^{H-1} x_c(j, w)
$$

1. The directional vectors are concatenated and transformed via a shared $1 \times 1$ convolution:
   $$
   \mathbf{f} = \delta\left( \text{BatchNorm}\left( \text{Conv}_{1\times 1}\left( [\mathbf{z}^h, \mathbf{z}^w] \right) \right) \right)
   $$
2. The intermediate tensor $\mathbf{f} \in \mathbb{R}^{C/r \times (H+W)}$ is split back into $\mathbf{f}^h \in \mathbb{R}^{C/r \times H}$ and $\mathbf{f}^w \in \mathbb{R}^{C/r \times W}$.
3. Two independent $1 \times 1$ convolutions and sigmoid ($\sigma$) activations yield coordinate attention weights:
   $$
   \mathbf{g}^h = \sigma\left(\text{Conv}_h(\mathbf{f}^h)\right), \quad \mathbf{g}^w = \sigma\left(\text{Conv}_w(\mathbf{f}^w)\right)
   $$
4. The output feature map is reweighted along both orthogonal directions:
   $$
   \mathbf{y}_c(i, j) = x_c(i, j) \times \mathbf{g}_c^h(i) \times \mathbf{g}_c^w(j)
   $$

### 3.3 Receptive Field Block (RFB)
The RFB module applies multi-rate dilated convolutions ($r \in \{1, 2, 3\}$) simulating human vision to capture context across multiple receptive scales without spatial downsampling.

---

## 4. Custom Multi-Task Composite Loss Formulation

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{obj}} \mathcal{L}_{\text{obj}} + \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}}
$$

### 4.1 Focal Objectness Loss ($\mathcal{L}_{\text{obj}}$)
To handle the $> 10,000 : 1$ background-to-foreground class imbalance, we apply Focal Loss with focusing parameter $\gamma = 2.0$ and weighting factor $\alpha_t = 0.25$:

$$
\mathcal{L}_{\text{obj}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

### 4.2 Complete-IoU (CIoU) Bounding Box Loss ($\mathcal{L}_{\text{box}}$)
Enforces overlap area, center Euclidean distance, and aspect ratio consistency simultaneously:

$$
\mathcal{L}_{\text{box}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{\text{gt}})}{c^2} + \alpha_{\text{ciou}} v
$$

where:

$$
v = \frac{4}{\pi^2}\left(\arctan\frac{w^{\text{gt}}}{h^{\text{gt}}} - \arctan\frac{w}{h}\right)^2, \quad \alpha_{\text{ciou}} = \frac{v}{(1 - \text{IoU}) + v}
$$

### 4.3 Label-Smoothed Classification Loss ($\mathcal{L}_{\text{cls}}$)

$$
\mathcal{L}_{\text{cls}} = -\sum_{k=1}^K \left[ y_k^{\text{ls}} \log(\hat{c}_k) + (1 - y_k^{\text{ls}}) \log(1 - \hat{c}_k) \right]
$$

where $y_k^{\text{ls}} = y_k(1 - \epsilon) + 0.5\epsilon$ with label smoothing factor $\epsilon = 0.05$.

### 4.4 Cosine Annealing Learning Rate Schedule

$$
\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_0 - \eta_{\text{min}})\left(1 + \cos\left(\frac{t}{T_{\text{max}}}\pi\right)\right)
$$
