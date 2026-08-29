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
   - Across a 640 x 640 spatial grid, the drone occupies less than 0.05% of the total pixels, creating a >10,000 : 1 ratio of background anchors to foreground targets.

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
| **Loss Formulation** | Smooth L1 + BCE | CIoU + Focal Loss | CIoU + Focal Loss + Cls Loss |
| **Parameters** | 1.17M | 3.87M | **4.12M** |
| **FLOPs (640x640)** | 12.8 GFLOPs | 21.6 GFLOPs | **24.8 GFLOPs** |
| **Inference (FPS)** | 180.2 FPS | 73.6 FPS | **68.8 FPS** |

---

## 3. Detailed Component Deep-Dive: Model 3 (DroneNet-FPN-Attention)

### 3.1 High-Resolution Feature Pyramid Network (HR-FPN)
Unlike standard FPNs that operate on strides {8, 16, 32} (P3, P4, P5), our **High-Resolution FPN** shifts receptive attention to strides {4, 8, 16} (P2, P3, P4):

- **$P_2$ (Stride 4, Resolution 160 x 160)**: Dedicated to tiny drones (4px - 24px). Preserves microscopic spatial features such as rotor tips and landing skids.
- **$P_3$ (Stride 8, Resolution 80 x 80)**: Dedicated to medium-scale drones (24px - 64px). Balances semantic context and localization precision.
- **$P_4$ (Stride 16, Resolution 40 x 40)**: Dedicated to larger, close-range UAVs (> 64px). Captures structural fuselage context.

### 3.2 Directional Coordinate Attention (CA)
Standard Squeeze-and-Excitation (SE) attention performs 2D global average pooling, which destroys precise spatial coordinate localization vital for pin-pointing tiny drones.

**Coordinate Attention** replaces 2D pooling with two 1D spatial pooling operations along the horizontal ($X$) and vertical ($Y$) axes:

$$\mathbf{z}_c^h(h) = \frac{1}{W} \sum_{0 \le i < W} x_c(h, i), \quad \mathbf{z}_c^w(w) = \frac{1}{H} \sum_{0 \le j < H} x_c(j, w)$$

1. The directional vectors are concatenated and transformed via a shared 1 x 1 convolution:
   $$\mathbf{f} = \delta\left( \text{BatchNorm}\left( \text{Conv}_{1\times 1}\left( [\mathbf{z}^h, \mathbf{z}^w] \right) \right) \right)$$
2. The intermediate tensor $\mathbf{f} \in \mathbb{R}^{C/r \times (H+W)}$ is split back into $\mathbf{f}^h \in \mathbb{R}^{C/r \times H}$ and $\mathbf{f}^w \in \mathbb{R}^{C/r \times W}$.
3. Two independent 1 x 1 convolutions and sigmoid activations yield coordinate attention weights:
   $$\mathbf{g}^h = \sigma\left(\text{Conv}_h(\mathbf{f}^h)\right), \quad \mathbf{g}^w = \sigma\left(\text{Conv}_w(\mathbf{f}^w)\right)$$
4. The output feature map is reweighted along both orthogonal directions:
   $$\mathbf{y}_c(i, j) = x_c(i, j) \times \mathbf{g}_c^h(i) \times \mathbf{g}_c^w(j)$$

### 3.3 Receptive Field Block (RFB)
The RFB module applies multi-rate dilated convolutions ($r \in \{1, 2, 3\}$) simulating human vision to capture context across multiple receptive scales without spatial downsampling.

---

## 4. Multi-Task Loss Formulation

$$\mathcal{L}_{\text{total}} = \lambda_{\text{cls}} \mathcal{L}_{\text{Focal}} + \lambda_{\text{obj}} \mathcal{L}_{\text{Obj}} + \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}}$$

- **Focal Loss** ($\gamma=2.0, \alpha=0.25$): Addresses extreme background anchor imbalance.
- **Complete-IoU (CIoU) Loss**: Enforces overlap area, center Euclidean distance, and aspect ratio consistency simultaneously:
  $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v$$
