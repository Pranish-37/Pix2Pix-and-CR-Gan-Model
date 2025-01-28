# Pix2Pix-and-CR-Gan-Model


# Generating Synthetic Images and Labels for BRVO Disease

### 🌍 Leveraging GAN Models to Enhance Ophthalmology Research

This project involves using **Generative Adversarial Networks (GANs)** to generate synthetic images and labels for **Branch Retinal Vein Occlusion (BRVO)**, a vision-threatening retinal vascular condition. By integrating GAN models with OCTA imaging data, this project aims to improve treatment prediction, enable data augmentation, and address research gaps in BRVO treatment analysis.

---

## 📌 Problem Statement
**BRVO** causes unpredictable visual impairment and varying treatment needs, complicating clinical decision-making. This project:
- Generates synthetic progression data for missing treatment records.
- Enables personalized and efficient treatment plans by forecasting disease trajectories using GAN models.

---

## ✨ Key Features
1. **Conditional Recurrent GAN (CRGAN)** (In Progress):
   - Generates synthetic progression images using patient ID, treatment stage, and noise.
   - Utilizes LSTM layers to capture sequential dependencies.

2. **Pix2Pix GAN**:
   - Generates label masks for BRVO images.
   - Improves image-to-label translation with encoder-decoder architecture and skip connections.

3. **Evaluation Metrics**:
   - Metrics like **MSE**, **SSIM**, and **Cosine Similarity** are used to compare generated results with ground truth.

4. **Future Integration Framework**:
   - Aims to combine the CRGAN and Pix2Pix models to generate both synthetic images and their corresponding labels.

---

## 📚 Methodology
### **CRGAN (In Progress)**
The **CRGAN** uses:
- **Recurrent Generator**:
  - Combines LSTM layers with transposed convolutional layers to generate realistic images.
  - Uses patient and stage embeddings to capture progression dynamics.

- **Conditional Discriminator**:
  - Takes an image and its corresponding patient ID and stage embeddings to classify it as real or fake.

Formula:
- **Generator Loss**:
  \[
  \text{L}_{G} = \mathbb{E}_{z,c}[\log(1 - D(G(z, c)))]
  \]
- **Discriminator Loss**:
  \[
  \text{L}_{D} = -\mathbb{E}_{x,c}[\log(D(x, c))] - \mathbb{E}_{z,c}[\log(1 - D(G(z, c)))]
  \]

---

### **Pix2Pix GAN**
The **Pix2Pix** model is trained to generate label masks for BRVO images:
- **Generator**:
  - Encoder-decoder with skip connections for improved spatial accuracy.
  - Outputs grayscale images (labels).

- **Discriminator**:
  - Classifies concatenated image-label pairs as real or fake.

Formula:
- **Generator Loss**:
  \[
  \text{L}_{G} = \text{BCE Loss} + \lambda_{L1} \cdot \text{L1 Loss}
  \]
- **Discriminator Loss**:
  \[
  \text{L}_{D} = \text{BCE Loss (Real)} + \text{BCE Loss (Fake)}
  \]

---

## 🔧 Technologies Used
- **Programming**: Python, PyTorch.
- **GAN Models**: CRGAN (Recurrent GAN), Pix2Pix.
- **Evaluation**: MSE, SSIM, Cosine Similarity.
- **Dataset**: OCTA images from the SOUL dataset.

---

## 🔗 How It Works
1. **Data Preprocessing**:
   - Resizes images and normalizes pixel values to stabilize training.

2. **GAN Models**:
   - **CRGAN**: Generates realistic images of retinal progression.
   - **Pix2Pix**: Generates label masks for the images.

3. **Evaluation**:
   - Compares generated labels with original labels using similarity metrics.

---

## 📁 Dataset
We are using the [SOUL Dataset](https://pubmed.ncbi.nlm.nih.gov/39095383/), which contains OCTA images related to BRVO. This dataset includes images of 53 patients across multiple treatment stages, collected from the Affiliated Hospital of Shandong Second Medical University (2020-2021).

---

## 📚 Results
### **Pix2Pix GAN**:
- Successfully generates label masks with high fidelity.
- Metrics:
  - **MSE**: Measures pixel-level error.
  - **SSIM**: Evaluates structural similarity.
  - **Cosine Similarity**: Assesses feature similarity.

### **CRGAN**:
- Currently under progress and expected to improve synthetic image generation with refined hyperparameters.

---

## 🚀 Future Scope
1. **Improve CRGAN**:
   - Enhance output quality to generate patient-specific images.
   - Address missing treatment stages with realistic synthetic data.

2. **Integrate Models**:
   - Combine CRGAN and Pix2Pix to create a unified framework for generating images and labels.

3. **Intensity Prediction**:
   - Develop a GAN model to quantify BRVO severity.

---

## 🔐 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Pranish-37/Pix2Pix-and-CR-Gan-Model
   cd Pix2Pix-and-CR-Gan-Model
   ```

2. Install dependencies:
   by running the notebook dependencies will be automatically downloaded.

3. Run the training scripts:
   ```bash
   python only label generator code.ipynb
   ```

   Note: The **CRGAN** code is in progress and will be updated once finalized.

---

Let us know if you have any feedback or would like to contribute!(ramababuinampudi@gmail.com)

