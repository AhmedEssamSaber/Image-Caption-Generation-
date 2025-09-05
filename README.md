# 📸 Image Caption Generation

This project implements **Image Captioning** using three architectures:  
1. **Vision Transformer (ViT) + Transformer Decoder**  
2. **CNN (ResNet-50) + LSTM with Attention**  
3. **CNN (ResNet-50) + LSTM**  

All models are trained and evaluated on the **Flickr30k dataset**, with metrics including **CIDEr, BLEU-1/2/3/4, and ROUGE-L**. Training includes **checkpointing, visualizations, and metrics logging**.

---

##  Features
- Transformer-based, LSTM-Attention, and standard LSTM captioning models.  
- Training loop with early stopping & learning rate scheduling.  
- Checkpoint saving and resuming.  
- Automatic plot generation:  
  - Training vs. Validation Loss  
  - CIDEr, BLEU, and ROUGE metrics over epochs  
- Vocabulary saving (`vocab.pkl`).  

---

##  Dataset Setup
We use [**Flickr30k**](https://www.kaggle.com/datasets/adityajn105/flickr30k).  

Download and unzip:
```bash
!kaggle datasets download -d adityajn105/flickr30k
!unzip -q flickr30k.zip -d flickr30k_dataset
```
---
##  Installation
Install the required Python packages:

```bash
pip install -r requirements.txt
```
---
##  Visualizations

- Training vs. Validation Loss
- CIDEr, BLEU, ROUGE over epochs
- Example of it :
<div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
    <img src="path/to/image1.png" alt="Image 1" width="45%" />
    <img src="path/to/image2.png" alt="Image 2" width="45%" />
    <img src="path/to/image3.png" alt="Image 3" width="45%" />
</div>


---

# Image Caption Generation Streamlit app 

This is the interface of the project .

![Sample Image](images/sample.png)

---
# Scores 


| Model          | Epochs | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-L | CIDEr  |
|----------------|--------|--------|--------|--------|--------|---------|---------|--------|
| VIT_Transformer| 5      | 0.6239 | 0.4141 | 0.2838 | 0.1901 | 0.4751  | 0.4516  | 0.4171 |
| CNN + Attention| 13     | 0.6287 | 0.4035 | 0.2756 | 0.1835 | 0.4656  | 0.4435  | 0.3931 |
| CNN + LSTM     | 14     | 0.6039 | 0.3839 | 0.2497 | 0.1579 | 0.4501  | 0.4280  | 0.3436 |


Note on Training:
The evaluation scores reported above are based on the training sessions indicated in the table. In general, the models can achieve higher BLEU, ROUGE, and CIDEr scores if trained for more epochs. However, in our experiments, we observed that the scores were increasing only marginally after the reported number of epochs, so training was stopped to save time and computational resources.
