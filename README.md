# Image Colorization using GAN and Inception-ResNet-v2

This project implements an image colorization model using a Generative Adversarial Network (GAN) with semantic feature extraction from Inception-ResNet-v2, as described in the paper by Beyer Bruvik and Penfold.

## Project Structure
```
.
├── data/                  # Dataset and processed data
├── models/               # Model definitions
│   ├── generator.py      # Generator network
│   ├── discriminator.py  # Discriminator network
│   └── inception.py      # Inception-ResNet-v2 feature extractor
├── utils/                # Utility functions
│   ├── data_loader.py    # Data loading and preprocessing
│   └── metrics.py        # Evaluation metrics
├── train.py              # Training script
├── evaluate.py           # Evaluation script
└── requirements.txt      # Project dependencies
```

## Requirements
- Python 3.8+
- PyTorch 1.8+
- torchvision
- numpy
- pillow
- matplotlib
- tqdm

## Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
The model is trained on the COCO dataset. Download instructions:
1. Visit the COCO dataset website: http://cocodataset.org/#download
2. Download the 2017 Train/Val images
3. Place the images in the `data/` directory

## Training
To train the model:
```bash
python train.py --data_dir /path/to/coco --batch_size 64 --epochs 15
```

## Evaluation
To evaluate the model:
```bash
python evaluate.py --model_path /path/to/checkpoint --test_dir /path/to/test/images
```

## Results
The model achieves strong performance in terms of both quantitative metrics (PSNR and SSIM) and visual quality. For detailed results and comparisons, please refer to the paper.

## Citation
If you use this code in your research, please cite:
```
@article{beyer2023image,
  title={Image Colorization using GAN and Inception-ResNet-v2},
  author={Beyer Bruvik, Olivia and Penfold, Mia},
  journal={Stanford University},
  year={2023}
}
``` 