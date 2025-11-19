# DDPM Image Generation for MNIST

This project implements a Denoising Diffusion Probabilistic Model (DDPM) for generating handwritten digit images using the MNIST dataset.

## Requirements

- Python >= 3.10
- CUDA-capable GPU (recommended)

## Installation

1. Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset Setup

### Download Training Data

The RGB MNIST dataset should be downloaded separately and is **NOT** included in this repository.

**Download Instructions:**

1. Download the RGB MNIST dataset from the course-provided Google Drive link (check your assignment description or course website)

2. Alternatively, you can download from this Google Drive link:
   ```
   [Insert Google Drive Link Here]
   ```

3.Alternatively,there's another way to download:
### Generate Images Dataset

This repository provides the **MNIST dataset zip file (`mnist.zip`)**, which can be used for image generation or other machine learning experiments.

#### Download the Dataset

Please download the dataset from the Hugging Face repository:

[MNIST Dataset on Hugging Face](https://huggingface.co/grandmaeatsadumpling/Generate-Images-Dataset)

You can either click the **Download** button or use Git + Git LFS:

```bash
git lfs install
git clone https://huggingface.co/grandmaeatsadumpling/Generate-Images-Dataset
````

After downloading, the dataset file will be located at:

```
Generate-Images-Dataset/mnist.zip
```

---

#### Usage

Extract `mnist.zip` to use the dataset:

```bash
unzip mnist.zip -d ./mnist_data
```

---

#### Notes

* Make sure Git LFS is installed; otherwise, large files cannot be downloaded correctly.
* The dataset file may be large, so please be patient while downloading.



4. Extract the downloaded dataset to create the following directory structure:
   ```
   data/mnist_rgb/
   ├── 0/
   │   ├── image1.png
   │   ├── image2.png
   │   └── ...
   ├── 1/
   │   ├── image1.png
   │   └── ...
   ├── 2/
   │   ├── ...
   └── 9/
       └── ...
   ```

5. Update the `--data_path` argument when running the training script to point to your dataset location.

**Important:** The `data/mnist_rgb/` directory containing training images should NOT be committed to the repository to reduce repository size.

## Training

To train the DDPM model, run:

```bash
cd src
python train.py --data_path /path/to/mnist_rgb --epochs 100 --batch_size 128
```

### Training Arguments

- `--data_path`: Path to the RGB MNIST dataset (default: `./data/mnist_rgb`)
- `--image_size`: Size of the images (default: `28`)
- `--timesteps`: Number of diffusion timesteps (default: `1000`)
- `--beta_start`: Starting value of beta schedule (default: `0.0001`)
- `--beta_end`: Ending value of beta schedule (default: `0.02`)
- `--time_emb_dim`: Dimension of time embedding (default: `128`)
- `--base_dim`: Base dimension for UNet (default: `64`)
- `--epochs`: Number of training epochs (default: `100`)
- `--batch_size`: Training batch size (default: `128`)
- `--lr`: Learning rate (default: `2e-4`)
- `--checkpoint_dir`: Directory to save checkpoints (default: `./checkpoints`)
- `--sample_dir`: Directory to save sample images during training (default: `./samples`)
- `--save_interval`: Checkpoint save interval in epochs (default: `10`)
- `--sample_interval`: Sample generation interval in steps (default: `500`)

### Example Training Command

```bash
python train.py \
    --data_path ../data/mnist_rgb \
    --epochs 100 \
    --batch_size 128 \
    --lr 2e-4 \
    --checkpoint_dir ../checkpoints \
    --sample_dir ../samples
```

## Image Generation

### Generate 10,000 Images for Submission

To generate 10,000 images for FID evaluation:

```bash
cd src
python generate.py \
    --checkpoint ../checkpoints/best_model.pt \
    --output_dir ../generated_images \
    --num_images 10000 \
    --batch_size 64 \
    --mode generate
```

The generated images will be saved as `00001.png`, `00002.png`, ..., `10000.png` in the output directory.

### Visualize Diffusion Process (for Report)

To visualize the diffusion process with intermediate steps:

```bash
python generate.py \
    --checkpoint ../checkpoints/best_model.pt \
    --output_dir ../diffusion_visualization \
    --mode visualize
```

This will generate a grid showing 8 samples across 8 timesteps (from noise to final image), which can be used in your report.

### Generation Arguments

- `--checkpoint`: Path to the trained model checkpoint (required)
- `--output_dir`: Directory to save generated images (default: `./generated_images`)
- `--num_images`: Number of images to generate (default: `10000`)
- `--batch_size`: Batch size for generation (default: `64`)
- `--mode`: Generation mode - `generate` or `visualize` (default: `generate`)
- Other model parameters should match those used during training

## Calculating FID Score

After generating images, you can calculate the FID score using `pytorch-fid`:

1. Install pytorch-fid:

```bash
pip install pytorch-fid
```

2. Calculate FID score:

```bash
# Using the training dataset
python -m pytorch_fid generated_images data/mnist_rgb

# Or using the precalculated statistics
python -m pytorch_fid generated_images mnist.npz
```

## Project Structure

```
.
├── src/
│   ├── model.py          # DDPM model implementation
│   ├── train.py          # Training script
│   └── generate.py       # Image generation script
├── requirements.txt      # Python dependencies
└── readme.md            # This file
```

## Model Architecture

The implementation uses a UNet architecture with:
- Time embeddings using sinusoidal position encodings
- GroupNorm for normalization
- SiLU (Swish) activation functions
- Skip connections between encoder and decoder

The diffusion process follows the DDPM paper:
- Forward process: Gradually adds Gaussian noise to images
- Reverse process: Learns to denoise images step by step
- Uses a linear beta schedule from `beta_start` to `beta_end`

## Tips for Better Results

1. **Training Duration**: Train for at least 80-100 epochs for good results
2. **Learning Rate**: Use a learning rate around 1e-4 to 2e-4
3. **Batch Size**: Use the largest batch size your GPU can handle (128 or 256 recommended)
4. **Timesteps**: 1000 timesteps is a good default
5. **Model Size**: Adjust `base_dim` based on your GPU memory (64 is a good balance)

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size`
- Reduce `base_dim` in the model
- Use gradient accumulation

### Poor Image Quality
- Train for more epochs
- Adjust learning rate
- Try different beta schedules
- Increase model capacity (`base_dim`)

### Slow Training
- Increase `batch_size` if possible
- Reduce `num_workers` in DataLoader
- Use mixed precision training (requires code modification)

## References

- DDPM Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- pytorch-fid: [GitHub Repository](https://github.com/mseitzer/pytorch-fid)
