# train_lol_combined.py
import json
import os
import itertools
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Tuple, Union, Dict

import torch
from torchvision.transforms import functional as TF
import math
from einops import rearrange
# from einops.layers.torch import Rearrange # Not explicitly used after MHA change
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
# matplotlib might cause issues on headless servers, comment out if unused for plotting directly
# from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision import transforms as T
# from torchvision.datasets import ImageFolder # Not used
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
from tqdm import tqdm # Added for potential progress bars if needed
import traceback # For detailed error printing

# Assuming consistency_models2.py is in the same directory or accessible in the Python path
try:
    from consistency_models2 import (
        ConsistencySamplingAndEditing,
        ImprovedConsistencyTraining,
        pseudo_huber_loss,
        update_ema_model_,
        karras_schedule,
        model_forward_wrapper # Not directly used now, but available
    )
except ImportError:
    print("Error: consistency_models2.py not found or contains errors.")
    print("Please ensure the file exists in the same directory and is runnable.")
    exit(1)


# -------------------------
# Dataset Implementations (No changes needed here from previous version)
# -------------------------

class PairedDataset(Dataset):
    """Loads paired images from a single source directory structure."""
    def __init__(
        self,
        visible_dir: str,
        infrared_dir: str, # Renaming 'target' to 'infrared' for consistency with variable names
        transform: Optional[Callable] = None,
        crop_size: Optional[Tuple[int, int]] = (128, 128), # Make crop optional for validation if needed
        resize_size: Optional[Tuple[int, int]] = (128, 128) # Keep but note it's not actively used resize
    ):
        self.visible_dir = visible_dir
        self.infrared_dir = infrared_dir
        self.transform = transform
        self.crop_size = crop_size
        self.resize_size = resize_size # Store resize size

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
        try:
            vis_files_raw = sorted([f for f in os.listdir(visible_dir) if f.lower().endswith(image_extensions)])
            inf_files_raw = sorted([f for f in os.listdir(infrared_dir) if f.lower().endswith(image_extensions)])
        except FileNotFoundError as e:
            print(f"Error: Directory not found - {e}. Please check dataset paths.")
            raise e

        # Simple pairing based on sorted order - assumes filenames correspond 1-to-1
        if len(vis_files_raw) != len(inf_files_raw):
            print(f"Warning: Mismatched number of images in {visible_dir} ({len(vis_files_raw)}) and {infrared_dir} ({len(inf_files_raw)}). Attempting to pair by filename prefix.")
            # Attempt pairing by base filename (more robust)
            vis_map = {os.path.splitext(f)[0]: f for f in vis_files_raw}
            inf_map = {os.path.splitext(f)[0]: f for f in inf_files_raw}
            common_keys = sorted(list(vis_map.keys() & inf_map.keys()))
            self.visible_images = [vis_map[k] for k in common_keys]
            self.infrared_images = [inf_map[k] for k in common_keys]
            print(f"Found {len(self.visible_images)} pairs based on common filename prefixes.")
        else:
             self.visible_images = vis_files_raw
             self.infrared_images = inf_files_raw

        if not self.visible_images:
            print(f"Warning: No image pairs found for {visible_dir} and {infrared_dir}")


    def __len__(self) -> int:
        return len(self.visible_images)

    def __getitem__(self, index: int) -> Optional[Tuple[Tensor, Tensor]]:
        if index >= len(self.visible_images):
             raise IndexError("Index out of bounds")

        visible_path = os.path.join(self.visible_dir, self.visible_images[index])
        infrared_path = os.path.join(self.infrared_dir, self.infrared_images[index])

        try:
            # Use L for grayscale, RGB for color. Assuming LOL is color.
            visible_image = Image.open(visible_path).convert("RGB")
            infrared_image = Image.open(infrared_path).convert("RGB")
        except Exception as e:
            # print(f"Error loading image pair: {visible_path}, {infrared_path}. Error: {e}")
            # Return next item or None (requires collate_fn handling)
            # Simple approach: skip by returning None and using a collate_fn or filter later
            return None # Signal to collate_fn to skip this sample

        # --- Apply synchronized transforms ---
        # Random Horizontal Flip (typically only for training)
        # Let's assume validation doesn't need random flips
        if self.crop_size: # Apply flips only if cropping (indicative of training)
            if torch.rand(1).item() > 0.5:
                visible_image = TF.hflip(visible_image)
                infrared_image = TF.hflip(infrared_image)

        # Random Crop (typically only for training)
        if self.crop_size:
            if visible_image.size != infrared_image.size:
                 # print(f"Skipping image pair {visible_path}, {infrared_path} during crop due to mismatched sizes {visible_image.size} vs {infrared_image.size}")
                 return None # Signal to skip
            try:
                 i, j, h, w = T.RandomCrop.get_params(visible_image, output_size=self.crop_size)
                 visible_image = TF.crop(visible_image, i, j, h, w)
                 infrared_image = TF.crop(infrared_image, i, j, h, w)
            except ValueError as e:
                 # print(f"Error during cropping {visible_path} (size {visible_image.size}) with crop size {self.crop_size}: {e}")
                 return None

        # Resize (Optional, apply if needed for model input, esp. during validation if not cropping)
        # If resize_size is set and different from current size (esp. after no crop)
        if self.resize_size and not self.crop_size: # Example: Resize during validation if not cropping
             visible_image = TF.resize(visible_image, self.resize_size, interpolation=TF.InterpolationMode.BICUBIC)
             infrared_image = TF.resize(infrared_image, self.resize_size, interpolation=TF.InterpolationMode.BICUBIC)

        # Apply final transform (ToTensor, Normalize)
        if self.transform:
            visible_image = self.transform(visible_image)
            infrared_image = self.transform(infrared_image)

        return visible_image, infrared_image


class CombinedPairedDataset(Dataset):
    """Loads paired images from multiple source directory structures for training."""
    def __init__(
        self,
        visible_dirs: List[str],
        infrared_dirs: List[str], # Renaming 'target' to 'infrared'
        transform: Optional[Callable] = None,
        crop_size: Tuple[int, int] = (128, 128),
        resize_size: Optional[Tuple[int, int]] = None # Keep resize optional
    ):
        assert len(visible_dirs) == len(infrared_dirs), "Must provide the same number of visible and infrared directories"
        self.transform = transform
        self.crop_size = crop_size
        self.resize_size = resize_size

        self.visible_files = []
        self.infrared_files = []
        self.source_indices = [] # Optional: track source

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')

        print("Initializing CombinedPairedDataset...")
        for i, (vis_dir, inf_dir) in enumerate(zip(visible_dirs, infrared_dirs)):
            print(f" Processing source {i}: {vis_dir} & {inf_dir}")
            try:
                vis_files_raw = sorted([f for f in os.listdir(vis_dir) if f.lower().endswith(image_extensions)])
                inf_files_raw = sorted([f for f in os.listdir(inf_dir) if f.lower().endswith(image_extensions)])
            except FileNotFoundError:
                print(f"  Warning: Directory not found, skipping source: {vis_dir} or {inf_dir}")
                continue

            paired_vis = []
            paired_inf = []
            # Attempt pairing by base filename (more robust)
            vis_map = {os.path.splitext(f)[0]: f for f in vis_files_raw}
            inf_map = {os.path.splitext(f)[0]: f for f in inf_files_raw}
            common_keys = sorted(list(vis_map.keys() & inf_map.keys()))
            paired_vis = [os.path.join(vis_dir, vis_map[k]) for k in common_keys]
            paired_inf = [os.path.join(inf_dir, inf_map[k]) for k in common_keys]

            if not paired_vis:
                 print(f"  Warning: No paired images found for source {i} ({vis_dir}).")
                 continue

            self.visible_files.extend(paired_vis)
            self.infrared_files.extend(paired_inf)
            self.source_indices.extend([i] * len(paired_vis)) # Store source index

            print(f"  -> Added {len(paired_vis)} pairs from source {i}.")

        if not self.visible_files:
             raise RuntimeError("No training data could be loaded. Check dataset paths and structure in 'datasets/' directory.")

        print(f"Combined dataset initialized: Found {len(self.visible_files)} paired images total.")


    def __len__(self) -> int:
        return len(self.visible_files)

    def __getitem__(self, index: int) -> Optional[Tuple[Tensor, Tensor]]:
        visible_path = self.visible_files[index]
        infrared_path = self.infrared_files[index]

        try:
            visible_image = Image.open(visible_path).convert("RGB")
            infrared_image = Image.open(infrared_path).convert("RGB")
        except Exception as e:
            # print(f"Error loading image pair: {visible_path}, {infrared_path}. Error: {e}")
            return None # Signal to skip

        if visible_image.size != infrared_image.size:
            # print(f"Skipping image pair {visible_path}, {infrared_path} due to mismatched sizes {visible_image.size} vs {infrared_image.size}")
            # Option 1: Resize to match (e.g., resize smaller to larger) - Can distort data
            # Option 2: Skip - Safest
            return None # Signal to skip

        # --- Apply synchronized transforms (typically only for training) ---
        # Random Horizontal Flip
        if torch.rand(1).item() > 0.5:
            visible_image = TF.hflip(visible_image)
            infrared_image = TF.hflip(infrared_image)

        # Random Crop
        if self.crop_size:
            try:
                i, j, h, w = T.RandomCrop.get_params(visible_image, output_size=self.crop_size)
                visible_image = TF.crop(visible_image, i, j, h, w)
                infrared_image = TF.crop(infrared_image, i, j, h, w)
            except ValueError as e:
                 # print(f"Error during cropping {visible_path} (size {visible_image.size}) with crop size {self.crop_size}: {e}")
                 return None # Signal to skip
        elif self.resize_size: # Apply resize if no crop is done and resize is specified
            visible_image = TF.resize(visible_image, self.resize_size, interpolation=TF.InterpolationMode.BICUBIC)
            infrared_image = TF.resize(infrared_image, self.resize_size, interpolation=TF.InterpolationMode.BICUBIC)


        # Apply final transform (ToTensor, Normalize)
        if self.transform:
            visible_image = self.transform(visible_image)
            infrared_image = self.transform(infrared_image)

        return visible_image, infrared_image

def collate_fn_skip_none(batch):
    """Collate function that filters out None items"""
    original_len = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # print(f"Warning: Entire batch was filtered out (original size: {original_len}).")
        return None # Return None if the whole batch was bad
    filtered_len = len(batch)
    # if original_len > filtered_len:
    #     print(f"Note: Filtered {original_len - filtered_len} samples from batch.")
    return torch.utils.data.dataloader.default_collate(batch)


# -------------------------
# DataModule (No changes needed here from previous version)
# -------------------------

@dataclass
class CombinedImageDataModuleConfig:
    # Base paths for each dataset component (Relative to script location)
    # lolv1_dir: str = "IRVI/single/rooftop_total"
    lolv2_real_dir: str = "IRVI/single/traffic"
    # lolv2_synthetic_dir: str = "datasets/LOLv2-synthetic"
    

    # Structure assumes Train/input, Train/target, Test/input, Test/target
    train_input_subdir: str = "trainB"
    train_target_subdir: str = "trainA"
    val_input_subdir: str = "testB"  # Using Test set for validation
    val_target_subdir: str = "testA"

    image_size_crop: Tuple[int, int] = (128, 128) # Crop size for training
    resize_size_val: Optional[Tuple[int, int]] = (128, 128) # Resize validation images

    batch_size: int = 16
    num_workers: int = 6
    pin_memory: bool = True
    persistent_workers: bool = True


class CombinedLOLDataModule(LightningDataModule):
    def __init__(self, config: CombinedImageDataModuleConfig) -> None:
        super().__init__()
        self.config = config
        self.train_vis_dirs = []
        self.train_inf_dirs = []
        self.val_sets_info = [] # Store info for creating validation datasets
        self._val_dataset_names = [] # Internal store for names

        # Define transforms
        self.transform_train = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1),  # Normalize to [-1, 1]
        ])
        # Validation transform might be different (e.g., no augmentation)
        self.transform_val = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1), # Normalize to [-1, 1]
        ])


    def setup(self, stage: str = None) -> None:
        print(f"\n--- DataModule Setup (Stage: {stage}) ---")
        # --- Prepare List of Datasets to Use ---
        # Use paths from config, which are now relative
        datasets_to_use = [
            #("lolv1", self.config.lolv1_dir),
            ("lolv2r", self.config.lolv2_real_dir),
        ]

        # --- Setup Training Data (if stage is 'fit' or None) ---
        if stage == "fit" or stage is None:
            self.train_vis_dirs = []
            self.train_inf_dirs = []
            print("--- Setting up Training Data ---")
            for name, base_dir in datasets_to_use:
                if base_dir and os.path.isdir(base_dir): # Check if base_dir exists
                    vis_dir = os.path.join(base_dir, self.config.train_input_subdir)
                    inf_dir = os.path.join(base_dir, self.config.train_target_subdir)
                    if os.path.isdir(vis_dir) and os.path.isdir(inf_dir):
                        self.train_vis_dirs.append(vis_dir)
                        self.train_inf_dirs.append(inf_dir)
                        print(f" Found training data for {name} in {base_dir}")
                    else:
                        print(f" Warning: Training subdirs '{self.config.train_input_subdir}'/'{self.config.train_target_subdir}' not found in {base_dir}")
                elif base_dir:
                     print(f" Warning: Base directory '{base_dir}' for {name} not found.")


            if not self.train_vis_dirs:
                 raise FileNotFoundError("CRITICAL: No valid training data directories found. Check config paths and 'datasets/' structure.")

            self.dataset_train = CombinedPairedDataset(
                visible_dirs=self.train_vis_dirs,
                infrared_dirs=self.train_inf_dirs,
                transform=self.transform_train,
                crop_size=self.config.image_size_crop,
                resize_size=None # No resize after random crop during training
            )
            print(f"Total combined training samples: {len(self.dataset_train)}")


        # --- Setup Validation Data (if stage is 'fit', 'validate', 'test' or None) ---
        # We use Test sets for validation here
        if stage in ["fit", "validate", "test", None]:
            self.val_sets_info = []
            self.datasets_val = []
            self._val_dataset_names = []
            print("\n--- Setting up Validation Data (using Test sets) ---")
            for name, base_dir in datasets_to_use:
                 if base_dir and os.path.isdir(base_dir):
                     vis_dir = os.path.join(base_dir, self.config.val_input_subdir)
                     inf_dir = os.path.join(base_dir, self.config.val_target_subdir)
                     if os.path.isdir(vis_dir) and os.path.isdir(inf_dir):
                         print(f" Found validation data for {name} in {base_dir}")
                         dataset = PairedDataset(
                             visible_dir=vis_dir,
                             infrared_dir=inf_dir,
                             transform=self.transform_val,
                             crop_size=None, # No random cropping during validation
                             resize_size=self.config.resize_size_val # Use specified resize size
                         )
                         if len(dataset) > 0:
                            self.datasets_val.append(dataset)
                            self._val_dataset_names.append(name) # Store name corresponding to dataset list
                            print(f"  Validation dataset '{name}' loaded with {len(dataset)} samples.")
                         else:
                             print(f"  Warning: Validation dataset '{name}' is empty.")
                     else:
                        print(f" Warning: Validation subdirs '{self.config.val_input_subdir}'/'{self.config.val_target_subdir}' not found in {base_dir}")
                 elif base_dir:
                     print(f" Warning: Base directory '{base_dir}' for {name} not found.")

            if not self.datasets_val:
                print("Warning: No validation datasets could be loaded.")


    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, 'dataset_train'):
             print("Warning: train_dataloader called before setup or setup failed for training.")
             # Force setup if needed, though Trainer should handle this
             # self.setup(stage='fit')
             if not hasattr(self, 'dataset_train'): # Check again after trying setup
                  raise RuntimeError("Training dataset not initialized. Setup failed.")

        print(f"Creating train dataloader with batch size {self.config.batch_size}...")
        return DataLoader(
            self.dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
            collate_fn=collate_fn_skip_none # Use collate_fn to handle None items
        )

    def val_dataloader(self) -> List[DataLoader]:
        if not hasattr(self, 'datasets_val') or not self.datasets_val:
             # Try running setup for validation stage if datasets_val doesn't exist
             # This might happen if only `trainer.validate()` is called without `trainer.fit()`
             # print("Validation datasets not found, attempting setup(stage='validate')...")
             # self.setup(stage='validate')
             if not hasattr(self, 'datasets_val') or not self.datasets_val: # Check again
                 print("No validation data loaded, returning empty list for val_dataloader.")
                 return [] # Return empty list if no validation data

        print(f"Creating {len(self.datasets_val)} validation dataloader(s)...")
        val_loaders = []
        for i, dataset in enumerate(self.datasets_val):
            val_loaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.config.batch_size, # Can use a different val batch size if needed
                    shuffle=False,
                    num_workers=self.config.num_workers,
                    pin_memory=self.config.pin_memory,
                    persistent_workers=self.config.persistent_workers if self.config.num_workers > 0 else False,
                    collate_fn=collate_fn_skip_none # Use collate_fn here too
                )
            )
        return val_loaders

    def get_val_dataset_names(self) -> List[str]:
        """Helper to get names corresponding to dataloader indices"""
        # Return the stored names populated during setup
        if not hasattr(self, '_val_dataset_names'):
            # Attempt setup if names weren't populated (e.g., direct validate call)
            self.setup(stage='validate')
        return self._val_dataset_names


# -------------------------
# Model Components (UNet etc.)
# -------------------------

def GroupNorm(channels: int) -> nn.GroupNorm:
    # Ensure num_groups is at least 1 and divides channels if possible
    if channels == 0: return nn.Identity() # Handle zero channels case
    num_groups = min(32, channels // 4) if channels >= 4 else 1
    # Ensure num_groups divides channels, if not, find largest divisor <= 32 or use 1
    if channels % num_groups != 0:
        found = False
        for ng in range(min(32, channels), 0, -1):
            if channels % ng == 0:
                num_groups = ng
                found = True
                break
        if not found: num_groups = 1 # Fallback if no suitable divisor found
    if num_groups <= 0: num_groups = 1 # Safeguard
    try:
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    except ValueError as e:
        # This might happen if channels=0 despite the check, or other reasons
        print(f"Warning: Error creating GroupNorm(num_groups={num_groups}, num_channels={channels}): {e}. Using nn.Identity().")
        return nn.Identity()


class SelfAttention(nn.Module):
    """ Multi-Head Self-Attention using PyTorch's MHA"""
    def __init__(
        self,
        in_channels: int,
        n_heads: int = 8,
        dropout: float = 0.1, # Reduced default dropout
    ) -> None:
        super().__init__()
        assert in_channels > 0, "in_channels must be positive for SelfAttention"
        self.dropout = dropout
        self.n_heads = n_heads
        # Ensure n_heads divides in_channels
        if in_channels % n_heads != 0:
            old_heads = n_heads
            # Find largest power of 2 <= old_heads that divides in_channels
            n_heads = old_heads
            while n_heads > 1 and in_channels % n_heads != 0 :
                n_heads //= 2
            if in_channels % n_heads != 0: n_heads = 1 # Fallback: use 1 head if no divisor found
            print(f"Warning: Adjusting SelfAttention n_heads from {old_heads} to {n_heads} for in_channels {in_channels}")
            self.n_heads = n_heads

        self.norm = GroupNorm(in_channels)
        # Use MHA which expects (Batch, SeqLen, EmbedDim)
        # EmbedDim = in_channels, SeqLen = H*W
        self.mha = nn.MultiheadAttention(in_channels, self.n_heads, dropout=dropout, batch_first=True)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1) # Project back


    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        residual = x
        x_norm = self.norm(x)

        # Reshape for MHA: [B, C, H, W] -> [B, H*W, C]
        x_norm = rearrange(x_norm, 'b c h w -> b (h w) c')

        # Apply Multihead Attention
        # Q, K, V are all derived from x_norm
        attn_output, _ = self.mha(x_norm, x_norm, x_norm, need_weights=False) # [B, L, C]

        # Reshape back to image format and project
        attn_output = rearrange(attn_output, 'b (h w) c -> b c h w', h=H, w=W) # [B, C, H, W]
        out = self.proj_out(attn_output)

        return out + residual


class UNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_level_channels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.noise_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(noise_level_channels, out_channels),
        )
        self.block1 = nn.Sequential(
            GroupNorm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.block2 = nn.Sequential(
            GroupNorm(out_channels),
            nn.SiLU(),
            nn.Dropout2d(dropout), # Apply dropout here
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, noise_level_emb: Tensor) -> Tensor:
        residual = self.residual_proj(x)
        h = self.block1(x)

        # Project noise level embedding and add
        noise_proj = self.noise_proj(noise_level_emb) # [B, out_channels]
        h = h + noise_proj[:, :, None, None] # Add broadcasted noise projection [B, C, 1, 1]

        h = self.block2(h)
        return h + residual


class UNetBlockWithSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        noise_level_channels: int,
        n_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.unet_block = UNetBlock(
            in_channels, out_channels, noise_level_channels, dropout
        )
        # Attention applied on the *output* channels of the block
        self.self_attention = SelfAttention(
            out_channels, n_heads, dropout
        )

    def forward(self, x: Tensor, noise_level: Tensor) -> Tensor:
        h = self.unet_block(x, noise_level)
        h = self.self_attention(h)
        return h


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # Use strided convolution for downsampling
        self.projection = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # Use ConvTranspose2d for upsampling
        self.projection = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)


class NoiseLevelEmbedding(nn.Module):
    """ Sinusoidal noise level embedding """
    def __init__(self, channels: int, scale: float = 16.0) -> None:
        super().__init__()
        self.channels = channels
        assert channels % 2 == 0, "NoiseLevelEmbedding channels must be even"
        half_dim = channels // 2
        emb = math.log(10000) / (half_dim - 1)
        # Store persistent buffer for embedding frequencies
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb), persistent=False)

        self.projection = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.SiLU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x is expected to be sigmas [B,]
        x = x[:, None] * self.emb[None, :] # Use stored emb [B, half_dim]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) # [B, channels]
        return self.projection(x) # Output shape [B, channels]


@dataclass
class UNetConfig:
    image_channels: int = 3 # Target image (Infrared/Normal Light)
    output_channels: int = 3 # Predicted Infrared/Normal Light
    conditional_channels: int = 3 # Visible/Low Light
    noise_level_channels: int = 256 # Dimension of noise level embedding
    noise_level_scale: float = 16.0 # Scale for noise level embedding
    n_heads: int = 4 # Reduced default heads
    base_channels: int = 64 # Starting number of channels
    channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8) # Channel multipliers per resolution
    blocks_per_resolution: int = 2 # Number of blocks at each resolution
    use_attention_at_resolution: Tuple[bool, ...] = (False, False, True, True) # Use attention at deeper resolutions
    dropout: float = 0.1 # Dropout rate


class UNet(nn.Module):
    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.base_channels
        noise_emb_dim = config.noise_level_channels

        self.noise_level_embedding = NoiseLevelEmbedding(
            noise_emb_dim, config.noise_level_scale
        )

        # Input projection combines noisy target and conditional image
        self.input_projection = nn.Conv2d(
            config.image_channels + config.conditional_channels, # noisy target + visible
            channels,
            kernel_size=3,
            padding=1, # Use integer padding
        )

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        skip_connection_channels = [] # Stores output channels of encoder blocks

        # --- Encoder ---
        current_channels = channels
        print("--- Building UNet Encoder ---")
        for i, mult in enumerate(config.channel_multipliers):
            out_ch = channels * mult
            use_attn = config.use_attention_at_resolution[i]
            print(f" Res {i}: In={current_channels}, Out={out_ch}, Attn={use_attn}")
            for block_idx in range(config.blocks_per_resolution):
                block_args = {
                    "in_channels": current_channels,
                    "out_channels": out_ch,
                    "noise_level_channels": noise_emb_dim,
                    "dropout": config.dropout,
                }
                if use_attn:
                    block_args["n_heads"] = config.n_heads
                    block = UNetBlockWithSelfAttention(**block_args)
                else:
                    block = UNetBlock(**block_args)

                self.encoder_blocks.append(block)
                current_channels = out_ch
                # Add the output channels of this block to skip connections
                skip_connection_channels.append(current_channels)
                print(f"  -> Enc Block {block_idx}: Appended Skip channels={current_channels}")


            # Downsample after blocks at this resolution, except the last one
            if i < len(config.channel_multipliers) - 1:
                print(f"  -> Downsampling from {current_channels} channels")
                self.encoder_blocks.append(Downsample(current_channels))
                # Don't add skip connection after downsampling

        print(f"--- Bottleneck (Channels: {current_channels}) ---")
        # --- Bottleneck ---
        self.bottleneck = nn.ModuleList([
            UNetBlockWithSelfAttention(current_channels, current_channels, noise_emb_dim, config.n_heads, config.dropout),
            UNetBlockWithSelfAttention(current_channels, current_channels, noise_emb_dim, config.n_heads, config.dropout),
        ])

        print("--- Building UNet Decoder ---")
        # --- Decoder ---
        # Iterate through multipliers in reverse
        for i, mult in reversed(list(enumerate(config.channel_multipliers))):
            target_out_ch = channels * mult # Target output channels for this resolution level
            use_attn = config.use_attention_at_resolution[i]
            print(f" Res {i}: Target Out={target_out_ch}, Current In={current_channels}, Attn={use_attn}")

            # Upsample before blocks, except for the first decoder resolution (following bottleneck)
            if i != len(config.channel_multipliers) - 1: # If not the deepest resolution being decoded
                 print(f"  -> Upsampling from {current_channels} channels")
                 upsample_layer = Upsample(current_channels)
                 self.decoder_blocks.append(upsample_layer)
                 # Assuming Upsample keeps the channel count (like ConvTranspose2d)

            # Number of blocks should match the encoder at this resolution level
            for block_idx in range(config.blocks_per_resolution):
                if not skip_connection_channels:
                     # This should not happen with the corrected logic
                     raise RuntimeError(f"CRITICAL Error: Ran out of skip connections during decoder Res {i}, Block {block_idx}!")

                skip_ch = skip_connection_channels.pop()
                # Input channels = current channels from below/upsample + skip connection channels
                in_ch_combined = current_channels + skip_ch
                print(f"  -> Dec Block {block_idx}: Skip channels={skip_ch}, Combined In={in_ch_combined}, Target Out={target_out_ch}")

                block_args = {
                    "in_channels": in_ch_combined,
                    "out_channels": target_out_ch, # Target output channels for this block
                    "noise_level_channels": noise_emb_dim,
                    "dropout": config.dropout,
                }
                if use_attn:
                    block_args["n_heads"] = config.n_heads
                    block = UNetBlockWithSelfAttention(**block_args)
                else:
                    block = UNetBlock(**block_args)

                self.decoder_blocks.append(block)
                current_channels = target_out_ch # Update current channels for the next block/level

        # --- Output Projection ---
        print(f"--- Output Projection (In: {current_channels}, Out: {config.output_channels}) ---")
        self.output_projection = nn.Sequential(
             GroupNorm(current_channels), # Normalize before final activation/conv
             nn.SiLU(),
             nn.Conv2d(current_channels, config.output_channels, kernel_size=3, padding=1)
        )
        print("--- UNet Build Complete ---")


    def forward(self, x: Tensor, noise_level: Tensor, v: Tensor) -> Tensor:
        """
        x: Noisy target image [B, C_img, H, W]
        noise_level: Sigma values [B,]
        v: Conditional visible image [B, C_cond, H, W]
        """
        # 1. Embed noise level
        noise_emb = self.noise_level_embedding(noise_level) # [B, noise_emb_dim]

        # 2. Concatenate input and condition, then project
        h = torch.cat([x, v], dim=1) # [B, C_img + C_cond, H, W]
        h = self.input_projection(h) # [B, base_channels, H, W]

        # 3. Encoder
        encoder_outputs = [] # Store outputs *before* downsampling for skip connections
        for layer in self.encoder_blocks:
            if isinstance(layer, Downsample):
                h = layer(h) # Apply downsampling
            else: # UNetBlock or UNetBlockWithSelfAttention
                h = layer(h, noise_emb)
                encoder_outputs.append(h) # Append output of the block

        # 4. Bottleneck
        for layer in self.bottleneck:
            h = layer(h, noise_emb)

        # 5. Decoder
        for layer in self.decoder_blocks:
            if isinstance(layer, Upsample):
                h = layer(h) # Apply upsampling first
            else: # UNetBlock or UNetBlockWithSelfAttention
                skip = encoder_outputs.pop() # Get corresponding skip connection
                # Ensure spatial dimensions match after upsampling before concat
                if h.shape[-2:] != skip.shape[-2:]:
                    # print(f"Decoder shape mismatch: h={h.shape}, skip={skip.shape}. Resizing h.") # Debug print
                    h = TF.resize(h, skip.shape[-2:], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

                h = torch.cat([h, skip], dim=1) # Concatenate along channel dimension
                h = layer(h, noise_emb)

        # 6. Output Projection
        output = self.output_projection(h)
        return output


    def save_pretrained(self, pretrained_path: str) -> None:
        os.makedirs(pretrained_path, exist_ok=True)
        # Save config as JSON
        config_path = os.path.join(pretrained_path, "config.json")
        # Use a helper to convert dataclass to dict, handling potential non-serializable types if needed
        # For this config, asdict should be fine.
        config_dict = asdict(self.config)
        with open(config_path, mode="w") as f:
            json.dump(config_dict, f, indent=4)
        # Save model state dict
        model_path = os.path.join(pretrained_path, "model.pt")
        torch.save(self.state_dict(), model_path)
        print(f"UNet config saved to {config_path}")
        print(f"UNet state_dict saved to {model_path}")

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "UNet":
        # Load config from JSON
        config_path = os.path.join(pretrained_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path, mode="r") as f:
            config_dict = json.load(f)

        # Recreate config object - handle potential tuples loaded as lists
        for key, value in config_dict.items():
            # Check if key exists in the dataclass definition
            if key in UNetConfig.__annotations__:
                field_type = UNetConfig.__annotations__[key]
                # Check if the field is supposed to be a tuple and value is a list
                if isinstance(value, list) and hasattr(field_type, '__origin__') and field_type.__origin__ is tuple:
                     config_dict[key] = tuple(value) # Convert list back to tuple

        # Filter out any keys in loaded dict that are not in the dataclass definition
        valid_keys = UNetConfig.__dataclass_fields__.keys()
        filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        config = UNetConfig(**filtered_config_dict)

        # Instantiate model
        model = cls(config)

        # Load state dict
        model_path = os.path.join(pretrained_path, "model.pt")
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model state_dict file not found at {model_path}")

        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        # Add strict=False if loading older checkpoints or slightly modified architectures
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"UNet state_dict loaded successfully from {model_path} (strict=True)")
        except RuntimeError as e:
            print(f"Warning: Error loading state_dict strictly: {e}. Trying with strict=False.")
            model.load_state_dict(state_dict, strict=False)
            print(f"UNet state_dict loaded with strict=False from {model_path}")

        return model


# -------------------------
# Lightning Module Config (No changes needed here from previous version)
# -------------------------

@dataclass
class LitImprovedConsistencyModelConfig:
    ema_decay_rate: float = 0.9999
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.995)
    lr_scheduler_start_factor: float = 1e-5
    lr_scheduler_iters: int = 10_000 # Steps for linear warmup
    sample_every_n_val_epochs: int = 1 # How often to log validation samples (every N val epochs)
    num_samples: int = 8 # Number of samples for logging grids
    # Define sigma sequence for validation/testing sampling
    val_sampling_sigmas: Tuple[float, ...] = (80.0, 39.17, 19.18, 9.39, 4.60, 2.25, 1.10, 0.54, 0.26, 0.13, 0.06) # Example fixed 11 steps


# -------------------------
# Lightning Module (No changes needed here from previous version)
# -------------------------

class LitImprovedConsistencyModel(LightningModule):
    def __init__(
        self,
        consistency_training: ImprovedConsistencyTraining,
        consistency_sampling: ConsistencySamplingAndEditing,
        model: UNet,
        ema_model: UNet,
        config: LitImprovedConsistencyModelConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'ema_model', 'consistency_training', 'consistency_sampling'])
        self.consistency_training = consistency_training
        self.consistency_sampling = consistency_sampling
        self.model = model
        self.ema_model = ema_model
        self.validation_step_outputs = []
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model.eval()
        self.val_dataset_names = []


    # --- MODIFIED: Hook to clear outputs and get dataset names ---
    def on_validation_epoch_start(self) -> None:
        # Determine number of validation dataloaders using the correct attribute
        num_val_loaders = 0
        if self.trainer.val_dataloaders: # Check if it exists and is not empty/None
             # Ensure it's a list before taking len (it should be)
             if isinstance(self.trainer.val_dataloaders, list):
                 num_val_loaders = len(self.trainer.val_dataloaders)
             else: # Should not happen with standard setup, but handle just in case
                 num_val_loaders = 1
                 print("Warning: trainer.val_dataloaders is not a list, assuming 1 loader.")
        else:
             print("Warning: trainer.val_dataloaders not available or empty at on_validation_epoch_start.")

        # Initialize list of lists for outputs
        self.validation_step_outputs = [[] for _ in range(num_val_loaders)]

        # Get validation dataset names (ensure datamodule is attached and setup)
        if self.trainer.datamodule and hasattr(self.trainer.datamodule, 'get_val_dataset_names'):
             self.val_dataset_names = self.trainer.datamodule.get_val_dataset_names()
             # Verify length matches num_val_loaders
             if len(self.val_dataset_names) != num_val_loaders:
                  print(f"Warning: Mismatch between found dataset names ({len(self.val_dataset_names)}) and number of loaders ({num_val_loaders}). Using fallback names.")
                  self.val_dataset_names = [f"val_{i}" for i in range(num_val_loaders)]
        else:
             print("Warning: Datamodule or get_val_dataset_names method not available at on_validation_epoch_start.")
             self.val_dataset_names = [f"val_{i}" for i in range(num_val_loaders)]
        print(f"Validation Epoch {self.current_epoch} Start. Expecting {num_val_loaders} dataloaders. Names: {self.val_dataset_names}")


    # ... (training_step, on_train_batch_end, configure_optimizers, validation_step, on_validation_epoch_end, sample_and_log_validation_images, __log_images remain exactly the same as the previous version) ...
    # Paste the rest of the LitImprovedConsistencyModel methods from the previous version here, unchanged.

    def training_step(self, batch: Union[Tensor, List[Tensor], None], batch_idx: int) -> Optional[Tensor]:
        if batch is None:
             return None
        visible_images, infrared_images = batch
        output = self.consistency_training(
            model=self.model,
            x=infrared_images,
            v=visible_images,
            current_training_step=self.global_step,
            total_training_steps=self.trainer.max_steps if self.trainer.max_steps > 0 else 1000000
        )
        loss = (pseudo_huber_loss(output.predicted, output.target) * output.loss_weights).mean()
        self.log_dict(
            {"train/loss": loss, "train/num_timesteps": float(output.num_timesteps)},
            on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def on_train_batch_end(self, outputs: Optional[Dict[str, Tensor]], batch: Any, batch_idx: int) -> None:
        if outputs is None or 'loss' not in outputs or torch.isnan(outputs['loss']):
            return
        update_ema_model_(self.ema_model, self.model, self.hparams.config.ema_decay_rate)


    def configure_optimizers(self):
    # Use AdamW instead of Adam
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.config.lr,
            betas=self.hparams.config.betas,
            weight_decay=1e-5 # Add a small weight decay (e.g., 1e-6 to 1e-4)
        )

        # --- Schedulers remain the same ---
        # Linear Warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.hparams.config.lr_scheduler_start_factor,
            total_iters=self.hparams.config.lr_scheduler_iters,
        )
        # Cosine Decay after warmup
        decay_steps = self.trainer.max_steps - self.hparams.config.lr_scheduler_iters
        # Ensure decay_steps is at least 1 to avoid errors if max_steps <= warmup_steps
        decay_steps = max(1, decay_steps)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=decay_steps,
            eta_min=self.hparams.config.lr * 0.01 # Decay to 1% of initial LR
        )
        # Chain the schedulers
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.hparams.config.lr_scheduler_iters]
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def validation_step(self, batch: Union[Tensor, List[Tensor], None], batch_idx: int, dataloader_idx: int = 0):
        if batch is None:
             return # Don't store anything for bad batches

        visible_images, infrared_images = batch
        batch_size = visible_images.shape[0]

        # Determine log prefix
        if dataloader_idx < len(self.val_dataset_names):
             log_prefix = f"val/{self.val_dataset_names[dataloader_idx]}/"
        else:
             # This case should be less likely now with initialization in on_validation_epoch_start
             print(f"Warning: validation_step dataloader_idx {dataloader_idx} out of range for names {self.val_dataset_names}.")
             log_prefix = f"val/dl_{dataloader_idx}/"

        self.ema_model.eval()
        psnr_batch_vals = []
        ssim_batch_vals = []
        sigmas = self.hparams.config.val_sampling_sigmas
        if not sigmas: sigmas = tuple(karras_schedule(11)[1:].tolist())
        max_sigma = sigmas[0]

        for i in range(batch_size):
            vis_img = visible_images[i].unsqueeze(0)
            inf_img_gt = infrared_images[i].unsqueeze(0)
            noise = torch.randn_like(inf_img_gt, device=self.device) * max_sigma
            predicted_inf_img = self.consistency_sampling(
                model=self.ema_model, y=noise, v=vis_img, sigmas=sigmas,
                clip_denoised=True, verbose=False
            )
            predicted_inf_img = predicted_inf_img.clamp(min=-1.0, max=1.0)
            gt_img_0_1 = (inf_img_gt + 1) / 2.0
            pred_img_0_1 = (predicted_inf_img + 1) / 2.0
            gt_np = gt_img_0_1.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_np = pred_img_0_1.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np_uint8 = (np.clip(gt_np, 0, 1) * 255).astype(np.uint8)
            pred_np_uint8 = (np.clip(pred_np, 0, 1) * 255).astype(np.uint8)
            H, W, _ = gt_np_uint8.shape
            win_size = min(7, H, W)
            if win_size % 2 == 0: win_size -= 1
            try:
                if win_size >= 3:
                    psnr_val = calculate_psnr(gt_np_uint8, pred_np_uint8, data_range=255)
                    ssim_val = calculate_ssim(
                        gt_np_uint8, pred_np_uint8, data_range=255.0,
                        channel_axis=-1, win_size=win_size, gaussian_weights=True, sigma=1.5
                    )
                    psnr_batch_vals.append(psnr_val)
                    ssim_batch_vals.append(ssim_val)
            except ValueError as e:
                 print(f"Metric calculation error (batch {batch_idx}, item {i}, DL {dataloader_idx}): {e}. Skipping.")
                 continue

        avg_psnr_batch = torch.tensor(np.nanmean(psnr_batch_vals) if psnr_batch_vals else float('nan'), device=self.device)
        avg_ssim_batch = torch.tensor(np.nanmean(ssim_batch_vals) if ssim_batch_vals else float('nan'), device=self.device)

        # --- NEW: Store outputs in the class attribute ---
        output_dict = {f"{log_prefix}psnr": avg_psnr_batch, f"{log_prefix}ssim": avg_ssim_batch}
        # Ensure dataloader_idx is valid before appending
        if dataloader_idx < len(self.validation_step_outputs):
            self.validation_step_outputs[dataloader_idx].append(output_dict)
        else:
             print(f"Error: dataloader_idx {dataloader_idx} invalid for self.validation_step_outputs (len {len(self.validation_step_outputs)})")

        # Optional: Return value (not used by on_validation_epoch_end anymore, but might be useful for other callbacks)
        # return output_dict


    # --- RENAMED and MODIFIED Hook ---
    # No longer receives outputs as argument
    def on_validation_epoch_end(self):
        print("\n--- Validation Epoch End ---")

        # --- Access stored outputs ---
        # self.validation_step_outputs is the list of lists populated by validation_step
        num_dataloaders = len(self.validation_step_outputs)
        if num_dataloaders != len(self.val_dataset_names):
            print(f"Warning: Mismatch between stored outputs ({num_dataloaders}) and dataset names ({len(self.val_dataset_names)}). Adjusting...")
            # Adjust based on available outputs
            num_dataloaders = min(num_dataloaders, len(self.val_dataset_names))


        all_epoch_metrics = {}

        for dataloader_idx in range(num_dataloaders):
            # Try to get dataset name, use fallback if index is out of bounds
            dataset_name = self.val_dataset_names[dataloader_idx] if dataloader_idx < len(self.val_dataset_names) else f"dl_{dataloader_idx}"
            log_prefix = f"val/{dataset_name}/"
            # --- Process stored outputs for this dataloader ---
            outputs_dl = self.validation_step_outputs[dataloader_idx]

            if not isinstance(outputs_dl, list):
                 print(f"Warning: Expected list of outputs for dataloader {dataloader_idx}, got {type(outputs_dl)}. Skipping.")
                 continue

            # Extract metrics from the stored dictionaries
            psnr_vals = [out[f'{log_prefix}psnr'] for out in outputs_dl if isinstance(out, dict) and f'{log_prefix}psnr' in out and isinstance(out[f'{log_prefix}psnr'], Tensor) and not torch.isnan(out[f'{log_prefix}psnr'])]
            ssim_vals = [out[f'{log_prefix}ssim'] for out in outputs_dl if isinstance(out, dict) and f'{log_prefix}ssim' in out and isinstance(out[f'{log_prefix}ssim'], Tensor) and not torch.isnan(out[f'{log_prefix}ssim'])]

            # Calculate average, handling case where no valid metrics were collected
            avg_psnr = torch.stack(psnr_vals).mean() if psnr_vals else torch.tensor(float('nan'), device=self.device)
            avg_ssim = torch.stack(ssim_vals).mean() if ssim_vals else torch.tensor(float('nan'), device=self.device)

            # Log epoch metrics
            epoch_metrics = {
                f"{log_prefix}psnr_epoch": avg_psnr,
                f"{log_prefix}ssim_epoch": avg_ssim
            }
            # Log using self.log_dict for proper handling by logger and callbacks
            # Use sync_dist=True for correct aggregation in DDP
            self.log_dict(epoch_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, add_dataloader_idx=False)
            # Add to prog bar manually if desired (only logs on rank 0)
            self.log(f"{log_prefix}psnr_prog", avg_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=False, rank_zero_only=True, add_dataloader_idx=False)

            all_epoch_metrics.update(epoch_metrics)
            # Print results (rank 0 only to avoid clutter)
            if self.trainer.is_global_zero:
                 print(f"  {dataset_name}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f} ({len(psnr_vals)} valid batches)")


        # --- Sampling logic remains the same, just called from here ---
        should_sample = (self.current_epoch + 1) % self.hparams.config.sample_every_n_val_epochs == 0
        if should_sample and self.trainer.is_global_zero:
             print(f"\nSampling validation images at end of epoch {self.current_epoch} (global step {self.global_step})")
             self.sample_and_log_validation_images()

        # --- Crucial: Clear the stored outputs for the next epoch ---
        # This is now handled by on_validation_epoch_start
        # self.validation_step_outputs.clear() # No longer needed here


    @torch.no_grad()
    def sample_and_log_validation_images(self):
        """Samples images from each validation dataloader and logs them."""
        self.ema_model.eval()
        try:
            val_dataloaders = self.trainer.val_dataloaders
            if not val_dataloaders: return
            if not isinstance(val_dataloaders, list): val_dataloaders = [val_dataloaders]
            dm_batch_size = self.trainer.datamodule.config.batch_size if hasattr(self.trainer.datamodule, 'config') else 4
            num_samples = min(self.hparams.config.num_samples, dm_batch_size)
            sigmas = self.hparams.config.val_sampling_sigmas
            if not sigmas: sigmas = tuple(karras_schedule(11)[1:].tolist())
            max_sigma = sigmas[0]
        except Exception as e:
            print(f"Error preparing for validation sampling: {e}")
            traceback.print_exc()
            return

        for dataloader_idx, loader in enumerate(val_dataloaders):
            dataset_name = self.val_dataset_names[dataloader_idx] if dataloader_idx < len(self.val_dataset_names) else f"dl_{dataloader_idx}"
            prefix = f"samples/{dataset_name}"
            print(f" Sampling from {dataset_name}...")
            try:
                batch = next(iter(loader))
                attempts = 0
                max_attempts = 5
                while batch is None and hasattr(loader, "__iter__") and attempts < max_attempts:
                    batch = next(iter(loader))
                    attempts += 1
                if batch is None:
                    print(f"Could not get a valid batch for sampling from {dataset_name} after {attempts} attempts.")
                    continue

                visible_images, infrared_images = batch
                vis_to_log = visible_images[:num_samples].to(self.device)
                inf_gt_to_log = infrared_images[:num_samples].to(self.device)
                noise = torch.randn_like(inf_gt_to_log, device=self.device) * max_sigma
                samples = self.consistency_sampling(
                    model=self.ema_model, y=noise, v=vis_to_log, sigmas=sigmas,
                    clip_denoised=True, verbose=False
                )
                samples = samples.clamp(min=-1.0, max=1.0)
                self.__log_images(inf_gt_to_log, f"{prefix}/0_ground_truth_target", self.global_step)
                self.__log_images(vis_to_log, f"{prefix}/1_input_visible", self.global_step)
                self.__log_images(samples, f"{prefix}/2_generated", self.global_step)
                print(f"  Logged samples for {dataset_name}")
            except StopIteration:
                print(f"Warning: Validation dataloader {dataloader_idx} ({dataset_name}) exhausted, cannot sample.")
            except Exception as e:
                print(f"Error during sampling for dataloader {dataloader_idx} ({dataset_name}): {e}")
                traceback.print_exc()


    @torch.no_grad()
    def __log_images(self, images: Tensor, title: str, global_step: int) -> None:
        """Logs a grid of images to TensorBoard."""
        if self.logger is None or not hasattr(self.logger.experiment, 'add_image'):
            return
        images = images.detach().float().cpu()
        images = images.clamp(-1.0, 1.0)
        try:
             nrow = int(math.sqrt(images.shape[0])) if images.shape[0] > 0 else 1
             grid = make_grid(images, nrow=max(1, nrow), normalize=True, value_range=(-1.0, 1.0))
             self.logger.experiment.add_image(title, grid, global_step)
        except Exception as e:
             print(f"Error creating or logging image grid '{title}': {e}")


# ... (Keep TrainingConfig, run_training, main functions exactly as before) ...

# -------------------------
# Training Configuration and Main Execution
# -------------------------

@dataclass
class TrainingConfig:
    data_config: CombinedImageDataModuleConfig # Holds dataset paths and params
    unet_config: UNetConfig # Holds UNet architecture params
    consistency_train_config: ImprovedConsistencyTraining # Holds ICT params
    consistency_sample_config: ConsistencySamplingAndEditing # Holds CSAE params
    lit_model_config: LitImprovedConsistencyModelConfig # Holds LitModule params (LR, EMA, etc.)
    trainer_config: Dict[str, Any] # Holds PyTorch Lightning Trainer params
    project_name: str = "lol_combined"
    seed: int = 42
    resume_from_ckpt: Optional[str] = None # Path to checkpoint to resume from


def run_training(config: TrainingConfig) -> None:
    """Sets up and runs the training process."""
    print("--- Initializing Training ---")
    print(f"Project Name: {config.project_name}")
    print(f"Seed: {config.seed}")
    seed_everything(config.seed, workers=True)

    # 1. Create DataModule
    print("\n--- Creating DataModule ---")
    dm = CombinedLOLDataModule(config.data_config)
    # Note: Trainer calls dm.setup() automatically before fit/validate/test

    # 2. Create Model and EMA Model
    print("\n--- Creating Models (UNet + EMA) ---")
    model = UNet(config.unet_config)
    ema_model = UNet(config.unet_config)
    # Initialize EMA weights same as model - crucial step
    ema_model.load_state_dict(model.state_dict())
    print(" Models created.")

    # Print model summary (optional, requires torchinfo)
    try:
        # Needs example input shape - use config values
        example_img_size = config.data_config.image_size_crop
        # Use placeholder batch size 1 for summary to avoid large allocation
        example_input_shape = (1, config.unet_config.image_channels, *example_img_size)
        example_cond_shape = (1, config.unet_config.conditional_channels, *example_img_size)
        example_sigma_shape = (1,)
        print("\n--- UNet Summary ---")
        # Limit depth for brevity
        summary(model, input_data=[torch.randn(example_input_shape), torch.randn(example_sigma_shape), torch.randn(example_cond_shape)], depth=5, verbose=0)
    except Exception as e:
        print(f"Could not print model summary: {e}")


    # 3. Create Lightning Module
    print("\n--- Creating Lightning Module ---")
    lit_icm = LitImprovedConsistencyModel(
        consistency_training=config.consistency_train_config,
        consistency_sampling=config.consistency_sample_config,
        model=model,
        ema_model=ema_model,
        config=config.lit_model_config, # Pass the LitModule specific config
    )
    print(" Lightning Module created.")

    # 4. Setup Logger and Callbacks
    print("\n--- Setting up Logger and Callbacks ---")
    # Define log directory relative to script location
    log_dir = "logs"
    # Add seed to version for better tracking if running multiple seeds
    logger_version = f"seed_{config.seed}_{config.data_config.batch_size}bs"
    logger = TensorBoardLogger(log_dir, name=config.project_name, version=logger_version)
    print(f" TensorBoard logs will be saved to: {logger.log_dir}")

    # Define checkpoint directory relative to script location
    checkpoint_dir = os.path.join("checkpoints", config.project_name, logger_version)
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Define metric to monitor for saving best checkpoints (use a specific validation set)
    # Ensure the metric name exactly matches what's logged in validation_epoch_end
    monitor_metric = "val/traffic/psnr_epoch" # <<< ENSURE 'lolv1' exists if monitoring it
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="ckpt-{epoch:03d}-{step:06d}",#-{monitor_metric:.2f}", # Filename template - metric needs formatting if included
        save_top_k=3,
        monitor=monitor_metric,
        mode="max", # Maximize PSNR
        save_last=True, # Save the latest checkpoint regardless of performance
        auto_insert_metric_name=False, # We manually control logging names
        every_n_epochs=config.trainer_config.get("check_val_every_n_epoch", 1) * 5 # Save checkpoint less frequently
    )
    print(f" Checkpoints will be saved to: {checkpoint_dir}")
    print(f" Monitoring '{monitor_metric}' for best checkpoints.")

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [lr_monitor, checkpoint_callback]
    print(" Callbacks created: LearningRateMonitor, ModelCheckpoint")

    # 5. Create Trainer
    print("\n--- Creating Trainer ---")
    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **config.trainer_config # Pass trainer settings from config
    )
    print(f" Trainer configured with: {config.trainer_config}")

    # 6. Start Training
    print("\n--- Starting Training ---")
    print(f" Resuming from checkpoint: {config.resume_from_ckpt}" if config.resume_from_ckpt else " Starting from scratch.")
    # Trainer handles setup call internally
    try:
        trainer.fit(lit_icm, datamodule=dm, ckpt_path=config.resume_from_ckpt)
        print("\n--- Training Completed ---")
    except Exception as e:
         print("\n--- Training Interrupted ---")
         print(f"Error: {e}")
         traceback.print_exc()
         # Optionally save state even if interrupted
         if trainer.is_global_zero:
             interrupted_ckpt_path = os.path.join(checkpoint_dir, "interrupted_last.ckpt")
             trainer.save_checkpoint(interrupted_ckpt_path)
             print(f"Saved intermediate state to {interrupted_ckpt_path}")


    # 7. Save Final Model Artifacts (Optional, as checkpoints are saved)
    # Check if training finished or was interrupted but we still want to save
    if trainer.state.status in ["finished", "interrupted"] and trainer.is_global_zero :
        final_model_dir = f"trained_models/{config.project_name}_final_step_{trainer.global_step}"
        print(f"\n--- Saving Final Model Artifacts (Rank 0) ---")
        print(f" Saving to: {final_model_dir}")
        os.makedirs(final_model_dir, exist_ok=True)
        try:
            # Save the online model config and weights
            lit_icm.model.save_pretrained(os.path.join(final_model_dir, "online_model"))
            # Save the EMA model config and weights
            lit_icm.ema_model.save_pretrained(os.path.join(final_model_dir, "ema_model"))
            print(" Final model artifacts saved.")
        except Exception as e:
            print(f"Error saving final models: {e}")


def main():
    # --- Core Configuration ---
    # All paths are now relative to where the script is run
    # Assumes 'datasets/', 'logs/', 'checkpoints_...', 'trained_models/' folders
    # will be created/used in the same directory as the script.

    # 1. Dataset Paths (Relative)
    # These should point to the directories created by the mkdir commands
    irvi_root     = "IRVI/single"
    traffic_dir   = os.path.join(irvi_root, "traffic")
    rooftop_dir   = os.path.join(irvi_root, "rooftop_total")
    # lolv1_base_path      = 
    lolv2_real_base_path = traffic_dir
    # lolv2_synthetic_base_path = "datasets/LOLv2-synthetic"

    # Set to None or "" to exclude a dataset (the DataModule will handle missing dirs)
    # e.g., lolv2_real_base_path = None

    # 2. Training Parameters
    PROJECT_NAME = "irvi_single_traffic_bs_1" # Name for logs/checkpoints
    SEED = 42
    BATCH_SIZE = 1 # ** ADJUST BASED ON YOUR GPU MEMORY **
    NUM_WORKERS = 6 # ** ADJUST BASED ON YOUR CPU CORES **
    MAX_TRAINING_STEPS = 500000000 # Total training steps
    VALIDATE_EVERY_N_EPOCHS = 5 # How often to run validation loop
    LEARNING_RATE = 1e-4
    LR_WARMUP_STEPS = 5000 # Steps for linear LR warmup
    EMA_DECAY = 0.9999
    PRECISION = "16-mixed" # Use "32" or "bf16-mixed" if needed/supported
    ACCELERATOR = "gpu"
    # For single GPU: [0] or 1. For multi-GPU: [0, 1, ...] or integer > 1 or "auto"
    DEVICES = [0]
    GRADIENT_CLIP_VAL = 1.0 # Optional gradient clipping

    # 3. Model Architecture
    IMG_SIZE = 256
    BASE_CHANNELS = 64
    # Ensure length matches use_attention_at_resolution length
    CHANNEL_MULTIPLIERS = (1, 2, 4, 4) # e.g., 64, 128, 256, 256
    ATTENTION_RESOLUTIONS = (False, False, True, True) # Use attention where channel dim is high
    BLOCKS_PER_RESOLUTION = 2
    DROPOUT = 0.1
    N_HEADS = 4 # Number of heads for attention layers

    # 4. Consistency Training Parameters
    # See ImprovedConsistencyTraining defaults or adjust as needed
    ICT_INITIAL_TIMESTEPS = 10
    ICT_FINAL_TIMESTEPS = 150 # N value for improved schedule - adjust as needed
    ICT_SIGMA_MIN = 0.002
    ICT_SIGMA_MAX = 80.0
    ICT_RHO = 7.0

    # 5. Sampling Parameters
    SAMPLE_EVERY_N_VAL_EPOCHS = 1 # Log samples every time validation runs
    NUM_SAMPLES_TO_LOG = 8 # Number of images in logged grids (should be <= batch size)
    # Define a fixed sequence for validation sampling
    VAL_SAMPLING_SIGMAS = tuple(karras_schedule(num_timesteps=11, sigma_min=ICT_SIGMA_MIN, sigma_max=ICT_SIGMA_MAX, rho=ICT_RHO)[1:].tolist())
    # Example: [80. , 39.17, 19.18,  9.39,  4.6 ,  2.25,  1.1 ,  0.54,  0.26, 0.13,  0.06]

    # 6. Resume Training (Optional)
    RESUME_CHECKPOINT_PATH = "/raid/home/avs/Conditional-Consistency-Models/amilb/irvi/checkpoints/irvi_single_traffic_bs_1/seed_42_1bs/interrupted_last.ckpt"

    print("--- Training Configuration ---")
    print(f" Project: {PROJECT_NAME}, Seed: {SEED}")
    print(f" Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    print(f" Max Steps: {MAX_TRAINING_STEPS}, Val Every: {VALIDATE_EVERY_N_EPOCHS} epochs")
    print(f" Precision: {PRECISION}, Devices: {DEVICES}")
    print(f" Image Size: {IMG_SIZE}, Base Channels: {BASE_CHANNELS}")
    print(f" Resume Checkpoint: {RESUME_CHECKPOINT_PATH}")
    print("-----------------------------")

    # --- Assemble Configuration Objects ---

    # data_cfg = CombinedImageDataModuleConfig(
    #     lolv1_dir=lolv1_base_path,
    #     lolv2_real_dir=lolv2_real_base_path,
    #     # lolv2_synthetic_dir=lolv2_synthetic_base_path,
    #     image_size_crop=(IMG_SIZE, IMG_SIZE),
    #     resize_size_val=None, # Resize validation images
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    # )
    
    data_cfg = CombinedImageDataModuleConfig(
        #lolv1_dir=rooftop_dir,
        lolv2_real_dir=traffic_dir,

        train_input_subdir="trainB",    # VIS (condition)
        train_target_subdir="trainA",   # IR  (target)
        val_input_subdir="testB",       # VIS
        val_target_subdir="testA",      # IR

        image_size_crop=(IMG_SIZE, IMG_SIZE),
        resize_size_val=None,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        )

    unet_cfg = UNetConfig(
        image_channels=3, # Target image (Normal Light)
        output_channels=3, # Predicted Normal Light
        conditional_channels=3, # Condition (Low Light)
        base_channels=BASE_CHANNELS,
        channel_multipliers=CHANNEL_MULTIPLIERS,
        blocks_per_resolution=BLOCKS_PER_RESOLUTION,
        use_attention_at_resolution=ATTENTION_RESOLUTIONS,
        dropout=DROPOUT,
        noise_level_channels=256, # Default from original code
        n_heads=N_HEADS
    )

    # Configure Improved Consistency Training
    ict_cfg = ImprovedConsistencyTraining(
         initial_timesteps=ICT_INITIAL_TIMESTEPS,
         final_timesteps=ICT_FINAL_TIMESTEPS,
         sigma_min=ICT_SIGMA_MIN,
         sigma_max=ICT_SIGMA_MAX,
         rho=ICT_RHO
    )

    # Configure Consistency Sampling (defaults are usually fine)
    csae_cfg = ConsistencySamplingAndEditing(
        sigma_min=ICT_SIGMA_MIN # Match sigma_min with training
    )

    # Configure Lightning Module Hyperparameters
    lit_model_cfg = LitImprovedConsistencyModelConfig(
        ema_decay_rate=EMA_DECAY,
        lr=LEARNING_RATE,
        lr_scheduler_iters=LR_WARMUP_STEPS,
        sample_every_n_val_epochs=SAMPLE_EVERY_N_VAL_EPOCHS,
        num_samples=NUM_SAMPLES_TO_LOG,
        val_sampling_sigmas=VAL_SAMPLING_SIGMAS,
    )

    # Configure PyTorch Lightning Trainer
    trainer_cfg = {
        "accelerator": ACCELERATOR,
        "devices": DEVICES,
        "max_steps": MAX_TRAINING_STEPS,
        "precision": PRECISION,
        "log_every_n_steps": 50, # How often to log metrics during training steps
        "check_val_every_n_epoch": VALIDATE_EVERY_N_EPOCHS,
        "gradient_clip_val": GRADIENT_CLIP_VAL,
        # "benchmark": True, # Can improve performance if input sizes are constant
        # Set strategy explicitly: "auto" lets Lightning choose based on devices
        "strategy": "auto",
    }

    # Assemble the final configuration object
    full_config = TrainingConfig(
        data_config=data_cfg,
        unet_config=unet_cfg,
        consistency_train_config=ict_cfg,
        consistency_sample_config=csae_cfg,
        lit_model_config=lit_model_cfg,
        trainer_config=trainer_cfg,
        project_name=PROJECT_NAME,
        seed=SEED,
        resume_from_ckpt=RESUME_CHECKPOINT_PATH
    )

    # --- Run Training ---
    run_training(full_config)


if __name__ == "__main__":
    # Check necessary imports are available
    print("Checking dependencies...")
    try:
        import lightning
        import skimage
        import einops
        import torchinfo
        import PIL
        import numpy
        import tqdm
        print(" All major dependencies found.")
    except ImportError as e:
        print(f"CRITICAL: Missing dependency: {e}.")
        print("Please install required packages:")
        print(" pip install torch torchvision torchaudio lightning scikit-image einops torchinfo Pillow tqdm numpy")
        exit(1)

    main()