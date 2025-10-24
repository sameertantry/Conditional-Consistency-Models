import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import shutil

from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import lpips
from DISTS_pytorch import DISTS


def split_concatenated_image(concat_path, output_real_fake_dir):
    """
    Split a concatenated image (HE | Real IHC | Fake IHC) into real and fake.
    
    Returns paths to saved real and fake images.
    """
    img = Image.open(concat_path).convert("RGB")
    width, height = img.size
    
    # Assuming 3 equal-width images concatenated horizontally
    single_width = width // 3
    
    # Extract middle (real) and right (fake) sections
    real_ihc = img.crop((single_width, 0, single_width * 2, height))
    fake_ihc = img.crop((single_width * 2, 0, width, height))
    
    # Generate filenames
    base_name = os.path.splitext(os.path.basename(concat_path))[0]
    if base_name.startswith("concatenated_"):
        base_name = base_name.replace("concatenated_", "")
    
    real_path = os.path.join(output_real_fake_dir, f"{base_name}_real_B.png")
    fake_path = os.path.join(output_real_fake_dir, f"{base_name}_fake_B.png")
    
    # Save images
    real_ihc.save(real_path)
    fake_ihc.save(fake_path)
    
    return real_path, fake_path


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img)


def to_tensor(img_np):
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t


def to_tensor_neg1_1(img_np):
    t = to_tensor(img_np) * 2 - 1
    return t


def compute_ssim_psnr(real_path, fake_path):
    real_np = load_image(real_path)
    fake_np = load_image(fake_path, size=real_np.shape[:2][::-1])
    s = ssim(real_np, fake_np, channel_axis=2, data_range=255)
    p = psnr(real_np, fake_np, data_range=255)
    return s, p


def compute_fid(split_folder, device="cuda"):
    """
    Compute FID using pytorch-fid.
    
    Args:
        split_folder: Folder containing both *_real_B.png and *_fake_B.png files
        device: Device to use for computation
    
    Returns:
        FID value or None if computation fails
    """
    # Create separate temporary folders for real and fake images
    temp_real = os.path.join(split_folder, "temp_fid_real")
    temp_fake = os.path.join(split_folder, "temp_fid_fake")
    
    os.makedirs(temp_real, exist_ok=True)
    os.makedirs(temp_fake, exist_ok=True)
    
    try:
        # Copy real and fake images to separate folders
        real_paths = glob.glob(os.path.join(split_folder, "*_real_B.png"))
        fake_paths = glob.glob(os.path.join(split_folder, "*_fake_B.png"))
        
        print(f"  Copying {len(real_paths)} real images to temp folder...")
        for src in real_paths:
            dst = os.path.join(temp_real, os.path.basename(src))
            shutil.copy2(src, dst)
        
        print(f"  Copying {len(fake_paths)} fake images to temp folder...")
        for src in fake_paths:
            dst = os.path.join(temp_fake, os.path.basename(src))
            shutil.copy2(src, dst)
        
        # Determine device flag
        device_flag = ""
        if device == "cuda" and torch.cuda.is_available():
            device_flag = "--device cuda"
        else:
            device_flag = "--device cpu"
        
        print(f"  Running pytorch-fid...")
        # Run pytorch-fid command
        cmd = f"python -m pytorch_fid {temp_real} {temp_fake} {device_flag}"
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Parse FID from output
        for line in result.stdout.split('\n'):
            if 'FID:' in line:
                fid_value = float(line.split('FID:')[1].strip())
                print(f"  ✓ FID computed: {fid_value:.4f}")
                return fid_value
        
        # If we couldn't find FID in the expected format, try alternative parsing
        if result.stdout:
            # Sometimes pytorch-fid just outputs the number
            try:
                fid_value = float(result.stdout.strip().split()[-1])
                print(f"  ✓ FID computed: {fid_value:.4f}")
                return fid_value
            except:
                pass
        
        print("  ✗ Could not parse FID from output")
        print(f"  stdout: {result.stdout}")
        print(f"  stderr: {result.stderr}")
        return None
        
    except subprocess.TimeoutExpired:
        print("  ✗ FID computation timed out")
        return None
    except Exception as e:
        print(f"  ✗ Error computing FID: {e}")
        return None
    finally:
        # Cleanup temp folders
        if os.path.exists(temp_real):
            shutil.rmtree(temp_real)
        if os.path.exists(temp_fake):
            shutil.rmtree(temp_fake)


def compute_metrics_from_concatenated(concat_folder, batch_size=8, num_workers=4, num=20):
    """
    Compute metrics from concatenated images.
    
    Args:
        concat_folder: Folder containing concatenated images (HE | Real | Fake)
        batch_size: Batch size for LPIPS/DISTS
        num_workers: Number of workers for parallel processing
    """
    print(f"\nProcessing concatenated images from: {concat_folder}")
    
    # Create temporary folder for split images
    split_folder = concat_folder + "_split"
    os.makedirs(split_folder, exist_ok=True)
    
    # Find all concatenated images
    concat_paths = sorted(glob.glob(os.path.join(concat_folder, "*.png"))) + \
                   sorted(glob.glob(os.path.join(concat_folder, "*.jpg")))
    
    print(f"Found {len(concat_paths)} concatenated images")
    
    # Split all concatenated images
    print("\nSplitting concatenated images...")
    real_paths = []
    fake_paths = []
    for concat_path in tqdm(concat_paths[:63], desc="Splitting"):
        real_path, fake_path = split_concatenated_image(concat_path, split_folder)
        real_paths.append(real_path)
        fake_paths.append(fake_path)
    
    print(f"✓ Split into {len(real_paths)} real/fake pairs in {split_folder}")
    
    # Match images
    real_dict = {os.path.basename(p).replace("_real_B.png", ""): p for p in real_paths}
    fake_dict = {os.path.basename(p).replace("_fake_B.png", ""): p for p in fake_paths}
    
    names = sorted(set(real_dict.keys()) & set(fake_dict.keys()))
    if not names:
        print("ERROR: No matching image pairs found!")
        return None
    
    print(f"Found {len(names)} matching pairs")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. SSIM and PSNR
    ssim_scores, psnr_scores = [], []
    real_paths_list = [real_dict[name] for name in names]
    fake_paths_list = [fake_dict[name] for name in names]
    
    print("\n1/4: Computing SSIM and PSNR...")
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_ssim_psnr, r, f) for r, f in zip(real_paths_list, fake_paths_list)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="SSIM/PSNR"):
            s, p = future.result()
            ssim_scores.append(s)
            psnr_scores.append(p)
    
    # 2. LPIPS and DISTS
    class PairDataset(torch.utils.data.Dataset):
        def __init__(self, names, real_dict, fake_dict):
            self.names = names
            self.real_dict = real_dict
            self.fake_dict = fake_dict
        
        def __len__(self):
            return len(self.names)
        
        def __getitem__(self, idx):
            name = self.names[idx]
            real_np = load_image(self.real_dict[name])
            fake_np = load_image(self.fake_dict[name], size=real_np.shape[:2][::-1])
            r01 = to_tensor(real_np).squeeze(0)
            f01 = to_tensor(fake_np).squeeze(0)
            r11 = to_tensor_neg1_1(real_np).squeeze(0)
            f11 = to_tensor_neg1_1(fake_np).squeeze(0)
            return r01, f01, r11, f11
    
    print("\n2/4: Computing LPIPS and DISTS...")
    dists_metric = DISTS().to(device)
    lpips_metric = lpips.LPIPS(net='alex').to(device)
    lpips_metric.eval()
    
    dataset = PairDataset(names, real_dict, fake_dict)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    
    dists_scores, lpips_scores = [], []
    with torch.no_grad():
        for r01, f01, r11, f11 in tqdm(loader, desc="LPIPS/DISTS"):
            r01, f01 = r01.to(device), f01.to(device)
            r11, f11 = r11.to(device), f11.to(device)
            
            d = dists_metric(r01, f01)
            l = lpips_metric(r11, f11)
            
            if isinstance(d, torch.Tensor):
                dists_scores.extend(d.cpu().numpy().tolist()) if d.dim() > 0 else dists_scores.append(d.item())
            else:
                dists_scores.append(float(d))
            
            if isinstance(l, torch.Tensor):
                lpips_scores.extend(l.cpu().numpy().tolist()) if l.dim() > 0 else lpips_scores.append(l.item())
            else:
                lpips_scores.append(float(l))
            
            del r01, f01, r11, f11, d, l
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # 3. FID
    print("\n3/4: Computing FID...")
    fid_value = compute_fid(split_folder, device)
    
    results = {
        "concat_folder": concat_folder,
        "num_images": len(names),
        "FID": fid_value if fid_value is not None else "N/A",
        "SSIM": np.mean(ssim_scores),
        "PSNR": np.mean(psnr_scores),
        "LPIPS": np.mean(lpips_scores),
        "DISTS": np.mean(dists_scores),
    }
    
    print("\n✓ Metrics computed successfully!")
    if fid_value is not None:
        print(f"  FID:    {fid_value:.4f}")
    else:
        print(f"  FID:    N/A (computation failed)")
    print(f"  SSIM:   {np.mean(ssim_scores):.4f}")
    print(f"  PSNR:   {np.mean(psnr_scores):.2f} dB")
    print(f"  LPIPS:  {np.mean(lpips_scores):.4f}")
    print(f"  DISTS:  {np.mean(dists_scores):.4f}")
    
    return results


if __name__ == "__main__":
    # Configuration
    concat_folder = "results_mist_her2"  # Folder with concatenated images
    
    # Compute metrics
    print("="*60)
    print("Computing metrics from concatenated images")
    print("="*60)
    
    metrics = compute_metrics_from_concatenated(
        concat_folder,
        batch_size=8,
        num_workers=4
    )
    
    if metrics is not None:
        # Save to CSV
        df = pd.DataFrame([metrics])
        csv_path = 'mist.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to {csv_path}")
        
        # Save to text file
        txt_path = 'mist.txt'
        with open(txt_path, 'w') as f:
            f.write(f"Folder: {metrics['concat_folder']}\n")
            f.write(f"Number of images: {metrics['num_images']}\n\n")
            if metrics['FID'] != "N/A":
                f.write(f"FID:    {metrics['FID']:.4f}\n")
            else:
                f.write(f"FID:    N/A\n")
            f.write(f"SSIM:   {metrics['SSIM']:.4f}\n")
            f.write(f"PSNR:   {metrics['PSNR']:.2f} dB\n")
            f.write(f"LPIPS:  {metrics['LPIPS']:.4f}\n")
            f.write(f"DISTS:  {metrics['DISTS']:.4f}\n")
        print(f"✓ Results saved to {txt_path}")
    else:
        print("\nERROR: Could not compute metrics!")
