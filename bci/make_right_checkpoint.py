import torch
import json
import os

# Start fresh - convert directly from the .ckpt checkpoint
ckpt_path = "checkpoints_bci/epoch=0-step=974.ckpt"
output_dir = "checkpoints_bci/pretrained_model_epoch0"

# Load the original Lightning checkpoint
print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location='cpu')

print(f"Checkpoint from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")
print(f"State_dict has {len(checkpoint['state_dict'])} keys")

# Check first key to see the prefix structure
first_key = list(checkpoint['state_dict'].keys())[0]
print(f"First key example: {first_key}")

# Remove prefixes from state_dict - KEEP FIRST OCCURRENCE ONLY
clean_state_dict = {}
skipped_count = 0

for key, value in checkpoint['state_dict'].items():
    # Remove all prefixes to get to the actual parameter name
    new_key = key.replace('ema_model.', '').replace('model.', '')
    
    # Only keep the FIRST occurrence of each key
    if new_key not in clean_state_dict:
        clean_state_dict[new_key] = value
    else:
        skipped_count += 1
        print(f"Skipping duplicate: {key} -> {new_key}")

print(f"\nCleaned state_dict has {len(clean_state_dict)} keys")
print(f"Skipped {skipped_count} duplicate keys")
print(f"First cleaned key: {list(clean_state_dict.keys())[0]}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Save config.json
config = {
    "channels": 3,
    "noise_level_channels": 256,
    "noise_level_scale": 0.02,
    "n_heads": 8,
    "top_blocks_channels": [128, 128],
    "top_blocks_n_blocks_per_resolution": [2, 2],
    "top_blocks_has_resampling": [True, True],
    "top_blocks_dropout": [0.0, 0.0],
    "mid_blocks_channels": [256, 512],
    "mid_blocks_n_blocks_per_resolution": [4, 4],
    "mid_blocks_has_resampling": [True, False],
    "mid_blocks_dropout": [0.0, 0.3]
}

with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

# Save cleaned state_dict as model.pt
torch.save(clean_state_dict, os.path.join(output_dir, "model.pt"))

print(f"\n✓ Saved config.json")
print(f"✓ Saved model.pt with {len(clean_state_dict)} keys")
print(f"✓ Model ready at: {output_dir}")
