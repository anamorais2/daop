import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import ast


DA_NAMES = [
    "Pad & RandomCrop", "HorizontalFlip", "VerticalFlip", "Rotate",
    "Affine (Translate)", "Affine (Shear)", "Perspective", "ElasticTransform",
    "ChannelShuffle", "ToGray", "GaussianBlur", "GaussNoise", "InvertImg",
    "Posterize", "Solarize", "Sharpen (Kernel)", "Sharpen (Gaussian)",
    "Equalize", "ImageCompression", "RandomGamma", "MedianBlur", "MotionBlur",
    "CLAHE", "RandomBrightnessContrast", "PlasmaBrightnessContrast",
    "CoarseDropout", "Blur", "HueSaturationValue", "ColorJitter",
    "RandomResizedCrop", "AutoContrast", "Erasing", "RGBShift",
    "PlanckianJitter", "ChannelDropout", "Illumination (Linear)",
    "Illumination (Corner)", "Illumination (Gaussian)", "PlasmaShadow",
    "RandomRain", "SaltAndPepper", "RandomSnow", "OpticalDistortion",
    "ThinPlateSpline"
]

print(f"Loaded {len(DA_NAMES)} augmentation names (hardcoded).")


def parse_genotype(genotype_str):
    """
    Try to parse and normalize a genotype string into a flat list of augmentations.
    Handles formats like: [[], [[id, params], ...]] or [[id, params], ...]
    Returns a list of augmentation entries or an empty list on failure.
    """
    try:
        data = ast.literal_eval(genotype_str)

        # Case 1: Full format [[], [[id, params], ...]] or [pretext, downstream]
        if isinstance(data, list) and len(data) == 2 and isinstance(data[0], list) and isinstance(data[1], list):
            # If the first list is empty (pure SL), the actions are in the second list
            if len(data[0]) == 0:
                return data[1]
            # Otherwise return the full structure (fallback)
            return data

        # Case 2: Just a list of augmentations -> [[id, params], ...]
        if isinstance(data, list):
            return data

        return []
    except Exception:
        # If parsing fails (e.g. malformed string), return empty
        return []


def analyze_da_distribution(folder_path, experiment_name="DA Distribution"):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not all_files:
        print(f"ERROR: No CSV files found in: {folder_path}")
        return

    print(f"Analyzing distribution across {len(all_files)} seeds...")

    da_counts = {i: 0 for i in range(len(DA_NAMES))}
    total_augmentations_count = 0
    total_seeds = 0

    for filename in all_files:
        try:
            # Read CSV
            df = pd.read_csv(filename, sep=';', on_bad_lines='skip')

            # Ensure required columns exist
            if 'best_fitness' not in df.columns or 'best_individual' not in df.columns:
                continue

            # Convert fitness to numeric and drop NaNs
            df['best_fitness'] = pd.to_numeric(df['best_fitness'], errors='coerce')
            df = df.dropna(subset=['best_fitness'])

            if df.empty:
                continue

            # Take the row with BEST FITNESS (the seed winner)
            best_row = df.loc[df['best_fitness'].idxmax()]
            genotype_str = best_row['best_individual']

            # Parse genotype
            augs_list = parse_genotype(genotype_str)

            # Count augmentations
            if augs_list:
                total_seeds += 1
                for item in augs_list:
                    # item expected: [ID, [params...]]
                    if isinstance(item, list) and len(item) > 0:
                        aug_id = item[0]
                        if isinstance(aug_id, int) and 0 <= aug_id < len(DA_NAMES):
                            da_counts[aug_id] += 1
                            total_augmentations_count += 1

        except Exception as e:
            print(f"Warning reading {os.path.basename(filename)}: {e}")

    # Filter only augmentations that appeared at least once
    active_augs = {DA_NAMES[k]: v for k, v in da_counts.items() if v > 0}

    if not active_augs:
        print("No augmentations found in the CSV files.")
        return

    # Sort by frequency
    sorted_augs = sorted(active_augs.items(), key=lambda item: item[1], reverse=True)
    labels = [k for k, v in sorted_augs]
    values = [v for k, v in sorted_augs]

    # --- PLOT ---
    plt.figure(figsize=(12, 8))

    bars = plt.bar(labels, values, color='#4c72b0', edgecolor='black', alpha=0.9)

    plt.xlabel('Data Augmentation Strategy', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Count)', fontsize=12, fontweight='bold')
    plt.title(f'Most Selected Augmentations - {experiment_name}\n(Across {total_seeds} seeds)', fontsize=14)

    # Rotate X labels to fit
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    max_y = max(values)
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_augmentations_count) * 100
        
        label_text = f"{percentage:.1f}%"
        
        plt.text(bar.get_x() + bar.get_width()/2, height + (max_y * 0.01), 
                 label_text, ha='center', va='bottom', fontsize=10, fontweight='bold', color='black')

    plt.ylim(0, max_y * 1.15) 
    plt.tight_layout()

    output_file = os.path.join(folder_path, f'DA_Distribution_{experiment_name}.png')
    plt.savefig(output_file, dpi=300)
    print(f"Chart saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        folder = sys.argv[1]
        name = sys.argv[2] if len(sys.argv) > 2 else "MedMNIST"
        analyze_da_distribution(folder, name)
    else:
        print("Usage: python da_distribution_medmnist.py <csv_folder> [Experiment_Name]")
