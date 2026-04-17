import ast


raw_policy = "[[], [[37, [0.81, 0.47, 0.62, 0.85, 0.57]], [4, [0.87, 0.42, 0.79, 0.58, 0.36]], [9, [0.97, 0.71, 0.9, 0.71, 0.18]], [9, [0, 0.89, 0.22, 0.23, 0.06]], [16, [0.12, 0.64, 0.68, 0.69, 0.44]]]]"

OP_NAMES = [
    "Pad & Crop", "HorizontalFlip", "VerticalFlip", "Rotate", "Affine (Translate)", 
    "Affine (Shear)", "Perspective", "ElasticTransform", "ChannelShuffle", "ToGray", 
    "GaussianBlur", "GaussNoise", "InvertImg", "Posterize", "Solarize", 
    "Sharpen (Kernel)", "Sharpen (Gaussian)", "Equalize", "ImageCompression", 
    "RandomGamma", "MedianBlur", "MotionBlur", "CLAHE", "RandomBrightnessContrast", 
    "PlasmaBrightnessContrast", "CoarseDropout", "Blur", "HueSaturationValue", 
    "ColorJitter", "RandomResizedCrop", "AutoContrast", "Erasing", "RGBShift", 
    "PlanckianJitter", "ChannelDropout", "Illumination (Linear)", "Illumination (Corner)", 
    "Illumination (Gaussian)", "PlasmaShadow", "RandomRain", "SaltAndPepper", 
    "RandomSnow", "OpticalDistortion", "ThinPlateSpline"
]

def translate_policy(policy_str):
    

    #Expected policy format: [<something>, [[op_idx, [prob, mag1, mag2, ...]], ...]]
    try:
        data = ast.literal_eval(policy_str)

        # Validate basic structure
        if not (isinstance(data, list) and len(data) == 2):
            print("Invalid format.")
            return

        operations_list = data[1]

        print("\n🔍 **POLICY TRANSLATION:**")
        print("=" * 50)

        for i, op in enumerate(operations_list):
            idx = op[0]          # operation index (e.g., 28)
            params = op[1]       # list of parameters
            prob = params[0]     # first is probability
            mags = params[1:]    # rest are magnitudes

            # lookup name
            if 0 <= idx < len(OP_NAMES):
                name = OP_NAMES[idx]
            else:
                name = f"Unknown ({idx})"

            # pretty print
            print(f"🔹 Operation {i+1}: {name}")
            print(f"   ↳ Probability: {prob:.1%} (p={prob})")
            print(f"   ↳ Parameters:  {mags}")
            print("-" * 50)

    except Exception as e:
        print(f"Error reading policy: {e}")


if __name__ == "__main__":
    translate_policy(raw_policy)