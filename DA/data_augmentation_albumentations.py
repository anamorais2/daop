import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2
import cv2
import albumentations as A

interpolation = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
border_type = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101]
grayscale_methods = ['weighted_average', 'from_lab', 'desaturation', 'average', 'max', 'pca']
illumination_effects = ['brighten', 'darken', 'both']

da_funcs = [
    lambda p0,p1,p2,p3,p4: A.Compose([A.Pad((int(p1*20), int(p2*20), int(p3*20), int(p4*20)), p=1.0), A.RandomCrop(height=32, width=32, p=1.0)], p=p0),
    lambda p0,p1,p2,p3,p4: A.HorizontalFlip(p=p0),
    lambda p0,p1,p2,p3,p4: A.VerticalFlip(p=p0),
    lambda p0,p1,p2,p3,p4: A.Rotate(limit=sorted((p1*180-90, p2*180-90)), interpolation=interpolation[int(p3*len(interpolation)*0.99)], border_mode=border_type[int(p4*len(border_type)*0.99)], p=p0),
    lambda p0,p1,p2,p3,p4: A.Affine(translate_percent=sorted((p1*2-1, p2*2-1)), interpolation=interpolation[int(p3*len(interpolation)*0.99)], border_mode=border_type[int(p4*len(border_type)*0.99)], p=p0),
    lambda p0,p1,p2,p3,p4: A.Affine(shear=sorted((p1*360-180, p2*360-180)), interpolation=interpolation[int(p3*len(interpolation)*0.99)], border_mode=border_type[int(p4*len(border_type)*0.99)], p=p0),
    lambda p0,p1,p2,p3,p4: A.Perspective(scale=sorted((p1*2,p2*2)), interpolation=interpolation[int(p3*len(interpolation)*0.99)], p=p0),
    lambda p0,p1,p2,p3,p4: A.ElasticTransform(alpha=p1*1000+1, sigma=p2*100+1, interpolation=interpolation[int(p3*len(interpolation)*0.99)], p=p0),
    lambda p0,p1,p2,p3,p4: A.ChannelShuffle(p=p0),
    lambda p0,p1,p2,p3,p4: A.ToGray(num_output_channels=3, method=grayscale_methods[int(p1*len(grayscale_methods)*0.99)], p=p0),
    lambda p0,p1,p2,p3,p4: A.GaussianBlur(blur_limit=sorted((int(p1*20), int(p2*20))), sigma_limit=sorted((p3*10+0.1, p4*10+0.1)), p=p0),
    lambda p0,p1,p2,p3,p4: A.GaussNoise(mean_range=sorted((p1*2-1, p2*2-1)), std_range=sorted((p3, p4)), p=p0),
    lambda p0,p1,p2,p3,p4: A.InvertImg(p=p0),
    lambda p0,p1,p2,p3,p4: A.Posterize(num_bits=sorted((int(p1*6.99)+1, int(p2*6.99)+1)), p=p0),
    lambda p0,p1,p2,p3,p4: A.Solarize(threshold_range=sorted((p1, p2)), p=p0),
    lambda p0,p1,p2,p3,p4: A.Sharpen(alpha=sorted((p1,p2)), lightness=sorted((p3,p4)), method='kernel', p=p0),
    lambda p0,p1,p2,p3,p4: A.Sharpen(alpha=sorted((p1,p2)), lightness=sorted((p3,p4)), method='gaussian', p=p0),
    lambda p0,p1,p2,p3,p4: A.Equalize(mode='cv' if p1 < 0.5 else 'pil', by_channels=p2 < 0.5, p=p0),
    lambda p0,p1,p2,p3,p4: A.ImageCompression(compression_type='jpeg' if p1 < 0.5 else 'webp', quality_range=sorted((int(p2*99.9)+1, int(p3*99.9)+1)), p=p0),
    lambda p0,p1,p2,p3,p4: A.RandomGamma(gamma_limit=sorted((p1*1000+1, p2*1000+1)), p=p0),
    lambda p0,p1,p2,p3,p4: A.MedianBlur(blur_limit=sorted((int(p1*100)+3, int(p2*100)+3)), p=p0),
    lambda p0,p1,p2,p3,p4: A.MotionBlur(blur_limit=sorted((int(p1*100)+3, int(p2*100)+3)), allow_shifted=p3 < 0.5, p=p0),
    lambda p0,p1,p2,p3,p4: A.CLAHE(clip_limit=sorted((int(p1*100)+1, int(p2*100)+1)), tile_grid_size=(int(p3*20)+2, int(p4*20)+2), p=p0),
    lambda p0,p1,p2,p3,p4: A.RandomBrightnessContrast(brightness_limit=sorted((p1*2-1, p2*2-1)), contrast_limit=sorted((p3*2-1, p4*2-1)), p=p0),
    lambda p0,p1,p2,p3,p4: A.PlasmaBrightnessContrast(brightness_range=sorted((p1*2-1, p2*2-1)), contrast_range=sorted((p3*2-1, p4*2-1)), p=p0),
    lambda p0,p1,p2,p3,p4: A.CoarseDropout(num_holes_range=sorted((int(p1*10)+1, int(p2*10)+1)), hole_height_range=sorted((p3, p4)), hole_width_range=sorted((p3, p4)), p=p0),
    lambda p0,p1,p2,p3,p4: A.Blur(blur_limit=sorted((int(p1*100)+3, int(p2*100)+3)), p=p0),
    lambda p0,p1,p2,p3,p4: A.HueSaturationValue(hue_shift_limit=sorted((p1*200-100, p2*200-100)), sat_shift_limit=sorted((p3*200-100, p4*200-100)), p=p0),
    lambda p0,p1,p2,p3,p4: A.ColorJitter(brightness=p1, contrast=p2, saturation=p3, hue=p4*0.5, p=p0),
    lambda p0,p1,p2,p3,p4: A.RandomResizedCrop((32, 32), scale=sorted((p1*0.99+0.01, p2*0.99+0.01)), ratio=sorted((p3+0.5, p4+0.5)), p=p0),
    lambda p0,p1,p2,p3,p4: A.AutoContrast(cutoff=p1*100, method='cdf' if p2 < 0.5 else 'pil', p=p0),
    lambda p0,p1,p2,p3,p4: A.Erasing(scale=sorted((p1*0.3+0.01, p2*0.3+0.01)), ratio=sorted((p3*3+0.1, p4*3+0.1)), p=p0),
    lambda p0,p1,p2,p3,p4: A.RGBShift(r_shift_limit=p1*200, g_shift_limit=p2*200, b_shift_limit=p3*200, p=p0),
    lambda p0,p1,p2,p3,p4: A.PlanckianJitter(mode='blackbody' if p1 < 0.5 else 'cied', sampling_method='gaussian' if p2 < 0.5 else 'uniform', p=p0),
    lambda p0,p1,p2,p3,p4: A.ChannelDropout(channel_drop_range=sorted((int(p1*1.99)+1, int(p2*1.99)+1)), fill=p3*255, p=p0),
    lambda p0,p1,p2,p3,p4: A.Illumination(mode='linear', intensity_range=sorted((p1*0.19+0.01, p2*0.19+0.01)), angle_range=sorted((p3*360, p4*360)), p=p0),
    lambda p0,p1,p2,p3,p4: A.Illumination(mode='corner', intensity_range=sorted((p1*0.19+0.01, p2*0.19+0.01)), effect_type=illumination_effects[int(p3*len(illumination_effects)*0.99)], p=p0),
    lambda p0,p1,p2,p3,p4: A.Illumination(mode='gaussian', intensity_range=sorted((p1*0.19+0.01, p2*0.19+0.01)), center_range=sorted((p3, p4)), p=p0),
    lambda p0,p1,p2,p3,p4: A.PlasmaShadow(shadow_intensity_range=sorted((p1, p2)), roughness=p3*5+0.1, p=p0),
    lambda p0,p1,p2,p3,p4: A.RandomRain(slant_range=sorted((p1*40-20, p2*40-20)), drop_length=int(p3*20)+1, drop_width=int(p4*20)+1, p=p0),
    lambda p0,p1,p2,p3,p4: A.SaltAndPepper(amount=sorted((p1,p2)), salt_vs_pepper=(p3,1-p3), p=p0),
    lambda p0,p1,p2,p3,p4: A.RandomSnow(snow_point_range=sorted((p1,p2)), brightness_coeff=p3*10+0.1, method='bleach' if p4 < 0.5 else 'texture', p=p0),
    lambda p0,p1,p2,p3,p4: A.OpticalDistortion(distort_limit=sorted((p1*2-1, p2*2-1)), mode='camera' if p3 < 0.5 else 'fisheye', interpolation=interpolation[int(p4*len(interpolation)*0.99)], p=p0),
    lambda p0,p1,p2,p3,p4: A.ThinPlateSpline(scale_range=sorted((p1,p2)), num_control_points=int(p3*6)+2, interpolation=interpolation[int(p4*len(interpolation)*0.99)], p=p0)
]


def da_funcs_probs(min_prob,max_prob, img_size):
    da_funcs = [
        lambda p0,p1,p2,p3,p4: A.Compose([A.Pad((int(p1*20), int(p2*20), int(p3*20), int(p4*20)), p=1.0), A.RandomCrop(height=img_size[0], width=img_size[1], p=1.0)], p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.HorizontalFlip(p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.VerticalFlip(p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Rotate(limit=sorted((p1*180-90, p2*180-90)), interpolation=interpolation[int(p3*len(interpolation)*0.99)], border_mode=border_type[int(p4*len(border_type)*0.99)], p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Affine(translate_percent=sorted((p1*2-1, p2*2-1)), interpolation=interpolation[int(p3*len(interpolation)*0.99)], border_mode=border_type[int(p4*len(border_type)*0.99)], p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Affine(shear=sorted((p1*360-180, p2*360-180)), interpolation=interpolation[int(p3*len(interpolation)*0.99)], border_mode=border_type[int(p4*len(border_type)*0.99)], p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Perspective(scale=sorted((p1*2,p2*2)), interpolation=interpolation[int(p3*len(interpolation)*0.99)], p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.ElasticTransform(alpha=p1*1000+1, sigma=p2*100+1, interpolation=interpolation[int(p3*len(interpolation)*0.99)], p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.ChannelShuffle(p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.ToGray(num_output_channels=3, method=grayscale_methods[int(p1*len(grayscale_methods)*0.99)], p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.GaussianBlur(blur_limit=sorted((int(p1*20), int(p2*20))), sigma_limit=sorted((p3*10+0.1, p4*10+0.1)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.GaussNoise(mean_range=sorted((p1*2-1, p2*2-1)), std_range=sorted((p3, p4)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.InvertImg(p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Posterize(num_bits=sorted((int(p1*6.99)+1, int(p2*6.99)+1)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Solarize(threshold_range=sorted((p1, p2)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Sharpen(alpha=sorted((p1,p2)), lightness=sorted((p3,p4)), method='kernel', p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Sharpen(alpha=sorted((p1,p2)), lightness=sorted((p3,p4)), method='gaussian', p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Equalize(mode='cv' if p1 < 0.5 else 'pil', by_channels=p2 < 0.5, p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.ImageCompression(compression_type='jpeg' if p1 < 0.5 else 'webp', quality_range=sorted((int(p2*99.9)+1, int(p3*99.9)+1)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.RandomGamma(gamma_limit=sorted((p1*1000+1, p2*1000+1)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.MedianBlur(blur_limit=sorted((int(p1*100)+3, int(p2*100)+3)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.MotionBlur(blur_limit=sorted((int(p1*100)+3, int(p2*100)+3)), allow_shifted=p3 < 0.5, p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.CLAHE(clip_limit=sorted((int(p1*100)+1, int(p2*100)+1)), tile_grid_size=(int(p3*20)+2, int(p4*20)+2), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.RandomBrightnessContrast(brightness_limit=sorted((p1*2-1, p2*2-1)), contrast_limit=sorted((p3*2-1, p4*2-1)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.PlasmaBrightnessContrast(brightness_range=sorted((p1*2-1, p2*2-1)), contrast_range=sorted((p3*2-1, p4*2-1)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.CoarseDropout(num_holes_range=sorted((int(p1*10)+1, int(p2*10)+1)), hole_height_range=sorted((p3, p4)), hole_width_range=sorted((p3, p4)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Blur(blur_limit=sorted((int(p1*100)+3, int(p2*100)+3)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.HueSaturationValue(hue_shift_limit=sorted((p1*200-100, p2*200-100)), sat_shift_limit=sorted((p3*200-100, p4*200-100)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.ColorJitter(brightness=p1, contrast=p2, saturation=p3, hue=p4*0.5, p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.RandomResizedCrop((img_size[0], img_size[1]), scale=sorted((p1*0.99+0.01, p2*0.99+0.01)), ratio=sorted((p3+0.5, p4+0.5)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.AutoContrast(cutoff=p1*100, method='cdf' if p2 < 0.5 else 'pil', p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Erasing(p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.RGBShift(r_shift_limit=p1*200, g_shift_limit=p2*200, b_shift_limit=p3*200, p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.PlanckianJitter(mode='blackbody' if p1 < 0.5 else 'cied', sampling_method='gaussian' if p2 < 0.5 else 'uniform', p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.ChannelDropout(channel_drop_range=sorted((int(p1*1.99)+1, int(p2*1.99)+1)), fill=p3*255, p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Illumination(mode='linear', intensity_range=sorted((p1*0.19+0.01, p2*0.19+0.01)), angle_range=sorted((p3*360, p4*360)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Illumination(mode='corner', intensity_range=sorted((p1*0.19+0.01, p2*0.19+0.01)), effect_type=illumination_effects[int(p3*len(illumination_effects)*0.99)], p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.Illumination(mode='gaussian', intensity_range=sorted((p1*0.19+0.01, p2*0.19+0.01)), center_range=sorted((p3, p4)), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.PlasmaShadow(shadow_intensity_range=sorted((p1, p2)), roughness=p3*5+0.1, p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.RandomRain(slant_range=sorted((p1*40-20, p2*40-20)), drop_length=int(p3*20)+1, drop_width=int(p4*20)+1, p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.SaltAndPepper(amount=sorted((p1,p2)), salt_vs_pepper=(p3,1-p3), p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.RandomSnow(snow_point_range=sorted((p1,p2)), brightness_coeff=p3*10+0.1, method='bleach' if p4 < 0.5 else 'texture', p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.OpticalDistortion(distort_limit=sorted((p1*2-1, p2*2-1)), mode='camera' if p3 < 0.5 else 'fisheye', interpolation=interpolation[int(p4*len(interpolation)*0.99)], p=p0*(max_prob-min_prob)+min_prob),
        lambda p0,p1,p2,p3,p4: A.ThinPlateSpline(scale_range=sorted((p1,p2)), num_control_points=int(p3*6)+2, interpolation=interpolation[int(p4*len(interpolation)*0.99)], p=p0*(max_prob-min_prob)+min_prob)
    ]

    return da_funcs


def map_augments(invididual, config):
    data_augmentations = []

    for aug in invididual:
        if aug[0] < len(config['da_funcs']):
            data_augmentations.append(config['da_funcs'][aug[0]](*aug[1]))

    return data_augmentations