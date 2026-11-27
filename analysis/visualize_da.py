import torch
import matplotlib.pyplot as plt
import numpy as np
import medmnist
from medmnist import INFO
import albumentations as A
import cv2

# ==========================================
# 1. BEST INDIVIDUAL - OrganCMNIST (Offline ResNet-18)
# ==========================================
# ID 40: SaltAndPepper    (p=0.66) [0.23, 0.00, 0.04, 0.72]
# ID 4:  Affine Translate (p=0.51) [0.70, 0.25, 0.03, 0.33]
# ID 43: ThinPlateSpline  (p=0.25) [0.16, 0.62, 0.72, 0.00]

BEST_INDIVIDUAL = [
    [40, [0.66, 0.23, 0.00, 0.04, 0.72]],
    [4,  [0.51, 0.70, 0.25, 0.03, 0.33]],
    [43, [0.25, 0.16, 0.62, 0.72, 0.00]]
]

interpolation = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
border_type = [cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101]

def get_da_functions(img_size, min_prob=0, max_prob=1):
    da_funcs = [None]*45 
    
    # ID 4: Affine
    da_funcs[4] = lambda p0,p1,p2,p3,p4: A.Affine(translate_percent=sorted((p1*2-1, p2*2-1)), interpolation=interpolation[int(p3*len(interpolation)*0.99)], border_mode=border_type[int(p4*len(border_type)*0.99)], p=p0*(max_prob-min_prob)+min_prob)
    # ID 40: SaltAndPepper
    da_funcs[40] = lambda p0,p1,p2,p3,p4: A.SaltAndPepper(amount=sorted((p1,p2)), salt_vs_pepper=(p3,1-p3), p=p0*(max_prob-min_prob)+min_prob)
    # ID 43: ThinPlateSpline
    da_funcs[43] = lambda p0,p1,p2,p3,p4: A.ThinPlateSpline(scale_range=sorted((p1,p2)), num_control_points=int(p3*6)+2, interpolation=interpolation[int(p4*len(interpolation)*0.99)], p=p0*(max_prob-min_prob)+min_prob)
    
    return da_funcs

# Resolução original do OrganCMNIST
IMG_HEIGHT, IMG_WIDTH = 28, 28
available_funcs = get_da_functions(img_size=(IMG_HEIGHT, IMG_WIDTH))

pipeline_steps = []
for gene in BEST_INDIVIDUAL:
    func_idx, params = gene[0], gene[1]    
    forced_params = [1.0] + params[1:] # Forçar probabilidade a 1.0 (100%) para ver o fenótipo real
    pipeline_steps.append(available_funcs[func_idx](*forced_params))

evolved_pipeline = A.Compose(pipeline_steps)

# ==========================================
# 2. LOAD ORGANCMNIST DATASET (28x28)
# ==========================================
info = INFO['organcmnist']
DataClass = getattr(medmnist, info['python_class'])
dataset = DataClass(split='train', download=True, size=28)

# O OrganCMNIST tem 11 classes (ex: 3 = Coração, 0 = Bexiga, 6 = Fígado). Vamos pegar no Coração.
target_img = None
for i in range(len(dataset)):
    img, label = dataset[i]
    if label.item() == 3: # Selecionar um órgão reconhecível
        # Converter grayscale para RGB para o Albumentations processar
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            target_img = np.stack((img_np,)*3, axis=-1)
        else:
            target_img = img_np
        break

# Aplicar o pipeline
transformed = evolved_pipeline(image=target_img)['image']

# ==========================================
# 3. SAVE SEPARATE PDFs (NO AXES, NO TITLES)
# ==========================================

# Imagem Original (28x28, com nearest para píxeis nítidos)
fig1, ax1 = plt.subplots(figsize=(4, 4))
ax1.imshow(target_img, interpolation='nearest') 
ax1.axis('off')
plt.savefig('organ_original_28x28.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
plt.close(fig1)

# Imagem Transformada
fig2, ax2 = plt.subplots(figsize=(4, 4))
ax2.imshow(transformed, interpolation='nearest')
ax2.axis('off')
plt.savefig('organ_phenotype_28x28.pdf', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)
plt.close(fig2)

print("Imagens guardadas com sucesso como 'organ_original_28x28.pdf' e 'organ_phenotype_28x28.pdf'.")