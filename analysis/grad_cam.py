import sys
import configs.config_visualize_da as config_visualize_da
import torch
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, GradCAMElementWise, GradCAMPlusPlus, XGradCAM, AblationCAM, ScoreCAM, EigenCAM, EigenGradCAM, LayerCAM 
from pytorch_grad_cam import FullGrad, DeepFeatureFactorization, KPCA_CAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget, FinerWeightedTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from net_models_torch import TrainResNet18Grad
import data_processing_rotnet_torch
from analysis.utils import np_entropy, np_neg_entropy, np_spatial_var, np_neg_spatial_var, max_abs_val, min_abs_val

config = config_visualize_da.config

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
ROTNET_CLASSES = ['0', '90', '180', '270']

# individual = [[10, [0.03, 0.58, 0.81, 0.87, 0.11]]]
# individual = [[43, [0.99, 0.55, 0.63, 0, 0.3]], [1, [0.55, 0.73, 0.62, 0.13, 0.14]]]
individual = []
num_batches = 1
skip_batches = 0
samples_per_batch = 128

FONT_SIZE = 24


def rescale_0_1(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def grad_cam_image(no_norm_images, norm_images, max_samples, output_prefix=None):
    model_paths = ["na_100_downstream_0.pt", "hf_100_downstream_0.pt", "tp_100_downstream_0.pt", "rn_100_downstream_0.pt"]
    # model_paths = ["na_100_pretext_0.pt", "hf_100_pretext_0.pt", "tp_100_pretext_0.pt", "rn_100_pretext_0.pt"]

    for mpi, model_path in enumerate(model_paths):
        model_instance = TrainResNet18Grad()
        model_instance.switch_to_downstream()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model
        model.eval()

        # target_layers_list = [model.layer1, model.layer1[-1], model.layer2, model.layer2[-1], model.layer3, model.layer3[-1], model.layer4, model.layer4[-1]]

        target_layers_list = [model.layer1, model.layer2]
        # target_layers_list = [model.layer1, model.layer2, model.layer3, model.layer4]
        target_layers = [[tl] for tl in target_layers_list]
        

        # target_layers = [model.layer2]

        norm_images = norm_images[:max_samples].to(config['device'])

        print(no_norm_images.shape)

        for tl_i, tl in enumerate(target_layers):
            with GradCAM(model=model, target_layers=tl) as cam:
                grayscale_cam = cam(input_tensor=norm_images, targets=None, aug_smooth=True, eigen_smooth=True)

                for i in range(min(max_samples, len(no_norm_images))):
                    cam_i = grayscale_cam[i, :]

                    # print(cam_i.shape)

                    cam_image = show_cam_on_image(rescale_0_1(no_norm_images[i]).numpy().transpose(1, 2, 0), cam_i, use_rgb=True)
                    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

                    output_image_name = f"model{mpi}_tl{tl_i}_{i}" if output_prefix is None else f"{output_prefix}_model{mpi}_tl{tl_i}_{i}"

                    if not os.path.exists("output_grad_cam"):
                        os.makedirs("output_grad_cam")

                    cv2.imwrite(os.path.join("output_grad_cam", output_image_name+".png"), cam_image)
                    if mpi == 0 and tl_i == 0:
                        cv2.imwrite(os.path.join("output_grad_cam", "no_cam_"+output_image_name+".png"), no_norm_images[i].numpy().transpose(1, 2, 0))

def grad_cam_diff(no_norm_images, norm_images, labels, max_samples, output_prefix=None):
    model_paths = ["rn_100_downstream_5.pt", "my_downstream_11.pt"]
    display_names = ["RotNet Augments", "Evolved Augments"]
    
    grayscale_cam = []
    predicted_labels = []

    for mpi, model_path in enumerate(model_paths):
        model_instance = TrainResNet18Grad()
        model_instance.switch_to_downstream()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()

        norm_images = norm_images[:max_samples].to(config['device'])

        labels = labels[:max_samples].to(config['device'])

        with torch.no_grad():
            outputs = model(norm_images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.append(predicted)

        print(no_norm_images.shape)

        with GradCAM(model=model, target_layers=[model.layer1]) as cam:
            grayscale_cam.append(cam(input_tensor=norm_images, targets=[ClassifierOutputTarget(predicted_labels[mpi][i]) for i in range(len(norm_images))]))

    for i in range(min(max_samples, len(no_norm_images))):
        cam_i_0 = grayscale_cam[0][i, :]
        cam_i_1 = grayscale_cam[1][i, :]

        pred_i_0 = predicted_labels[0][i]
        pred_i_1 = predicted_labels[1][i]

        print(cam_i_0.shape, cam_i_1.shape)
        print(cam_i_0.max(), cam_i_0.min(), cam_i_1.max(), cam_i_1.min())

        cam_image_0 = show_cam_on_image(rescale_0_1(no_norm_images[i]).numpy().transpose(1, 2, 0), cam_i_0, use_rgb=True)
        # cam_image_0 = cv2.cvtColor(cam_image_0, cv2.COLOR_RGB2BGR)

        cam_image_1 = show_cam_on_image(rescale_0_1(no_norm_images[i]).numpy().transpose(1, 2, 0), cam_i_1, use_rgb=True)
        # cam_image_1 = cv2.cvtColor(cam_image_1, cv2.COLOR_RGB2BGR)

        cam_i_diff = cam_i_1 - cam_i_0
        cam_i_diff = rescale_0_1(torch.tensor(cam_i_diff)).numpy()
        print(cam_i_diff.max(), cam_i_diff.min())

        # print(cam_i.shape)

        cam_image_diff = show_cam_on_image(rescale_0_1(no_norm_images[i]).numpy().transpose(1, 2, 0), cam_i_diff, use_rgb=True)
        # cam_image_diff = cv2.cvtColor(cam_image_diff, cv2.COLOR_RGB2BGR)

        # show the 3 cam_image and original image (no_norm) in the same plot
        figure, axes = plt.subplots(2, 2)
        figure.subplots_adjust(hspace=0.5)
        axes[0,0].imshow(no_norm_images[i].numpy().transpose(1, 2, 0))
        axes[0,0].set_title(f"Original\nLabel: {CIFAR_CLASSES[labels[i]]}")
        axes[0,1].imshow(cam_image_diff)
        axes[0,1].set_title("Diff")
        axes[1,0].imshow(cam_image_0)
        axes[1,0].set_title(f"{display_names[0]}\nPredicted Label: {CIFAR_CLASSES[pred_i_0]}")
        axes[1,1].imshow(cam_image_1)
        axes[1,1].set_title(f"{display_names[1]}\nPredicted Label: {CIFAR_CLASSES[pred_i_1]}")
        plt.show()


def grad_cam_diff_pretext(no_norm_images, norm_images, labels, max_samples, output_prefix=None):
    model_paths = ["rn_100_pretext_5.pt", "my_pretext_11.pt"]
    display_names = ["RotNet Augments", "Evolved Augments"]
    
    grayscale_cam = []
    predicted_labels = []

    for mpi, model_path in enumerate(model_paths):
        model_instance = TrainResNet18Grad()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()

        norm_images = norm_images[:max_samples].to(config['device'])
        labels = labels[:max_samples].to(config['device'])

        with torch.no_grad():
            outputs = model(norm_images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.append(predicted)

        print(no_norm_images.shape)

        with GradCAM(model=model, target_layers=[model.layer3]) as cam:
            grayscale_cam.append(cam(input_tensor=norm_images, targets=[ClassifierOutputTarget(predicted_labels[mpi][i]) for i in range(len(norm_images))]))

    for i in range(min(max_samples, len(no_norm_images))):
        cam_i_0 = grayscale_cam[0][i, :]
        cam_i_1 = grayscale_cam[1][i, :]

        pred_i_0 = predicted_labels[0][i]
        pred_i_1 = predicted_labels[1][i]

        print(cam_i_0.shape, cam_i_1.shape)
        print(cam_i_0.max(), cam_i_0.min(), cam_i_1.max(), cam_i_1.min())

        cam_image_0 = show_cam_on_image(rescale_0_1(no_norm_images[i]).numpy().transpose(1, 2, 0), cam_i_0, use_rgb=True)
        # cam_image_0 = cv2.cvtColor(cam_image_0, cv2.COLOR_RGB2BGR)

        cam_image_1 = show_cam_on_image(rescale_0_1(no_norm_images[i]).numpy().transpose(1, 2, 0), cam_i_1, use_rgb=True)
        # cam_image_1 = cv2.cvtColor(cam_image_1, cv2.COLOR_RGB2BGR)

        cam_i_diff = cam_i_1 - cam_i_0
        cam_i_diff = rescale_0_1(torch.tensor(cam_i_diff)).numpy()
        print(cam_i_diff.max(), cam_i_diff.min())

        # print(cam_i.shape)

        cam_image_diff = show_cam_on_image(rescale_0_1(no_norm_images[i]).numpy().transpose(1, 2, 0), cam_i_diff, use_rgb=True)
        # cam_image_diff = cv2.cvtColor(cam_image_diff, cv2.COLOR_RGB2BGR)

        # show the 3 cam_image and original image (no_norm) in the same plot
        figure, axes = plt.subplots(2, 2)
        figure.subplots_adjust(hspace=0.5)
        axes[0,0].imshow(no_norm_images[i].numpy().transpose(1, 2, 0))
        axes[0,0].set_title(f"Original\nLabel: {ROTNET_CLASSES[labels[i]]}")
        axes[0,1].imshow(cam_image_diff)
        axes[0,1].set_title("Diff")
        axes[1,0].imshow(cam_image_0)
        axes[1,0].set_title(f"{display_names[0]}\nPredicted Label: {ROTNET_CLASSES[pred_i_0]}")
        axes[1,1].imshow(cam_image_1)
        axes[1,1].set_title(f"{display_names[1]}\nPredicted Label: {ROTNET_CLASSES[pred_i_1]}")
        plt.show()


def grad_cam_average_batch(no_norm_images, norm_images, labels, max_samples, output_prefix=None):
    model_path = "na_100_downstream_0.pt"
    display_name = "No Augments"
    
    avg_cams = []
    for target_class in range(10):
        grayscale_cam = None
        predicted_labels = []

        model_instance = TrainResNet18Grad()
        model_instance.switch_to_downstream()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()

        # get indexes of images with target class
        target_class_idx = torch.where(labels == target_class)[0]
        print(f"Number of images with target class {CIFAR_CLASSES[target_class]}: {len(target_class_idx)}")
        # print(torch.where(labels == target_class))

        norm_images_target = norm_images[target_class_idx].to(config['device'])
        # norm_images_target = norm_images[:max_samples].to(config['device'])

        labels_target = labels[target_class_idx].to(config['device'])
        # labels_target = labels[:max_samples].to(config['device'])

        with torch.no_grad():
            # for i in range(norm_images_target.shape[0]):
            #     plt.imshow(norm_images_target[i].cpu().numpy().transpose(1, 2, 0))
            #     plt.show()
            outputs = model(norm_images_target)
            _, predicted_labels = torch.max(outputs.data, 1)
            # print(predicted_labels)
            # print(outputs)

        correct_guesses = torch.where(predicted_labels == labels_target)[0].cpu()
        print(f"Number of correct guesses: {len(correct_guesses)}")

        with KPCA_CAM(model=model, target_layers=[model.layer2]) as cam:
            grayscale_cam = cam(input_tensor=norm_images_target[correct_guesses], targets=[ClassifierOutputTarget(target_class) for i in range(len(correct_guesses))], aug_smooth=True, eigen_smooth=True)
            # grayscale_cam = cam(input_tensor=norm_images_target[correct_guesses], targets=[ClassifierOutputTarget(target_class) for i in range(len(correct_guesses))])
 
        # no_norm_images_target = no_norm_images[target_class_idx]
        # no_norm_images_target = no_norm_images_target[correct_guesses]
        # print(no_norm_images_target.shape)
        # for i in range(min(max_samples, len(no_norm_images_target))):
        #     avg_cam = np.mean(grayscale_cam, axis=0)

        #     # cam_image= show_cam_on_image(rescale_0_1(no_norm_images_target[i]).numpy().transpose(1, 2, 0), avg_cam, use_rgb=True)

        #     figure, axes = plt.subplots(1, 2)
        #     figure.subplots_adjust(hspace=0.5)
        #     axes[0].imshow(no_norm_images_target[i].numpy().transpose(1, 2, 0))
        #     axes[0].set_title(f"Original\nLabel: {CIFAR_CLASSES[labels[i]]}")
        #     axes[1].imshow(rescale_0_1(torch.tensor(avg_cam)).numpy())
        #     axes[1].set_title(f"{display_name}\nPredicted Label: {CIFAR_CLASSES[predicted_labels[i]]}")
        #     plt.show()

        avg_cams.append(np.mean(grayscale_cam, axis=0))

    figure, axes = plt.subplots(2, 5)
    figure.suptitle(f"Average CAMs\n{display_name}", fontsize=FONT_SIZE)
    for i, avg_cam in enumerate(avg_cams):
        axes[i//5, i%5].imshow(rescale_0_1(torch.tensor(avg_cam)).numpy())
        axes[i//5, i%5].set_title(f"Label: {CIFAR_CLASSES[i]}", fontsize=FONT_SIZE)
        axes[i//5, i%5].axis('off')
    plt.show()


def grad_cam_average_multiple_comparison(norm_images, labels):
    model_paths = ["my_downstream_11.pt", "rn_100_downstream_11.pt", "na_100_downstream_0.pt"]
    display_names = ["Evolved Augments", "RotNet Augments", "No Augments"]

    models = []

    for model_path in model_paths:
        model_instance = TrainResNet18Grad()
        model_instance.switch_to_downstream()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()
        models.append(model)

    avg_cams = []
    for target_class in range(10):
        # get indexes of images with target class
        target_class_idx = torch.where(labels == target_class)[0]
        print(f"Number of images with target class {CIFAR_CLASSES[target_class]}: {len(target_class_idx)}")

        norm_images_target = norm_images[target_class_idx].to(config['device'])

        labels_target = labels[target_class_idx].to(config['device'])

        correct_guesses = []
        for model in models:
            predicted_labels = []
            with torch.no_grad():
                outputs = model(norm_images_target)
                _, predicted_labels = torch.max(outputs.data, 1)

            current_correct_guesses = torch.where(predicted_labels == labels_target)[0].cpu()
            print(f"Number of correct guesses: {len(current_correct_guesses)}")
            correct_guesses.append(current_correct_guesses)

        # print(correct_guesses)
        all_correct_guesses = torch.cat(correct_guesses)
        unique_correct_guesses, counts = torch.unique(all_correct_guesses, return_counts=True)
        shared_correct_guesses = unique_correct_guesses[counts == len(model_paths)]
        print(f"Number of shared correct guesses: {len(shared_correct_guesses)}")
        # print(shared_correct_guesses)

        grayscale_cams = []
        for model in models:
            with GradCAMPlusPlus(model=model, target_layers=[model.layer1]) as cam:
                grayscale_cams.append(cam(input_tensor=norm_images_target[shared_correct_guesses], targets=[ClassifierOutputTarget(target_class) for _ in range(len(shared_correct_guesses))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target[shared_correct_guesses], targets=[ClassifierOutputSoftmaxTarget(target_class) for _ in range(len(shared_correct_guesses))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target[shared_correct_guesses], targets=None))
                # grayscale_cams.append(cam(input_tensor=norm_images_target[shared_correct_guesses], targets=[FinerWeightedTarget(target_class, [i for i in range(10) if i != target_class], 0.1) for _ in range(len(shared_correct_guesses))]))

        avg_cams.append([np.mean(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])

    for model_i in range(len(model_paths)):
        figure, axes = plt.subplots(2, 5)
        figure.suptitle(f"Average CAMs\n{display_names[model_i]}", fontsize=FONT_SIZE)
        for i, avg_cam in enumerate(avg_cams):
            axes[i//5, i%5].imshow(rescale_0_1(torch.tensor(avg_cam[model_i])).numpy())
            axes[i//5, i%5].set_title(f"Label: {CIFAR_CLASSES[i]}", fontsize=FONT_SIZE)
            axes[i//5, i%5].axis('off')
        plt.show()


def grad_cam_average_disjunction(norm_images, labels):
    model_paths = ["my_downstream_11.pt", "rn_100_downstream_5.pt", "na_100_downstream_0.pt"]
    display_names = ["Evolved Augments", "RotNet Augments", "No Augments"]

    models = []

    for model_path in model_paths:
        model_instance = TrainResNet18Grad()
        model_instance.switch_to_downstream()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()
        models.append(model)

    avg_cams = []
    var_cams = []
    ent_cams = []
    neg_ent_cams = []
    spvar_cams = []
    neg_spvar_cams = []
    for target_class in range(10):
        # get indexes of images with target class
        target_class_idx = torch.where(labels == target_class)[0]
        print(f"Number of images with target class {CIFAR_CLASSES[target_class]}: {len(target_class_idx)}")

        norm_images_target = norm_images[target_class_idx].to(config['device'])

        labels_target = labels[target_class_idx].to(config['device'])

        correct_guesses = []
        for i, model in enumerate(models):
            predicted_labels = []
            with torch.no_grad():
                outputs = model(norm_images_target)
                _, predicted_labels = torch.max(outputs.data, 1)

            if i == 0:
                current_correct_guesses = torch.where(predicted_labels == labels_target)[0].cpu()
            else:
                current_correct_guesses = torch.where(predicted_labels != labels_target)[0].cpu()
            print(f"Number of correct guesses: {len(current_correct_guesses)}")
            correct_guesses.append(current_correct_guesses)

        # print(correct_guesses)
        all_correct_guesses = torch.cat(correct_guesses)
        unique_correct_guesses, counts = torch.unique(all_correct_guesses, return_counts=True)
        shared_correct_guesses = unique_correct_guesses[counts == len(model_paths)]
        print(f"Number of shared correct guesses: {len(shared_correct_guesses)}")
        # print(shared_correct_guesses)

        grayscale_cams = []
        for model in models:
            with GradCAMPlusPlus(model=model, target_layers=[model.layer1]) as cam:
                grayscale_cams.append(cam(input_tensor=norm_images_target[shared_correct_guesses], targets=[ClassifierOutputTarget(target_class) for _ in range(len(shared_correct_guesses))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target[shared_correct_guesses], targets=[ClassifierOutputSoftmaxTarget(target_class) for _ in range(len(shared_correct_guesses))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target[shared_correct_guesses], targets=None))
                # grayscale_cams.append(cam(input_tensor=norm_images_target[shared_correct_guesses], targets=[FinerWeightedTarget(target_class, [i for i in range(10) if i != target_class], 0.1) for _ in range(len(shared_correct_guesses))]))

        avg_cams.append([np.mean(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])
        var_cams.append([np.var(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])
        ent_cams.append([np_entropy(np.mean(grayscale_cams[i], axis=0)) for i in range(len(grayscale_cams))])
        neg_ent_cams.append([np_neg_entropy(np.mean(grayscale_cams[i], axis=0)) for i in range(len(grayscale_cams))])
        spvar_cams.append([np.array(np_spatial_var(np.mean(grayscale_cams[i], axis=0))) for i in range(len(grayscale_cams))])
        neg_spvar_cams.append([np.array(np_neg_spatial_var(np.mean(grayscale_cams[i], axis=0))) for i in range(len(grayscale_cams))])


    for model_i in range(1,len(model_paths)):
        figure, axes = plt.subplots(2, 5)
        figure.suptitle(f"Average CAM diff between {display_names[0]} and {display_names[model_i]}", fontsize=FONT_SIZE)
        for i, avg_cam in enumerate(avg_cams):
            print(f"Mean min_abs_val spatial variance diff sum for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {min_abs_val(np.sum(spvar_cams[i][0]), np.sum(neg_spvar_cams[i][0])) - min_abs_val(np.sum(spvar_cams[i][model_i]), np.sum(neg_spvar_cams[i][model_i]))}")
            diff_cam = rescale_0_1(torch.tensor(avg_cam[0] - avg_cam[model_i])).numpy()
            # print(f"Entropy of diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np_entropy(diff_cam)}")
            axes[i//5, i%5].imshow(diff_cam)
            axes[i//5, i%5].set_title(f"Label: {CIFAR_CLASSES[i]}", fontsize=FONT_SIZE)
            axes[i//5, i%5].axis('off')
        plt.show()


def grad_cam_wrong_guesses(norm_images, labels):
    model_paths = ["my_downstream_11.pt", "rn_100_downstream_11.pt", "na_100_downstream_0.pt"]
    display_names = ["Evolved Augments", "RotNet Augments", "No Augments"]

    models = []

    for model_path in model_paths:
        model_instance = TrainResNet18Grad()
        model_instance.switch_to_downstream()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()
        models.append(model)

    avg_cams = []
    for target_class in range(10):
        # get indexes of images with target class
        target_class_idx = torch.where(labels == target_class)[0]
        print(f"Number of images with target class {CIFAR_CLASSES[target_class]}: {len(target_class_idx)}")

        norm_images_target = norm_images[target_class_idx].to(config['device'])

        labels_target = labels[target_class_idx].to(config['device'])

        wrong_guesses = []
        for model in models:
            predicted_labels = []
            with torch.no_grad():
                outputs = model(norm_images_target)
                _, predicted_labels = torch.max(outputs.data, 1)

            current_wrong_guesses = torch.where(predicted_labels != labels_target)[0].cpu()
            print(f"Number of wrong guesses: {len(current_wrong_guesses)}")
            wrong_guesses.append(current_wrong_guesses)


        grayscale_cams = []
        for model_i, model in enumerate(models):
            with GradCAM(model=model, target_layers=[model.layer1]) as cam:
                grayscale_cams.append(cam(input_tensor=norm_images_target[wrong_guesses[model_i]], targets=[ClassifierOutputTarget(target_class) for _ in range(len(wrong_guesses[model_i]))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target[wrong_guesses[model_i]], targets=None))
                # grayscale_cams.append(cam(input_tensor=norm_images_target[wrong_guesses[model_i]], targets=[FinerWeightedTarget(target_class, [i for i in range(10) if i != target_class], 0.9) for _ in range(len(wrong_guesses[model_i]))]))

        avg_cams.append([np.mean(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])

    for model_i in range(len(model_paths)):
        figure, axes = plt.subplots(2, 5)
        figure.suptitle(f"Average CAMs\n{display_names[model_i]}", fontsize=FONT_SIZE)
        for i, avg_cam in enumerate(avg_cams):
            axes[i//5, i%5].imshow(rescale_0_1(torch.tensor(avg_cam[model_i])).numpy())
            axes[i//5, i%5].set_title(f"Label: {CIFAR_CLASSES[i]}", fontsize=FONT_SIZE)
            axes[i//5, i%5].axis('off')
        plt.show()


def grad_cam_average_comparison_all(norm_images, labels):
    model_paths = ["my_downstream_11.pt", "rn_100_downstream_5.pt", "na_100_downstream_0.pt"]
    display_names = ["Evolved Augments", "RotNet Augments", "No Augments"]

    models = []

    for model_path in model_paths:
        model_instance = TrainResNet18Grad()
        model_instance.switch_to_downstream()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()
        models.append(model)

    avg_cams = []
    var_cams = []
    for target_class in range(10):
        target_class_idx = torch.where(labels == target_class)[0]
        print(f"Number of images with target class {CIFAR_CLASSES[target_class]}: {len(target_class_idx)}")

        norm_images_target = norm_images[target_class_idx].to(config['device'])

        grayscale_cams = []
        for model in models:
            with GradCAM(model=model, target_layers=[model.layer1]) as cam:
                grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[ClassifierOutputTarget(target_class) for _ in range(len(norm_images_target))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[ClassifierOutputSoftmaxTarget(target_class) for _ in range(len(norm_images_target))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=None))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[FinerWeightedTarget(target_class, [i for i in range(10) if i != target_class], 0.9) for _ in range(len(norm_images_target))]))


        avg_cams.append([np.mean(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])
        var_cams.append([np.var(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])

    for model_i in range(len(model_paths)):
        figure, axes = plt.subplots(2, 5)
        figure.suptitle(f"Average CAMs\n{display_names[model_i]}", fontsize=FONT_SIZE)
        for i, avg_cam in enumerate(avg_cams):
            # print(f"Mean variance for model {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np.mean(var_cams[i][model_i])}")
            axes[i//5, i%5].imshow(rescale_0_1(torch.tensor(avg_cam[model_i])).numpy())
            axes[i//5, i%5].set_title(f"Label: {CIFAR_CLASSES[i]}", fontsize=FONT_SIZE)
            axes[i//5, i%5].axis('off')
        plt.show()


def grad_cam_average_comparison_all_pretext(norm_images, labels):
    model_paths = ["my_pretext_11.pt", "rn_100_pretext_5.pt", "na_100_pretext_0.pt"]
    display_names = ["Evolved Augments", "RotNet Augments", "No Augments"]

    models = []

    for model_path in model_paths:
        model_instance = TrainResNet18Grad()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()
        models.append(model)

    avg_cams = []
    var_cams = []
    for target_class in range(4):
        target_class_idx = torch.where(labels == target_class)[0]
        print(f"Number of images with target class {ROTNET_CLASSES[target_class]}: {len(target_class_idx)}")

        norm_images_target = norm_images[target_class_idx].to(config['device'])

        grayscale_cams = []
        for model in models:
            with GradCAM(model=model, target_layers=[model.layer3]) as cam:
                grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[ClassifierOutputTarget(target_class) for _ in range(len(norm_images_target))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[ClassifierOutputSoftmaxTarget(target_class) for _ in range(len(norm_images_target))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=None))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[FinerWeightedTarget(target_class, [i for i in range(10) if i != target_class], 0.9) for _ in range(len(norm_images_target))]))


        avg_cams.append([np.mean(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])
        var_cams.append([np.var(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])

    for model_i in range(len(model_paths)):
        figure, axes = plt.subplots(2, 2)
        figure.suptitle(f"Average CAMs\n{display_names[model_i]}", fontsize=FONT_SIZE)
        for i, avg_cam in enumerate(avg_cams):
            # print(f"Mean variance for model {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np.mean(var_cams[i][model_i])}")
            axes[i//2, i%2].imshow(rescale_0_1(torch.tensor(avg_cam[model_i])).numpy())
            axes[i//2, i%2].set_title(f"Label: {ROTNET_CLASSES[i]}", fontsize=FONT_SIZE)
            axes[i//2, i%2].axis('off')
        plt.show()



def grad_cam_average_diff(norm_images, labels):
    model_paths = ["my_downstream_11.pt", "rn_100_downstream_5.pt", "na_100_downstream_0.pt"]
    display_names = ["Evolved Augments", "RotNet Augments", "No Augments"]

    models = []

    for model_path in model_paths:
        model_instance = TrainResNet18Grad()
        model_instance.switch_to_downstream()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()
        models.append(model)

    avg_cams = []
    var_cams = []
    ent_cams = []
    neg_ent_cams = []
    spvar_cams = []
    neg_spvar_cams = []
    for target_class in range(10):
        target_class_idx = torch.where(labels == target_class)[0]
        print(f"Number of images with target class {CIFAR_CLASSES[target_class]}: {len(target_class_idx)}")

        norm_images_target = norm_images[target_class_idx].to(config['device'])

        grayscale_cams = []
        for model in models:
            with GradCAM(model=model, target_layers=[model.layer1]) as cam:
                grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[ClassifierOutputTarget(target_class) for _ in range(len(norm_images_target))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[ClassifierOutputSoftmaxTarget(target_class) for _ in range(len(norm_images_target))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=None))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[FinerWeightedTarget(target_class, [i for i in range(10) if i != target_class], 0.9) for _ in range(len(norm_images_target))]))


        avg_cams.append([np.mean(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])
        var_cams.append([np.var(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])
        ent_cams.append([np_entropy(np.mean(grayscale_cams[i], axis=0)) for i in range(len(grayscale_cams))])
        neg_ent_cams.append([np_neg_entropy(np.mean(grayscale_cams[i], axis=0)) for i in range(len(grayscale_cams))])
        spvar_cams.append([np.array(np_spatial_var(np.mean(grayscale_cams[i], axis=0))) for i in range(len(grayscale_cams))])
        neg_spvar_cams.append([np.array(np_neg_spatial_var(np.mean(grayscale_cams[i], axis=0))) for i in range(len(grayscale_cams))])

    for model_i in range(1,len(model_paths)):
        figure, axes = plt.subplots(2, 5)
        figure.suptitle(f"Average CAM diff between {display_names[0]} and {display_names[model_i]}", fontsize=FONT_SIZE)
        for i, avg_cam in enumerate(avg_cams):
            # print(f"Mean variance diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np.mean(var_cams[i][0] - var_cams[i][model_i])}")
            # print(f"Mean entropy diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {ent_cams[i][0] - ent_cams[i][model_i]}")
            # print(f"Mean min_abs_val entropy diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {min_abs_val(np.sum(ent_cams[i][0]), np.sum(neg_ent_cams[i][0])) - min_abs_val(np.sum(ent_cams[i][model_i]), np.sum(neg_ent_cams[i][model_i]))}")
            # print(f"Mean max_abs_val entropy diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {max_abs_val(np.sum(ent_cams[i][0]), np.sum(neg_ent_cams[i][0])) - max_abs_val(np.sum(ent_cams[i][model_i]), np.sum(neg_ent_cams[i][model_i]))}")
            # print(f"Mean spatial variance diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {tuple(spvar_cams[i][0] - spvar_cams[i][model_i])}")
            # print(f"Mean spatial variance diff sum for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np.sum(spvar_cams[i][0] - spvar_cams[i][model_i])}")
            # print(f"Mean negative spatial variance diff sum for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np.sum(neg_spvar_cams[i][0] - neg_spvar_cams[i][model_i])}")
            # print(f"Mean max_abs_val spatial variance diff sum for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {max_abs_val(np.sum(spvar_cams[i][0]), np.sum(neg_spvar_cams[i][0])) - max_abs_val(np.sum(spvar_cams[i][model_i]), np.sum(neg_spvar_cams[i][model_i]))}")
            print(f"Mean min_abs_val spatial variance diff sum for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {min_abs_val(np.sum(spvar_cams[i][0]), np.sum(neg_spvar_cams[i][0])) - min_abs_val(np.sum(spvar_cams[i][model_i]), np.sum(neg_spvar_cams[i][model_i]))}")
            diff_cam = rescale_0_1(torch.tensor(avg_cam[0] - avg_cam[model_i])).numpy()
            # print(f"Entropy of diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np_entropy(diff_cam)}")
            axes[i//5, i%5].imshow(diff_cam)
            axes[i//5, i%5].set_title(f"Label: {CIFAR_CLASSES[i]}", fontsize=FONT_SIZE)
            axes[i//5, i%5].axis('off')
        plt.show()

def grad_cam_average_diff_pretext(norm_images, labels):
    model_paths = ["my_pretext_11.pt", "rn_100_pretext_5.pt", "na_100_pretext_0.pt"]
    display_names = ["Evolved Augments", "RotNet Augments", "No Augments"]

    models = []

    for model_path in model_paths:
        model_instance = TrainResNet18Grad()
        model_instance.load_weights_from_path(os.path.join("models", model_path))

        model = model_instance.model.to(config['device'])
        model.eval()
        models.append(model)

    avg_cams = []
    var_cams = []
    ent_cams = []
    neg_ent_cams = []
    spvar_cams = []
    neg_spvar_cams = []
    for target_class in range(4):
        target_class_idx = torch.where(labels == target_class)[0]
        print(f"Number of images with target class {ROTNET_CLASSES[target_class]}: {len(target_class_idx)}")

        norm_images_target = norm_images[target_class_idx].to(config['device'])

        grayscale_cams = []
        for model in models:
            with GradCAM(model=model, target_layers=[model.layer3]) as cam:
                grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[ClassifierOutputTarget(target_class) for _ in range(len(norm_images_target))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[ClassifierOutputSoftmaxTarget(target_class) for _ in range(len(norm_images_target))]))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=None))
                # grayscale_cams.append(cam(input_tensor=norm_images_target, targets=[FinerWeightedTarget(target_class, [i for i in range(10) if i != target_class], 0.9) for _ in range(len(norm_images_target))]))


        avg_cams.append([np.mean(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])
        var_cams.append([np.var(grayscale_cams[i], axis=0) for i in range(len(grayscale_cams))])
        ent_cams.append([np_entropy(np.mean(grayscale_cams[i], axis=0)) for i in range(len(grayscale_cams))])
        neg_ent_cams.append([np_neg_entropy(np.mean(grayscale_cams[i], axis=0)) for i in range(len(grayscale_cams))])
        spvar_cams.append([np.array(np_spatial_var(np.mean(grayscale_cams[i], axis=0))) for i in range(len(grayscale_cams))])
        neg_spvar_cams.append([np.array(np_neg_spatial_var(np.mean(grayscale_cams[i], axis=0))) for i in range(len(grayscale_cams))])

    for model_i in range(1,len(model_paths)):
        figure, axes = plt.subplots(2, 2)
        figure.suptitle(f"Average CAM diff between {display_names[0]} and {display_names[model_i]}", fontsize=FONT_SIZE)
        for i, avg_cam in enumerate(avg_cams):
            # print(f"Mean variance diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np.mean(var_cams[i][0] - var_cams[i][model_i])}")
            # print(f"Mean entropy diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {ent_cams[i][0] - ent_cams[i][model_i]}")
            # print(f"Mean min_abs_val entropy diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {min_abs_val(np.sum(ent_cams[i][0]), np.sum(neg_ent_cams[i][0])) - min_abs_val(np.sum(ent_cams[i][model_i]), np.sum(neg_ent_cams[i][model_i]))}")
            # print(f"Mean max_abs_val entropy diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {max_abs_val(np.sum(ent_cams[i][0]), np.sum(neg_ent_cams[i][0])) - max_abs_val(np.sum(ent_cams[i][model_i]), np.sum(neg_ent_cams[i][model_i]))}")
            # print(f"Mean spatial variance diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {tuple(spvar_cams[i][0] - spvar_cams[i][model_i])}")
            # print(f"Mean spatial variance diff sum for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np.sum(spvar_cams[i][0] - spvar_cams[i][model_i])}")
            # print(f"Mean negative spatial variance diff sum for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np.sum(neg_spvar_cams[i][0] - neg_spvar_cams[i][model_i])}")
            # print(f"Mean max_abs_val spatial variance diff sum for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {max_abs_val(np.sum(spvar_cams[i][0]), np.sum(neg_spvar_cams[i][0])) - max_abs_val(np.sum(spvar_cams[i][model_i]), np.sum(neg_spvar_cams[i][model_i]))}")
            print(f"Mean min_abs_val spatial variance diff sum for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {min_abs_val(np.sum(spvar_cams[i][0]), np.sum(neg_spvar_cams[i][0])) - min_abs_val(np.sum(spvar_cams[i][model_i]), np.sum(neg_spvar_cams[i][model_i]))}")
            diff_cam = rescale_0_1(torch.tensor(avg_cam[0] - avg_cam[model_i])).numpy()
            # print(f"Entropy of diff for models {display_names[0]} and {display_names[model_i]} class {CIFAR_CLASSES[i]}: {np_entropy(diff_cam)}")
            axes[i//2, i%2].imshow(diff_cam)
            axes[i//2, i%2].set_title(f"Label: {ROTNET_CLASSES[i]}", fontsize=FONT_SIZE)
            axes[i//2, i%2].axis('off')
        plt.show()

def test_individual(individual, max_batches, skip_batches, max_samples_per_batch, output_prefix=None, hide_da=True):
    for da in individual:
        print(config['da_funcs'][da[0]](*da[1]))

    individual = [[], individual]

    config['dataset_transforms'] = data_processing_rotnet_torch.dataset_transforms_no_norm

    if hide_da:
        no_norm_dataset_vars = config['load_dataset_func'](individual, config)
    else:
        no_norm_dataset_vars = config['load_dataset_func'](individual, config)

    no_norm_loaders = config['data_loader_func'](*no_norm_dataset_vars, config)

    config['dataset_transforms'] = data_processing_rotnet_torch.dataset_transforms

    norm_dataset_vars = config['load_dataset_func'](individual, config)

    norm_loaders = config['data_loader_func'](*norm_dataset_vars, config)

    # for loader in [loaders[0]]:
    # for no_norm_loader, norm_loader in zip(no_norm_loaders, norm_loaders):
    for no_norm_loader, norm_loader in zip([no_norm_loaders[2]], [norm_loaders[2]]):
    # for no_norm_loader, norm_loader in zip([no_norm_loaders[3]], [norm_loaders[3]]):
        max_i = 0
        skip_i = 0
        for (no_norm_x, no_norm_y), (norm_x, norm_y) in zip(no_norm_loader, norm_loader):
            if skip_i < skip_batches:
                skip_i += 1
                continue

            # grad_cam_image(no_norm_x, norm_x, max_samples_per_batch, output_prefix=output_prefix)
            # grad_cam_diff(no_norm_x, norm_x, norm_y, max_samples_per_batch, output_prefix=output_prefix)
            grad_cam_diff_pretext(no_norm_x, norm_x, norm_y, max_samples_per_batch)
            # grad_cam_average_batch(norm_x, norm_y, max_samples_per_batch, output_prefix=output_prefix)
            # grad_cam_average_multiple_comparison(norm_x, norm_y)
            # grad_cam_wrong_guesses(norm_x, norm_y)
            # grad_cam_average_comparison_all(norm_x, norm_y)
            # grad_cam_average_comparison_all_pretext(norm_x, norm_y)
            # grad_cam_average_disjunction(norm_x, norm_y)
            # grad_cam_average_diff(norm_x, norm_y)
            # grad_cam_average_diff_pretext(norm_x, norm_y)
            max_i += 1
            if max_i >= max_batches:
                break
        break


test_individual(individual, num_batches, skip_batches, samples_per_batch)