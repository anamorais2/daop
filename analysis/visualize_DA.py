import configs.config_visualize_da as config_visualize_da
import torch
import matplotlib.pyplot as plt

config = config_visualize_da.config

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# individual = [[10, [0.03, 0.58, 0.81, 0.87, 0.11]]]
# individual = [[43, [0.99, 0.55, 0.63, 0, 0.3]], [1, [0.55, 0.73, 0.62, 0.13, 0.14]]]
INDIVIDUAL = [[43, [0.96, 0.95, 0.38, 0, 0.16]], [0, [0.3, 0.22, 0.82, 0.29, 0.65]], [27, [0.19, 0.29, 0.88, 0.34, 0.33]], [1, [0.3, 0.37, 0.94, 0.62, 0.63]]]
# INDIVIDUAL = [[0, [1.0, 0.2, 0.2, 0.2, 0.2]], [1, [0.5, 0.5, 0.5, 0.5, 0.5]]]
# INDIVIDUAL = []
NUM_BATCHES = 1


def visualize_transformed_images(images, labels):
    # 4 by 4 plot of images with labels
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(images[i * 4 + j].permute(1, 2, 0))
            axs[i, j].set_title(CIFAR_CLASSES[labels[i * 4 + j]])
            axs[i, j].axis('off')

    plt.show()

def test_individual(individual, num_batches):
    for da in individual:
        print(config['da_funcs'][da[0]](*da[1]))

    dataset_vars = config['load_dataset_func']([[],individual], config)

    loader = config['data_loader_func'](*dataset_vars, config)[1]

    i = 0
    for x, y in loader:
        visualize_transformed_images(x, y)
        i += 1
        if i == num_batches:
            break


test_individual(INDIVIDUAL, NUM_BATCHES)