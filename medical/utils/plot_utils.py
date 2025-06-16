import matplotlib.pyplot as plt


def plot_sample(array_list, file_name, color_map = 'nipy_spectral'):
    '''
    Input : array_list = [ img_tensor, seg_tensor ]   shape of img (H,W)
    Example: [ tensor.shape=[(224,224)], tensor.shape=[(224,224)] ]
    '''
    # fig = plt.figure(figsize=(10,8), dpi=100)

    plt.subplot(1,3,1)
    plt.imshow(array_list[0], cmap='bone')
    plt.title('Original Image')
    plt.axis('off')
    #presumably a medical image, converts it to a PyTorch tensor, and applies windowing using the parameters from dicom_windows.liver. It uses the 'bone' color map for visualization, adds a title "Windowed Image", and turns off the axis.

    plt.subplot(1,3,2)
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Mask')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(array_list[0], cmap='bone')
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title('Liver & Mask')
    plt.axis('off')
    
    # Save the figure
    plt.savefig(file_name, bbox_inches='tight')

    plt.show()