from fastai.imports import *

def convert_img_to_plt(img, NCHW):
    img = np.squeeze(img)
    if NCHW and len(img.shape) == 3:
        img = (np.moveaxis(img, 0, -1) + 1.0) / 2
    return img


def show_image_list(images, NCHW=False):
    if len(images) == 1:
        im = plt.imshow(convert_img_to_plt(images[0], NCHW))
    else:
        fig, axs = plt.subplots(1, len(images), figsize=(20, 20))
        for img, ax in zip(images, axs):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            im = ax.imshow(convert_img_to_plt(img, NCHW))

    #        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.2)
    plt.show()