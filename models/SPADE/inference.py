import numpy as np
from matplotlib import pyplot as plt

from models.SPADE.load_model import load_model as load_model_spade
from models.SPADE.preprocessing import preprocess, postprocess
from models.pix2pixHD.load_model import load_model as load_model_hd


def inference(mask, filename, model_type, dataset_type, model_version, custom_url='', save=True):
    spade = model_type.lower() == 'spade'
    if spade:
        model = load_model_spade(dataset=dataset_type, version_=model_version)
    else:
        model = load_model_hd(dataset=dataset_type, version_=model_version)

    model.eval()
    mask, mask_orig, size = preprocess(mask)
    model_output = model.inference(mask.unsqueeze(0).to('cuda')).squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
    model_output = postprocess(spade, model_output, filename, size, dataset=dataset_type)
    two_images = np.concatenate((mask_orig, model_output), axis=1)
    path = None
    if save:
        path = save_inference(filename, two_images, datapath=custom_url if custom_url else 'api/static/result/')
    return model_output, path


def save_inference(filename, res, datapath):
    plt.show()
    path = datapath + filename.split('.')[-2] + '_.png'
    plt.imsave(path, res / 255.0)
    return path
