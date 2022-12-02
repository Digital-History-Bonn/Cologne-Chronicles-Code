"""Module for loading Model and predicting images"""
import argparse
import datetime
import os
from typing import Union, List, Any

import numpy as np
import sklearn  # type: ignore
import tensorflow as tf  # type: ignore
import torch  # type: ignore
import tqdm  # type: ignore
from PIL import Image  # type: ignore
from numpy import ndarray
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import train
from model import DhSegment
from news_dataset import NewsDataset

IN_CHANNELS, OUT_CHANNELS = 3, 10


def _get_model(path: str) -> DhSegment:
    # create model
    result: DhSegment = DhSegment([3, 4, 6, 4], in_channels=IN_CHANNELS, out_channel=OUT_CHANNELS)
    result = result.float()
    result.load(path)
    result.to(DEVICE)
    result.eval()
    return result


def get_file(file: str, scale=0.25) -> torch.Tensor:
    """
    loads a image as tensor
    :param file: path to file
    :param scale: scale
    :return: image as torch.Tensor
    """
    img = Image.open(file).convert('RGB')
    shape = int(img.size[0] * scale), int(img.size[1] * scale)
    img = img.resize(shape, resample=Image.BICUBIC)

    w_pad, h_pad = (32 - (shape[0] % 32)), (32 - (shape[1] % 32))
    img_np = np.pad(np.asarray(img), ((0, h_pad), (0, w_pad), (0, 0)), 'constant', constant_values=0)
    img_t = np.transpose(torch.tensor(img_np), (2, 0, 1))
    return torch.unsqueeze(torch.tensor(img_t), dim=0)


def get_output_filenames(output_path: Union[str, List[Any]], input_path: Union[str, List[Any]]) -> \
        Union[str, List[Any]]:
    """returns generated output name or output name from args"""

    return output_path or list(map(lambda f: f'{os.path.splitext(f)[0]}_OUT.png', input_path))


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='model.pt', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--scale', '-s', metavar='scale', type=float, default=0.25,
                        help='Scale factor for the input images')
    parser.add_argument('--with-validation', '-v', dest='val', action='store_true',
                        help='If True, news_dataset must be linked to a Directory containing validation data, '
                             'similar to data loaded in train.py. This data will be validated, results are logged '
                             'with tensor board ')
    parser.add_argument('--name', '-n', metavar='NAME', type=str,
                        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                        help='name of run in tensorboard')

    return parser.parse_args()


def run_validation(scale: float):
    """runs validation on data of NewsDataset"""
    dataset = NewsDataset(scale=scale, crop=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    loss_fn = CrossEntropyLoss(weight=torch.tensor(train.LOSS_WEIGHTS)).to(DEVICE)
    step = 0
    for data in tqdm.tqdm(loader, desc='validation_round', total=len(loader)):
        val_image = data[0].to(DEVICE)
        target = data[1].to(DEVICE)

        prediction = model(val_image)
        loss = loss_fn(prediction, target)

        prediction = prediction.detach().cpu().numpy()
        loss = loss.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        val_image = val_image.detach().cpu()

        prediction = np.argmax(prediction, axis=1)
        jaccard_score = sklearn.metrics.jaccard_score(target.flatten(), prediction.flatten(), average='macro')
        accuracy_score = sklearn.metrics.accuracy_score(target.flatten(), prediction.flatten())
        target = torch.tensor(target)
        prediction = torch.tensor(prediction)

        with summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=step)
            tf.summary.scalar('accuracy', accuracy_score, step=step)
            tf.summary.scalar('jaccard score', jaccard_score, step=step)
            tf.summary.image('image', torch.permute(val_image, (0, 2, 3, 1)),
                             step=step)
            tf.summary.image('target', torch.unsqueeze(target.float(), 3) / OUT_CHANNELS, step=step)
            tf.summary.image('prediction', torch.unsqueeze(prediction.float(), 3) / OUT_CHANNELS, step=step)

            step += 1

        del val_image, target, prediction, loss
        torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args.output, args.input)

    train_log_dir = 'logs/runs/' + args.name
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = _get_model(args.model)

    if args.val:
        run_validation(args.scale)
    else:
        for i, filename in enumerate(in_files):
            image = get_file(filename, scale=args.scale)
            pred: ndarray = np.array(model.predict(torch.tensor(image).to(DEVICE)))
            result_img = Image.fromarray(pred).convert('RGB')
            result_img.save(out_files[i])