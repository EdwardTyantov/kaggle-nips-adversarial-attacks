#-*- coding: utf8 -*-
import os, sys, logging
import numpy as np
from optparse import OptionParser
import torch, torch.nn as nn
from .models import factory
from .folder import ImageTestFolder
from .transform_rules import imagenet_like
from .routine import test_model
from .utils import read_labels
from . import LABEL_FILE


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


class Runner(object):
    def __init__(self, input_dir, output_file, checkpoint_path):
        self.input_dir = input_dir
        self.output_file = output_file
        self.checkpoint_path = checkpoint_path

    def _load_model(self, model_name):
        logger.info('Loading model %s', model_name)
        model = factory(model_name)
        state_dict = torch.load(self.checkpoint_path)
        model.load_state_dict(state_dict)
        model = model.cuda()
        # TODO: add torch.nn.DataParallel
        logger.info('Loaded')
        return model

    def _test_model(self, config, model):
        logger.info('Testing model')
        tr = imagenet_like()['test']
        folder = ImageTestFolder(self.input_dir, transform=tr)
        logger.info('Number of files %d', len(folder))

        results = []
        crop_num = len(tr.transforms[0])
        for index in range(crop_num):
            # iterate over tranformations
            logger.info('Testing transformation %d/%d', index + 1, crop_num)
            folder.transform.transforms[0].index = index
            loader = torch.utils.data.DataLoader(folder, batch_size=config.batch_size, num_workers=config.workers,
                                                 pin_memory=True)
            names, crop_results = test_model(loader, model, activation=nn.Softmax())
            results.append(crop_results)
            #break

        final_results = [sum(map(lambda x: x[i].numpy(), results)) / float(crop_num) for i in
                         range(len(folder.imgs))]

        return names, final_results

    def _write_output(self, names, results):
        logger.info('Writing output, len of results %d', len(results))
        mapping = read_labels(LABEL_FILE)

        with open(self.output_file, 'w') as wf:
            for filename, probs in zip(names, results):
                label_idx = np.argmax(probs) # .numpy()
                label = label_idx + 1 # like TF, see https://github.com/tensorflow/models/tree/master/inception
                #label = mapping[label_idx]
                wf.write('{0},{1}\n'.format(filename, label))

    def run(self, config):
        logger.info('Start predicting')
        # Init auxiliary
        model = self._load_model(config.model_name)
        # Run model
        names, results = self._test_model(config, model)
        self._write_output(names, results)


def main():
    parser = OptionParser()
    parser.add_option("--input_dir", action='store', type='str', dest='input_dir', default=None)
    parser.add_option("--output_file", action='store', type='str', dest='output_file', default=None)
    parser.add_option("--checkpoint_path", action='store', type='str', dest='checkpoint_path', default=None)
    parser.add_option("--model_name", action='store', type='str', dest='model_name', default=None)
    parser.add_option("--batch_size", action='store', type='int', dest='batch_size', default=128)
    parser.add_option("--workers", action='store', type='int', dest='workers', default=2) # TODO: make 2, but shm issue
    config, _ = parser.parse_args()
    input_dir = config.input_dir
    output_file = config.output_file
    checkpoint_path = config.checkpoint_path
    assert input_dir is not None and os.path.exists(input_dir)
    assert checkpoint_path is not None and os.path.exists(checkpoint_path)

    logger.info('Config: %s', str(config))
    app = Runner(input_dir, output_file, checkpoint_path)
    app.run(config)


if __name__ == '__main__':
    sys.exit(main())
