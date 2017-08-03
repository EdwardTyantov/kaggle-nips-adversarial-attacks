#-*- coding: utf8 -*-
import torch, logging


logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)


def test_model(test_loader, model, activation=None):
    model.eval()
    names, results = [], []
    for i, (input, name_batch) in enumerate(test_loader):
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        output = model(input_var)
        if activation is not None:
            output = activation(output)
        names.extend(name_batch)
        results.extend(output.data.cpu())
        logger.info('Batch %d',i)

    return names, results
