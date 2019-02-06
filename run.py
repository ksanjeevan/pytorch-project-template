#!/usr/bin/env python3
import argparse, json, os
import torch

from utils import Logger
import data as data_module
import net as net_module

from train import Trainer

from eval import ClassificationEvaluator, ImageInference


def _get_transform(config, name):
    tsf_name = config['transforms']['type']
    tsf_args = config['transforms']['args']
    return getattr(data_module, tsf_name)(name, **tsf_args)

def _get_model_att(checkpoint):
    m_name = checkpoint['config']['model']['type']
    sd = checkpoint['state_dict']
    classes = checkpoint['classes']
    return m_name, sd, classes


def eval_main(checkpoint):

    config = checkpoint['config']
    data_config = config['data']

    tsf = _get_transform(config, 'val')

    data_manager = getattr(data_module, config['data']['type'])(config['data'])
    test_loader = data_manager.get_loader('test', tsf)

    m_name, sd, classes = _get_model_att(checkpoint)
    model = getattr(net_module, m_name)(classes, state_dict=sd)

    model.load_state_dict(checkpoint['state_dict'])

    num_classes = len(classes)
    metrics = getattr(net_module, config['metrics'])(num_classes)

    evaluation = ClassificationEvaluator(test_loader, model)
    ret = evaluation.evaluate(metrics)
    print(ret)
    return ret


def infer_main(file, config, checkpoint):
    if checkpoint is None:
        model = getattr(net_module, config['model']['type'])()
    else:
        m_name, sd, classes = _get_model_att(checkpoint)
        model = getattr(net_module, m_name)(classes, state_dict=sd)
        model.load_state_dict(checkpoint['state_dict'])

    tsf = _get_transform(config, 'val')
    inference = ImageInference(model, transforms=tsf)
    label, conf = inference.infer(file_path)
    print(label, conf)
    inference.draw(file_path, label, conf)


def train_main(config, resume):
    train_logger = Logger()

    data_config = config['data']

    tsf_name = config['transforms']['type']
    tsf_args = config['transforms']['args']
    
    t_transforms = _get_transform(config, 'train')
    v_transforms = _get_transform(config, 'val')

    data_manager = getattr(data_module, config['data']['type'])(config['data'])
    classes = data_manager.classes

    t_loader = data_manager.get_loader('train', t_transforms)
    v_loader = data_manager.get_loader('val', v_transforms)

    m_name = config['model']['type']
    model = getattr(net_module, m_name)(classes, mode=config['net_mode'])
    num_classes = len(classes)
    print(model)

    loss = getattr(net_module, config['train']['loss'])
    metrics = getattr(net_module, config['metrics'])(num_classes)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    opt_name = config['optimizer']['type']
    opt_args = config['optimizer']['args']
    optimizer = getattr(torch.optim, opt_name)(trainable_params, **opt_args)
    trainer = Trainer(model, loss, metrics, optimizer, 
                      resume=resume,
                      config=config,
                      data_loader=t_loader,
                      valid_data_loader=v_loader,
                      train_logger=train_logger)

    trainer.train()



def _test_loader(config):

    def disp_batch(batch):
        ret = []
        for b in batch:

            if len(b.size()) != 1:
                ret.append(b.shape)
            else:
                ret.append(b)
        return ret

    tsf = _get_transform(config, 'val')
    data_manager = getattr(data_module, config['data']['type'])(config['data'])
    loader = data_manager.get_loader('train', tsf)
    print(tsf.transfs)
    for batch in loader:
        print(disp_batch(batch))




if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='PyTorch Template')

    argparser.add_argument('action', type=str,
                           help='what action to take (train, test, eval)')
    
    argparser.add_argument('-c', '--config', default=None, type=str,
                           help='config file path (default: None)')
    argparser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    argparser.add_argument('--net_mode', default='init', type=str,
                           help='type of transfer learning to use')

    args = argparser.parse_args()


    # Resolve config vs. resume
    checkpoint = None
    if args.config:
        config = json.load(open(args.config))
        config['net_mode'] = args.net_mode
    elif args.resume:
        checkpoint = torch.load(args.resume)
        config = checkpoint['config']

    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    # Pick mode to run
    if args.action == 'train':
        train_main(config, args.resume)

    elif args.action == 'eval':
        eval_main(checkpoint)

    elif args.action == 'testloader':
        _test_loader(config)

    elif os.path.isfile(args.action):
        file_path = args.action
        infer_main(file_path, config, checkpoint)
