'''
Tests a model.
'''

from pathlib import Path

import numpy as np
import torch
import copy
from pytorch_lightning import Trainer

from core.model import Model
from core.datamodule import DataModule

from .utils.gif import make_gif
from .utils.plots import make_error_plots
from .utils.utils import Logger


import json


def evaluate(config, misc_args, trainer, datamodule, model, test_quantization, print_compression=True, prefix="real_"):
    # compute testing statistics
    if misc_args.get('compute_stats'):
        # run on test
        datamodule.setup('test')
        model.prefix = prefix
        model.denormalize = datamodule.test.denorm_f

        with torch.inference_mode():
            trainer.test(model=model, datamodule=datamodule)

            if print_compression:
                print("Compression Factor", datamodule.test.size / model.size)
                trainer.logger.log_metrics(
                    {f"{prefix}compression_factor": datamodule.test.size / model.size},
                    step=0,
                )

    # make error plots
    if misc_args.get("make_error_plots"):
        make_error_plots(trainer, datamodule, model, prefix)

    # make GIF
    if misc_args.get('make_gif'):
        make_gif(trainer, datamodule, model, prefix)

    if misc_args.get('export'):
        with torch.inference_mode():
            data = trainer.predict(model=model, datamodule=datamodule)

        data = torch.cat(data).reshape(datamodule.predict.num_snapshots, -1, datamodule.output_shape[2])

        data = datamodule.predict.denorm_f(data)

        np.save(f'{trainer.logger.log_dir}/{prefix}reconstruction.npy', data.detach().cpu().numpy())

    return


def test(log_dir, config):

    #Extract args
    data_args = config['data']
    model_args = config['model']
    trainer_args = config['train']
    misc_args = config['misc']
    # if quantization is a section in the config, check test_quantization flag under it
    test_quantization = (misc_args.get("quantization", {}).get("test_quantization", False))


    #Setup datamodule
    datamodule = DataModule(**data_args)

    #Build model
    try:
        ckpt_path = list(Path(log_dir, 'checkpoints').rglob('best-epoch=*.ckpt'))[0]
    except:
        ckpt_path = list(Path(log_dir, 'checkpoints').rglob('last.ckpt'))[0]

    if 'ckpt_path' in model_args.keys():
        model_args.pop('ckpt_path')

    model = Model.load_from_checkpoint(ckpt_path, input_shape=datamodule.input_shape, output_shape=datamodule.output_shape, **model_args)

    #Build trainer
    logger = Logger(save_dir=log_dir, name='', version='', default_hp_metric=False)

    trainer_args["logger"] = logger
    trainer_args["devices"] = 1
    
    if "profiler" in trainer_args.keys():
        trainer_args.pop("profiler")

    if "strategy" in trainer_args.keys():
        trainer_args.pop("strategy")

    trainer = Trainer(**trainer_args, inference_mode=True)

    evaluate(config, misc_args, trainer, datamodule, model, test_quantization)

    if test_quantization:
        quantization_test(config, model, trainer, datamodule)

    return


def quantization_test(config, model, trainer, datamodule):
    from core.utils.quantization import quantize_parameters, dequantize_parameters
    misc_args = config['misc']
    if 'quantization' in config['misc']:
        quantization_types = config['misc']['quantization'].get('types', None)
        bits_list = config['misc']['quantization'].get('bits', [8])
        compand = config['misc']['quantization'].get('compand', True)

    orig_model = model

    if quantization_types is not None:
        for i in range(len(quantization_types)):
            model = copy.deepcopy(orig_model)
            quantization_type = quantization_types[i]
            bits = bits_list[i]
            print(f"Testing quantization type: {quantization_type}")

            quantized_parameters, metadata = quantize_parameters(model, method=quantization_type,
                                                                               bits=bits, block_size=None, seed=0, compand=compand)
            dequantized_parameters = dequantize_parameters(quantized_parameters, metadata,
                                                           method=quantization_type, bits=bits, seed=0, compand=compand)

            # Compute the compression factor by comparing the total bits in the datamodule and the quantized model parameters
            orig_size = datamodule.test.size * datamodule.test.features.element_size() * 8
            metadata_size = sum(torch.tensor(value).numel() * torch.tensor(value).element_size() * 8 for value in metadata.values())
            quantized_size = quantized_parameters.numel() * quantized_parameters.element_size() * 8 + metadata_size
            compression_factor = orig_size / quantized_size
            # report the compression factor
            trainer.logger.log_metrics(
                {
                    f"{quantization_type}_{bits}_compression_factor":
                        compression_factor
                },
                step=0,
            )
            print(f"Compression factor for {quantization_type} with {bits} bits: {compression_factor}")

            # reconstruct the model with dequantized parameters
            torch.nn.utils.vector_to_parameters(dequantized_parameters, model.network.hypernet.parameters())

            # test the model again
            evaluate(config, misc_args, trainer, datamodule, model, test_quantization=False, print_compression=False, prefix=f"{quantization_type}_{bits}_")
