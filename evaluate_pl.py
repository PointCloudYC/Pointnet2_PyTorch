import os
import numpy as np
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger

from pointnet2.data import PSNet3DSemSeg
from torch.utils.data import DataLoader, DistributedSampler

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


CHECK_POINT_PATH=f"/media/yinchao/Mastery/career/Pointnet2_PyTorch/outputs-psnet-ssg/sem-ssg/epoch=2-val_loss=1.44-val_acc=0.597.ckpt"
# CHECK_POINT_PATH = os.path.join('./',CHECK_POINT_PATH)

def hydra_params_to_dotdict(hparams):
    """convert dict object to dot dict form(i.e. naive form of dict w/o nested dict)
    Args:
        hparams (dict): key-value collection
    """
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)


@hydra.main("pointnet2/config/config.yaml")
def main(cfg):
    def log_string(str):
        logger.info(str)
        print(str)

    # create the model using the hydra config
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))

    # load pre-trained weights
    model.load_from_checkpoint(CHECK_POINT_PATH)
    print(model)

    # obtain the test set
    val_dset = PSNet3DSemSeg(model.hparams["num_points"], train=False)
    test_loader= DataLoader(
            val_dset,
            batch_size=model.hparams["batch_size"],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

    # evaluate loop





    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for pc, labels in test_loader:
    #         pc = pc.to(device)
    #         labels = labels.to(device)
    #         logits = model(pc)
    #         loss = F.cross_entropy(logits, labels)
    #         acc = (torch.argmax(logits, dim=1) == labels).float().mean()
    #         # total += labels.size(0)
    #         # correct += (predicted == labels).sum().item()
    #     print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))




    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.epochs,
        # early_stop_callback=early_stop_callback,
        # checkpoint_callback=checkpoint_callback,
        # distributed_backend=cfg.distrib_backend,
    )

    # # trainer.fit(model)
    metrics= trainer.test(model, test_dataloaders=test_loader)
    print(metrics)


if __name__ == "__main__":
    main()
