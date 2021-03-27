"""
Author: YINChao
Date: Mar. 2021

main code:
    create the dataset and dataloader for the test set
    create the model w. pretrained weights from pth
    make prediction for each point
    compute evaluate metrics(i.e. accuracy and IoU)
    plot confusion matrix and precision, recall and f1-score
"""

import argparse
import os
from pointnet2.data import Indoor3DSemSeg
from pointnet2.data import PSNet3DSemSeg
# from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
# from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np

import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from pointnet2.models.pointnet2_ssg_sem import PointNet2SemSegSSG
from pointnet2.models.pointnet2_msg_sem import PointNet2SemSegMSG

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = BASE_DIR
# sys.path.append(os.path.join(ROOT_DIR, 'models'))
label2class = {
0: 'ibeam',
1: 'pipe',
2: 'pump',
3: 'rectangularbeam',
4: 'tank',
}
classes=list(label2class.values())
num_classes = len(classes)
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

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


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Semantic Segmentation Model for PSNet reduced 5 dataset')
    parser.add_argument('--model', type=str, default='ssg', help='semantic segmentation model type, ssg or msg')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=3, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3, help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool




def main(args,model,test_loader):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = 'log/sem_seg/' + args.log_dir
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = num_classes
    # model.hparams["num_points"], model.hparams["batch_size"]
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    # root = 'data/s3dis/stanford_indoor3d/'

    # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    TEST_DATASET_WHOLE_SCENE = PSNet3DSemSeg(NUM_POINT, train=False, download=False, data_precent=1.0)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    classifier = model
    # MODEL_TYPE=args.model
    # if MODEL_TYPE == 'ssg':
    #     pass
    # elif MODEL_TYPE == 'msg':
    #     pass
    # else:
    #     print('Unknown model %s. Exiting...'%(FLAGS.model))
    #     exit(-1)

    # model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    # MODEL = importlib.import_module(model_name)
    # classifier = MODEL.get_model(NUM_CLASSES).cuda()


    # checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    # classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier = classifier.eval()
    with torch.no_grad():
        # scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        # scene_id = [x[:-4] for x in scene_id]
        # scene_id = ['Area_3']
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            # print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            print("Inference [%d/%d] ..." % (batch_idx + 1, num_batches))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                # fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                # fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')
                fout = open(os.path.join(visual_dir, 'Area_3' + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, 'Area_3' + '_gt.obj'), 'w')

            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 9))

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                    batch_data[:, :, 3:6] /= 1.0

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=np.float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]
                color_gt = g_label2color[whole_scene_label[i]]
                if args.visual:
                    fout.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
            if args.visual:
                fout.close()
                fout_gt.close()

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")

@hydra.main("pointnet2/config/config.yaml")
def load_model_dataloader(cfg):

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

    return model, test_loader

if __name__ == '__main__':
    model, test_loader=load_model_dataloader()
    args = parse_args()
    main(args,model,test_loader)