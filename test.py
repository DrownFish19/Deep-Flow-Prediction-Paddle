################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Compute errors for a test set and visualize. This script can loop over a range of models in 
# order to compute an averaged evaluation. 
#
################

import math
import os

import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import DataLoader

import utils
from DfpNet import TurbNetG
from dataset import TurbDataset
from swin_transformer import SwinTransformer
from utils import log

suffix = ""  # customize loading & output if necessary

model_name = "TurbNetG"
# model_name = "SwinTransformer"
prefix = model_name

expo = 6
dataset = TurbDataset(None, mode=TurbDataset.TEST, dataDir="dataset/train/reg/", dataDirTest="dataset/test/")
testLoader = DataLoader(dataset, batch_size=1, shuffle=False)

if model_name == "TurbNetG":
    netG = TurbNetG(channelExponent=expo)
else:
    netG = SwinTransformer(img_size=128, embed_dim=128, in_chans=3, depths=[2, 6], num_heads=[4, 4], window_size=4, drop_path_rate=0.1)

lf = "./" + prefix + "_testout{}.txt".format(suffix)
utils.makeDirs(["results_test"])

# loop over different trained models
avgLoss = 0.
losses = []
models = []

if model_name == "TurbNetG":
    modelFn = "ckpt/TurbNetG_modelG"
else:
    modelFn = "ckpt/SwinTransformer_modelG"

models.append(modelFn)
log(lf, "Loading " + modelFn)
netG.set_state_dict(paddle.load(modelFn))
log(lf, "Loaded " + modelFn)

criterionL1 = nn.L1Loss()
L1val_accum = 0.0
L1val_dn_accum = 0.0
lossPer_p_accum = 0
lossPer_v_accum = 0
lossPer_accum = 0

netG.eval()

for i, data in enumerate(testLoader, 0):
    inputs, targets = data
    inputs = inputs.astype('float32')
    targets = targets.astype('float32')

    outputs = netG(inputs)
    lossL1 = criterionL1(outputs, targets)
    L1val_accum += lossL1.item()

    outputs_cpu = outputs.cpu().numpy()[0]
    targets_cpu = targets.cpu().numpy()[0]

    # precentage loss by ratio of means which is same as the ratio of the sum
    lossPer_p = np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])) / np.sum(np.abs(targets_cpu[0]))
    lossPer_v = (np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2]))) / (
            np.sum(np.abs(targets_cpu[1])) + np.sum(np.abs(targets_cpu[2])))
    lossPer = np.sum(np.abs(outputs_cpu - targets_cpu)) / np.sum(np.abs(targets_cpu))
    lossPer_p_accum += lossPer_p.item()
    lossPer_v_accum += lossPer_v.item()
    lossPer_accum += lossPer.item()

    log(lf, "Test sample %d" % i)
    log(lf, "    pressure:  abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu[0] - targets_cpu[0])), lossPer_p.item()))
    log(lf, "    velocity:  abs. difference, ratio: %f , %f " % (
        np.sum(np.abs(outputs_cpu[1] - targets_cpu[1])) + np.sum(np.abs(outputs_cpu[2] - targets_cpu[2])), lossPer_v.item()))
    log(lf, "    aggregate: abs. difference, ratio: %f , %f " % (np.sum(np.abs(outputs_cpu - targets_cpu)), lossPer.item()))

    # Calculate the norm
    input_ndarray = inputs.cpu().numpy()[0]
    v_norm = (np.max(np.abs(input_ndarray[0, :, :])) ** 2 + np.max(np.abs(input_ndarray[1, :, :])) ** 2) ** 0.5

    outputs_denormalized = dataset.denormalize(outputs_cpu, v_norm)
    targets_denormalized = dataset.denormalize(targets_cpu, v_norm)

    # denormalized error
    outputs_denormalized_comp = np.array([outputs_denormalized])
    outputs_denormalized_comp = paddle.to_tensor(outputs_denormalized_comp)
    targets_denormalized_comp = np.array([targets_denormalized])
    targets_denormalized_comp = paddle.to_tensor(targets_denormalized_comp)

    outputs_dn, targets_dn = outputs_denormalized_comp, targets_denormalized_comp

    lossL1_dn = criterionL1(outputs_dn, targets_dn)
    L1val_dn_accum += lossL1_dn.item()

    # write output image, note - this is currently overwritten for multiple models
    os.chdir("./results_test/")
    utils.imageOut("%04d" % i, outputs_cpu, targets_cpu, normalize=False, saveMontage=True)  # write normalized with error
    os.chdir("../")

log(lf, "\n")
L1val_accum /= len(testLoader)
lossPer_p_accum /= len(testLoader)
lossPer_v_accum /= len(testLoader)
lossPer_accum /= len(testLoader)
L1val_dn_accum /= len(testLoader)
log(lf, "Loss percentage (p, v, combined) (relative error): %f %%    %f %%    %f %% " % (
    lossPer_p_accum * 100, lossPer_v_accum * 100, lossPer_accum * 100))
log(lf, "L1 error: %f" % L1val_accum)
log(lf, "Denormalized error: %f" % L1val_dn_accum)
log(lf, "\n")

avgLoss += lossPer_accum
losses.append(lossPer_accum)

if len(losses) > 1:
    avgLoss /= len(losses)
    lossStdErr = np.std(losses) / math.sqrt(len(losses))
    log(lf, "Averaged relative error and std dev across models:   %f , %f " % (avgLoss, lossStdErr))
