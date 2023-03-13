################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Main training script
#
################

import random
import sys

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader

import dataset
import utils
from DfpNet import weights_init
from swin_transformer import SwinTransformer

# 初始化并行环境
dist.init_parallel_env()

######## Settings ########

# number of training iterations
iterations = 10000
# batch size
batch_size = 64
# learning rate, generator
lrG = 0.0006
# decay learning rate?
decayLr = True
# channel exponent to control network size
expo = 5
# data set config
# prop = None  # by default, use all from "../data/train"
prop = [1000, 0.75, 0, 0.25]  # mix data from multiple directories
# save txt files with per epoch loss?
saveL1 = True

reg = True
dataDir = "data/train/"
dataDirTest = "data/test/"
##########################

prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]
    print("Output prefix: {}".format(prefix))
    prop[0] = int(sys.argv[2])

if prop is not None:
    prefix += str(prop[0]) + "_"

dropout = 0.  # note, the original runs from https://arxiv.org/abs/1810.08217 used slight dropout, but the effect is minimal; conv layers "shouldn't need" dropout, hence set to 0 here.
doLoad = ""  # optional, path to pre-trained model

print("LR: {}".format(lrG))
print("LR decay: {}".format(decayLr))
print("Iterations: {}".format(iterations))
print("Dropout: {}".format(dropout))

##########################

seed = random.randint(0, 2 ** 32 - 1)
print("Random seed: {}".format(seed))
random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)

# create pytorch data object with dfp dataset
data = dataset.TurbDataset(prop, dataDir=dataDir, dataDirTest=dataDirTest, shuffle=1)
trainLoader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)
print("Training batches: {}".format(len(trainLoader)))
dataValidation = dataset.ValiDataset(data)
valiLoader = DataLoader(dataValidation, batch_size=batch_size, shuffle=False, drop_last=True)
print("Validation batches: {}".format(len(valiLoader)))

# setup training
epochs = int(iterations / len(trainLoader) + 0.5)
epochs = 500
# netG = TurbNetG(channelExponent=expo, dropout=dropout)
# netG = SwinTransformer(img_size=128, embed_dim=96, in_chans=3, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=4, drop_path_rate=0.2)
# netG = SwinTransformer(img_size=128, embed_dim=256, in_chans=3, depths=[2, 6, 2], num_heads=[4, 8, 8], window_size=4, drop_path_rate=0.2)
netG = SwinTransformer(img_size=128, embed_dim=128, in_chans=3, depths=[2, 6], num_heads=[4, 4], window_size=4, drop_path_rate=0.1)
netG = paddle.DataParallel(netG)
print(netG)  # print full net
params = sum([np.prod(p.shape) for p in netG.parameters() if p.trainable])
print("Initialized TurbNet with {} trainable params ".format(params))

netG.apply(weights_init)
if len(doLoad) > 0:
    netG.set_state_dict(paddle.load(doLoad))
    print("Loaded model " + doLoad)

criterionL1 = nn.L1Loss()
optimizerG = optim.Adam(parameters=netG.parameters(), learning_rate=lrG, beta1=0.5, beta2=0.999, weight_decay=0.0)
##########################

for epoch in range(epochs):
    print("Starting epoch {} / {}".format((epoch + 1), epochs))

    netG.train()
    L1_accum = 0.0
    for i, traindata in enumerate(trainLoader):
        inputs, targets = traindata
        inputs = inputs.astype('float32')
        targets = targets.astype('float32')

        # compute LR decay
        if decayLr:
            currLr = utils.computeLR(epoch, epochs, lrG * 0.1, lrG)
            if currLr < lrG:
                optimizerG.set_lr(currLr)

        netG.clear_gradients()
        gen_out = netG(inputs)

        lossL1 = criterionL1(gen_out, targets)
        lossL1.backward()

        optimizerG.step()

        lossL1viz = lossL1.item()
        L1_accum += lossL1viz

        if i == len(trainLoader) - 1:
            print("Epoch: {}, batch-idx: {}, L1: {}\n".format(epoch, i, lossL1viz), flush=True)

    # validation
    netG.eval()
    L1val_accum = 0.0
    for i, validata in enumerate(valiLoader, 0):
        inputs, targets = validata
        inputs = inputs.astype('float32')
        targets = targets.astype('float32')

        outputs = netG(inputs)
        outputs_cpu = outputs.cpu().numpy()

        lossL1 = criterionL1(outputs, targets)
        L1val_accum += lossL1.item()

        if i == 0:
            input_ndarray = inputs.cpu().numpy()[0]
            v_norm = (np.max(np.abs(input_ndarray[0, :, :])) ** 2 + np.max(np.abs(input_ndarray[1, :, :])) ** 2) ** 0.5

            outputs_denormalized = data.denormalize(outputs_cpu[0], v_norm)
            targets_denormalized = data.denormalize(targets.cpu().numpy()[0], v_norm)
            utils.makeDirs(["{}_results_train".format(prefix)])
            utils.imageOut("{}_results_train/epoch{}_{}".format(prefix, epoch, i), outputs_denormalized, targets_denormalized, saveTargets=True)

    # data for graph plotting
    L1_accum /= len(trainLoader)
    L1val_accum /= len(valiLoader)
    if saveL1:
        if epoch == 0:
            utils.resetLog(prefix + "L1.txt")
            utils.resetLog(prefix + "L1val.txt")
        utils.log(prefix + "L1.txt", "{} ".format(L1_accum), True)
        utils.log(prefix + "L1val.txt", "{} ".format(L1val_accum), True)

    paddle.save(netG.state_dict(), "{}_results_train/{}_modelG".format(prefix, epoch))
