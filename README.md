# 飞桨黑客马拉松第四期 科学计算 Deep Learning Methods for Reynolds-Averaged Navier-Stokes Simulationsof Airfoil Flows

## 1.简介

本项目基于paddle框架复现。

本研究侧重于现代化的 U-net 架构，并评估大量经过训练的神经网络在计算压力和速度分布方面的准确性。 特别是，它说明了训练数据的大小和权重的数量如何影响解决方案的准确性。 借助最佳模型，这项研究得出的平均相对压力和速度误差小于 3%，涵盖一系列以前未见过的翼型形状。 此外，所有源代码都是公开的，以确保可重复性，并为对物理问题的深度学习方法感兴趣的研究人员提供一个起点。 虽然这项工作侧重于 RANS 解决方案，但神经网络架构和学习设置非常通用，适用于笛卡尔网格上的各种偏微分方程边值问题。


论文主要点如下：
* 作者提出了一种基于UNet神经网络的模型；
* 作者评估了大量训练的神经网络在计算压力和速度分布方面的准确性，说明了训练数据的大小和权重的数量如何影响解决方案的准确性；
* 作者借助最佳模型得出的平均相对压力和速度误差小于 3%。

本项目关键技术要点：

* 使用swin transformer模型替代原论文模型并得到相近结果；
* 针对更高版本openfoam，例如openfoamv10，实现相关文件修改。


实验结果要点：
* 成功复现论文代码框架及全流程运行测试；
* 使用swin transformer提升论文精度；
* openfoamv10版本可应用本项目代码进行数据生成与测试，代码运行环境更加宽松。

论文信息：
Thuerey N, Weißenow K, Prantl L, et al. Deep learning methods for Reynolds-averaged Navier–Stokes simulations of airfoil flows[J]. AIAA Journal, 2020, 58(1): 25-36.

参考GitHub地址：
https://github.com/thunil/Deep-Flow-Prediction

项目aistudio地址：
https://aistudio.baidu.com/aistudio/projectdetail/5671596

模型结构
![](https://ai-studio-static-online.cdn.bcebos.com/f0f823f4480e4f9ca465897eacb90618f4cfcb76d5c5427784b395c9dd3d11c9)





## 2.数据集

本项目数据集通过作者提供的dataGen.py代码生成，生成后保存为npz文件，已上传aistudio[数据集](https://aistudio.baidu.com/aistudio/datasetdetail/193595)并关联本项目。

本项目关联数据集为通过原作者[链接](https://dataserv.ub.tum.de/s/m1459172/download?path=%2F&files=data_full.tar.gz)下载，后续会逐渐上传通过openfoamv10版本制作的数据集，可在数据集页面和项目说明中查看。

### 2.1 高版本openfoam使用

#### openfoam安装
```bash
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get update
sudo apt-get -y install openfoam10
```
#### openfoam代码修改

修改code/data/OpenFOAM/0/p文件：
```
    inlet
    {
        type            freestreamPressure;
        freestreamValue uniform 0;  # 增加此行
    }

    exit
    {
        type            freestreamPressure;
        freestreamValue uniform 0;  # 增加此行
    }

    top
    {
        type            freestreamPressure;
        freestreamValue uniform 0;  # 增加此行
    }

    bottom
    {
        type            freestreamPressure;
        freestreamValue uniform 0;  # 增加此行
    }
```

修改code/data/OpenFOAM/system/internalCloud文件：
```
sets
(
    cloud
    {
        type    points; # 此处修改
        axis    xyz;
        points  $points;
        ordered yes; # 此处增加
    }
);
```

修改code/data/dataGen.py文件：

p_ufile在v5版本中为分开输出，在v10版本中为合并输出，此处需要根据数据输出形式进行修改。
修改后在代码文件中修改数据提取dim。
```python
def outputProcessing(basename, freestreamX, freestreamY, dataDir=output_dir, p_ufile='OpenFOAM/postProcessing/internalCloud/500/cloud_p_U.xy',
                     res=128, imageIndex=0):
    
    
    ···
    
    ar = np.loadtxt(p_ufile) # 此处为加载的输出文件
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf) < 1e-4 and abs(ar[curIndex][1] - yf) < 1e-4:
                npOutput[3][x][y] = ar[curIndex][3] #此处输出为压力场   需要按照输出文件情况进行修改，如果输出文件表示为p_U，表示先p后U，则无需修改
                curIndex += 1
                # fill input as well
                npOutput[0][x][y] = freestreamX
                npOutput[1][x][y] = freestreamY
            else:
                npOutput[3][x][y] = 0
                # fill mask
                npOutput[2][x][y] = 1.0

    ar = np.loadtxt(p_ufile) # 此处为加载的输出文件
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf) < 1e-4 and abs(ar[curIndex][1] - yf) < 1e-4:
                npOutput[4][x][y] = ar[curIndex][4]  #此处输出为 X 方向速度场
                npOutput[5][x][y] = ar[curIndex][5]  #此处输出为 Y 方向速度场
                curIndex += 1
            else:
                npOutput[4][x][y] = 0
                npOutput[5][x][y] = 0
    ···
```

### 2.2 数据生成过程

请注意，如果您下载下面的训练数据包，则可以跳过接下来的两个步骤。 只需确保源目录中有 data/train 和 data/test，然后就可以继续训练步骤。

* (1）下载机翼数据

进入data目录，通过运行./download_airfoils.sh下载机翼配置文件，这将创建 airfoil_database 和 airfoil_database_test 目录。

* （2）生成数据
运行```python ./dataGen.py``` 生成一组机翼数据。 此脚本执行 openfoam 并运行 gmsh 以对机翼剖面进行网格划分。

* （3）生成过程
首先通过gmsh对机翼剖面进行网格划分，然后通过openfoam计算网格点的压强和速度（包含x方向和y方向）。

### 2.3 数据说明

数据保存为为压缩的numpy数组。 每个文件中的张量大小为 6x128x128，维度为：通道、x、y。 

前三个通道表示source，包含x和y方向上的自由流速度的两个字段和一个包含机翼几何形状的掩码。 最后三个通道代表target，包含一个压力场和两个速度场。

source：[free-stream x, free-stream y, mask]
target：[pressure, flow-velocity x, flow-velocity y]

## 3.环境依赖

### python依赖
* paddle
* matplotlib

### 环境依赖
* openfoam （任意版本）aistudio无法安装，此处不进行安装，可在本地机器进行安装运行


## 4.快速开始


```python
!cd work
```


```python
!tar zxf /home/aistudio/data/data197778/dataset.tgz -C /home/aistudio/work
```

### 模型训练

默认使用TurbNetG（原论文)模型进行训练

使用swin transformer 模型请在train.py中修改
```
line 43 model_name = "TurbNetG"
line 44 # model_name = "SwinTransformer"
```


```python
!python train.py # 4.1.1 使用TurbNetG（原论文)模型进行训练
```

### 模型测试

默认使用TurbNetG（原论文)模型进行测试

使用swin transformer 模型请在test.py中修改
```
line 26 model_name = "TurbNetG"
line 27 # model_name = "SwinTransformer"
```


```python
!cd work && python test.py # 4.1.2 使用TurbNetG（原论文)模型进行测试
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.9/site-packages/matplotlib/__init__.py:107: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, and in 3.10 it will stop working
      from collections import MutableMapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.9/site-packages/matplotlib/rcsetup.py:20: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, and in 3.10 it will stop working
      from collections import Iterable, Mapping
    /opt/conda/envs/python35-paddle120-env/lib/python3.9/site-packages/matplotlib/colors.py:53: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated since Python 3.3, and in 3.10 it will stop working
      from collections import Sized
    2.4.1
    Reducing data to load for tests
    Number of data loaded: 10
    Using fixed maxima [4.65, 2.04, 2.37]
    Data stats, input  mean 0.190220, max  0.961052;   targets mean 0.289502 , max 1.000000 
    W0313 21:44:46.654492  4140 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
    W0313 21:44:46.658141  4140 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
    Loading ckpt/TurbNetG_modelG
    Loaded ckpt/TurbNetG_modelG

    ......
    
    Loss percentage (p, v, combined) (relative error): 14.640122 %    2.204290 %    2.691775 % 
    L1 error: 0.005441
    Denormalized error: 0.012608
    
    


## 5.代码结构
```text
.
├── ckpt                            # 预训练模型文件
│   ├── SwinTransformer_modelG          # 使用swin transformer模型进行训练
│   └── TurbNetG_modelG                 # 使用原论文模型进行训练
├── data
│   ├── dataGen.py                  # 数据生成脚本
│   ├── download_airfoils.sh        # 下载机翼配置文件脚本
│   ├── OpenFOAM                    # openfoam配置文件
│   ├── shearAirfoils.py
│   └── utils.py
├── dataset                         # 数据集存放位置
│   ├── test
│   └── train
├── dataset.py                      # dataloader
├── DfpNet.py                       # 原论文模型
├── swin_transformer.py             # swin transformer模型
├── test.py                         # 测试文件
├── train.py                        # 训练文件
└── utils.py                        
```

## 6.复现结果

### 6.1 精度对齐

复现后模型精度基本与论文报告精度保持一致。

针对模型TurbNetG模型（7.7m参数）
| | Validation loss | Test data L1 error | Test data relative error     |
| -------- | -------- | -------- | -------- |
| 原论结果     | 0.004     | 0.0055     |0.0026     |
| 复现结果     | 0.004     | 0.0054     |0.0026     |


### 6.2 生成样本可视化展示
![](https://ai-studio-static-online.cdn.bcebos.com/52df2e068f7f4ea480075346e09ce7c04498c972c0ce41b89d9fa7fe1f6d72fc)

### 6.3 Swin Transformer精度

使用Swin Transformer能在参数缩减的情况下取得和TurbNetG几乎持平的效果。

同时受益于Swin Transformer架构，针对更大的数据，例如224*224, 384*384, 依旧能够保持良好的性能优势。

| | Validation loss | Test data L1 error | Test data relative error     |
| -------- | -------- | -------- | -------- |
| TurbNetG  (30.9m参数)    | 0.004     | 0.005     |0.0024     |
| Swin Transformer (10m参数)    | 0.004     | 0.005     |0.0025     |



## 7.模型信息

| 信息                | 说明| 
| --------          | -------- | 
| 发布者               | 朱卫国 (DrownFish19)    | 
| 发布时间              | 2023.03     | 
| 框架版本              | paddle 2.4.1     | 
| 支持硬件              | GPU、CPU     | 
| aistudio              | [notebook](https://aistudio.baidu.com/aistudio/projectdetail/5541961)     | 

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
