id: tensorflow-mnist-standalone

## mnist 

我们这里主要提供一个mnist数据集的简单的demo，支持整个训练，预测，保存saved_model的整个流程，训练也支持多gpu训练

#### 数据准备

代码会自动去下载mnist的数据，只需要在mnist_main.py中调整data_dir的位置，这个默认是/tmp/mnist_data。也可以自己下载：

训练数据：

[train-images-idx3-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz)

[train-labels-idx1-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz)

测试数据：

[t10k-images-idx3-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz)
[t10k-labels-idx1-ubyte.gz](https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz)

#### 训练模型

- cpu>=2
- memory >= 2g
- gpu  >= gt-1030, memory >= 2g

```shell
python main.py \
	--num_gpus 2 \ 		# 选择使用gpu个数，默认
    --export_dir ./		# 选择 saved_model 保存路径
```



#### 测试模型：

```shell
python infer.py \
	example3.png		# 选择测试图片
    --model_dir 		# save_models/1578488839/
# - predict num: 3

```

#### tips:

可以通过 -h 或者 --help 查询其他 args 的作用