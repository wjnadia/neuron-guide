# 딥러닝 프레임워크 병렬화 (Horovod)

## 가. Tensorflow에서 Horovod 사용법

다중노드에서 멀티 GPU를 활용할 경우 Horovod를 Tensorflow와 연동하여 병렬화가 가능하다. 아래 예시와 같이 Horovod 사용을 위한 코드를 추가해주면 Tensorflow와 연동이 가능하다. Tensorflow 및 Tensorflow에서 활용 가능한 Keras API 모두 Horovod와 연동이 가능하며 우선 Tensorflow에서 Horovod와 연동하는 방법을 소개한다.\
(예시: MNIST Dataset 및 LeNet-5 CNN 구조)

※ Tensorflow에서 Horovod 활용을 위한 자세한 사용법은 Horovod 공식 가이드 참조\
(https://github.com/horovod/horovod#usage)



* Tensorflow에서 Horovod 사용을 위한 import 및 메인 함수에서 Horovod 초기화

```python
import horovod.tensorflow as hvd
...
hvd.init()
```

※ horovod.tensorflow: Horovod를 Tensorflow와 연동하기 위한 모듈

※ Horovod를 사용하기 위하여 초기화한다.



* 메인 함수에서 Horovod 활용을 위한 Dataset 설정

```python
(x_train, y_train), (x_test, y_test) = \
keras.datasets.mnist.load_data('MNIST-data-%d' % hvd.rank())
```

※ 각 작업별로 접근할 dataset을 설정하기 위하여 Horovod rank에 따라 설정 및 생성한다.



* 메인 함수에서 optimizer에 Horovod 관련 설정 및 broadcast, 학습 진행 수 설정

```python
opt = tf.train.AdamOptimizer(0.001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
global_step = tf.train.get_or_create_global_step()
train_op = opt.minimize(loss, global_step=global_step)
hooks = [hvd.BroadcastGlobalVariablesHook(0),
tf.train.StopAtStepHook(last_step=20000 // hvd.size()), ... ]
```

※ Optimizer에 Horovod 관련 설정을 적용하고 각 작업에 broadcast를 활용하여 전달함

※ 각 작업들의 학습과정 step을 Horovod 작업 수에 따라 설정함



* Horovod의 프로세스 rank 에 따라 GPU Device 할당

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
```

※ 각 GPU 별로 하나의 작업을 Horovod의 local rank에 따라 할당함



* Rank 0 작업에 Checkpoint 설정

```python
checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None
...
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
hooks=hooks,
config=config) as mon_sess:
```

※ Checkpoint 저장 및 불러오는 작업은 하나의 프로세스에서 수행되어야 하므로 rank 0번에 설정함



## 나. Keras에서 Horovod 사용법

Tensorflow에서는 Keras API를 활용할 경우에도 Horovod와 연동하여 병렬화가 가능하다. 아래 예시와 같이 Horovod 사용을 위한 코드를 추가해주면 Keras와 연동이 가능하다.\
(예시: MNIST Dataset 및 LeNet-5 CNN 구조)

※ Keras에서 Horovod 활용을 위한 자세한 사용법은 Horovod 공식 가이드 참조\
([https://github.com/horovod/horovod/blob/master/docs/keras.rst](https://github.com/horovod/horovod/blob/master/docs/keras.rst))



* Keras에서 Horovod 사용을 위한 import 및 메인 함수에서 Horovod 초기화

```python
import horovod.tensorflow.keras as hvd
...
hvd.init()
```

※ horovod.tensorflow.keras: Horovod를 Tensorflow 내의 Keras와 연동하기 위한 모듈

※ Horovod를 사용하기 위하여 초기화한다.



* Horovod의 프로세스 rank 에 따라 GPU Device 할당

```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
```

※ 각 GPU 별로 하나의 작업을 Horovod의 local rank에 따라 할당함



* 메인 함수에서 optimizer에 Horovod 관련 설정 및 broadcast, 학습 진행 수 설정

```python
epochs = int(math.ceil(12.0 / hvd.size()))
...
opt = keras.optimizers.Adadelta(1.0 * hvd.size())
opt = hvd.DistributedOptimizer(opt)
callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0), ]
```

※ 각 작업들의 학습과정 step을 Horovod 작업 수에 따라 설정함

※ Optimizer에 Horovod 관련 설정을 적용하고 각 작업에 broadcast를 활용하여 전달함



* Rank 0 작업에 Checkpoint 설정

```python
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
```

※ Checkpoint 저장 및 불러오는 작업은 하나의 프로세스에서 수행되어야 하여 rank 0번에 설정함



* Horovod의 프로세스 rank 에 따라 GPU Device 할당

```python
model.fit(x_train, y_train, batch_size=batch_size, callbacks=callbacks, epochs=epochs,
verbose=1 if hvd.rank() == 0 else 0, validation_data=(x_test, y_test))
```

※ 학습 중 출력되는 문구를 Rank 0번 작업에서만 출력하기 위하여 Rank 0번 작업만 verbose 값을 1로 설정함



## 다. PyTorch에서 Horovod 사용법

다중노드에서 멀티 GPU를 활용할 경우 Horovod를 PyTorch와 연동하여 병렬화가 가능하다. 아래 예시와 같이 Horovod 사용을 위한 코드를 추가해주면 PyTorch와 연동이 가능하다.\
(예시: MNIST Dataset 및 LeNet-5 CNN 구조)

※ PyTorch에서 Horovod 활용을 위한 자세한 사용법은 Horovod 공식 가이드 참조\
([https://github.com/horovod/horovod/blob/master/docs/pytorch.rst](https://github.com/horovod/horovod/blob/master/docs/pytorch.rst))



* PyTorch에서 Horovod 사용을 위한 import 및 메인 함수에서 Horovod 초기화 및 설정

```python
import torch.utils.data.distributed
import horovod.torch as hvd
...
hvd.init()
if args.cuda:
    torch.cuda.set_device(hvd.local_rank())
    torch.set_num_threads(1)
```

※ torch.utils.data.distributed: PyTorch에서 distributed training을 수행하기 위한 모듈

※ horovod.torch: Horovod를 PyTorch와 연동하기 위한 모듈

※ Horovod 초기화 및 초기화 과정에서 설정된 rank에 따라 작업을 수행할 device를 설정한다.

※ 각 작업별로 CPU thread 1개를 사용하기 위해 torch.set\_num\_threads(1)를 사용한다.



* Training 과정에 Horovod 관련 내용 추가

```python
def train(args, model, device, train_loader, optimizer, epoch):
...
train_sampler.set_epoch(epoch)
...
    if batch_idx % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_sampler),
                100.* batch_idx / len(train_loader), loss.item()))
```

※ train\_sampler.set\_epoch(epoch): train sampler의 epoch 설정

※ Training dataset이 여러 작업들에 나뉘어서 처리되므로 전체 dataset 크기 확인을 위하여 len(train\_sampler)을 사용한다.



* Horovod를 활용하여 평균값 계산

```python
def metric_average(val, name):
tensor = torch.tensor(val)
avg_tensor = hvd.allreduce(tensor, name=name)
return avg_tensor.item()
```

※ 여러 노드에 걸쳐 평균값을 계산하기 위하여 Horovod의 Allreduce 통신을 활용하여 계산한다.



* Test 과정에 Horovod 관련 내용 추가

```python
test_loss /= len(test_sampler)
test_accuracy /= len(test_sampler)
test_loss = metric_average(test_loss, 'avg_loss')
test_accuracy = metric_average(test_accuracy, 'avg_accuracy')
if hvd.rank() == 0:
    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
           test_loss, 100. * test_accuracy))
```

※ 여러 노드에 걸쳐 평균값을 계산해야 하므로 위에서 선언된 metric\_average 함수를 활용한다.

※ 각 노드별로 Allreduce 통신을 거쳐 loss 및 accuracy에 대해 계산된 값을 동일하게 가지고 있으므로 rank 0번에서 print 함수를 수행한다.



* 메인 함수에서 Horovod 활용을 위한 Dataset 설정

```python
train_dataset = datasets.MNIST('data-%d' % hvd.rank(), train=True, download=True,
transform=transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,)) ]))
train_sampler = torch.utils.data.distributed.DistributedSampler(
train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)
test_dataset = datasets.MNIST('data-%d' % hvd.rank(), train=False, transform=transforms.Compose([
transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)) ]))
test_sampler = torch.utils.data.distributed.DistributedSampler(
test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
sampler=test_sampler, **kwargs)
```

※ 각 작업별로 접근할 dataset을 설정하기 위하여 Horovod rank에 따라 설정 및 생성한다.

※ PyTorch의 distributed sampler를 설정하여 이를 data loader에 할당한다.



* &#x20;메인 함수에서 optimizer에 Horovod 관련 설정 및 training, test 과정에 sampler 추가

```python
optimizer = optim.SGD(model.parameters(), lr=args.lr * hvd.size(), momentum=args.momentum)
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
for epoch in range(1, args.epochs + 1):
    train(args, model, train_loader, optimizer, epoch, train_sampler)
    test(args, model, test_loader, test_sampler)
```

※ Optimizer에 Horovod 관련 설정을 적용하고 각 작업에 broadcast를 활용하여 전달함

※ Training 및 test 과정에 sampler를 추가하여 각 함수에 전달함

{% hint style="info" %}
2022년 9월 22일에 마지막으로 업데이트되었습니다.
{% endhint %}
