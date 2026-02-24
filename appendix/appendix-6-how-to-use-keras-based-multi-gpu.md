# Keras 기반 Multi GPU 사용법

Keras(케라스)는 파이썬으로 작성된 오픈 소스 신경망 라이브러리로, MXNet, Deeplearning4j, 텐서플로, Microsoft Cognitive Toolkit 또는 Theano 위에서 수행할 수 있는 High-level Neural Network API 입니다. NEURON의cas\_v100\_2, cas\_v100nv\_4, cas\_v100nv\_8, amd\_a100nv\_8 큐에는 한 노드에 2개, 4개, 8개의 GPU가 장착되어 있어, 단일노드를 이용할 때에도 GPU를 여러 대 사용하여 신경망 학습을 할 수 있는 환경이 구축되어 있습니다.



## 가. Multi-GPU 사용을 위한 코드 변경 및 작업 제출 방법

### 1. \[from keras.utils import multi\_gpu\_model] 모듈 추가

| <p>import keras</p><p>from keras.datasets import cifar10</p><p>from keras.models import Sequential</p><p>from keras.layers import Dense, Dropout, Activation, Flatten</p><p>from keras.layers import Conv2D, MaxPooling2D</p><p><mark style="color:red;">from keras.utils import multi_gpu_model</mark></p><p>import os</p> |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

### 2. 코드 내 multi-gpu사용 선언

|                                                                                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <p># initiate RMSprop optimizer<br>opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)<br># multi-gpu<br><mark style="color:red;">model = multi_gpu_model(model, gpus=2)</mark><br># Let's train the model using RMSprop<br>model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])</p> |

※ 사용하고자 하는 GPU 개수만큼 gpus를 설정합니다. (ex. cas\_v100nv\_4노드의 경우에는 gpus=4라고 설정)

### 3. 작업제출 스크립트

```bash
#!/bin/sh
#SBATCH -J keras
#SBATCH --time=24:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p cas_v100_4
#SBATCH --comment tensorflow
#SBATCH --gres=gpu:2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8

module purge
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1 conda/tensorflow_2.4.1

srun python example.py
```

{% hint style="info" %}
2022년 11월 9일에 마지막으로 업데이트 되었습니다.
{% endhint %}
