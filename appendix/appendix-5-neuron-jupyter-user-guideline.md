# 뉴론 Jupyter

## 가. Jupyter 개요

### **1. JupyterHub**

* JupyterHub 란 멀티 사용자 환경에서 주피터 랩/노트북을 사용할 수 있는 오픈소스 소프트웨어를 뜻합니다.
* JupyterHub는 다양한 환경 (JupyterLab, Notebook, RStudio, Nteract 등)을 지원할 뿐만 아니라 인증 서버 (OAuth, LDAP, GitHub 등) 및 배치 스케줄러와도 (PBSPro, Slurm, LSF 등) 유연하게 연동 가능합니다.
* JupyterHub는 컨테이너 관리 플랫폼인 Kubernetes와도 연동이 쉬워 컨테이너 기반의 클라우드 환경에 쉽게 연동 가능합니다.

※ Neuron 기반 JupyterHub는 5호기 Bright LDAP, OTP 인증기능을 추가하였고 Slurm 배치 스케줄러와 연동하여 자원을 할당하여 Jupyter 실행하고 현재 default로 Jupyter Notebook을 제공하고 추가로 JupyterLab 제공합니다.

### **2. Jupyter Notebook**

* Jupyter Notebook은 웹 기반의 오픈소스 어플리케이션으로 프로그래머들에게 문서 생성, 코드 생성 및 실행, 수학적 라이브러리를 사용한 데이터 시각화, 통계 모델링, 머신러닝/딥러닝 프로그래밍에 사용합니다.
* 40 여개의 프로그래밍 언어 즉 Python, R, Julia, Scala등을 지원합니다.
* 프로그래밍 언어로 작성한 코드는 HTML, 이미지, 동영상 파일, LaTeX 등 다양한 타입으로 변환 가능합니다.
* Apache Spark, Pandas, Scikit-learn, ggplot2, Tensorflow 등 다양한 툴/라이브러리들과 연동 가능합니다.

![](<../.gitbook/assets/Jupyter Notebook.png>)

### **3. JupyterLab**

* JupyterLab은 Jupyter Notebook 인터페이스에 사용자 편의를 위한 기능들을 추가하여 확장 가능한 모듈로 구성됩니다.
* Jupyter Notebook과 달리 하나의 작업 화면에 Tabs 와 Splitters를 사용하여 여러 개의 도큐먼트 또는 다른 기능을 제공합니다.

![](../.gitbook/assets/JupyterLab.png)

## 나. 스크립트를 통한 Jupyter 실행

### **1. 개요**&#x20;

* **로그인 노드 접속**&#x20;

&#x20;`$ ssh [사용자ID]@neuron.ksc.re.kr`&#x20;

* **(필수) Jupyter 관련 패키지 설치 스크립트 실행**&#x20;

&#x20; `$ sh /apps/jupyter/kisti_conda_jupyter.sh`&#x20;

&#x20; ※ notebook conda env 생성 및 jupyter notebook, jupyterlab, cudatoolkit 11.6, cudnn 패키지 설치



* **(옵션) 필요에 따라 기타 패키지 자동 설치 스크립트 실행 혹은 직접 실행**

&#x20; ※ notebook conda env 활성화 : **$ conda activate notebook**



#### &#x20;   **1) tensorflow 작업 환경**&#x20;

&#x20;   `(notebook) $ sh  /apps/jupyter/kisti_conda_tensorflow.sh` &#x20;

#### &#x20;  **2) pytorch 작업 환경**

&#x20;    `(notebook) $ sh  /apps/jupyter/kisti_conda_pytorch.sh`

<mark style="color:red;">**※ 최초 한번만 실행하며 환경설정이 완료되면 즉시 웹 페이지 접속하여 (다. JupyterLab 사용방법) JupyterLab/Notebook을 사용 가능합니다.**</mark>

### **2. 스크립트 실행**

* 터미널로 로그인 노드 (<mark style="color:red;">**neuron.ksc.re.kr**</mark>) 에 접속하여 다음 스크립트 /apps/jupyter/kisti\_conda\_jupyter.sh 를 실행합니다.
* 스크립트를 실행하면 <mark style="color:red;">**/scratch/\[사용자ID]/.conda/envs**</mark> 디렉터리에 notebook Conda 환경이 만들어지고 jupyterhub, jupyterLab, notebook 패키지들이 자동으로 설치되고 멀티 GPU 환경에 필요한 cudatoolkit=11.6 과 cudnn이 설치됩니다.&#x20;

**※ 이 파일은 한번만 실행하면 되고 그 다음부터는 바로 웹 페이지 접속하여 사용 가능합니다.**\
※ 실행파일은 공유 디렉터리에서 /apps/jupyter/kisti\_conda\_jupyter.sh 로 바로 실행 가능합니다.\
※ 아래 테스트는 사용자ID _<mark style="color:red;">**a1113a01**</mark>_ 로 진행하였습니다.

```shell-session
[a1113a01@glogin02 ~]$ sh /apps/jupyter/kisti_conda_jupyter.sh
... ...
modified /home01/a1113a01/.bashrc
...prepare conda environment for jupyter user.
Exporting CONDA ENVS and PKGS PATH to bash File.
Downloading and Extracting Packages
#################################################################### | 100%
#################################################################### | 100%
#################################################################### | 100%
#################################################################### | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: / WARNING conda.core.prefix_data:_load_single_record(167):
done
```

* shell을 다시 시작하고 base 환경 자동 활성화 기능을 꺼야 합니다. (한번만 실행)

```shell-session
[a1113a01@glogin01 ~]$ source ~/.bashrc
(base) [a1113a01@glogin01 ~]$ conda config --set auto_activate_base false
(base) [a1113a01@glogin01 ~]$ conda deactivate
```

※ base 환경 자동 활성화 기능을 false 로 설정함으로 다음에 base 환경으로 자동 활성화 되는 것을 방지합니다. (만약 base 환경으로 활성화 되지 않았으면 source \~/.bashrc 이후 바로 conda activate notebook 명령어를 실행)

* conda notebook 환경을 다음 명령어로 활성화 합니다.

```shell-session
[a1113a01@glogin01 ~]$ conda activate notebook
(notebook) [a1113a01@glogin01 ~]$
```

* Tensorflow(tensorboard 포함) 혹은 Pytorch 등의 사용을 원하는 사용자는 KISTI에서 제공하는 자동 설치 스크립트를 실행하여 설치할 수 있습니다.&#x20;

※ 주의: 반드시 notebook 사용자 환경에서 실행해야 한다.

```shell-session
## tensorflow-gpu, tensorboard 패키지와 jupyter_notebook 및 jupyter_lab을 위한 tensorboard extenstion 설치
(notebook) 757% [a1113a01@glogin01 ~]$ sh /apps/jupyter/kisti_conda_tensorflow.sh (약 10분 소요)

## pytorch, torchvision, torchaudio 패키지 설치
(notebook) 757% [a1113a01@glogin01 ~]$ sh /apps/jupyter/kisti_conda_pytorch.sh (약 5분 소요)
```

※ 이제부터 사용자는 직접 웹에 접속하여 Jupyter 노트북을 사용할 수 있습니다. (여기까지 작업들은 한번만 실행하면 됨)

### **3. JupyterHub 웹 페이지 접속**

* <mark style="color:red;">https://jupyter.ksc.re.kr</mark> 에 접속하여 신청 받은 뉴론 계정, OTP, 비밀번호를 입력합니다.

![](<../.gitbook/assets/JupyterHub 웹 페이지 접속.png>)

* 메인 화면에서 자원 사용현황 확인 및 Refresh 버튼을 클릭하여 자원 사용 현황을 업데이트 할 수 있습니다.

<figure><img src="../.gitbook/assets/jupyter_queue_selection.png" alt=""><figcaption></figcaption></figure>

### **4. 큐 (Queue) 선택 및 Jupyter 실행**

* Jupyter를 실행하기 전에 <mark style="color:red;">**Refresh**</mark> 버튼을 클릭하여 자원 현황을 확인
  * Total CPUs: 전체 CPU 코어 개수
  * Alloc CPUs: 사용중인 CPU 코어 개수
  * Total GPUs: 전체 GPU 개수
  * Alloc GPUs: 사용중인 GPU 개수 &#x20;
  * Node Usage: 노드 활용률
* Queue 정보 확인
  * jupyter queue (무료): 환경 설치, 전처리, 디버깅 용도&#x20;
  * other queues (유료): 딥러닝/머신러닝 등 모델 실행 및 시각화 용도

※ jupyter queue는 현재 2개 노드로 최대 20개(노드 당 10개) Jupyter Lab/Notebook 실행 가능합니다. (여러 사용자가 노드의 CPU+GPU\[v100] 공유)&#x20;

※ jupyter queue의 GPU는 공유자원이기 때문에 할당 정보(Alloc GPUs)는 0으로 표시됩니다.

※ 큐 선택 시 대기중 작업 존재 여부를 확인하시고 노트북 실행하시기 바랍니다.

<mark style="color:red;">**※ jupyter 큐 외 기타 큐 선택 시 가급적 \*:gpu=1 을 선택하여 주피터를 실행하시기 바랍니다.**</mark>

※ 유료 과금 정책은 기존 Neuron 시스템 과금 정책을 따르고 정보는 국가슈퍼컴퓨팅 홈페이지 요금 안내 페이지 (https://www.ksc.re.kr/jwjg/gjbg/ygan) 에서 확인 가능합니다.

* Job queue 에서 해당 queue를 선택하고 Submit버튼을 클릭하여 Jupyter Notebook 실행 (other queues로도 실행 가능하나, 다만 과금 발생함, 과금 정보는 KSC 홈페이지 Neuron 과금 정보 참고)

<figure><img src="../.gitbook/assets/jupyter_queue_selection_v2.png" alt=""><figcaption></figcaption></figure>

* 다음과 같은 화면이 몇 초간 진행 되면서 자원 할당이 진행된다.

<figure><img src="../.gitbook/assets/processing (1).png" alt=""><figcaption></figcaption></figure>

* Default로 https://jupyter.ksc.re.kr/user/a1113a01/lab JupyterLab 화면이 실행된다.

<figure><img src="../.gitbook/assets/jupyterlab.png" alt=""><figcaption></figcaption></figure>

{% embed url="https://youtu.be/HN-Uw3NYmmA" %}

## 다. JupyterLab 사용 방법

#### **1) Jupyter 작업 환경**

* <mark style="color:red;">**Jupyter 환경 디렉터리: /scratch/\[사용자ID]/.conda/envs/notebook**</mark>
* <mark style="color:red;">**로그 저장 디렉터리: /scratch/\[사용자ID]/log/작업ID.log**</mark>
* <mark style="color:red;">**작업 파일 저장 디렉터리: /scratch/\[사용자ID]/workspace/**</mark>

※ 사용자는 본인이 필요로 하는 머신러닝/딥러닝 라이브러리들을 .../notebook conda 환경에 설치하기 때문에 기본 쿼터가 큰 /scratch/사용자ID/ 에 설치된다. (Jupyter 실행 후 발생하는 로그파일도 /scratch/사용자ID 에 저장)

※ 사용자가 작성한 코드는 /scratch/사용자ID/에 저장된다.

※ conda 환경 백업을 위한 conda 환경 내보내기 및 가져오기 관련 정보는 KISTI 홈페이지 소프트웨어 지침서에서 확인할 수 있다.



* Terminal 실행, Launcher 탭에서 Terminal 아이콘을 클릭한다.
  * Launcher 탭이 보이지 않을 경우 Menu Bars에서 + 아이콘을 클릭한다.

![](../.gitbook/assets/eDtreex0IDwyL3d.png)

![](../.gitbook/assets/1pQTEGKnXpoCpqm.png)

* Tensorboard 실행, Menu Bars->+아이콘->Launcher->Tensorboard를 클릭한다.

![](../.gitbook/assets/zoYS0S9gCXUvx4R.png)

![](../.gitbook/assets/t8EJplkhAqtCUtR.png)

#### **2) 실행중인 세션 종료**

* 다음과 같이, Left Side Bar에서 Session 탭을 클릭하여 실행중인 Terminal Sessions 이나 Kernel Sessions들을 Shut Down 버튼을 클릭하여 종료한다.

※ 세션을 종료시키지 않고 JupyterHub 웹페이지를 종료하는 경우, 다음 Jupyter 실행 시에도 그대로 남아있게 된다. (과금은 진행되지 않음)

<figure><img src="../.gitbook/assets/shutdown.png" alt=""><figcaption></figcaption></figure>

#### **3) Jupyter 종료**

File -> Hub Control Panel -> Stop My Server

<figure><img src="../.gitbook/assets/stopserver.png" alt=""><figcaption></figcaption></figure>

<figure><img src="../.gitbook/assets/stop.png" alt=""><figcaption></figcaption></figure>

<mark style="color:red;">**※ 홈페이지 Logout 시 실행중인 Jupyter 및 세션들은 모두 자동으로 종료된다.**</mark>

\




## 라. 머신러닝/딥러닝 예제 코드 실행하기

### **1. 예제 코드 실행에 필요한 라이브러리 설치**

* Launcher에서 Terminal 클릭하여 머신러닝/딥러닝에 필요한 라이브러리 설치한다.

<figure><img src="../.gitbook/assets/1.png" alt=""><figcaption></figcaption></figure>

* 터미널 환경에서 conda activate notebook 명령어로 notebook 환경을 활성화하고 notebook 환경에 필요한 라이브러리를 설치한다.

※ 반드시 notebook conda 환경에 설치해야 Jupyter 웹 화면에 적용된다.

<figure><img src="../.gitbook/assets/2.png" alt=""><figcaption></figcaption></figure>

* notebook 환경에 사용자가 원하는 머신러닝/딥러닝 관련 라이브러리 설치 예시et

```shell-session
772% [a1113a01@gpu06 workspace]$ conda activate notebook
(notebook) 773% [a1113a01@gpu06 workspace]$ conda install -c conda-forge [conda 패키지 명]-y
```

### **2. 예제 코드 작성 및 실행**

* 사용자 작업 디렉터리에서 예제 파일 iris\_ex.ipynb를 클릭한다.
* 프로그램 편집/실행 창에서 Shift+Enter로 예제 코드를 실행한다.

※ 실행 과정에 나오는 warning 들은 무시 가능하며, 동일 코드 재실행 시 warning 메시지 출력되지 않는다. (warning 내용은 코딩 시 버전에 따른 문법적 제시 안내)

<figure><img src="../.gitbook/assets/3.png" alt=""><figcaption></figcaption></figure>

* matplotlib 라이브러리를 사용한 그래프 출력

![](../.gitbook/assets/SX6ZxYNxMoZSub3.png)

![](../.gitbook/assets/g1MZ0fOQKvwUBWy.png)

### **3. Tensorboard 실행**

* Menu Bar -> Files 에서 tfboard\_ex.ipynb를 클릭한다.
* Shifter+Enter로 코드 실행한다. (약1분 소요)

<figure><img src="../.gitbook/assets/4.png" alt=""><figcaption></figcaption></figure>

* Tensorboard 실행한다.&#x20;

※ logs 폴더에 로그 데이터가 저장된다.

<figure><img src="../.gitbook/assets/5.png" alt=""><figcaption></figcaption></figure>

* TensorBoard -> Scalars

![](../.gitbook/assets/nNt6sLANd0iPvq2.png)

* TensorBoard -> Graphs

![](../.gitbook/assets/E7Q3W0udQjkzOXy.png)

* TensorBoard -> Distributions

![](../.gitbook/assets/YhCiSqGRPGn79il.png)

* TensorBoard -> Histograms

![](../.gitbook/assets/idcwepwUTPBQ8v8.png)

### **4. 새로운 Launcher 만들기 및 Python 코드 작성**

* 아래와 같이 New -> Python 3 메뉴를 클릭하여 새로운 Python 코드의 작성이 가능하다.

<figure><img src="../.gitbook/assets/6.png" alt=""><figcaption></figcaption></figure>

* Python 3 커널을 사용할 수 있는 새로운 Jupyter Notebook Launcher가 실행된다.

<figure><img src="../.gitbook/assets/7.png" alt=""><figcaption></figcaption></figure>



## 마. Jupyter 종료 방법

### **1. 실행중인 세션 종료**

* 다음과 같이, Left Side Bar에서 Session 탭을 클릭하여 실행중인 Terminal Sessions 이나 Kernel Sessions들을 Shut Down 버튼을 클릭하여 종료한다.

※ 세션을 종료시키지 않고 JupyterHub 웹페이지를 종료하는 경우, 다음 Jupyter 실행 시에도 그대로 남아있게 된다. (과금은 진행되지 않음)

<figure><img src="../.gitbook/assets/8.png" alt=""><figcaption></figcaption></figure>

### **2. Jupyter 종료**

* (JupyterLab) Jupyter 사용이 끝나면 반드시 Jupyter를 종료시켜 자원을 반납해야 한다.
* File 메뉴에서 Hub Control Panel 클릭하여 Home 페이지로 와서 Stop My Server 클릭하여 자원을 반납할 수 있다.

&#x20;

<figure><img src="../.gitbook/assets/stop_my_server.png" alt=""><figcaption></figcaption></figure>

&#x20;



## 바. Jupyter 환경 초기화 방법

* conda 가상 환경 notebook 에 pip 으로 설치 할 경우 기존 conda install 로 설치한 패키지들과 버전 충돌이 발생하여 Jupyter 노크북이 실행이 안될 경우 다음과 같은 명령어로 환경 초기화를 해줄 수 있습니다.
* 터미널로 로그인 노드에서 /apps/jupyter/reset\_env.sh 를 실행합니다.
* 해당 스크립트를 실행하면 /scratch/\[사용자ID]/.conda/envs 디렉터리에 만들어졌던 notebook 가상환경에 설치되었던 모든 패키지들이 삭제되고 처음 jupyter 실행을 위한 기본 패키지들이 다시 설치 됩니다.
* /sratch/\[사용자ID]/workspace/에 데이터는 보존됩니다.

```shell-session
[a1113a01@glogin02 ~]$ sh /apps/jupyter/reset_env.sh

Remove all packages in environment /scratch/a1113a01/.conda/envs/notebook:

Preparing transaction: done

Verifying transaction: done

If you need another packages, you can run the installation scripts for some packages such as below ! 
1.(mandatary) conda activate notebook
2.(option)
  a. [tensorfow] sh /apps/jupyter/kisti_conda_tensorflow.sh
  b. [pytorch] sh /apps/jupyter/kisti_conda_pytorch.sh
  c. [etc] sh /apps/jupyter/kisti_conda_etc.sh

[a1113a01@glogin02 ~]$ conda activate notebook

## 필요에 따라 기타 패키지에 대한 자동 설치 스크립트 실행

(notebook)[a1113a01@glogin02 ~]$ /apps/jupyter/kisti_conda_tensorflow.sh  
(notebook)[a1113a01@glogin02 ~]$ /apps/jupyter/kisti_conda_pytorch.sh

```

{% embed url="https://youtu.be/bMvwXXJvwq4" %}

{% hint style="info" %}
2023년 6월 30일에 마지막으로 업데이트 되었습니다.
{% endhint %}
