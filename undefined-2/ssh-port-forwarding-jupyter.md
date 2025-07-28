---
description: 뉴론에서 작업 제출로 Jupyter 실행하는 방법
---

# SSH Port Forwarding 을 통한 Jupyter 작업 실행

Jupyter Notebook은 웹 기반의 오픈소스 어플리케이션으로 프로그래머들에게 문서 생성, 코드 생성 및 실행, 수학적 라이브러리를 사용한 데이터 시각화, 통계 모델링, 머신러닝/딥러닝 프로그래밍에 사용되고 있습니다.&#x20;

이 문서에서는 뉴론 시스템에서 SSH Port Forwarding 을 통한 Jupyter Notebook 작업 실행 방법에 대해 안내 드립니다.

<mark style="color:red;">**※ 작업 디렉터리 : /scratch/$USER**</mark> <mark style="color:red;">**(사용자 작업은 /scratch/$USER 및 하위 경로에서만 제출 가능)**</mark>



## 가. 작업 스크립트 작성 및 제출 <a href="#job_script_submit" id="job_script_submit"></a>

다음과 같이 작업 제출 스크립트 jupyter\_run.sh 를 작성하여 sbatch 로 작업 제출합니다.

### 1. 작성 스크립트 작성 <a href="#jobscript" id="jobscript"></a>

```bash
#!/bin/bash
#SBATCH --comment=pytorch
#SBATCH --partition=eme_h200nv_8 
#SBATCH --time=12:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=4     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the port and node name
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command

echo "load module-environment"
module load gcc/10.2.0 cuda/12.3  # pre-load module needed  

echo "execute jupyter"
source ~/.bashrc
conda activate notebook 
cd /scratch/$USER  
# the root/work directory of Jupyter lab/notebook
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} 
#jupyter token: your account ID
echo "end of the job"
```

<mark style="color:red;">※ jupyter 파티션은 작업 제출이 불가</mark>

<mark style="color:red;">※ --gres, --cpus-per-task로 GPU, CPU 개수 설정 가능</mark>

<mark style="color:red;">※ conda 가상환경 notebook에 jupyter notebook, jupyterlab 등 설치되어 있어야 합니다.</mark> \ <mark style="color:red;">(기존</mark> [<mark style="color:red;">Neuron Jupyter 사용자</mark>](broken-reference)<mark style="color:red;">는 별도로 설치할 필요가 없습니다.)</mark>

### 2. 작업 스크립트 제출 <a href="#submit" id="submit"></a>

```bash
$ sbatch jupyter_run.sh 
Submitted batch job [job_id]
```



## 나. 작업 제출 디렉터리의 Port Forwarding Command 파일 확인

```bash
$ cat port_forwarding_command
ssh -L localhost:8888:gpu30:22782 $USER@neuron.ksc.re.kr
```

<mark style="color:red;">※ port\_forwarding\_command 파일의 gpu30:22782 는 실행할 때 마다 바뀌기 때문에 꼭 확인 필요</mark>



## 다. SSH Port Forwarding 설정

사용자 본인의 PC에서 새로운 SSH Client (e.g., Putty, MobaXterm, PowerShell, Command Prompt, etc)를 실행하여 다음과 같이 명령어를 복사하여 실행합니다. (뉴론 로그인 시 사용했던 OTP, 비밀번호를 입력합니다.)

### 1. Windows OS일 경우 명령 프롬프트 예시 <a href="#windows" id="windows"></a>

<figure><img src="../.gitbook/assets/image (3).png" alt=""><figcaption></figcaption></figure>

### 2. Mac OS일 경우 터미널 예시 <a href="#mac" id="mac"></a>

```bash
user1@001-ghlee-Mac-Studio ~ % ssh -L localhost:8888:gpu30:22782 
$USER@neuron.ksc.re.kr
Password(OTP): 
```

<mark style="color:red;">※ 로그인 성공 상태에서만 jupyter 접속이 가능하니 exit하여 빠져나올 경우 다시 로그인 해야 합니다.</mark>



## 라. 웹 브라우저 접속 <a href="#web_browser" id="web_browser"></a>

로그인 성공 후, 본인 PC 웹 브라우저에 localhost:8888 로 접속 (Password or token: 본인 아이디)

<mark style="color:red;">※ 비밀번호가 아닌 아이디입니다. (예: x0010a01)</mark>

<figure><img src="../.gitbook/assets/image (1) (1).png" alt=""><figcaption></figcaption></figure>

<mark style="color:red;">※ 해당 작업은 인터렉티브 작업으로 실행이 되기 때문에 사용이 완료되면</mark> \ <mark style="color:red;">scancel \[job\_id] 명령으로 작업을 삭제하시기 바랍니다. (MAX WALLTIME 은 24시간)</mark>
