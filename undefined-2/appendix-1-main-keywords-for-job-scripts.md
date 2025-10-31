# 작업 스크립트 주요 키워드

\
작업 스크립트 내에서 적절한 키워드를 사용하여 원하는 작업을 위한 자원 할당 방법을 명시해야 합니다. 주요 키워드는 아래와 같으며, 사용자는 이들 중에서 몇 가지만 사용하여 작업 스크립트 파일을 작성할 수 있습니다.

***

* **job-name ( -J, --job-name )**\
  작업의 이름을 지정하며, 명시하지 않으면 스크립트 파일 이름이 작업 이름으로 지정됩니다.

***

* **time ( -t, --time )**\
  예상되는 작업 소요 시간을 의미하며, 실제 예상되는 작업 소요 시간보다 약간 더 길게 설정해 주는 것이 안전합니다. 해당 파티션의 Wall time limit을 초과하면, 작업이 제출되지 않습니다. 지정된 시간에 이르렀는데 작업이 완료되지 않으면 SLURM 스케줄러가 작업을 강제 종료 시킵니다.

***

* **partition ( -p, --partition )**\
  작업 수행을 위한 SLURM 파티션을 지정합니다. 파티션 명은 sinfo 명령어로 확인이 가능합니다.

***

* **nodes ( -N, --nodes )**\
  작업을 위해 할당할 노드의 수를 지정합니다.

***

* **ntasks ( -n, --ntasks )**\
  작업을 위해 할당할 프로세스의 수를 지정합니다.

***

* **ntasks-per-node ( --ntasks-per-node )**\
  노드당 할당할 프로세스의 수를 지정합니다.

***

* **input ( -i, --input )**\
  Standard input을 지정합니다.

***

* **cpus-per-task ( -c, --cpus-per-task )**\
  작업 태스크 당 필요한 CPU 개수를 명시합니다.

***

* **output ( -o, --output )**\
  Standard output을 지정합니다.
  * %x : "job name" 으로 지정한 명칭을 파일 명으로 사용합니다.
  * %j : 작업 제출 시 부여되는 "job ID" 를 파일 명으로 사용합니다.
  * %a : "job array ID" (index) 번호를 파일 명으로 사용합니다.
  * %u : "user ID" 를 파일 명 으로 사용합니다.

***

* **error ( -e, --error )**\
  Standard error를 지정합니다.
  * %x : "job name" 으로 지정한 명칭을 파일 명으로 사용합니다.
  * %j : 작업 제출 시 부여되는 "job ID" 를 파일 명으로 사용합니다.
  * %a : "job array ID" (index) 번호를 파일 명으로 사용합니다.
  * %u : "user ID"를 파일 명으로 사용합니다.

***

* **dependency ( -d, --dependency )**\
  작업 의존성을 설정합니다. 설정된 작업이 종료된 후에 작업이 시작됩니다.



※ SLURM 상세 매뉴얼 : http://slurm.schedmd.com

{% hint style="info" %}
2022년 9월 22일에 마지막으로 업데이트 되었습니다.
{% endhint %}
