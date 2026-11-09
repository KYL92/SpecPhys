#!/bin/bash
# [69 문서] 체인 128 레인 7개 기동(pro6000 컨테이너 안에서 실행: docker exec -d -u user vface_train bash -lc 'bash scripts/launch_chain128.sh')
cd /SPECPHYS_ROOT
for i in 1 2 3 4 5 6 7; do nohup bash scripts/run_chain128_contrib.sh P$i pro6000 >> /tmp/chain128_P$i.log 2>&1 & sleep 15; done
