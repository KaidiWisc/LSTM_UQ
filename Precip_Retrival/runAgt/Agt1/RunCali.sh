#!/bin/bash

# wait until streamflow_sim shows up 

rm ../Agt1/submit/*streamflow_sim_1.csv
cp Input.txt ../Agt1/submit/Input1.txt
cp Input.txt ../Agt1/submit/Input_1.txt
cp run.info ../Agt1/submit/

while [ ! -e "../Agt1/submit/train_streamflow_sim_1.csv" ]; do
  sleep 5 
done

mv ../Agt1/submit/train_streamflow_sim_1.csv Output.txt

itnum=$(grep 'group_id' run.info | awk -F', ' '{print $2}')


[ ! -d "../Agt1/submit/stream${itnum}" ] && mkdir "../Agt1/submit/stream${itnum}"

mv ../Agt1/submit/validation_streamflow_sim_1.csv ../Agt1/submit/stream${itnum}/
mv ../Agt1/submit/test_streamflow_sim_1.csv ../Agt1/submit/stream${itnum}/

rm Input.txt