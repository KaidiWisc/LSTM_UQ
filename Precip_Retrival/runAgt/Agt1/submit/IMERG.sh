#!/bin/bash


for ii in {1..10}
do

mkdir copy${ii}
cp -r runLSTM copy${ii}/
cp -r runCSGD copy${ii}/
cp Input$(( $1 * 10 + ii )).txt copy${ii}/runCSGD/Input.txt
cp Input$(( $1 * 10 + ii )).txt copy${ii}/runLSTM/
rm Input$(( $1 * 10 + ii )).txt
touch Input$(( $1 * 10 + ii )).txt
touch test_streamflow_sim_$(( $1 * 10 + ii )).csv
touch train_streamflow_sim_$(( $1 * 10 + ii )).csv
touch validation_streamflow_sim_$(( $1 * 10 + ii )).csv
done

for ii in {1..10}
do

cd copy${ii}
cd runCSGD
nohup python TrainCSGD.Main.py 01532000
cd ..

cd runLSTM
beta=$(grep "^obs_beta" Input$(( $1 * 10 + ii )).txt | awk '{print $2}')
eta=$(grep "^obs_eta" Input$(( $1 * 10 + ii )).txt | awk '{print $2}')
sed -i "s/\(obserrormodel_beta: \).*/\1$beta/" MTS_basin.yml
sed -i "s/\(obserrormodel_eta: \).*/\1$eta/" MTS_basin.yml 
nohup python MTScase.py 01532000 & 
cd ..
cd ..

done

wait

for ii in {1..10}
do
mv copy${ii}/runLSTM/*streamflow_sim_*.csv ./
done
