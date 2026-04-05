rm -r Agt1/submit/stream*
rm Agt1/submit/*.csv
rm Agt1/submit/*.log
rm Agt1/submit/*.err
rm Agt1/submit/*.txt
rm Agt1/submit/*.out
rm Agt1/submit/run.info

for ii in {1..100}
do
touch Agt1/submit/Input${ii}.txt
done


for ii in {2..100}
do

mkdir Agt${ii}
rm Agt${ii}/*
cp Agt1/*.* Agt${ii}/
cp Agt1/pestpp-ies Agt${ii}/
sed -i "s/Input_1/Input_${ii}/" Agt${ii}/RunCali.sh
sed -i "s/Input1/Input${ii}/" Agt${ii}/RunCali.sh
sed -i "s/sim_1/sim_${ii}/" Agt${ii}/RunCali.sh
cd Agt${ii}
chmod +x *.sh
chmod +x pestpp-ies
rm Agt${ii}/Input.txt
rm Agt${ii}/Output.txt
rm Agt${ii}/nohup.out
rm Agt${ii}/run.info
nohup ./pestpp-ies MyCali.pst /H 128.105.68.127:4294 &

cd ..
done


cd Agt1
chmod +x *.sh
chmod +x pestpp-ies
rm Input.txt
rm Output.txt
rm nohup.out
rm run.info
nohup ./pestpp-ies MyCali.pst /H 128.105.68.127:4294 &
cd submit
chmod +x *.sh
nohup ./Wait_submit.sh &

