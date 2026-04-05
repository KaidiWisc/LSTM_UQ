#!/bin/bash

lastcnt=0
count_files() {
  
  cnt=0
  for i in {1..100} 
  do
    if [ -f "Input_${i}.txt" ]
    then
    cnt=$((cnt + 1))
    fi
  done
  echo $cnt
}


while true 
do
thiscnt=$(count_files)
echo ${thiscnt}

if [ $thiscnt -eq $lastcnt ] && [ $thiscnt -gt 0 ]
then

   condor_submit IMERG.sub  
	 rm Input_*.txt

else
   lastcnt=${thiscnt}
fi

sleep 30
done

