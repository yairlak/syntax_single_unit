
#for decimate in "--decimate 50" " ";
#do

decimate=4
smooth=10
for data_type in "micro" "spike";
do
    for filter in "raw" "high-gamma";
    do
    nohup python 3by3_decoding.py --level phone --patient 479_11 --data-type $data_type --filter $filter --probe-name LSTG RPSTG --patient 479_25 --data-type $data_type --filter $filter --probe-name LSTG RPSTG --patient 482 --data-type $data_type --filter $filter --probe-name LSTG --patient 487 --data-type $data_type --filter $filter --probe-name LSTG --patient 505 --data-type $data_type --filter $filter --probe-name LSTG RSTG --patient 515 --data-type $data_type --filter $filter --probe-name LSTG --decimate $decimate --smooth $smooth &
    done
done

nohup python 3by3_decoding.py --level phone --patient 479_11 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 479_25 --data-type spike --filter gaussian-kernel --probe-name LSTG RPSTG --patient 482 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 487 --data-type spike --filter gaussian-kernel --probe-name LSTG --patient 505 --data-type spike --filter gaussian-kernel --probe-name LSTG RSTG --patient 515 --data-type spike --filter gaussian-kernel --probe-name LSTG --decimate $decimate --smooth $smooth &

nohup python 3by3_decoding.py --level phone --patient 479_11 --data-type spike --filter gaussian-kernel-25 --probe-name LSTG RPSTG --patient 479_25 --data-type spike --filter gaussian-kernel-25 --probe-name LSTG RPSTG --patient 482 --data-type spike --filter gaussian-kernel-25 --probe-name LSTG --patient 487 --data-type spike --filter gaussian-kernel-25 --probe-name LSTG --patient 505 --data-type spike --filter gaussian-kernel-25 --probe-name LSTG RSTG --patient 515 --data-type spike --filter gaussian-kernel-25 --probe-name LSTG --decimate $decimate --smooth smooth &
#done
