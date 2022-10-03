
PATIENTS="479_11 479_25 482 499 502 505 510 513 515 530 538 539 540 541 543 544 549 551"
PATIENTS="540"

for PATIENT in $PATIENTS
do
    echo $PATIENT
    python generate_logs_with_phonemes.py --patient $PATIENT 
done
