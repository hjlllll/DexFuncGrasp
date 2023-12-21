#for i in $seq 0 1 2 4 6 7 8 9 10 11 12;do
#python process_train_data.py --category_id $i
#done
for i in $seq 0 1 4 6 9 10 15 16 17 18 19 20;do
python process_train_data.py --category_id $i
done