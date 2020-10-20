for i in 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 
do
  echo "=============START MODEL TRAINING FOR BENCHMARK $i============="
  python train.py --dataset Traffic --out_target $i --batch_size 128 --experiment Traffic_out_$i --max_epoch 200
  echo "==============END MODEL TRAINING FOR BENCHMARK $i=============="
done