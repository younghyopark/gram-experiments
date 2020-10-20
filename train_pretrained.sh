for i in 0 1 2 3 4 5 6 7 8 9
do
  echo "=============START MODEL TRAINING FOR BENCHMARK $i============="
  python train_pretrained.py --dataset Traffic --out_target $i --batch_size 128 --experiment Traffic_new_pretrained_out_$i --max_epoch 40 --pre_trained 1
  python train_pretrained.py --dataset Traffic --out_target $i --batch_size 128 --experiment Traffic_new_scratch_out_$i --max_epoch 200 --pre_trained 0
  echo "==============END MODEL TRAINING FOR BENCHMARK $i=============="
done