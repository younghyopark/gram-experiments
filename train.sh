for i in 1 2 3
do
  echo "=============START MODEL TRAINING FOR BENCHMARK $i============="
  python train.py --dataset SVHN --out_target $i --batch_size 256 --experiment SVHN_out_$i --max_epoch 200
  echo "==============END MODEL TRAINING FOR BENCHMARK $i=============="
done