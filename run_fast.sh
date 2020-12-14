export CUBLAS_WORKSPACE_CONFIG=':4096:8'

BATCH=$6

python imagenet_main.py $4 --arch $1 --gpu 0 --batch-size $BATCH --seed 0 --epoch 1 --learning-rate $5 --workers $7

python imagenet_main.py $4 --arch $1 --world-size 1 --multiprocessing-distributed --dist-backend $2 --dist-url $3 --rank 0 --batch-size $BATCH --seed 0 --epochs 1 --learning-rate $5 --workers $7

python imagenet_main.py $4 --arch $1 --world-size 2 --multiprocessing-distributed --dist-backend $2  --dist-url $3  --rank 0 --batch-size $BATCH --seed 0 --epochs 1 --learning-rate $5 --workers $7 

python imagenet_main.py $4 --arch $1 --world-size 4 --multiprocessing-distributed --dist-backend $2 --dist-url $3 --rank 0 --batch-size $BATCH --seed 0 --epochs 1 --learning-rate $5 --workers $7 


