set CUBLAS_WORKSPACE_CONFIG=:4096:8

python imagenet_main.py %2 --arch %1 --gpu 0 --batch-size %4 --seed 0 --epochs 1 --learning-rate %3

python imagenet_main.py %2 --arch %1 --world-size 1 --multiprocessing-distributed --dist-backend gloo --dist-url file:///D:\pg --rank 0 --batch-size %4 --seed 0 --epochs 1 --learning-rate %3

sleep 5
del D:\pg
sleep 5

python imagenet_main.py %2 --arch %1 --world-size 2 --multiprocessing-distributed --dist-backend gloo --dist-url file:///D:\pg --rank 0 --batch-size %4 --seed 0 --epochs 1 --learning-rate %3

sleep 5
del D:\pg
sleep 5

python imagenet_main.py %2 --arch %1 --world-size 4 --multiprocessing-distributed --dist-backend gloo --dist-url file:///D:\pg --rank 0 --batch-size %4 --seed 0 --epochs 1 --learning-rate %3
