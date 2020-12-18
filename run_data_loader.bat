set CUBLAS_WORKSPACE_CONFIG=:4096:8

python imagenet_main.py D:\medium --arch resnet50 --gpu 0 --batch-size 128 --seed 0 --epochs 1 --data-loader --workers 1
python imagenet_main.py D:\medium --arch resnet50 --gpu 0 --batch-size 128 --seed 0 --epochs 1 --data-loader --workers 4
python imagenet_main.py D:\medium --arch resnet50 --gpu 0 --batch-size 128 --seed 0 --epochs 1 --data-loader --workers 6
python imagenet_main.py D:\medium --arch resnet50 --gpu 0 --batch-size 128 --seed 0 --epochs 1 --data-loader --workers 8