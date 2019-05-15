python3 train.py --model=cmu --datapath={datapath} --batchsize=64 --lr=0.001 --modelpath={path-to-save}


python3 train.py --model=mobilenet_v2_1.4 --datapath=../../cocoapi/annotations --imgpath=../../cocoapi --batchsize=2 --lr=0.001


CUDA_VISIBLE_DEVICE=2  python3 train.py --model=mobilenet_v2_1.4 --datapath=../cocoapi/annotations --imgpath=../cocoapi --batchsize=2 --lr=0.001 --gpus=1

