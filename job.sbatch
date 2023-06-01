# demo
python3 -m demo.demo --config-file configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml --input input1.jpg input2.jpg --output ./output --opts MODEL.WEIGHTS fsdet://coco/tfa_cos_1shot/model_final.pth

# train
python3 -m tools.train_net --num-gpus 1 --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_base1.yaml

