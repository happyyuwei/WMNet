cd ../

python main_script.py --data=../../data/flower/train^
    --model=../../trained_models/flower_unet_l1_cw^
    --watermark=../../watermark/wm_color.png^
    --is_binary=False^
    --decoder=../../trained_models/auto_cifar^
    --attack=crop

@pause