cd ../

python main_script.py --data=../../data/flower/train^
    --model=../../trained_models/flower_unet_l1_bw_noise^
    --watermark=../../watermark/wm_binary.png^
    --is_binary=True^
    --decoder=../../trained_models/auto_mnist^
    --attack=noise

@pause