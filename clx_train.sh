echo "=============================================================================================="
echo "QC_code recognize : ghostnet "
echo "=============================================================================================="
python train.py --training_dataset ~/chelx/dataset/QR_1080p/retina_format/train/label.txt \
                --network ghostnet \
                --save_folder ~/chelx/ckpt/QC_code/Retinaface_1280/g_ghostnet_bs8_base/

wait

echo "=============================================================================================="
echo "QC_code recognize : mobile0.25 "
echo "=============================================================================================="
python train.py --training_dataset ~/chelx/dataset/QR_1080p/retina_format/train/label.txt \
                --network mobile0.25 \
                --save_folder ~/chelx/ckpt/QC_code/Retinaface_1280/g_mobile0.25_bs16_base/
wait

echo "\nTHE SYSTEM WILL BE SHUTDOWN NOW !!! \n"
shutdown -h now
