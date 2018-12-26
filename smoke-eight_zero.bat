python sr-train.py --scale=2 --dataset=tests/zero_x --test_dataset=tests/eight --training_images=20 --layers=6 --filters=32 --batch_num=1 --lr_decay_epoch=1 --end_lr=2e-5 --model_name=smoke-eight_zero-6x32
python sr-image.py --scale=2 --file_glob=data/tests/eightzero-small/*.png --layers=6 --filters=32 --batch_num=1 --output_dir=output/smoke/eight-zero --model_name=smoke-eight_zero-6x32
