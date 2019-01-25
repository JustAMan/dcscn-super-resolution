python sr-train.py --scale=2 --dataset=tests/nanorama-train --test_dataset=tests/nanorama-test --training_images=20 --layers=8 --filters=64 --batch_num=1 --lr_decay_epoch=2 --end_lr=2e-5 --model_name=smoke-nanorama-8x64
python sr-image.py --scale=2 --file_glob=data/tests/fry-small/*.png --layers=8 --filters=64 --batch_num=1 --output_dir=output/smoke/nanorama --model_name=smoke-nanorama-8x64
