python train.py --scale=2 --dataset=tests/eight --test_dataset=tests/zero --training_images=20 --layers=6 --filters=32 --batch_num=1 --lr_decay_epoch=1 --end_lr=2e-5
python sr0.py --scale=2 --file_glob=data/tests/eight-small/eight-small.png --layers=6 --filters=32 --batch_num=1
