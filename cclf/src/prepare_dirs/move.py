import os
import random

TEST_SIZE = 0.1

names= ['winnie', 'magnus']

def move_files(name):
    src_dir =f'../data/{name}'
    train_dir =f'../data/{name}'
    test_dir =f'../data/{name}'

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    files = os.listdir(src_dir)

    random.shuffle(files)
    test_files = files[:int(len(files) * TEST_SIZE)]
    train_files = files[int(len(files) * TEST_SIZE):]

    for file in test_files:
        os.rename(os.path.join(src_dir, file), os.path.join(test_dir, file))

    for file in train_files:
        os.rename(os.path.join(src_dir, file), os.path.join(train_dir, file))

[move_files(name) for name in names]