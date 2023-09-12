#! /bin/bash

echo "Generating binarized VGG3 models with learning rate: 0,0001 and step-size: 10"

echo "binarized"

python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=10 --gamma=0.5 --model=VGG3 --seed=1867001

python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=10 --gamma=0.5 --model=VGG3 --seed=8646684

python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=10 --gamma=0.5 --model=VGG3 --seed=4527078

echo "Generating binarized VGG3 models with learning rate: 0,001 and step-size: 10"

python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --model=VGG3 --seed=1867001

python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --model=VGG3 --seed=8646684

python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --model=VGG3 --seed=4527078

echo "Generating binarized VGG3 models with learning rate: 0,0001 and step-size: 5"

python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=5 --gamma=0.5 --model=VGG3 --seed=1867001

python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=5 --gamma=0.5 --model=VGG3 --seed=8646684

python3 run_fashion_binarized_fi.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=5 --gamma=0.5 --model=VGG3 --seed=4527078

echo "----------------------------------------------------------------------------"