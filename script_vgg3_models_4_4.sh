#! /bin/bash

echo "Generating VGG3 models with learning rate: 0,0001 and step-size: 10"

echo "weight: 4, input: 4"

python3 run_fashion_quant_gen.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=10 --gamma=0.5 --weight=4 --input=4 --model=VGG3 --seed=1867001

python3 run_fashion_quant_gen.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=10 --gamma=0.5 --weight=4 --input=4 --model=VGG3 --seed=8646684

python3 run_fashion_quant_gen.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=10 --gamma=0.5 --weight=4 --input=4 --model=VGG3 --seed=4527078

echo "Generating VGG3 models with learning rate: 0,001 and step-size: 10"

python3 run_fashion_quant_gen.py --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --weight=4 --input=4 --model=VGG3 --seed=1867001

python3 run_fashion_quant_gen.py --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --weight=4 --input=4 --model=VGG3 --seed=8646684

python3 run_fashion_quant_gen.py --batch-size=256 --epochs=100 --lr=0.001 --step-size=10 --gamma=0.5 --weight=4 --input=4 --model=VGG3 --seed=4527078

echo "Generating VGG3 models with learning rate: 0,0001 and step-size: 5"

python3 run_fashion_quant_gen.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=5 --gamma=0.5 --weight=4 --input=4 --model=VGG3 --seed=1867001

python3 run_fashion_quant_gen.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=5 --gamma=0.5 --weight=4 --input=4 --model=VGG3 --seed=8646684

python3 run_fashion_quant_gen.py --batch-size=256 --epochs=100 --lr=0.0001 --step-size=5 --gamma=0.5 --weight=4 --input=4 --model=VGG3 --seed=4527078