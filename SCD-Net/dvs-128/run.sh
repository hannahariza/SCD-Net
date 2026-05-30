# for cifar10-dvs
python train.py -c ./cifar10dvs.yaml
python train_causal_cifar10dvs.py -c ./cifar10dvs.yaml --model max_former_causal
# for dvsgesture
# python train.py -c ./dvsgesture.yaml
python train_causal_dvs128.py -c ./dvsgesture.yaml --model max_former_causal