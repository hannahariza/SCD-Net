export HOST_NODE_ADDR=127.0.0.1:2963
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo

DATASET=${1:-imagenet100}
if [ "$DATASET" = "imagenet1k" ]; then
  DATA_PATH=${2:-/root/lanyun-pub/imagenet-1k}
  CONFIG=${CONFIG:-./conf/10_512_t4.yml}
  NAME=${NAME:-eval_imagenet1k_10_512_t4}
else
  DATA_PATH=${2:-./data/imagenet_100}
  CONFIG=${CONFIG:-./conf/10_512_t4.yml}
  NAME=${NAME:-eval_imagenet100_10_512_t4}
fi

LOG_DIR=${LOG_DIR:-./log/$NAME}
OUTPUT_DIR=${OUTPUT_DIR:-./output/$NAME}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
RESUME=${RESUME:-./checkpoints/10-512-T4.pth.tar}

NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes=1 --nproc_per_node=$NPROC_PER_NODE --rdzv_endpoint=$HOST_NODE_ADDR test.py \
 --pin_mem --dist_eval -c $CONFIG --exp $NAME --dataset $DATASET --data_path $DATA_PATH \
 --log_dir $LOG_DIR --output_dir $OUTPUT_DIR --resume $RESUME
