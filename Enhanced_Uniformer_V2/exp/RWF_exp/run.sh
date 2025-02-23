NUM_SHARDS=1
NUM_GPUS=2
BATCH_SIZE=4
BASE_LR=1.5e-6
work_path=./exp/RWF_exp
PYTHONPATH=$PYTHONPATH:./slowfast \
python3.8 tools/run_net_multi_node.py \
  --init_method tcp://localhost:10126 \
  --cfg $work_path/config.yaml \
  --num_shards $NUM_SHARDS \
  DATA.PATH_TO_DATA_DIR /data/DERI-AVA/data_dirs/rwf_2000/data_raw_2/data_paths \
  DATA.PATH_PREFIX /data/DERI-AVA/data_dirs/rwf_2000/data_raw_2 \
  DATA.PATH_LABEL_SEPARATOR " " \
  TRAIN.EVAL_PERIOD 1 \
  TRAIN.CHECKPOINT_PERIOD 52 \
  TRAIN.BATCH_SIZE $BATCH_SIZE \
  TRAIN.SAVE_LATEST True \
  NUM_GPUS $NUM_GPUS \
  NUM_SHARDS $NUM_SHARDS \
  SOLVER.MAX_EPOCH 50 \
  SOLVER.BASE_LR $BASE_LR \
  SOLVER.BASE_LR_SCALE_NUM_SHARDS False \
  SOLVER.WARMUP_EPOCHS 1. \
  TEST.NUM_ENSEMBLE_VIEWS 4 \
  TEST.NUM_SPATIAL_CROPS 3 \
  TEST.TEST_BEST True \
  TEST.ADD_SOFTMAX True \
  TEST.BATCH_SIZE 6 \
  RNG_SEED 7 \
  OUTPUT_DIR $work_path
