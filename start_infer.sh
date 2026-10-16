export ZERO_WAM_PREDICTION_ROOT="$PWD/outputs/prediction_traces"
export ALOHA_PREPARED_ROOT=/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3 OMP_NUM_THREADS=4 ASCEND_RT_VISIBLE_DEVICES=7 MODEL_PATH="$PWD/outputs/aloha_step6000_validation/model" PORT=29056 
bash evaluation/aloha/launch_server.sh  --steps 50  --chunk-frames 2
