PROJECT_PATH=/workspace/MedHEval
CONFIG_NAME=vti_v0.9_coco_t0.9_coco_fix

##########################
# Inference
##########################

CURRENT_PATH=$(pwd)
cd ${PROJECT_PATH}/code/baselines/Med-LVLMs/llava-med-1.5

# Slake
python ${PROJECT_PATH}/code/baselines/Med-LVLMs/llava-med-1.5/llava/eval/eval_batch.py --num-chunks 1  --model-name liuhaotian/llava-v1.5-7b \
    --question-file ${PROJECT_PATH}/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json \
    --image-folder ${PROJECT_PATH}/dataset/Slake1.0/imgs \
    --answers-file ${PROJECT_PATH}/runs/${CONFIG_NAME}/slake/slake_answer.jsonl

# VQA-RAD
python ${PROJECT_PATH}/code/baselines/Med-LVLMs/llava-med-1.5/llava/eval/eval_batch.py --num-chunks 1  --model-name liuhaotian/llava-v1.5-7b \
    --question-file ${PROJECT_PATH}/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/rad_vqa_pairs.json \
    --image-folder ${PROJECT_PATH}/dataset/vqa_rad/VQA_RAD_image \
    --answers-file ${PROJECT_PATH}/runs/${CONFIG_NAME}/vqa_rad/vqa_rad_answer.jsonl

# IU-Xray
python ${PROJECT_PATH}/code/baselines/Med-LVLMs/llava-med-1.5/llava/eval/eval_batch.py --num-chunks 1  --model-name liuhaotian/llava-v1.5-7b \
    --question-file ${PROJECT_PATH}/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json \
    --image-folder ${PROJECT_PATH}/dataset/iu_xray/images \
    --answers-file ${PROJECT_PATH}/runs/${CONFIG_NAME}/xray/xray_answer.jsonl

cd ${CURRENT_PATH}

##########################
# Evaluation
##########################

# Slake
python ${PROJECT_PATH}/code/evaluation/close_ended_evaluation/eval_type1_single.py \
    --question-file ${PROJECT_PATH}/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/slake_qa_pairs.json \
    --prediction-file ${PROJECT_PATH}/runs/${CONFIG_NAME}/slake/slake_answer.jsonl \
    --output-file ${PROJECT_PATH}/runs/${CONFIG_NAME}/slake/slake_results.txt

# VQA-RAD
python ${PROJECT_PATH}/code/evaluation/close_ended_evaluation/eval_type1_single.py \
    --question-file ${PROJECT_PATH}/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/rad_vqa_pairs.json \
    --prediction-file ${PROJECT_PATH}/runs/${CONFIG_NAME}/vqa_rad/vqa_rad_answer.jsonl \
    --output-file ${PROJECT_PATH}/runs/${CONFIG_NAME}/vqa_rad/vqa_rad_results.txt

# IU-Xray
python ${PROJECT_PATH}/code/evaluation/close_ended_evaluation/eval_type1_single.py \
    --question-file ${PROJECT_PATH}/benchmark_data/Visual_Misinterpretation_Hallucination/close-ended/fine-grained/xray_closed_pairs.json \
    --prediction-file ${PROJECT_PATH}/runs/${CONFIG_NAME}/xray/xray_answer.jsonl \
    --output-file ${PROJECT_PATH}/runs/${CONFIG_NAME}/xray/xray_results.txt

# remove output files (optional)
rm test_type*.csv

##########################
# Merge Results
##########################

# TODO
