export daytime=''
export expname=''
export test_data=''
export order=0
export label=-1
export HYDRA_FULL_ERROR=1
export output=''

python scripts/evaluate_flow.py --model_path ${output}/${daytime}/${expname} --num_evals 1 --label ${label} --order ${order} --test_data ${test_data}
python scripts/compute.py --pt_path  ${output}/${daytime}/${expname}/ckpt${label}_1_${order}.pt
python scripts/evaluate_flow.py --model_path ${output}/${daytime}/${expname} --num_evals 20 --label ${label} --order ${order} --test_data ${test_data}
python scripts/compute.py --pt_path  ${output}/${daytime}/${expname}/ckpt${label}_20_${order}.pt --multi_eval

