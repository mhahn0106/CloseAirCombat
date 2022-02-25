#!/bin/sh
env="gym"
scenario="sumo-ants-v0"   
algo="ppo"
exp="fsp"
seed=2021

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
python render/render_gym.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --model-dir "D:\jax\code\CloseAirCombat\run_ant"\
    --act-hidden-size "" \
    --seed ${seed} \
    --use-selfplay --selfplay-algorithm "fsp" --use-eval --render-opponent-index "0"