# Get current directory
CURRENT_DIR=$(pwd)

# Run test script
for ic in 6; do
    for trial in 5; do
        for ckpt in 3500; do
            python scripts/run_agent_v2.py \
                --llm openai/gpt-4o-mini \
                --agent_type rap_flex \
                --db_path "$CURRENT_DIR/data/alfworld_pbt_6ic_seg20_zero_trial_${trial}_backups/${ckpt}/learning.db/learning.db" \
                --log_name pbt_zero_trial_${trial}_${ic}ic_${ckpt} \
                --num_passes 1 \
                --env alfworld_test \
                --num_ic $ic \
                --parallel 5 \
                --num_tasks 134 &
            sleep 5
        done
        wait
    done
done