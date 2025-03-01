# Get current directory
CURRENT_DIR=$(pwd)

# Migrate to chroma
python data/transform_chroma.py /mnt/ssd/agent_algo_bench/data/alfworld_expel_3ic_backups/2400/learning.db/learning.db /mnt/ssd/agent_algo_bench/data/alfworld_expel_3ic_backups/2400/learning.db/

# Run test script
for test_ic in 3 6; do
    for ckpt in 2400; do
        python scripts/run_agent_v2.py \
            --llm openai/gpt-4o-mini \
            --agent_type rap_flex \
            --db_path "$CURRENT_DIR/data/alfworld_expel_3ic_backups/${ckpt}/learning.db/" \
            --log_name expel_test_3ic_${test_ic}ic_chroma \
            --num_passes 1 \
            --env alfworld_test \
            --num_ic $test_ic \
            --db_type chroma \
            --num_tasks 134 &
    done
    wait
done