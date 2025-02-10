# Get current directory
CURRENT_DIR=$(pwd)

# Create new database folders by copying existing one in parallel
for ic in 1 3 6 10; do
    cp -r "$CURRENT_DIR/data/webshop" "$CURRENT_DIR/data/webshop_train_${ic}ic" &
done

# Wait for all copy operations to complete
wait

# Run training script for each number of in-context examples in parallel
for ic in 1 3 6 10; do 
    python scripts/run_agent_v2.py \
        --llm openai/gpt-4o-mini \
        --agent_type rap_flex \
        --db_path "$CURRENT_DIR/data/webshop_train_${ic}ic/learning.db" \
        --db_name webshop_train_${ic}ic \
        --num_passes 1 \
        --env webshop \
        --store_episodes \
        --num_ic $ic \
        --num_tasks 1000 &
done

# Wait for all training processes to complete
wait