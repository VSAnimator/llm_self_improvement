#!/bin/bash
# Config loader script

# Default config paths
BASE_CONFIG="default"
ENV_CONFIG=""
AGENT_CONFIG=""
LLM_CONFIG=""
CUSTOM_CONFIG=""

# Process config arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --base)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --env)
      ENV_CONFIG="$2"
      shift 2
      ;;
    --agent)
      AGENT_CONFIG="$2"
      shift 2
      ;;
    --llm)
      LLM_CONFIG="$2"
      shift 2
      ;;
    --custom)
      CUSTOM_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown config option: $1"
      exit 1
      ;;
  esac
done

# Load configs in order (base → env → agent → llm → custom)
if [[ -n "$BASE_CONFIG" && -f "agent_configs/base/${BASE_CONFIG}.sh" ]]; then
  source "agent_configs/base/${BASE_CONFIG}.sh"
fi

if [[ -n "$ENV_CONFIG" && -f "agent_configs/env/${ENV_CONFIG}.sh" ]]; then
  source "agent_configs/env/${ENV_CONFIG}.sh"
fi

if [[ -n "$AGENT_CONFIG" && -f "agent_configs/agent/${AGENT_CONFIG}.sh" ]]; then
  source "agent_configs/agent/${AGENT_CONFIG}.sh"
fi

if [[ -n "$LLM_CONFIG" && -f "agent_configs/llm/${LLM_CONFIG}.sh" ]]; then
  source "agent_configs/llm/${LLM_CONFIG}.sh"
fi

if [[ -n "$CUSTOM_CONFIG" && -f "agent_configs/custom/${CUSTOM_CONFIG}.sh" ]]; then
  source "agent_configs/custom/${CUSTOM_CONFIG}.sh"
fi 