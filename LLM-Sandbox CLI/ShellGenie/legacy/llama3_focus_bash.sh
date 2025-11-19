#!/bin/bash

function shellgenie() {
    local prompt="$1"
    local model_path="$2"   # Optional path to Llama3 model (required for llama.cpp)

    if [[ -z "$prompt" ]]; then
        echo "Usage: shellgenie 'your command in natural language' [model_path]"
        return
    fi

    if [[ -z "$model_path" ]]; then  # Use Ollama by default
        local command=$(ollama run llama2 "$prompt") 
    else  # Use llama.cpp
        # Replace with actual llama.cpp execution command
        local command=$(llama_cpp/main -m "$model_path" -p "$prompt")  
    fi

    # Safety Check (add your own logic)
    if [[ "$command" =~ (rm|sudo|dangerous_command) ]]; then
        echo "Potentially dangerous command detected. Please review:"
    fi

    echo "Generated command:"
    echo "$command"
    read -p "Execute? (y/n): " confirm
    if [[ "$confirm" == "y" ]]; then
        eval "$command"
    fi
}
