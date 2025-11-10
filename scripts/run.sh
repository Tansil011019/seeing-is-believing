#!/bin/bash
set -e

WORKING_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Working directory: $WORKING_DIR"
cd "$WORKING_DIR"

if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "Using virtual environment at $VIRTUAL_ENV"
    source "$VIRTUAL_ENV/bin/activate"
elif [[ -f "./venvs/bin/activate" ]]; then   
    echo "Activating virtual environment from ./venvs"
    source "./venvs/bin/activate"
else
    echo "No virtual environment detected."
    exit 1
fi

ENTRYPOINT=""
EXTRA_ARGS=()

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "--help                    Display this help message"
    echo "--train                   Run in training mode"
    echo "Example:"
    echo "  $0 --train"
    exit 1
}

SHORT_OPTS="h"
LONG_OPTS="train,help"

PARSED_OPTS=$(getopt --options $SHORT_OPTS --longoptions $LONG_OPTS --name "$0" -- "$@")

eval set -- "$PARSED_OPTS"

if [[ $? -ne 0 ]]; then
    usage
fi

while [[ -n "$1" ]]; do
    case "$1" in
        --train)
            ENTRYPOINT="train"
            shift
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unexpected option: $1"
            usage
            ;;
    esac
done

CMD="python3 -m $ENTRYPOINT ${EXTRA_ARGS[*]}"
echo "Running command $CMD"
echo "-----------------------------------"
echo "Start training"
eval $CMD
echo "Training complete"
echo "-----------------------------------"