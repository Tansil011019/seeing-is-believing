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
LONG_OPTS="train,help,train-rgan"

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
        --train-rgan)
            ENTRYPOINT="train_rgan"
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
echo "Start Experiment"
eval $CMD
echo "Experiment Complete"
echo "-----------------------------------"


# --- 🚀 EMAIL NOTIFICATION ---
# This path should point to your email python script
PYTHON_EMAIL_SCRIPT="/data/marthen/send_email.py"
PYTHON_EXE=$(which python3)

# Check if the notification script exists
if [ ! -f "$PYTHON_EMAIL_SCRIPT" ]; then
    echo "Notification script $PYTHON_EMAIL_SCRIPT not found. Skipping email."
    exit 0 # Exit script normally
fi

echo "Attempting to send email notification..."

# Prepare the email content
SUBJECT="✅ Wilson's Training Completed"
BODY="Start the training for the next model ASAP!"

# Call the python script. It will use the hardcoded credentials.
"$PYTHON_EXE" "$PYTHON_EMAIL_SCRIPT" "$SUBJECT" "$BODY"

if [ $? -eq 0 ]; then
    echo "Email notification sent."
else
    echo "ERROR: Email notification failed to send. Check logs above."
fi
