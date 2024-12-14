#!/bin/bash

submit_job() {
    local THREADS=$1
    local INPUT_SIZE=$2
    local TIME_LIMIT=$3
    local INPUT_FILE="input_${INPUT_SIZE}_${THREADS}.txt"
    local OUTPUT_FILE="result_${INPUT_SIZE}_${THREADS}.out"

    if [ "$THREADS" -eq 32 ]; then
        local CORE_AFFINITY="affinity[core($((THREADS / 2)))]"
    else
        local CORE_AFFINITY="affinity[core($THREADS)]"
    fi

    echo "Submitting job: THREADS=$THREADS, INPUT_SIZE=$INPUT_SIZE, TIME_LIMIT=$TIME_LIMIT"
    bsub -n "$THREADS" -W "$TIME_LIMIT" -o "$OUTPUT_FILE" -R "$CORE_AFFINITY" \
        OMP_NUM_THREADS="$THREADS" /polusfs/1sf/openmp/launchOpenMP.py "$INPUT_FILE"
}

# Time limits in HH:MM format
TIME_LIMIT_SMALL="00:01"
TIME_LIMIT_MEDIUM="00:05"
TIME_LIMIT_LARGE="00:10"
TIME_LIMIT_XLARGE="00:15"

for INPUT_SIZE in 99 316 999 3162; do
    case $INPUT_SIZE in
        100)
            TIME_LIMIT=$TIME_LIMIT_SMALL
            ;;
        500)
            TIME_LIMIT=$TIME_LIMIT_MEDIUM
            ;;
        1000)
            TIME_LIMIT=$TIME_LIMIT_LARGE
            ;;
        3000)
            TIME_LIMIT=$TIME_LIMIT_XLARGE
            ;;
    esac

    for THREADS in 1 2 4 8 16 32; do
        submit_job "$THREADS" "$INPUT_SIZE" "$TIME_LIMIT"
    done
done