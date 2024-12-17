#!/bin/bash

python=
config=
cluster=
resources=

handle_options() {
    while [ $# -gt 0 ]; do
        case $1 in
            -s | --stage)
                stage_name=$2
                ;;
            -m | --method)
                method=$2
                ;;
            -j | --jobmon)
                jobmon=1
                ;;
        esac
        shift
    done

    if [ -z "${method}" ]; then
        method=run
    fi
    if [ -z "${jobmon}" ]; then
        jobmon=0
    fi
}

handle_options "$@"

cmd="onemod --config $config --method $method"
if [ -n "${stage_name}" ]; then
    cmd="${cmd} --stage_name $stage_name --from_pipeline"
fi
if [ $jobmon -eq 1 ]; then
    cmd="${cmd} --backend jobmon --cluster $cluster --resources $resources"
fi

eval "source activate $python"
echo $cmd
eval $cmd
