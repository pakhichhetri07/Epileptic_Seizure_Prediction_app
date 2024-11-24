#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process run_model {
    input:
    path signal_data
    path model_file

    output:
    path 'Model.pkl'

    script:
    """
    python3 ${model_file} ${signal_data}
    """
}

workflow {
    signal_data = file("Signal_Data.csv")
    model_file = file("src.py")

    run_model(signal_data, model_file)
}
