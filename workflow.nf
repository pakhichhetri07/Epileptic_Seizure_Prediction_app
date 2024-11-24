#!/usr/bin/env nextflow

nextflow.enable.dsl=2

process run_model {
    input:
    path signal_data   // Input EEG data file
    path model_file    // Input Python script for model training

    // Define outputs
    output:
    path 'Model.pkl'   // Trained model output file

    // Command to run the Python script
    script:
    """
    python3 ${model_file} ${signal_data}
    """
}

// Workflow definition
workflow {
    signal_data = file("Signal_Data.csv")  
    model_file = file("src.py")            

    run_model(signal_data, model_file)
}

