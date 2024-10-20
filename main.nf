//process
process run_model {
    input:
    path signal_data_file

    output:
    path "Model.pkl"

    script:
    """
    python3 /home/pakhi/PAKHI/Project/nextflow_pro/Model.py ${signal_data_file}
    """
}


// Workflow 
workflow {
    signal_data_file = file("/home/pakhi/PAKHI/Project/nextflow_pro/Signal_Data.csv")
    run_model(signal_data_file)
}

