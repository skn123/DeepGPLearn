# 2 stage Gaussian Process simulation by Mont Carlo

## Main folder:
- script: (TEST: An Experiment of a simulation; test: an experiments of a function)
    - TEST_data_generation: Generate synthetic data by known GPs
    - TEST_MAIN: sample by annealed Importance Sampling
    - TEST_SA: find peaks by Simulated Annealing, and then sample by M-H
    - TEST_plot_hiD: Try to plot high dimrensional posterior
    - TEST_Gibbs: sample by Gibbs sampling
    - test_congau: test conditional Gaussian distribution function
- functions
    - kfcn: Gaussian kernel function
    - pos_bond: convert range (-inf,inf) to [0,positive number]
    - logmvnpdf: log pdf of multivariabel Gaussian distribution
    - mvhist: plot histogram of high dimensional samples
    - resample: resample the same number of samples by  weights
    - Ly_Given_z: likelihood of z by observed output y
    - Pz_Given_x: probability of z given observed inpit x
    - my_fitrgp: Gaussian Process Regression by given kernel
    - congau: find parameters of conditional Gaussian distribution
- data file
    - data: generated data by TEST_data_generation
- figure
    - figure of TEST_MAIN
        - 6 IOs across 10 seconds
            - Pred_y_ALL_z: prediction of y using all samples
            - Pred_y_Half_z: prediction of y using samples that z1 > z2
        - 6 IOs across 5 seconds
            - Pred_y_ALL_z2: prediction of y using all samples
            - Pred_y_Half_z2: prediction of y using samples that z1 > z2
        - 6 IOs across 5 seconds, using all samples, but learn with fake kernels 
            - Pred_y_too_narrow: using kernels that are narrower than ground truth
            - Pred_y_too_wide: using kernels that are wider than ground truth
    - figure of TEST_Gibbs
        - gibbs6: 6 IOs across 5 seconds
        - gibbs10:  11 IOs across 10 seconds

## test the number of sections:
- script
    - test: find the growth of number of sections along with the groth of dimensions
- functions
    - section: recursive function to finding number of sections where d dimensional space divide by L hyperplanes
    - NS: find the number of sections in GP problem
    
## history: archieved files that does not have much use