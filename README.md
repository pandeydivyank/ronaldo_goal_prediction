# Ronaldo Goal Prediction

A model that predicts if, given certain circumstances, Ronaldo will be able to score a goal.

Code Flow:

                            Raw Data
                                |
                          Import Export (Import Data)
                                |
                         Data Engineering
                                |
                       Feature Engineering
                                |
        Feature Selection       |
                |               |
                |_______________|
                                |
    Dimensionality Reduction    |                            Model Hyperparameter Tuning
                |               |                                        |
                |_______________|                                        |
                                |                                        |
                        Dataset Generation (Processed Data)     Model Architectures
                                        |        |       ________________|
                                        |        |      |
                                        |         --> Model
                                        |               |     Training
                                        |               |________|
                                        |               |
                                        |__             |     Testing
                                           |            |________|
            Import Export (Export Results) |            |
                          |_____________   |  __________|
                                        |  | |
                                        Main (run)

