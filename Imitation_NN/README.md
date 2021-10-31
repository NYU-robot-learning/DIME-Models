# Non Parametric Imitation using Nearest Neighbors

- This is a general template for INN.

## To use this model:

- Create a data folder with the following structure

```
Imitation_NN
│   README.md
│   non_parametric_imitation.py   
|
└───utils
│   │   dataloader.py
│   
└───data
    └───train
    |   │   |
    |   |   └───states
    |   │   |   │   ...
    |   │   └───actions
    |   │   |   │   ...
    |   
    └───test
    |   │   |
    |   |   └───states
    |   │   |   │   ...
    |   │   └───actions
    |   │   |   │   ...
```

- Pass the device and data path while initializing the model as parameters:
    ```
        INN = ImitationNearestNeighbors(device = "cuda", data_path = "./data")
    ```