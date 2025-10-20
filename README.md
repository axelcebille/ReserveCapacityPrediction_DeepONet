# ReserveCapacityPrediction_DeepONet
Predict reserve capacity of steel columns based on their geometrical properties and deformated shape using DeepONet framework, with the additional help of a GNN as the branch net..


# Structure

- __data.py__: data class to load and preprocess the raw data. (not my code, Sergei Garmaev's)
  
- __dataConfig.py__: class to configurate the raw input data's parameters. (path, etc...) (not my code, Sergei Garmaev's)
  
- __dataset.py__: dataset class, with _init_, _len_, _getitem_, to process and return the data in the right format: gives graph data, trunk input and targets.
  
- __models.py__: contains all the models class used --> _GNN_ class for the branch network, _DeepONet_ class for the full model.

- __main.py__: contains the main script with the data loading and the training loop. I separated the data into train, id-val, ood-test. 

  
