# temporal_ViT
This is the develpment of a temporal ViT using the MNIST dataset.

## Setting up the environment
To set up the environment, you can use the following command:
```bash
source environments/setup_envs.sh
```

## The analysis
We download and process the MNIST dataset.
This involves performing a sliding rolling window over the images to create a temporal dataset.
We will then train a temporal ViT model on this dataset.

