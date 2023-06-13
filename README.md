# AI 

## How to use
To install all the dependecies first run `pip install -r requirements.txt`

### Testing the neural network
`python3 drawingArea.py`

### Creating and training a new neural network 
First create the dataset:
 ```
python3 data/numberGenerator.py 10000 1000
python3 data/mnistModifier.py
```

Once finished you can start the training by running : `python3 main.py`
If you want to customize the training here is the parameters that can be changed in `main.py`:
 1. The Hyper Parameters class as you wish and try other Activation/Cost functions
 2. The proportions of each DataSet to use at line 23
