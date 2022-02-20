# Welcome

By: Group 4, Practical Course: Machine Learning and Computer Vision for HCI

# Running this project

Please install all required packages:

```bash
pip3 install -r requirements.txt
```

After that you are free to use `main.py` and to explore other runnable scripts.

Call for `VisQA/main.py` like this (adjust for your VisQA download location):

```bash
python3 VisQA/main.py
    --dataset_dir SOME/PATH/DATASET/VisQA/
```

or if you already have the dataset.csv provided you can skip preproccesing:

```bash
python3 VisQA/main.py
    --dataset_dir SOME/PATH/DATASET/VisQA/
    --dataset_csv SOME/PATH/DATASET/VisQA/dataset.csv
```

Dataset_dir is still necessary for later output.

Additionally, --analyze True will run some automatically runable analysis,
i.e. a call this if you want every output we can provide:

```bash
python3 VisQA/main.py
    --dataset_dir SOME/PATH/DATASET/VisQA/
    --analyze True
```

You can also specify all paths yourself: See below for options.