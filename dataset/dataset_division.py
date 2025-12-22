from datasets import load_dataset

# Load dataset from the hub
dataset = load_dataset("json", data_files="training_data_advanced.json", split="train")

# rename columns
dataset = dataset.rename_column("text1", "anchor")
dataset = dataset.rename_column("text2", "positive")

 
# Add an id column to the dataset
dataset = dataset.add_column("id", list(range(len(dataset))))
 
# split dataset into a 10% test set
dataset = dataset.train_test_split(test_size=0.1)
 
# save datasets to disk
dataset["train"].to_json("train_dataset.json", orient="records")
dataset["test"].to_json("test_dataset.json", orient="records")