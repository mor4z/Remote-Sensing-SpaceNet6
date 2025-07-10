import os
import random

def split_dataset(input_path, test_split = 0.1, val_split = 0.1, random = True):
    # Get all the files in the input path
    files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    # Shuffle the files randomly
    if (random == True):
        random.shuffle(files)

    # Calculate the number of files for each split
    total_files = len(files)
    train_files = int(total_files * (1-test_split-val_split))
    val_files = int(total_files * val_split)
    test_files = int(total_files * test_split)

    # Split the files into training, validation, and test sets
    train_set = files[:train_files]
    val_set = files[train_files:train_files+val_files]
    test_set = files[train_files+val_files:]

    # Write the file paths to separate text files
    print(f"La Current Working Directory del notebook è: {os.getcwd()}")
    write_to_file(train_set, 'train.txt', 'Remote-Sensing-SpaceNet6/split')
    write_to_file(val_set, 'val.txt', 'Remote-Sensing-SpaceNet6/split')
    write_to_file(test_set, 'test.txt', 'Remote-Sensing-SpaceNet6/split')

def write_to_file(file_list, output_file, output_dir):
    output_path = os.path.join(output_dir, output_file)
    print(output_path)
    with open(output_path, 'w') as f:
        for file in file_list:
            f.write('../' + file + '\n')

if __name__ == "__main__":
    # Create the actual splits
    input_path = 'data/train/AOI_11_Rotterdam/PS-RGB'
    split_dataset(input_path, test_split=0.1, val_split=0.1, random=False)