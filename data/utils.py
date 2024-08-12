import os
import argparse
import shutil


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task to run')
    parser.add_argument('--data_dir', type=str, help='Data directory')
    args = parser.parse_args()

    if args.task == 'rearrange_images':
        # rearrange images into class folders that comply with ImageFolder dataset
        # First, read all files in the data directory
        data_dir = args.data_dir
        sub_dirs = [dir for dir in os.listdir(data_dir) if dir != '.DS_Store']
        classes = set([f.split('_')[0] for f in os.listdir(os.path.join(data_dir, sub_dirs[0]))])
        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(f"{data_dir}-folders", f'{sub_dir}'), 
                        exist_ok=True)
            for class_ in classes:
                os.makedirs(os.path.join(f"{data_dir}-folders", f'{sub_dir}', f'{class_}'), 
                            exist_ok=True)

        
        for sub_dir in sub_dirs:
            files = os.listdir(os.path.join(data_dir, sub_dir))
            for file in files:
                class_ = file.split('_')[0]
                # copy file to sub_dir
                shutil.copy(os.path.join(data_dir, sub_dir, file), 
                            os.path.join(f"{data_dir}-folders", sub_dir, class_, file))            


