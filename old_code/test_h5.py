import os
import h5py

directories = [
    "/home/ubuntu/DeFINO/datasets/dataset_NS_batch1/samples",
    "/home/ubuntu/DeFINO/datasets/dataset_NS_batch2/samples",
    "/home/ubuntu/DeFINO/datasets/dataset_NS_rand_batch1/samples",
    "/home/ubuntu/DeFINO/datasets/dataset_NS_rand_batch2/samples",
    "/home/ubuntu/DeFINO/datasets/dataset_NS_rand_batch3/samples",
    "/home/ubuntu/DeFINO/datasets/dataset_NS_rand_batch4/samples",
]

sample_paths = []
for directory in directories:
    for file in os.listdir(directory):
        if file.endswith(".h5"):
            sample_paths.append(os.path.join(directory, file))

for path in sample_paths:
    try:
        with h5py.File(path, "r") as f:
            pass
    except Exception as e:
        print(f"Error loading {path}: {e}")
        
# scp -i ~/Downloads/Richard-Macbook.pem -r ./dataset_NS_batch1/samples ubuntu@138.2.226.141:/home/ubuntu/DeFINO/datasets/dataset_NS_batch1/
# scp -i ~/Downloads/Richard-Macbook.pem -r ./dataset_NS_batch2/samples ubuntu@138.2.226.141:/home/ubuntu/DeFINO/datasets/dataset_NS_batch2/
# scp -i ~/Downloads/Richard-Macbook.pem -r ./dataset_NS_rand_batch1/samples ubuntu@138.2.226.141:/home/ubuntu/DeFINO/datasets/dataset_NS_rand_batch1/
# scp -i ~/Downloads/Richard-Macbook.pem -r ./dataset_NS_rand_batch2/samples ubuntu@138.2.226.141:/home/ubuntu/DeFINO/datasets/dataset_NS_rand_batch2/
# scp -i ~/Downloads/Richard-Macbook.pem -r ./dataset_NS_rand_batch3/samples ubuntu@138.2.226.141:/home/ubuntu/DeFINO/datasets/dataset_NS_rand_batch3/
# scp -i ~/Downloads/Richard-Macbook.pem -r ./dataset_NS_rand_batch4/samples ubuntu@138.2.226.141:/home/ubuntu/DeFINO/datasets/dataset_NS_rand_batch4/
