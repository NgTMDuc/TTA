import tarfile
import os
def extract_tar_file(file_path, destination_path):
    try:
        # Open the tar file
        with tarfile.open(file_path, 'r') as tar:
            # Extract all contents to the destination directory
            tar.extractall(path=destination_path)
        print(f"Extracted {file_path} to {destination_path}")
    except Exception as e:
        print(f"Error occurred: {e}")

#Target Directory
destination_directory = "/mnt/ducntm/DATA/ImageNet-C/"

root_data = "/mnt/ducntm/DATA"
all_files = ["blur.tar", 
            #  "digital.tar", 
            #  "extra.tar", 
            #  "noise.tar", 
            #  "weather.tar"
             ]

if __name__ == "__main__":
    for file in all_files:
        path = os.path.join(root_data, file)
        extract_tar_file(path, destination_directory)