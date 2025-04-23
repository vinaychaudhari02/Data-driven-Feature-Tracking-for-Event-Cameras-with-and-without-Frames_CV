import h5py

from pathlib import Path

 

def inspect_h5_file(h5_file_path):

    """

    Inspect the structure of an H5 file and print its groups and datasets.

    :param h5_file_path: Path to the H5 file.

    """

    h5_file_path = Path(h5_file_path)

    if not h5_file_path.exists():

        print(f"[ERROR] File {h5_file_path} does not exist.")

        return

   

    with h5py.File(h5_file_path, 'r') as f:

        def print_structure(name, obj):

            indent = "  " * name.count('/')

            if isinstance(obj, h5py.Dataset):

                print(f"{indent}- Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")

            elif isinstance(obj, h5py.Group):

                print(f"{indent}- Group: {name}")

       

        print(f"Structure of {h5_file_path}:")

        f.visititems(print_structure)

 

if __name__ == "__main__":

    import argparse

 

    parser = argparse.ArgumentParser(description="Inspect H5 file structure.")

    parser.add_argument('--h5_file', type=str, required=True, help="Path to the H5 file to inspect.")

    args = parser.parse_args()

 

    inspect_h5_file(args.h5_file)