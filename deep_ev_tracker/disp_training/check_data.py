import h5py
import yaml
import os

def check_sequence(sequence_dir):
    print(f"\nChecking sequence directory: {sequence_dir}")
    
    # Check rectified_data.h5
    data_path = os.path.join(sequence_dir, 'rectified_data.h5')
    print("\nChecking rectified_data.h5:")
    with h5py.File(data_path, 'r') as f:
        print("Available datasets:", list(f.keys()))
        if 'evleft_sbtmax' in f:
            print("evleft_sbtmax shape:", f['evleft_sbtmax'].shape)
        if 'ovcleft' in f:
            print("ovcleft shape:", f['ovcleft'].shape)
    
    # Check rectified_tracks.h5
    tracks_path = os.path.join(sequence_dir, 'rectified_tracks.h5')
    print("\nChecking rectified_tracks.h5:")
    with h5py.File(tracks_path, 'r') as f:
        print("Available timestamps:", list(f.keys())[:5], "...")
        if len(f.keys()) > 0:
            first_timestamp = list(f.keys())[0]
            print(f"\nContents of first timestamp ({first_timestamp}):")
            print("Available datasets:", list(f[first_timestamp].keys()))
            for key in f[first_timestamp].keys():
                print(f"{key} shape:", f[first_timestamp][key].shape)

if __name__ == "__main__":
    sequence_dir = r"F:\multiflow\M3ED_output\car_urban_day_horse"
    check_sequence(sequence_dir) 