import h5py

file_path = "./data/VDISC_train.hdf5"  # 파일 경로 수정
with h5py.File(file_path, 'r') as f:
    print("HDF5 file contents:")
    f.visit(print)
