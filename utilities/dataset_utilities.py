import numpy as np
import os
import pickle
import torch

def load_data_3d2d_modelnet40(data_folder, dataset_split, preprocessed=True):
    if preprocessed:
        var_name_list = ["p2d", "p3d", "R_gt", "t_gt", "W_gt", "num_points_2d", "num_points_3d"]
        subfolder = 'preprocessed'
        encoding='ASCII'
    else:
        var_name_list = ["matches", "R", "t"]
        subfolder = ''
        encoding='latin1' # Python2
    data = {}
    if dataset_split == "train":
        cur_folder = "/".join([data_folder,'modelnet40_train',subfolder])
        for var_name in var_name_list:
            in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp, encoding=encoding)
                else:
                    data[var_name] = pickle.load(ifp, encoding=encoding)
    elif dataset_split == "valid":
        cur_folder = "/".join([data_folder,'modelnet40_test',subfolder])
        for var_name in var_name_list:
            in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp, encoding=encoding)
                else:
                    data[var_name] = pickle.load(ifp, encoding=encoding)
    return data

def load_data_3d2d_data61_2d3d(data_folder, dataset_split, preprocessed=True):
    # preprocessed only
    var_name_list = ["p2d", "p3d", "R_gt", "t_gt", "W_gt", "num_points_2d", "num_points_3d"]
    subfolder = 'preprocessed'
    encoding='ASCII'
    data = {}
    if dataset_split == "train":
        cur_folder = "/".join([data_folder,'data61_2d3d_train',subfolder])
        for var_name in var_name_list:
            in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp, encoding=encoding)
                else:
                    data[var_name] = pickle.load(ifp, encoding=encoding)
    elif dataset_split == "valid":
        cur_folder = "/".join([data_folder,'data61_2d3d_test',subfolder])
        for var_name in var_name_list:
            in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp, encoding=encoding)
                else:
                    data[var_name] = pickle.load(ifp, encoding=encoding)
    return data

def load_data_3d2d_nyu_nonoverlap(data_folder, dataset_split, preprocessed=True):
    var_name_list = ["matches", "R", "t"]
    # Let's unpickle and save data
    data = {}
    if dataset_split == "train":
        cur_folder = "/".join([data_folder,'nyu_train'])
        for var_name in var_name_list:
            in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp, encoding='latin1')
                else:
                    data[var_name] = pickle.load(ifp, encoding='latin1')
    elif dataset_split == "valid":
        cur_folder = "/".join([data_folder,'nyu_test'])
        for var_name in var_name_list:
            in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp, encoding='latin1')
                else:
                    data[var_name] = pickle.load(ifp, encoding='latin1')
    return data

def load_data_3d2d_megadepth(data_folder, dataset_split, preprocessed=True):
    if preprocessed:
        var_name_list = ["p2d", "p3d", "R_gt", "t_gt", "W_gt", "num_points_2d", "num_points_3d"]
        subfolder = 'preprocessed'
        encoding='ASCII'
        data = {}
        if dataset_split == "train":
            cur_folder = "/".join([data_folder,'megadepth_train',subfolder])
            for var_name in var_name_list:
                in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
                with open(in_file_name, "rb") as ifp:
                    if var_name in data:
                        data[var_name] += pickle.load(ifp, encoding=encoding)
                    else:
                        data[var_name] = pickle.load(ifp, encoding=encoding)
        elif dataset_split == "valid":
            cur_folder = "/".join([data_folder,'megadepth_test',subfolder])
            for var_name in var_name_list:
                in_file_name = os.path.join(cur_folder, var_name) + ".pkl"
                with open(in_file_name, "rb") as ifp:
                    if var_name in data:
                        data[var_name] += pickle.load(ifp, encoding=encoding)
                    else:
                        data[var_name] = pickle.load(ifp, encoding=encoding)
    else:
        var_name_list = ["matches", "R", "t"]
        subfolder = ''
        encoding='latin1' # Python2
        # check whether the split file is available
        data_split = data_folder + "/megadepth_split_SIFT.pkl"
        trainValTestSplit = {}
        if os.path.isfile(data_split):
            with open(data_split, "rb") as ifp:
                trainValTestSplit = pickle.load(ifp)
        else:
            print("the train/test split file is missing!")
        data = {}
        if dataset_split == "train":
            for l in trainValTestSplit["train"]:
                cur_folder = "/".join([data_folder, 'megadepth_train', l[:-4]])
                for var_name in var_name_list:
                    cur_var_name = var_name
                    in_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"
                    if os.path.isfile(in_file_name):
                        with open(in_file_name, "rb") as ifp:
                            if var_name in data:
                                data[var_name] += pickle.load(ifp, encoding='latin1')
                            else:
                                data[var_name] = pickle.load(ifp, encoding='latin1')
                    else:
                        print("Could not find {}".format(in_file_name))
        elif dataset_split == "valid":
            for l in trainValTestSplit["val"]:
                cur_folder = "/".join([data_folder, 'megadepth_test', l[:-4]])
                for var_name in var_name_list:
                    cur_var_name = var_name
                    in_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"
                    if os.path.isfile(in_file_name):
                        with open(in_file_name, "rb") as ifp:
                            if var_name in data:
                                data[var_name] += pickle.load(ifp, encoding='latin1')
                            else:
                                data[var_name] = pickle.load(ifp, encoding='latin1')
                    else:
                        print("Could not find {}".format(in_file_name))
    return data

class Dataset(torch.utils.data.Dataset):
  def __init__(self, dataset_split, args, batch_size, preprocessed=True):
    self.batch_size = batch_size

    if args.dataset == "modelnet40":
        self.data = load_data_3d2d_modelnet40(args.data_dir, dataset_split, preprocessed)
    elif args.dataset == "nyurgbd":
        self.data = load_data_3d2d_nyu_nonoverlap(args.data_dir, dataset_split, preprocessed)
    elif args.dataset == "megadepth":
        self.data = load_data_3d2d_megadepth(args.data_dir, dataset_split, preprocessed)
    elif args.dataset == "data61_2d3d":
        self.data = load_data_3d2d_data61_2d3d(args.data_dir, dataset_split, preprocessed)
        preprocessed = True # only preprocessed data
        # Note: W_gt is inverted for this dataset (ie an m-vector)
    else:
        raise Exception("Unknown dataset")

    if preprocessed:
        self.len = len(self.data["t_gt"])
    else:
        self.len = len(self.data["t"])

        self.data["R_gt"] = self.data.pop("R") # Change name
        self.data["t_gt"] = self.data.pop("t") # Change name
        self.data["p2d"] = {}
        self.data["p3d"] = {}
        self.data["W_gt"] = {}
        self.data["num_points_2d"] = {}
        self.data["num_points_3d"] = {}
        matches = self.data.pop("matches")

        # Loop over samples:
        for i in range(len(matches)):
            print('{} of {}'.format(i+1, len(matches)))
            p2d3d = matches[i]
            repeat_times = 1
            if dataset_split == "train":
                # Force every point-set to have args.num_points_train:
                if p2d3d.shape[0] > args.num_points_train:
                    # Randomly select correspondences to keep:
                    idx = np.random.randint(p2d3d.shape[0], size=args.num_points_train)
                    p2d3d = p2d3d[idx, :]
                elif p2d3d.shape[0] < args.num_points_train:
                    # Determine how many times to replicate correspondences
                    repeat_times = args.num_points_train // p2d3d.shape[0] + 1
            # Split matches into 2D and 3D coordinates:
            p2d = p2d3d[:, :2]
            p3d = p2d3d[:, 2:]
            num_points_2d = p2d.shape[0]
            num_points_3d = p3d.shape[0]
            # Randomise p3d and save correspondences:
            W_gt = np.random.permutation(p3d.shape[0])
            p3d = p3d[W_gt, :]
            # Replicate correspondences:
            # This is for batch feature extraction only,
            # duplicates are ignored by the rest of the system
            if repeat_times > 1:
                p2d = np.tile(p2d, (repeat_times, 1))
                p2d = p2d[:args.num_points_train, :]
                p3d = np.tile(p3d, (repeat_times, 1))
                p3d = p3d[:args.num_points_train, :]
                W_gt = np.tile(W_gt, (repeat_times))
                W_gt = W_gt[:args.num_points_train]
            # Store:
            self.data["p2d"][i] = p2d
            self.data["p3d"][i] = p3d
            self.data["W_gt"][i] = W_gt
            self.data["num_points_2d"][i] = np.array([num_points_2d])
            self.data["num_points_3d"][i] = np.array([num_points_3d])

        # Save pickle:
        for var_name in self.data:
            print(var_name, len(self.data[var_name]), self.data[var_name][8].shape)
            in_file_name = var_name + ".pkl"
            with open(in_file_name, "wb") as ifp:
                pickle.dump(self.data[var_name], ifp)

  def __getitem__(self, index):
    return self.data["p2d"][index], self.data["p3d"][index], self.data["R_gt"][index], self.data["t_gt"][index], self.data["W_gt"][index], self.data["num_points_2d"][index], self.data["num_points_3d"][index]
    
  def __len__(self):
    return self.len
