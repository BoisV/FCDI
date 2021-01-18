from lib.dataset.utils import copyProcess, createDataset, make_txt_file, printClasses, copy


"""
准备数据集
"""

if __name__ == "__main__":
    root_path = './data/sp-society-camera-model-identification/train/train'
    save_path = './data/patches/phaseAp64s64/train_original'
    copy_path = './data/patches/phaseAp64s64/train'
    # 1. 从sp-society-camera-model-identification数据集中生成patch数据集
    createDataset(root=root_path, save_path=save_path,
                  patch_size=64, stride=64)
    make_txt_file(save_path)

    # 2. 打印相机模型
    printClasses(save_path)

    # 3. 从patch数据集取一定数量的patch作为新数据集
    copy(root=save_path, new_root=copy_path, number=100000)
    make_txt_file(copy_path)
