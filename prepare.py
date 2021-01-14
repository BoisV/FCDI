from lib.dataset.utils import createDataset, make_txt_file, printClasses


"""
准备数据集
"""
if __name__ == "__main__":
    root = '../../data/sp-society-camera-model-identification/train/train'
    save_path = './data/patches/phaseA/train'
    # createDataset(root=root, save_path=save_path, patch_size=256)
    make_txt_file(save_path)
    printClasses(save_path)
