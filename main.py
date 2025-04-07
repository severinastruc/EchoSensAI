from src.data_loader import get_audio

DATASET_PATH = "./data/UrbanSound8K/"

paths_list, class_list, fold_number = get_audio(DATASET_PATH)

for i in range(10):
    print("Path: :", paths_list[i])
    print("class: :", class_list[i])
    print("fold: :", fold_number[i])
