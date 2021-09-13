from os import path
import shutil

file = open("cell_images/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data_labels.csv", 'r')
img = []
for line in file:
    img.append([line.rstrip().split(',')[0],line.rstrip().split(',')[1]])
file.close()

for i in img:
    source_path = "cell_images/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data/" + i[1]
    if path.exists(source_path):
        type = ""
        if i[0].find("hem")>(-1):
            type = "hem/"
        else:
            type = "all/"
        destination_path = "cell_images/C-NMC_Leukemia/validation_data/" + type + i[1]
        shutil.move(source_path, destination_path)
    else:
        print(f"img {i[1]} not found")
