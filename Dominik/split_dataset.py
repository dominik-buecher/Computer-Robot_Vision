import random

def split_dataset(input_datei, ausgabe_datei_1, ausgabe_datei_2, trennungsprozentsatz=5):
    with open(input_datei, 'r') as datei:
        zeilen = datei.readlines()
    
    random.shuffle(zeilen)
    
    trennungspunkt = int(len(zeilen) * (trennungsprozentsatz / 100))
    ausgabe_1 = zeilen[:trennungspunkt]
    ausgabe_2 = zeilen[trennungspunkt:]
    
    with open(ausgabe_datei_1, 'w') as datei_1:
        datei_1.writelines(ausgabe_1)
    
    with open(ausgabe_datei_2, 'w') as datei_2:
        datei_2.writelines(ausgabe_2)



if __name__ == "__main__":
    input_datei = r'Dominik\annotation_files\pos_own.txt'
    ausgabe_datei_1 = r'Dominik\annotation_files\test.txt'
    ausgabe_datei_2 = r'Dominik\annotation_files\train.txt'
    split_dataset(input_datei, ausgabe_datei_1, ausgabe_datei_2, trennungsprozentsatz=5)
