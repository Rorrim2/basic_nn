import struct


f_train_labels =  open('train-labels.idx1-ubyte', 'rb')
f_train_images =  open('train-images.idx3-ubyte', 'rb')

f_test_labels =  open('t10k-labels.idx1-ubyte', 'rb')
f_test_images =  open('t10k-images.idx3-ubyte', 'rb')

labels_magic_number = struct.unpack('>i', f_train_labels.read(4))[0]
images_magic_number = struct.unpack('>i', f_train_images.read(4))[0]

labels_number_of_labels = struct.unpack('>i', f_train_labels.read(4))[0]
images_number_of_labels = struct.unpack('>i', f_train_images.read(4))[0]


print(labels_magic_number)
print(images_magic_number)
print(labels_number_of_labels)
print(images_number_of_labels)

byte_s = f_train_labels.read(1)[0]
print(byte_s)

row_num = struct.unpack('>i', f_train_images.read(4))[0]
col_num = struct.unpack('>i', f_train_images.read(4))[0]

for i in range(row_num):
    for j in range(col_num):
        num = f_train_images.read(1)[0]
        num = round(num / 255.0, 3)
        print(num, end = ' ')
    print('')



f_train_labels.close()
f_train_images.close()

f_test_labels.close()
f_test_images.close()
