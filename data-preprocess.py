import os

in_f = open('/home/vicky/NetInterpret-Lite/dataset/val_256/categories_places365.txt', 'r')
out_f = open('/home/vicky/NetInterpret-Lite/dataset/val_256/categories_places365_hive.txt', 'w')
for line in in_f.readlines():
    line_list = ['Places365']
    line_list.extend(line.strip().split(' '))
    print(line_list)
    new_line = line_list[0] + '\t' + line_list[2] +  '\t' + line_list[1] + '\n'
    out_f.write(new_line)

