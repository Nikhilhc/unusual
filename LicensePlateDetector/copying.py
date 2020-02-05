import glob
import shutil
import os



#
alp = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
            'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'V', 'W', 'X', 'Y', 'Z'
        ]

for i in range(10,35):
    count = 0
    src_dir = r"E:\Nikhil\python\LicensePlateDetector\Fnt\{}".format(alp[i])
    print(os.listdir(src_dir))
    for image in os.listdir(src_dir):

        dst_dir = r"E:\Nikhil\python\LicensePlateDetector\train20X20\{}\{}_{}.jpg".format(alp[i],alp[i],count)

        shutil.copy(src_dir + '/' + image, dst_dir)
        count = count + 1
    i=i+1
'''
for jpgfile in glob.iglob(os.path.join(src_dir, "*.png")):
    shutil.copy(jpgfile, dst_dir)
'''
