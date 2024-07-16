from easyocr.easyocr import *
import os


# GPU 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def get_files(path):
    file_list = []

    files = [f for f in os.listdir(path) if not f.startswith('.')]  # skip hidden file
    files.sort()
    abspath = os.path.abspath(path)
    for file in files:
        file_path = os.path.join(abspath, file)
        file_list.append(file_path)

    return file_list, len(file_list)


if __name__ == '__main__':

    reader = Reader(['ko'], gpu=True,
                    model_storage_directory='/home/prudent13/deep-text-recognition-benchmark/customset/user_network_dir',
                    user_network_directory='/home/prudent13/deep-text-recognition-benchmark/customset/user_network_dir',
                    recog_network='custom')

    files, count = get_files('/home/prudent13/deep-text-recognition-benchmark/demo_image')
    print('files, count ', files, count )

    for idx, file in enumerate(files):
        filename = os.path.basename(file)

        result = reader.readtext(file)

        # ./easyocr/utils.py 733 lines
        # result[0]: bbox
        # result[1]: string
        # result[2]: confidence
        for (bbox, string, confidence) in result:
            print("filename: '%s', confidence: %.4f, string: '%s'" % (filename, confidence, string))
            # print('bbox: ', bbox)


         