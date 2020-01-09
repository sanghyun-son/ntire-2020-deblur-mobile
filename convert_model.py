import argparse
from tensorflow import lite
import model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_h', type=int, defalut=720)
    parser.add_argument('--img_w', type=int, defalut=1280)
    parser.add_argument('--load_from', type=str, defalut='deblur.hdf5')
    parser.add_argument('--save_to', type=str, defalut='deblur.tflite')
    cfg = parser.parse_args()

    net = model.SRCNN(cfg.img_h, cfg.img_w)
    net.load_weights(cfg.load_from)

    converter = lite.TFLiteConverter.from_keras_model(net)
    lite_model = converter.convert()
    with open(cfg.save_to, 'wb') as f:
        f.write(lite_model)

if __name__ == '__main__':
    main()