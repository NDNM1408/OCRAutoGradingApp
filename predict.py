from vietocr.model.transformerocr import VietOCR
from vietocr.model.vocab import Vocab
from vietocr.predict import Predictor
import torch
from PIL import Image


config = {'vocab': 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ-',
          'device': 'cpu',
          'seq_modeling': 'transformer',
          'transformer': {'d_model': 256,
                          'nhead': 8,
                          'num_encoder_layers': 6,
                          'num_decoder_layers': 6,
                          'dim_feedforward': 2048,
                          'max_seq_length': 1024,
                          'pos_dropout': 0.1,
                          'trans_dropout': 0.1},
          'optimizer': {'max_lr': 0.0005, 'pct_start': 0.1},
          'trainer': {'batch_size': 64,
                      'print_every': 200,
                      'valid_every': 3000,
                      'iters': 120000,
                      'export': './weights/transformerocr.pth',
                      'checkpoint': './checkpoint/transformerocr_checkpoint.pth',
                      'log': './train.log',
                      'metrics': 10000},
          'dataset': {'name': 'hw1',
                      'data_root': '/kaggle/input/my-data/new_train/new_train/',  # forder chứa ảnh training
                      # forder chứa ảnh muốn dự đoán
                      'data_test_root': "/kaggle/input/my-data/public_test_data/new_public_test/",
                      'train_annotation': 'training_gt.txt',
                      'valid_annotation': 'valid_gt.txt',
                      'image_height': 32,
                      'image_min_width': 32,
                      'image_max_width': 256},
          'dataloader': {'num_workers': 3, 'pin_memory': True},
          'aug': {'data_aug': True, 'masked_language_model': True},
          'predictor': {'beamsearch': False},
          'quiet': False,
          'pretrain': '',
          'weights': 'final1.pth',
          'backbone': 'vgg19_bn',
          'cnn': {'pretrained': True,
                  'ss': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                  'ks': [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]],
                  'hidden': 256},
          'create_data_set': True}


def build_model(config):
    vocab = Vocab(config['vocab'])
    device = config['device']

    model = VietOCR(len(vocab),
                    config['backbone'],
                    config['cnn'],
                    config['transformer'],
                    config['seq_modeling'])

    return model, vocab


if __name__ == '__main__':
    model, vocab = build_model(config=config)
    img = Image.open('test3.png')
    detector = Predictor(config=config)
    res = detector.predict(img=img)
    print(res)
