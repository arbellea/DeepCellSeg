import DataHandeling
import os
from datetime import datetime
from typing import Tuple

ROOT_SAVE_DIR = '/newdisk/arbellea/DeepCellSegOut'
ROOT_DATA_DIR = '/HOME/Data/'
ISBI_DATA_ROOT = '/newdisk/arbellea/Data/ISBI'

TRAIN_IMAGE_SIZES = {
    # should be a dict of dicts. first key is data set name, second key seq_id and last val is
    # tuple of two ints
    'Fluo-N2DH-SIM+': {1: (690, 628), 2: (773, 739)},
    'Fluo-C2DL-MSC': {1: (832, 992), 2: (782, 1200)},
    'Fluo-N2DH-GOWT1': {1: (1024, 1024), 2: (1024, 1024)},
    'Fluo-N2DL-HeLa': {1: (700, 1100), 2: (700, 1100)},
    'DIC-C2DH-HeLa': {1: (512, 512), 2: (512, 512)},
    'PhC-C2DL-PSC': {1: (576, 720), 2: (576, 720)},
}
CHALLENGE_IMAGE_SIZES = {
    'Fluo-N2DH-SIM+': {1: (718, 660), 2: (790, 664)},
    'Fluo-C2DL-MSC': {1: (832, 992), 2: (782, 1200)},
    'Fluo-N2DH-GOWT1': {1: (1024, 1024), 2: (1024, 1024)},
    'Fluo-N2DL-HeLa': {1: (700, 1100), 2: (700, 1100)},
    'DIC-C2DH-HeLa': {1: (512, 512), 2: (512, 512)},
    'PhC-C2DL-PSC': {1: (576, 720), 2: (576, 720)},
}
FOV_VALUES = {
    'Fluo-N2DH-SIM+': 0,
    'Fluo-C2DL-MSC': 50,
    'Fluo-N2DH-GOWT1': 50,
    'Fluo-N2DL-HeLa': 25,
    'DIC-C2DH-HeLa': 50,
    'PhC-C2DL-PSC': 25,
}


class Sequence(object):
    def __init__(self, images_dir: str, file_format: str, image_size: Tuple[int, int], bitness=16):
        self.images_dir = images_dir
        self.file_format = file_format
        self.images_size = image_size
        self.bitness = bitness


class DataSet(object):
    def __init__(self, data_set: str, data_root_dir: str):
        self.name = data_set
        self.root_dir = data_root_dir
        self.seq_root_dir = os.path.join(data_root_dir, data_set)
        self.sequences = {}

    def add_sequence(self, seq_id: int, image_size: Tuple[int, int], file_format: str):
        sequence_string = '{0:0>2}'.format(seq_id)
        sequence_dir = os.path.join(self.seq_root_dir, sequence_string)
        seq = Sequence(sequence_dir, file_format, image_size)
        self.sequences[seq_id] = seq


class ParamBaseISBI(object):
    @staticmethod
    def _create_data_base(data_root=ISBI_DATA_ROOT, train_or_challenge='Challenge'):
        data_set_names = ['Fluo-N2DH-SIM+', 'Fluo-C2DL-MSC', 'Fluo-N2DH-GOWT1', 'Fluo-N2DL-HeLa', 'DIC-C2DH-HeLa',
                          'PhC-C2DL-PSC']

        data_sets = {}
        file_format = 't*.tif'
        for _data_set in data_set_names:

            data_sets[_data_set] = DataSet(_data_set, data_root)
            if train_or_challenge == 'Training':
                for seq_id, im_size in TRAIN_IMAGE_SIZES[_data_set].items():
                    data_sets[_data_set].add_sequence(seq_id, im_size, file_format)
            else:
                for seq_id, im_size in CHALLENGE_IMAGE_SIZES[_data_set].items():
                    data_sets[_data_set].add_sequence(seq_id, im_size, file_format)
        return data_sets


class ParamsEvalIsbiBiGRU(ParamBaseISBI):
    # Data and Data Provider

    data_provider_class = DataHandeling.DIRSegReaderEvalBiLSTM
    training_or_challlenge = 'Training'
    data_root_dir = ''
    selected_data_set = 'Fluo-N2DH-SIM+'
    selected_seq = 2
    norm = 2 ** 8
    # image_size = (718, 660)
    # # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-02/')
    # # image_size = (790, 664)
    # # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-01/')
    # # image_size = (832, 992)
    # # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-02/')
    # # image_size = (782, 1200)
    # norm = 2**15
    #
    q_capacity = 1000
    data_format = 'NCHW'

    # Eval Regime
    seq_length = 1

    # Loading Checkpoints
    load_checkpoint = True
    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_144653/model_210000.ckpt'  # SIM-01
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_144633/model_161925.ckpt' #SIM-02
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_145948/model_190000.ckpt' #MSC-02
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_150020/model_250000.ckpt' #MSC-01
    # Save Outputs
    dry_run = False
    experiment_name = 'BiGRU_Seg'
    save_out_dir = ROOT_SAVE_DIR
    final_out_dir = ISBI_DATA_ROOT

    # Hardware
    useGPU = True
    gpu_id = 0

    # Net Architecture
    net_params = {
        'conv_kxy': 3,
        'kout1': 32,
        'kout2': 64,
        'kout3': 128,
        'kout4': 256,
        'lstm_kxy': [7, 7],
        'lstm_kout1': 32,
        'lstm_kout2': 64,
        'lstm_kout3': 128,
        'lstm_kout4': 256
    }

    def __init__(self, params_dict: dict):
        self._override_params_(params_dict)
        self._data_preps_()

    def _override_params_(self, params_dict: dict):
        for key, val in params_dict.items():
            setattr(self, key, val)

    def _data_preps_(self):
        train_data_sets, challenge_data_sets = self._create_data_base()
        if self.training_or_challlenge == 'Training':
            selected_sequence = train_data_sets[self.selected_data_set].sequences[self.selected_seq]
        elif self.training_or_challlenge == 'Challenge':
            selected_sequence = challenge_data_sets[self.selected_data_set].sequences[self.selected_seq]

        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_out_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string)
        self.experiment_tmp_fw_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string, 'tmp',
                                                  'fw')
        self.experiment_tmp_bw_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string, 'tmp',
                                                  'bw')
        if self.final_out_dir:
            self.experiment_isbi_out = self.final_out_dir
            self.experiment_isbi_out = os.path.join(self.final_out_dir, self.training_or_challlenge,
                                                    self.selected_data_set, '{0:0>2}_RES'.format(self.selected_seq))
        else:

            self.experiment_isbi_out = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string,
                                                    '{0:0>2}_RES'.format(self.selected_seq))

        os.makedirs(self.experiment_out_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_fw_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_bw_dir, exist_ok=True)
        os.makedirs(self.experiment_isbi_out, exist_ok=True)

        train_data_sets, challenge_data_sets = self._create_data_base()
        if self.training_or_challlenge == 'training':
            selected_sequence = train_data_sets[self.selected_data_set].sequences[self.selected_seq]
        elif self.training_or_challlenge == 'challenge':
            selected_sequence = challenge_data_sets[self.selected_data_set].sequences[self.selected_seq]

        self.image_size = selected_sequence.images_size
        if '.tif' in selected_sequence.file_format:
            tmp_imgs_dir = os.path.join(self.experiment_out_dir, 'png_imgs')
            print('Converting Data from TIF to PNG')
            DataHandeling.tif2png_dir(data_dir=selected_sequence.images_dir, out_dir=tmp_imgs_dir,
                                      filename_format=selected_sequence.file_format)
            selected_sequence.images_dir = tmp_imgs_dir
            selected_sequence.file_format = selected_sequence.file_format.replace('.tif', '.png')

        self.data_provider = self.data_provider_class(data_dir=selected_sequence.images_dir,
                                                      filename_format=selected_sequence.file_format,
                                                      image_size=self.image_size,
                                                      capacity=self.q_capacity,
                                                      data_format=self.data_format,
                                                      )

        # os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)


class ParamsEvalIsbiLSTM(ParamBaseISBI):
    # Data and Data Provider

    data_provider_class = DataHandeling.CSVSegReaderSequenceEval
    # training_or_challlenge = 'Training'
    training_or_challlenge = 'Challenge'
    data_root_dir = os.path.join(ISBI_DATA_ROOT, training_or_challlenge)
    # selected_data_set = 'Fluo-N2DH-SIM+'
    # selected_data_set = 'Fluo-N2DH-GOWT1'
    # selected_data_set = 'Fluo-N2DL-HeLa'
    # selected_data_set = 'PhC-C2DL-PSC'
    # selected_data_set = 'Fluo-C2DL-MSC'
    selected_data_set ='DIC-C2DH-HeLa'
    selected_seq = 2
    multi_split = True

    norm = 2 ** 15
    renorm_factor = 1# 2**-1 # 0.0068#2**8.8/2**16 #0.0068
    q_capacity = 1000
    data_format = 'NCHW'

    min_cell_size = 100
    max_cell_size = 100000
    #FOR PSC:
    # min_cell_size = 20
    # max_cell_size = 1000

    # Eval Regime
    seq_length = 1



    # Loading Checkpoints
    load_checkpoint = True

    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/ISBI-Fluo-N2DH-GOWT1-01/2018-02-22_212805/model_120000.ckpt'
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/ISBI-PhC-C2DL-PSC-01/2018-02-24_203733/model_113500.ckpt'
    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/ISBI-DIC-C2DH-HeLa-01/2018-02-24_210007/model_104000.ckpt'
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/ISBI-Fluo-N2DL-HeLa-01/2018-02-24_205933/model_117500.ckpt'
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/ISBI-Fluo-C2DL-MSC-02/2018-01-29_114015/model_71400.ckpt' # MSC 02
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/2018-01-19_133810/model_48500.ckpt'  # SIM-01
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/2018-01-19_133827/model_48400.ckpt'  # SIM02
    # load_ch5eckpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/2018-01-13_202745/model_116500.ckpt'  # SIM-01
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/2018-01-14_233514/model_102600.ckpt'  # SIM02

    # Save Outputs
    dry_run = False
    experiment_name = 'LSTM_Seg_seq'
    save_out_dir = ROOT_SAVE_DIR
    final_out_dir = os.path.join(ISBI_DATA_ROOT, training_or_challlenge)

    # Hardware
    gpu_id = 2
    useGPU = True

    # Net Architecture
    net_params = {
        'conv_kxy': 3,
        'kout1': 32,
        'kout2': 32,
        'kout3': 64,
        'kout4': 128,
        'lstm_kxy': [5, 5],
        'lstm_kout1': 32,
        'lstm_kout2': 32,
        'lstm_kout3': 62,
        'lstm_kout4': 128,
    }

    def __init__(self, params_dict: dict):
        self.pad_x = 0
        self.pad_y = 0
        self._override_params_(params_dict)

        self._data_preps_()

    def _override_params_(self, params_dict: dict):
        for key, val in params_dict.items():
            setattr(self, key, val)

    def _data_preps_(self):
        if not self.useGPU:
            self.data_format = 'NHWC'
        data_sets = self._create_data_base(self.data_root_dir, self.training_or_challlenge)
        selected_sequence = data_sets[self.selected_data_set].sequences[self.selected_seq]

        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        run_info = '{0}-{1:02d}_{2}_{3}'.format(self.selected_data_set, self.selected_seq, self.training_or_challlenge,
                                                now_string)
        self.experiment_out_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs',
                                               run_info)
        self.experiment_tmp_vis_dir = os.path.join(self.experiment_out_dir, 'tmp',
                                                   'seg_vis')
        self.experiment_tmp_overlay_dir = os.path.join(self.experiment_out_dir, 'tmp',
                                                       'overlay_vis')
        if self.final_out_dir:

            self.experiment_isbi_out = os.path.join(self.final_out_dir,
                                                    self.selected_data_set, '{0:0>2}_RES'.format(self.selected_seq))
        else:

            self.experiment_isbi_out = os.path.join(self.experiment_out_dir,
                                                    '{0:0>2}_RES'.format(self.selected_seq))

        os.makedirs(self.experiment_out_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_vis_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_overlay_dir, exist_ok=True)
        os.makedirs(self.experiment_isbi_out, exist_ok=True)

        self.image_size = selected_sequence.images_size
        self.pad_y, self.pad_x = (0, 0)
        self.data_provider = self.data_provider_class(data_dir=selected_sequence.images_dir,
                                                      filename_format=selected_sequence.file_format,
                                                      image_size=self.image_size,
                                                      queue_capacity=self.q_capacity,
                                                      data_format=self.data_format,
                                                      )
        self.pad_y, self.pad_x = self.data_provider.pad_end
        self.image_size = (self.image_size[0] + self.pad_y + 32, self.image_size[1] + self.pad_x + 32)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        self.fov = FOV_VALUES[self.selected_data_set]


class ParamsEvalIsbiLSTM_Unet(ParamBaseISBI):
    # Data and Data Provider

    data_provider_class = DataHandeling.CSVSegReaderSequenceEval
    training_or_challlenge = 'Training'
    # training_or_challlenge = 'Challenge'
    data_root_dir = os.path.join(ISBI_DATA_ROOT, training_or_challlenge)
    # selected_data_set = 'Fluo-N2DH-SIM+'
    selected_data_set = 'Fluo-N2DH-GOWT1'
    # selected_data_set = 'Fluo-N2DL-HeLa'

    # selected_data_set = 'Fluo-C2DL-MSC'
    # selected_data_set ='DIC-C2DH-HeLa'
    selected_seq = 2

    norm = 2 ** 15
    renorm_factor = 2 ** 2  # 0.0068#2**8.8/2**16 #0.0068
    q_capacity = 1000
    data_format = 'NCHW'
    min_cell_size = 500
    max_cell_size = 100000

    # Loading Checkpoints
    load_checkpoint = True
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/ISBI-DIC-C2DH-HeLa-01/2018-02-05_122613/model_231600.ckpt'

    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/' \
                           'ISBI-Fluo-N2DH-SIM+-/2018-02-12_155105/model_87500.ckpt'  # MSC 01
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/ISBI-Fluo-C2DL-MSC-02/2018-01-29_114015/model_71400.ckpt' # MSC 02
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/2018-01-19_133810/model_48500.ckpt'  # SIM-01
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/2018-01-19_133827/model_48400.ckpt'  # SIM02
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/2018-01-13_202745/model_116500.ckpt'  # SIM-01
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg_seq/2018-01-14_233514/model_102600.ckpt'  # SIM02


    # Save Outputs
    dry_run = False
    experiment_name = 'LSTM_Seg_seq'
    save_out_dir = ROOT_SAVE_DIR
    final_out_dir = os.path.join(ISBI_DATA_ROOT, training_or_challlenge)

    # Hardware
    gpu_id = 2
    useGPU = True

    # Net Architecture
    net_params = {
        'conv_kxy': 3,
        'kout1': 32,
        'kout2': 32,
        'kout3': 64,
        'kout4': 128,
        'lstm_kxy': [5, 5],
        'lstm_kout1': 32,
        'lstm_kout2': 32,
        'lstm_kout3': 62,
        'lstm_kout4': 128,
    }
    # Eval Regime
    seq_length = 1

    def __init__(self, params_dict: dict):
        self.pad_x = 0
        self.pad_y = 0
        self._override_params_(params_dict)

        self._data_preps_()

    def _override_params_(self, params_dict: dict):
        for key, val in params_dict.items():
            setattr(self, key, val)

    def _data_preps_(self):
        if not self.useGPU:
            self.data_format = 'NHWC'
        data_sets = self._create_data_base(self.data_root_dir, self.training_or_challlenge)
        selected_sequence = data_sets[self.selected_data_set].sequences[self.selected_seq]

        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        run_info = '{0}-{1:02d}_{2}_{3}'.format(self.selected_data_set, self.selected_seq, self.training_or_challlenge,
                                                now_string)
        self.experiment_out_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs',
                                               run_info)
        self.experiment_tmp_vis_dir = os.path.join(self.experiment_out_dir, 'tmp',
                                                   'seg_vis')
        self.experiment_tmp_overlay_dir = os.path.join(self.experiment_out_dir, 'tmp',
                                                       'overlay_vis')
        if self.final_out_dir:

            self.experiment_isbi_out = os.path.join(self.final_out_dir,
                                                    self.selected_data_set, '{0:0>2}_RES'.format(self.selected_seq))
        else:

            self.experiment_isbi_out = os.path.join(self.experiment_out_dir,
                                                    '{0:0>2}_RES'.format(self.selected_seq))

        os.makedirs(self.experiment_out_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_vis_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_overlay_dir, exist_ok=True)
        os.makedirs(self.experiment_isbi_out, exist_ok=True)

        self.image_size = selected_sequence.images_size
        self.pad_y, self.pad_x = (0, 0)
        self.data_provider = self.data_provider_class(data_dir=selected_sequence.images_dir,
                                                      filename_format=selected_sequence.file_format,
                                                      image_size=self.image_size,
                                                      queue_capacity=self.q_capacity,
                                                      data_format=self.data_format,
                                                      )
        self.image_size = (self.image_size[0] + self.pad_y + 32, self.image_size[1] + self.pad_x + 32)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
