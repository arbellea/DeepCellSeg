import DataHandeling
import os
from datetime import datetime

ROOT_SAVE_DIR = '/newdisk/arbellea/DeepCellSegOut'
ROOT_DATA_DIR = '/HOME/Data/'


class ParamsBase(object):
    pass


class ParamsLSTM(ParamsBase):
    # Data and Data Provider
    root_data_dir = ROOT_DATA_DIR
    data_provider_class = DataHandeling.CSVSegReaderRandomLSTM
    # one_seg = False
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-01/')
    # image_size = (690, 628)
    one_seg = False
    norm = 2 ** 9
    data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-02/')
    image_size = (773, 739)
    # one_seg = True
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-01/')
    # image_size = (832, 992)
    # norm = 2**15
    train_csv_file = 'train_lstm.csv'
    val_csv_file = 'val_lstm.csv'
    train_crop_size = (128, 128)
    crops_per_image = 10

    num_train_data_threads = 4
    num_val_data_threads = 1
    train_q_capacity = 1000
    val_q_capacity = 1000
    train_q_min_after_dq = 100
    val_q_min_after_dq = 100
    data_format = 'NCHW'
    num_train_examples = None

    # Training Regime
    batch_size = 20
    num_iterations = 1000000
    learning_rate = 0.001
    skip_t_loss = [0, 1]
    class_weights = [0.2, 0.2, 0.6]

    # Validation
    validation_interval = 10

    # Loading Checkpoints
    load_checkpoint = True
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg/2017-10-25_230737/model_40000.ckpt'
    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg/2017-11-04_212207/model_140000.ckpt'
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg/2017-10-30_154659/model_60000.ckpt'
    dry_run = False
    # Saving Checkpoints
    experiment_name = 'LSTM_Seg'
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 10000
    save_checkpoint_every_N_hours = 3
    save_checkpoint_max_to_keep = 5

    # Tensorboard
    write_to_tb_interval = 10
    save_log_dir = ROOT_SAVE_DIR

    # Hardware
    gpu_id = 1
    profile = False

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
        'lstm_kout4': 256,
    }

    def __init__(self):
        self.train_data_provider = self.data_provider_class(filenames=[self.train_csv_file],
                                                            base_folder=self.data_base_folder,
                                                            image_size=self.image_size,
                                                            crop_size=self.train_crop_size,
                                                            crops_per_image=self.crops_per_image,
                                                            one_seg=self.one_seg,
                                                            num_threads=self.num_train_data_threads,
                                                            capacity=self.train_q_capacity,
                                                            min_after_dequeue=self.train_q_min_after_dq,
                                                            data_format=self.data_format,
                                                            num_examples=self.num_train_examples
                                                            )
        self.val_data_provider = self.data_provider_class(filenames=[self.train_csv_file],
                                                          base_folder=self.data_base_folder,
                                                          image_size=self.image_size,
                                                          crop_size=self.train_crop_size,
                                                          crops_per_image=self.crops_per_image,
                                                          one_seg=self.one_seg,
                                                          num_threads=self.num_val_data_threads,
                                                          capacity=self.val_q_capacity,
                                                          min_after_dequeue=self.val_q_min_after_dq,
                                                          data_format=self.data_format,
                                                          num_examples=None
                                                          )

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_log_dir = os.path.join(self.save_log_dir, self.experiment_name, now_string)
        self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.experiment_name, now_string)
        os.makedirs(self.experiment_log_dir, exist_ok=True)
        os.makedirs(self.experiment_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)


class ParamsBiLSTM(ParamsBase):
    # Data and Data Provider
    root_data_dir = ROOT_DATA_DIR
    data_provider_class = DataHandeling.CSVSegReaderRandomLSTM
    # one_seg = False
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-01/')
    # image_size = (690, 628)
    # one_seg = False
    # norm = 2 ** 9
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-02/')
    # image_size = (773, 739)
    one_seg = True
    data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-01/')
    image_size = (832, 992)
    norm = 2 ** 15
    train_csv_file = 'train_Bilstm.csv'
    val_csv_file = 'val_Bilstm.csv'
    # train_csv_file = 'train_lstm.csv'
    # val_csv_file = 'val_lstm.csv'
    train_crop_size = (128, 128)
    crops_per_image = 10
    # train_crop_size = (832, 992)
    # crops_per_image = 1

    num_train_data_threads = 4
    num_val_data_threads = 1
    train_q_capacity = 1000
    val_q_capacity = 1000
    train_q_min_after_dq = 100
    val_q_min_after_dq = 100
    data_format = 'NCHW'
    num_train_examples = None

    # Training Regime
    batch_size = 20
    num_iterations = 1000000
    learning_rate = 0.001
    skip_t_loss = [0, 1]
    class_weights = [0.2, 0.2, 0.6]

    # Validation
    validation_interval = 10

    # Loading Checkpoints
    load_checkpoint = False
    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiLSTM_Seg/2017-11-06_204529/model_23424.ckpt'
    dry_run = False
    # Saving Checkpoints
    experiment_name = 'NormBiLSTM_Seg'
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 10000
    save_checkpoint_every_N_hours = 3
    save_checkpoint_max_to_keep = 5

    # Tensorboard
    write_to_tb_interval = 10
    save_log_dir = ROOT_SAVE_DIR

    # Hardware
    gpu_id = 1
    profile = False

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
        'lstm_kout4': 256,
    }

    def __init__(self):
        self.train_data_provider = self.data_provider_class(filenames=[self.train_csv_file],
                                                            base_folder=self.data_base_folder,
                                                            image_size=self.image_size,
                                                            crop_size=self.train_crop_size,
                                                            crops_per_image=self.crops_per_image,
                                                            one_seg=self.one_seg,
                                                            num_threads=self.num_train_data_threads,
                                                            capacity=self.train_q_capacity,
                                                            min_after_dequeue=self.train_q_min_after_dq,
                                                            data_format=self.data_format,
                                                            num_examples=self.num_train_examples
                                                            )
        self.val_data_provider = self.data_provider_class(filenames=[self.train_csv_file],
                                                          base_folder=self.data_base_folder,
                                                          image_size=self.image_size,
                                                          crop_size=self.train_crop_size,
                                                          crops_per_image=self.crops_per_image,
                                                          one_seg=self.one_seg,
                                                          num_threads=self.num_val_data_threads,
                                                          capacity=self.val_q_capacity,
                                                          min_after_dequeue=self.val_q_min_after_dq,
                                                          data_format=self.data_format,
                                                          num_examples=None
                                                          )

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_log_dir = os.path.join(self.save_log_dir, self.experiment_name, now_string)
        self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.experiment_name, now_string)
        os.makedirs(self.experiment_log_dir, exist_ok=True)
        os.makedirs(self.experiment_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)


class ParamsBiGRU(ParamsBase):
    # Data and Data Provider
    root_data_dir = ROOT_DATA_DIR
    data_provider_class = DataHandeling.CSVSegReaderRandomLSTM
    one_seg = False
    data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-01/')
    image_size = (690, 628)
    norm = 2 ** 9
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-02/')
    # image_size = (773, 739)

    train_csv_file = 'train_lstm.csv'
    val_csv_file = 'val_lstm.csv'
    # one_seg = True
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-01/')
    # image_size = (832, 992)
    # norm = 2**15
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-02/')
    # image_size = (782, 1200)
    # norm = 2**15
    # train_csv_file = 'train_Bilstm.csv'
    # val_csv_file = 'val_Bilstm.csv'
    # train_csv_file = 'train_lstm.csv'
    # val_csv_file = 'val_lstm.csv'
    train_crop_size = (128, 128)
    crops_per_image = 1
    # train_crop_size = (832, 992)
    # crops_per_image = 1

    num_train_data_threads = 4
    num_val_data_threads = 1
    train_q_capacity = 1000
    val_q_capacity = 1000
    train_q_min_after_dq = 100
    val_q_min_after_dq = 100
    data_format = 'NCHW'
    num_train_examples = None

    # Training Regime
    batch_size = 4
    num_iterations = 1000000
    learning_rate = 0.001
    skip_t_loss = [0, 1]
    class_weights = [0.2, 0.2, 0.6]

    # Validation
    validation_interval = 10

    # Loading Checkpoints
    load_checkpoint = True
    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_144653/model_100000.ckpt'
    dry_run = False
    # Saving Checkpoints
    experiment_name = 'BiGRU_Seg'
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 5000
    save_checkpoint_every_N_hours = 12
    save_checkpoint_max_to_keep = 5

    # Tensorboard
    write_to_tb_interval = 10
    save_log_dir = ROOT_SAVE_DIR

    # Hardware
    gpu_id = 0
    profile = False

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
        'lstm_kout4': 256,
    }

    def __init__(self):
        self.train_data_provider = self.data_provider_class(filenames=[self.train_csv_file],
                                                            base_folder=self.data_base_folder,
                                                            image_size=self.image_size,
                                                            crop_size=self.train_crop_size,
                                                            crops_per_image=self.crops_per_image,
                                                            one_seg=self.one_seg,
                                                            num_threads=self.num_train_data_threads,
                                                            capacity=self.train_q_capacity,
                                                            min_after_dequeue=self.train_q_min_after_dq,
                                                            data_format=self.data_format,
                                                            num_examples=self.num_train_examples
                                                            )
        self.val_data_provider = self.data_provider_class(filenames=[self.train_csv_file],
                                                          base_folder=self.data_base_folder,
                                                          image_size=self.image_size,
                                                          crop_size=self.train_crop_size,
                                                          crops_per_image=self.crops_per_image,
                                                          one_seg=self.one_seg,
                                                          num_threads=self.num_val_data_threads,
                                                          capacity=self.val_q_capacity,
                                                          min_after_dequeue=self.val_q_min_after_dq,
                                                          data_format=self.data_format,
                                                          num_examples=None
                                                          )

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_log_dir = os.path.join(self.save_log_dir, self.experiment_name, now_string)
        self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.experiment_name, now_string)
        os.makedirs(self.experiment_log_dir, exist_ok=True)
        os.makedirs(self.experiment_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)


class ParamsEvalLSTM(ParamsBase):
    # Data and Data Provider
    root_data_dir = ROOT_DATA_DIR
    data_provider_class = DataHandeling.CSVSegReaderEvalLSTM
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-01/')
    # image_size = (718, 660)
    data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-02/')
    image_size = (790, 664)
    norm = 2 ** 9
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-01/')
    # image_size = (832, 992)
    # norm = 2**15

    csv_file = 'test_lstm.csv'

    num_data_threads = 4
    q_capacity = 1000
    data_format = 'NCHW'

    # Eval Regime
    seq_length = 1

    # Loading Checkpoints
    load_checkpoint = True
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg/2017-10-25_230737/model_100000.ckpt'
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg/2017-10-27_195111/model_90000.ckpt'
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg/2017-10-30_154659/model_60000.ckpt'
    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/LSTM_Seg/2017-11-03_170011/model_110000.ckpt'
    # Save Outputs
    dry_run = False
    experiment_name = 'LSTM_Seg'
    save_out_dir = ROOT_SAVE_DIR

    # Hardware
    gpu_id = 1

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

    def __init__(self):
        self.data_provider = self.data_provider_class(filenames=[self.csv_file],
                                                      base_folder=self.data_base_folder,
                                                      image_size=self.image_size,
                                                      num_threads=self.num_data_threads,
                                                      capacity=self.q_capacity,
                                                      data_format=self.data_format,
                                                      )

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_out_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string)
        os.makedirs(self.experiment_out_dir, exist_ok=True)


class ParamsEvalBiLSTM(ParamsBase):
    # Data and Data Provider
    root_data_dir = ROOT_DATA_DIR
    data_provider_class = DataHandeling.CSVSegReaderEvalBiLSTM
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-01/')
    # image_size = (718, 660)
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-02/')
    # image_size = (790, 664)
    # norm = 2**9
    data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-01/')
    image_size = (832, 992)
    norm = 2 ** 15

    csv_file = 'test_lstm.csv'
    # csv_file = 'test_train_lstm.csv'

    num_data_threads = 1
    q_capacity = 1000
    data_format = 'NCHW'

    # Eval Regime
    seq_length = 1

    # Loading Checkpoints
    load_checkpoint = True
    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiLSTM_Seg/2017-11-06_100311/model_88165.ckpt'
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiLSTM_Seg/2017-11-06_204529/model_20000.ckpt'
    # Save Outputs
    dry_run = False
    experiment_name = 'BiLSTM_Seg'
    save_out_dir = ROOT_SAVE_DIR

    # Hardware
    gpu_id = 2

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

    def __init__(self):
        self.data_provider = self.data_provider_class(filenames=[self.csv_file],
                                                      base_folder=self.data_base_folder,
                                                      image_size=self.image_size,
                                                      num_threads=self.num_data_threads,
                                                      capacity=self.q_capacity,
                                                      data_format=self.data_format,
                                                      )

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_out_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string)
        self.experiment_tmp_fw_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string, 'tmp',
                                                  'fw')
        self.experiment_tmp_bw_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string, 'tmp',
                                                  'bw')
        os.makedirs(self.experiment_out_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_fw_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_bw_dir, exist_ok=True)


class ParamsEvalBiGRU(ParamsBase):
    # Data and Data Provider
    root_data_dir = ROOT_DATA_DIR
    data_provider_class = DataHandeling.CSVSegReaderEvalBiLSTM
    data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-01/')
    image_size = (718, 660)
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-02/')
    # image_size = (790, 664)
    norm = 2 ** 9
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-01/')
    # image_size = (832, 992)
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-02/')
    # image_size = (782, 1200)
    # norm = 2**15

    csv_file = 'test_lstm.csv'
    # csv_file = 'test_train_lstm.csv'

    num_data_threads = 1
    q_capacity = 1000
    data_format = 'NCHW'

    # Eval Regime
    seq_length = 1

    # Loading Checkpoints
    load_checkpoint = True
    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_144653/model_210000.ckpt'  # SIM-01
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_144633/model_160000.ckpt' #SIM-02
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_145948/model_190000.ckpt' #MSC-02
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_150020/model_250000.ckpt' #MSC-01
    # Save Outputs
    dry_run = False
    experiment_name = 'BiGRU_Seg'
    save_out_dir = ROOT_SAVE_DIR
    final_out_dir = None

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

    def __init__(self):

        self._data_preps_()

    def _override_params_(self, **kwargs):
        for key, val in kwargs:
            setattr(self, key, val)

    def _data_preps_(self):
        self.data_provider = self.data_provider_class(filenames=[self.csv_file],
                                                      base_folder=self.data_base_folder,
                                                      image_size=self.image_size,
                                                      num_threads=self.num_data_threads,
                                                      capacity=self.q_capacity,
                                                      data_format=self.data_format,
                                                      )

        # os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_out_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string)
        self.experiment_tmp_fw_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string, 'tmp',
                                                  'fw')
        self.experiment_tmp_bw_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string, 'tmp',
                                                  'bw')
        if self.final_out_dir:
            self.experiment_isbi_out = self.final_out_dir
        else:

            self.experiment_isbi_out = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string,
                                                    'RES')

        os.makedirs(self.experiment_out_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_fw_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_bw_dir, exist_ok=True)
        os.makedirs(self.experiment_isbi_out, exist_ok=True)


class ParamsEvalIsbiBiGRU(ParamsBase):
    # Data and Data Provider

    root_data_dir = ROOT_DATA_DIR
    data_provider_class = DataHandeling.CSVSegReaderEvalBiLSTM
    data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-01/')
    image_size = (718, 660)
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-N2DH-SIM+-02/')
    # image_size = (790, 664)
    norm = 2 ** 9
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-01/')
    # image_size = (832, 992)
    # data_base_folder = os.path.join(ROOT_DATA_DIR, 'ISBI-Fluo-C2DL-MSC-02/')
    # image_size = (782, 1200)
    # norm = 2**15

    csv_file = 'test_lstm.csv'
    # csv_file = 'test_train_lstm.csv'

    num_data_threads = 1
    q_capacity = 1000
    data_format = 'NCHW'

    # Eval Regime
    seq_length = 1

    # Loading Checkpoints
    load_checkpoint = True
    load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_144653/model_210000.ckpt'  # SIM-01
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_144633/model_160000.ckpt' #SIM-02
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_145948/model_190000.ckpt' #MSC-02
    # load_checkpoint_path = '/newdisk/arbellea/DeepCellSegOut/BiGRU_Seg/2017-11-08_150020/model_250000.ckpt' #MSC-01
    # Save Outputs
    dry_run = False
    experiment_name = 'BiGRU_Seg'
    save_out_dir = ROOT_SAVE_DIR
    final_out_dir = None

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

    def __init__(self):

        self._data_preps_()

    def _override_params_(self, **kwargs):
        for key, val in kwargs:
            setattr(self, key, val)

    def _data_preps_(self):
        self.data_provider = self.data_provider_class(filenames=[self.csv_file],
                                                      base_folder=self.data_base_folder,
                                                      image_size=self.image_size,
                                                      num_threads=self.num_data_threads,
                                                      capacity=self.q_capacity,
                                                      data_format=self.data_format,
                                                      )

        # os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.experiment_out_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string)
        self.experiment_tmp_fw_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string, 'tmp',
                                                  'fw')
        self.experiment_tmp_bw_dir = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string, 'tmp',
                                                  'bw')
        if self.final_out_dir:
            self.experiment_isbi_out = self.final_out_dir
        else:

            self.experiment_isbi_out = os.path.join(self.save_out_dir, self.experiment_name, 'outputs', now_string,
                                                    'RES')

        os.makedirs(self.experiment_out_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_fw_dir, exist_ok=True)
        os.makedirs(self.experiment_tmp_bw_dir, exist_ok=True)
        os.makedirs(self.experiment_isbi_out, exist_ok=True)
