from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--which_epoch',
            type=str,
            default='vae_lr_0002_bs_64_scale_1_npoints_128_radius_02_latent_size_2_370best/370',
            help='which epoch to load? set to latest to use latest cached model'
        )

        self.is_train = False

