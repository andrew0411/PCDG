import torch
import torch.nn as nn
import torch.nn.functional as F

from domainbed.optimizers import get_optimizer
from domainbed.networks.ur_networks import URFeaturizer
from domainbed.lib import misc
from domainbed.algorithms import Algorithm

class ForwardModel(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return self.predict(x)

    def predict(self, x):
        return self.network(x)


class Decoder(nn.Module):
  def __init__(self):
    super(Decoder, self).__init__()
    
    # Decoder
    self.dec_cnn1 = nn.Sequential(
                    nn.ConvTranspose2d(256, 64, kernel_size = 4, stride = 2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU())
    
    self.dec_cnn2 = nn.Sequential(
                    nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding=1),
                    nn.Sigmoid())
          
          
  def forward(self, x):

      output = self.dec_cnn1(x)
      output = self.dec_cnn2(output)


      return output



class PCDG(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.ld = hparams.ld_mif
        self.aux_network = Decoder()

        parameters = [
            {'params': self.network.parameters()},
            {'params': self.aux_network.parameters()}
        ]
        self.optimizer = get_optimizer(
            hparams['optimizer'],
            parameters,
            lr=self.hparams['lr'],
            weight_decay = self.hparams['weight_decay']
        )

    def update(self, x, y, **kwargs):

        all_x = torch.cat(x)
        all_y = torch.cat(y)
        feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        logit = self.classifier(feat)
        loss = F.cross_entropy(logit, all_y)

        # Phase Consistent
        _, all_pha = self.extract(all_x)
        pred_pha = self.aux_network(inter_feats[1])
        reg_loss = F.mse_loss(pred_pha, all_pha)

        loss += reg_loss * self.ld


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model


    def extract(self, batch):
        # batch : [96, 3, 224, 224]
        x_fft = torch.rfft(batch, signal_ndim=2, onesided=False)
        fft_amp = x_fft[:, :, :, :, 0]**2 + x_fft[:, :, :, :, 1]**2
        fft_amp = torch.sqrt(fft_amp)
        fft_pha = torch.atan2(x_fft[:, :, :, :, 1], x_fft[:, :, :, :, 0])

        # fft_amp, fft_pha = [96, 3, 224, 224]
        return fft_amp, fft_pha
