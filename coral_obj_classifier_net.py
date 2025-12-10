import torch
import torch.nn as nn
import torch.nn.functional as F

class Boost_Classifier(nn.Module):
    def __init__(self, latent_dim, num_classes, main_spectra):
        super(Boost_Classifier, self).__init__()
        
        self.main_spectra = main_spectra
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            nn.Conv2d(main_spectra.shape[0], self.latent_dim, kernel_size=3, padding=1),#3
            nn.LeakyReLU(),

            nn.Dropout(0.5),
            nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),#5
            nn.LeakyReLU(),
        )

        self.net2 = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.latent_dim, num_classes),
        )

    def forward(self, spectrum, return_center_only=True):
        '''
            input x shape: (batch_size, latent_dim)
        '''
        batch_size, channel_num, img_height, img_width = spectrum.shape

        x = spectrum
        features_origin = self.net(x)#(batch_size, num_classes, img_height, img_width)
        features = features_origin.permute(0, 2, 3, 1)
        
        logits = self.net2(features)
    
        return logits