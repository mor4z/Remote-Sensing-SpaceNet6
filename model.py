import torch
import torch.nn as nn
import torchvision.transforms.functional as TF # Per il resize in caso di problemi di dimensioni

class DoubleConv(nn.Module):
    # Blocco di doppia convoluzione con BatchNorm e ReLU usato nella U-Net.

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # Prima convoluzione
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Seconda convoluzione
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):

    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):

        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Parte di Downsampling (Encoder)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature # Aggiorna in_channels per il prossimo blocco

        # Parte di Upsampling (Decoder)
        for feature in reversed(features):
            # Convoluzione trasposta per l'upsampling
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            # Blocco di doppia convoluzione dopo la concatenazione con la skip connection
            self.ups.append(DoubleConv(feature * 2, feature)) # feature*2 perché si concatena

        # Bottleneck (parte centrale della U-Net)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # Convoluzione finale 1x1 per mappare all'output desiderato
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder (Downsampling Path)
        for down in self.downs:
            x = down(x) # Applica il blocco DoubleConv
            skip_connections.append(x) 
            x = self.pool(x) # Applica il MaxPooling

        # Bottleneck
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]

        # Decoder (Upsampling Path)
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) 
            # Recupera la skip connection corrispondente
            skip_connection = skip_connections[idx // 2]

            # Gestione del mismatch di dimensioni (se l'immagine non è un multiplo di 2^N)
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            # Concatena la skip connection con l'output dell'upsampling
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            x = self.ups[idx + 1](concat_skip)

        # Convoluzione finale
        return self.final_conv(x)

# Funzione di test per verificare che la rete funzioni correttamente
def test():
    # Test della U-Net con un input di esempio di dimensioni non multiple di 2^N
    x = torch.randn((3, 4, 161, 161)) # Batch size 3, 4 canali, 161x161
    model = UNET(in_channels=4, out_channels=1) # in_channels=4 per SAR, out_channels=1 per maschera binaria
    preds = model(x)
    
    
    assert preds.shape[0] == x.shape[0], f"Batch size mismatch! Expected {x.shape[0]}, got {preds.shape[0]}" # La dimensione del batch deve coincidere
    assert preds.shape[1] == 1, f"Output channels mismatch! Expected 1, got {preds.shape[1]}" # Il canale di output deve essere 1
    assert preds.shape[2] == x.shape[2], f"Height mismatch! Expected {x.shape[2]}, got {preds.shape[2]}" # Le dimensioni spaziali devono coincidere
    assert preds.shape[3] == x.shape[3], f"Width mismatch! Expected {x.shape[3]}, got {preds.shape[3]}" # Le dimensioni spaziali devono coincidere

    print(f"Test superato! Input shape: {x.shape}, Output shape: {preds.shape}")


# Questo blocco viene eseguito solo se il file model.py viene eseguito direttamente
if __name__ == "__main__":
    test()