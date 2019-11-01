import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        
        # creating character dictionaries
        self.vocab_size = vocab_size
        
        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        
        # lstm cell
        #self.lstm = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first = True, dropout = 0.5, num_layers = self.num_layers)
    
        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    
        # activations
        self.softmax = nn.Softmax(dim=1)
        
        # initialize the weights
        self = self.apply(weights_init)

    def forward(self, features, captions):
        # setup the device
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        # batch size
        batch_size = features.size(0)
        
        # init the hidden and cell states to zeros
        #hidden_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        #cell_state = torch.zeros((batch_size, self.hidden_size)).to(device)
        
        hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
    
        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).to(device)

        # embed the captions
        captions_embed = self.embed(captions)
        
        vals = torch.cat((features.reshape(features.size(0), -1, features.size(1)), captions_embed), dim = 1)
        hidden_state, cell_state = self.lstm(vals, (hidden_state, cell_state))
        
        
        '''
        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm(features, (hidden_state, cell_state))
                
            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm(captions_embed[:, t, :], (hidden_state, cell_state))
            
            # output of the attention mechanism
            out = self.fc_out(hidden_state)
            
            # build the output tensor
            outputs[:, t, :] = out
         ''' 
        
        outputs = self.fc_out(hidden_state[:,1:,:])
            
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass