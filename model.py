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
        
        # lstm unit(s)
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
        hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)

        # embed the captions
        captions_embed = self.embed(captions)
        
        # pass through lstm unit(s)
        vals = torch.cat((features.reshape(features.size(0), -1, features.size(1)), captions_embed), dim = 1)
        outputs, (hidden_state, cell_state) = self.lstm(vals, (hidden_state, cell_state))
        
        # pass through the linear unit
        outputs = self.fc_out(outputs)
            
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
        output = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        
        # initialize hidden state
        hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
    
        while True: 
            # pass through lstm unit(s)
            lstm_out, (hidden_state, cell_state) = self.lstm(inputs, (hidden_state, cell_state)) # lstm_out shape : (1, 1, hidden_size)
            
            # pass through linear unit
            outputs = self.fc_out(lstm_out)  # outputs shape : (1, 1, vocab_size)
            
            # predict the most likely next word, max_indice shape : (1)
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) 
            
            # storing the word predicted
            output.append(max_indice.cpu().numpy()[0].item()) 
            
            if (max_indice == 1 or len(output) >= max_len):
                # We predicted the <end> word, so there is no further prediction to do
                break
            
            ## Prepare to embed the last predicted word to be the new input of the lstm
            inputs = self.embed(max_indice) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)
            
        return output