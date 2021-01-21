# MODEL

class ATM_ide(nn.Module):
    def __init__(self):
        super(ATM_ide,self).__init__()
        self.LSTM1 = nn.LSTM(
                input_size = 257, 
                hidden_size = 300,
                num_layers = 1,  
                batch_first=True)
        self.LSTM2 = nn.LSTM(
                input_size = 300, 
                hidden_size = 300,
                num_layers = 1,  
                batch_first=True)
        self.dense = nn.Linear(364,300)
        self.dense_rec = nn.Linear(300,257)

        #self.LSTM1.register_backward_hook(hook_fn) 
        self.dense1 = nn.Linear(3300,1024)
        self.dense2 = nn.Linear(1024,1024)
        self.dense3 = nn.Linear(1024,256)
        self.dense4 = nn.Linear(256,7)
        
        self.dense_mask1 = nn.Linear(256,300)
        self.dense_mask2 = nn.Linear(300,300)

        self.softmax = torch.nn.Softmax(dim=1)
        self.relu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.5)
        self.pad = torch.nn.ReflectionPad2d((0,0,5,5)) 

    def forward(self, input):

        lstm1_out, (h_n,h_c) = self.LSTM1(input, None)

        lstm2_out, (h_n,h_c) = self.LSTM2(lstm1_out, None)

        tmp = lstm2_out.view(1,1,input.shape[1],300)
        tmp = self.pad(tmp)
        tmp = tmp.squeeze()

        lstm2_con = torch.zeros((input.shape[1],3300), dtype = torch.float32, device = 'cuda', requires_grad=True)
        with torch.no_grad():
            for i in range(input.shape[1]):
                lstm2_con[i,:] = torch.cat((tmp[i+5-5,:],tmp[i+5-4,:],tmp[i+5-3,:],tmp[i+5-2,:],tmp[i+5-1,:],tmp[i+5,:],tmp[i+5+1,:],tmp[i+5+2,:],tmp[i+5+3,:],tmp[i+5+4,:],tmp[i+5+5,:]),0)

        dense1_out = self.relu(self.drop(self.dense1(lstm2_con)))
        dense2_out = self.relu(self.drop(self.dense2(dense1_out)))
        dense3_out = self.relu(self.drop(self.dense3(dense2_out)))
        output_speaker = self.dense4(dense3_out)

        dense_mask = self.relu(self.dense_mask1(dense3_out))
        dense_mask = self.relu(self.dense_mask2(dense_mask))

        dense_mask = dense_mask.view(1,input.shape[1],300)
        lstm2_masked = lstm2_out * dense_mask
        #lstm2_out, (h_n,h_c) = self.LSTM2(lstm1_masked, None)  

        output_enh = self.dense_rec(lstm2_masked.squeeze())

        return output_enh, output_speaker