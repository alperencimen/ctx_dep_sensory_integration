
# import libraries
import torch
import torch.nn as nn
import numpy as np

class BaseRNN(nn.Module):
    '''
    This is a Leaky-RNN
    It's dynamical equation is:
    h(t) = (1 - alpha) * h(t-1) + alpha * tanh(W_in * x(t-1) + W_rec * h(t-1))
    '''
    def __init__(self, input_dims, hidden_dims, output_dims, K, reg_ratio = 0.5, g=None,alpha = 0.5, device="cpu", loss_fn=nn.MSELoss(),seed =0):
        super(BaseRNN, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.K = K
        self.alpha = alpha
        self.device = device
        self.loss_fn = loss_fn
        self.seed = seed;
        self.reg_ratio = reg_ratio;
        # Raise error if tau is less than deltaT
        assert alpha <= 1, "Tau cannot be less than deltaT!"
        

        if K == hidden_dims:
            self.W_rec = nn.Parameter(torch.randn(hidden_dims, hidden_dims))
            if g is None:
                nn.init.xavier_uniform_(self.W_rec)
            else:
                nn.init.normal_(self.W_rec,0,g/np.sqrt(hidden_dims))
        else:
            # K rank
            self.S_l = nn.Parameter(torch.randn(hidden_dims, K))
            self.S_r = nn.Parameter(torch.randn(K, hidden_dims))
            nn.init.xavier_uniform_(self.S_l)
            nn.init.xavier_uniform_(self.S_r)  

        # initialize weights
        self.W_in = nn.Parameter(torch.randn(input_dims, hidden_dims))
        self.W_out = nn.Parameter(torch.randn(hidden_dims, output_dims))
        nn.init.xavier_uniform_(self.W_in)
        nn.init.xavier_uniform_(self.W_out)
        # move everything to device
        self.to(device)

    def forward(self,x,h):
        if self.K == self.hidden_dims:
            return self.forward_full(x, h)
        else:
            return self.forward_lr(x, h)
        
    def forward_full(self, x, h):
        """
        The shape of x is (batch_size, seq_len, input_dims)
        The shape of h is (batch_size, hidden_dims)
        The shape of output is (batch_size, seq_len, output_dims)
        The shape of hidden_states is (seq_len, batch_size, hidden_dims)
        """
        hidden_states = [h]
        outputs = []
        reg = 0;
        n_reg = round(h.shape[1]*self.reg_ratio)
        z = torch.zeros_like(h);
        
        for t in range(x.size(1)):
            
            h_from_prev = (1 - self.alpha) * h
            h_rec = torch.matmul(h, self.W_rec)
            h_in = torch.matmul(x[:, t], self.W_in)
            if t>0:
                z_old = z.clone();
            h = h_from_prev + self.alpha * torch.tanh(h_rec + h_in)
            z = h_rec + h_in
            hidden_states.append(h)
            reg = reg + torch.sum(( h[:,:n_reg] - h_from_prev[:,:n_reg]/(1 - self.alpha))**2)

            if t>0:
                reg = reg + torch.sum(( z[:,:n_reg] - z_old[:,:n_reg])**2)

            # Compute the output at each timestep
            output = torch.matmul(h, self.W_out)
            outputs.append(output)
        self.tcr_reg = reg/x.size(1)/h.shape[0]/n_reg
        return outputs, hidden_states

    def forward_lr(self, x, h):
        """
        The shape of x is (batch_size, seq_len, input_dims)
        The shape of h is (batch_size, hidden_dims)
        The shape of output is (batch_size, seq_len, output_dims)
        The shape of hidden_states is (seq_len, batch_size, hidden_dims)
        """
        hidden_states = [h]
        outputs = []
        reg = 0;
        n_reg = round(h.shape[1]/10)
        for t in range(x.size(1)):
            h_from_prev = (1 - self.alpha) * h
            h_rec = torch.matmul(torch.matmul(h, self.S_l), self.S_r)
            h_in = torch.matmul(x[:, t], self.W_in)
            h = h_from_prev + self.alpha * torch.tanh(h_rec + h_in)
            hidden_states.append(h)
            reg = reg + torch.sum((h[:,:n_reg] - h_from_prev[:,:n_reg]/(1 - self.alpha))**2)

            # Compute the output at each timestep
            output = torch.matmul(h, self.W_out)
            outputs.append(output)
        
        self.tcr_reg = reg/x.size(1)/h.shape[0]/n_reg
        return outputs, hidden_states

    def get_params(self):
        # return the weights of the network, as numpy arrays
        # in a dictionary
        if self.K == self.hidden_dims:
            return {"W_in": self.W_in.detach().cpu().numpy(),
                "W_rec": self.W_rec.detach().cpu().numpy(),
                "W_out": self.W_out.detach().cpu().numpy(),
                'alpha': self.alpha}
        else:
            return {"W_in": self.W_in.detach().cpu().numpy(),
                "W_rec": (self.S_l @ self.S_r).detach().cpu().numpy(),
                "W_out": self.W_out.detach().cpu().numpy(),
                'alpha': self.alpha}

    
    def run_rnn(self, inputs, device="cpu",seed = None):
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.eval()
        with torch.no_grad():
            # inputs and outputs are numpy arrays
            x = torch.from_numpy(inputs).float() # shape (batch_size, seq_len, input_dims)
    
            # move x,y,h to device
            x = x.to(device)
    
            # Forward pass
            batch_size = x.shape[0]
            h = torch.randn(batch_size, self.hidden_dims)
            h = h.to(device)
            output, h = self(x, h)
            output = torch.stack(output, dim=1)
            h = torch.stack(h, dim=1) # shape (batch_size, seq_len, hidden_dims)
    
            # convert output and h to numpy
            output = output.detach().cpu().numpy()
            h = h.detach().cpu().numpy()
        
        self.train()
        return output, h
