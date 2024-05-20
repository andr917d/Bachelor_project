import torch
from torch.nn.parameter import Parameter
import os
import wandb

#Feedforward neural network architectures

#Simple feedforward neural network
class FFNN_simple(torch.nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5):
        super(FFNN_simple, self).__init__()
        self.layers = torch.nn.ModuleList()
        

        for i in range(len(hidden_sizes)):
            if i == 0:
                self.layers.append(torch.nn.Linear(input_size, hidden_sizes[i]))
            else:
                self.layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(torch.nn.ReLU())
            self.layers.append(torch.nn.Dropout(p=dropout_prob))
        self.layers.append(torch.nn.Linear(hidden_sizes[-1], output_size))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def train(self, train_loader, num_epochs, log_interval=30):
        losses = []
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()
                if batch_idx % log_interval == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                        f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                losses.append(loss.item())
            self.scheduler.step()
        return losses
    
    def save_model(self, directory='models', filename='FFNN_simple.pt'):
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            print(f'Directory {directory} does not exist. please try again.')
            return
        torch.save(self.state_dict(), os.path.join(directory, filename))

    @classmethod
    def load_model(cls, directory='models', filename='FFNN_simple.pt', *args, **kwargs):
        directory = os.path.join(os.getcwd(), directory)
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(os.path.join(directory, filename)))
        return model
    

    
class FFNN_DeepEnsemble:
    def __init__(self, num_models, input_sizes, hidden_sizes, output_size, dropout_prob=0.5):
        self.models = [FFNN_simple(input_sizes, hidden_sizes, output_size, dropout_prob) for _ in range(num_models)]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.001) for model in self.models]
        self.scheduler = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) for optimizer in self.optimizers]
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_models = num_models

    def train(self, train_loader, num_epochs, log_interval=30):
        losses = [[] for _ in range(self.num_models)]
        for epoch in range(num_epochs):
            for model, optimizer in zip(self.models, self.optimizers):
                for batch, (x, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    loss = self.criterion(model(x), y)
                    loss.backward()
                    optimizer.step()
                    losses[self.models.index(model)].append(loss.item())
                print(f'Model {self.models.index(model)}: Epoch {epoch+1}, Loss: {losses[self.models.index(model)][-1]}')

            # Step the scheduler
            for scheduler in self.scheduler:
                scheduler.step()

        return losses
    

    def save_models(self, directory='models'):
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            print(f'Directory {directory} does not exist. please try again.')
            return

        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(directory, f'DeepEnsembleFFNN_model_{i}.pt'))

    def load_models(self, directory='models'):
        directory = os.path.join(os.getcwd(), directory)
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(directory, f'DeepEnsembleFFNN_model_{i}.pt')))

    @classmethod
    def load(cls, num_models, conv_layers, directory='models'):
        ensemble = cls(num_models, conv_layers)
        ensemble.load_models(directory)
        return ensemble


# Lightweight Normal distribution class
class Normal():
    def __init__(self, mu, std):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.mu = mu.to(self.device)
        self.std = std.to(self.device)
        self.shape = torch.broadcast_shapes(mu.shape, std.shape)
        
        
    def log_prob(self, x):
        return torch.sum(- (x - self.mu)**2 / (2*self.std**2) - .5*torch.log(2*torch.pi*self.std**2))
    
    def rsample(self):
        return torch.randn(self.shape, device=self.device) * self.std + self.mu
    
# Bayesian linear layer
class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, prior_mu=0.0, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Scale the representation of the log-variances to effectively reduce their learning rate
        self.log_scaler = 0.1

        self.weight = torch.randn((out_features, in_features))
        self.weight_mean = Parameter(self.weight)
        self.weight_log_std = Parameter(torch.zeros_like(self.weight) - 5 / self.log_scaler)

        self.bias = torch.zeros(out_features)
        self.bias_mean = Parameter(self.bias)
        self.bias_log_std = Parameter(torch.zeros_like(self.bias) - 5 / self.log_scaler)

        self.epsilon = 1e-6

        # Prior parameters
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma


        self.prior_mu_weight = torch.zeros_like(self.weight)+prior_mu
        self.prior_sigma_weight = torch.zeros_like(self.weight)+prior_sigma

        self.prior_mu_bias = torch.zeros_like(self.bias)+prior_mu
        self.prior_sigma_bias = torch.zeros_like(self.bias)+prior_sigma


    def forward(self, input):
        return torch.nn.functional.linear(input, self.weight, self.bias)
    
    @property
    def weight_std(self):
        return torch.exp(self.log_scaler*self.weight_log_std)

    @property
    def bias_std(self):
        return torch.exp(self.log_scaler*self.bias_log_std)
    
    def distributions(self):
        weight_dist = Normal(self.weight_mean, self.weight_std+self.epsilon)
        bias_dist = Normal(self.bias_mean, self.bias_std+self.epsilon)
        return (weight_dist, bias_dist)
    
    def variables(self):
        return (self.weight, self.bias)
    
    def sample(self):
        self.weight = self.distributions()[0].rsample()
        self.bias = self.distributions()[1].rsample()
        return
    
    def log_prob_q(self):

        log_prop_q = self.distributions()[0].log_prob(self.weight).sum() + self.distributions()[1].log_prob(self.bias).sum()
        return log_prop_q
    
    def log_prob_p(self):
        log_prop_p = Normal(self.prior_mu_weight, self.prior_sigma_weight).log_prob(self.weight).sum() + Normal(self.prior_mu_bias, self.prior_sigma_bias).log_prob(self.bias).sum()
        return log_prop_p

    def kl_closed_form(self):
        # Closed-form KL divergence for Gaussian prior and posterior
        def kl_divergence(mu_q, sigma_q, mu_p, sigma_p):
            kl = torch.log(sigma_p / sigma_q) + (sigma_q**2 + (mu_q - mu_p)**2) / (2 * sigma_p**2) - 0.5
            return kl.sum()

        kl_weight = kl_divergence(self.weight_mean, self.weight_std, self.prior_mu, self.prior_sigma)
        kl_bias = kl_divergence(self.bias_mean, self.bias_std, self.prior_mu, self.prior_sigma)

        return kl_weight + kl_bias   

#Bayesian neural network
class BNN(torch.nn.Module):
    # def __init__(self, input_size, hidden_size1, hidden_size2, output_size, mu_prior=0.0, sigma_prior=1.0):
    #     super().__init__()
    #     self.blinear1 = Linear(input_size, hidden_size1, mu_prior, sigma_prior)
    #     self.blinear2 = Linear(hidden_size1, hidden_size2, mu_prior, sigma_prior)
    #     self.blinear3 = Linear(hidden_size2, output_size, mu_prior, sigma_prior)
    #     self.relu = torch.nn.ReLU()
    #     self.log_variance = Parameter(torch.tensor(0.))

    # def forward(self, x):
    #     x = self.blinear1(x)
    #     x = self.relu(x)
    #     x = self.blinear2(x)
    #     x = self.relu(x)
    #     x = self.blinear3(x)
    #     return x

    # def sample(self):
    #     self.blinear1.sample()
    #     self.blinear2.sample()
    #     self.blinear3.sample()
    #     return
    
    # def log_prob_q(self):
    #     return self.blinear1.log_prob_q() + self.blinear2.log_prob_q() + self.blinear3.log_prob_q()
    
    # def log_prob_p(self):
    #     return self.blinear1.log_prob_p() + self.blinear2.log_prob_p() + self.blinear3.log_prob_p()
    
    
    # def __init__(self, input_size, hidden_sizes, output_size, mu_prior=0.0, sigma_prior=1.0):
    def __init__(self, config):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_sizes = config.model.hidden_sizes
        self.input_size = config.model.input_size
        self.output_size = config.model.output_size
        self.input_size = config.model.input_size
        self.output_size = config.model.output_size
        self.sigma_prior = config.hyper.sigma_prior
        self.mu_prior = config.hyper.mu_prior
        self.config = config
  

        self.linears = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        

        # Create linear layers
        prev_size = self.input_size
        for size in self.hidden_sizes:
            linear = Linear(prev_size, size, self.mu_prior, self.sigma_prior)
            self.linears.append(linear)
            prev_size = size

        self.output_linear = Linear(prev_size, self.output_size, self.mu_prior, self.sigma_prior)

        self.log_variance = Parameter(torch.tensor(0.))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.hyper.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.hyper.step_size, gamma=config.hyper.gamma)
        self.to(self.device)

    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
            x = self.relu(x)
        x = self.output_linear(x)
        return x

    def sample(self):
        for linear in self.linears:
            linear.sample()
        self.output_linear.sample()
        return
    
    def log_prob_q(self):
        log_prob_q = 0
        for linear in self.linears:
            log_prob_q += linear.log_prob_q()
        log_prob_q += self.output_linear.log_prob_q()
        return log_prob_q
    
    def log_prob_p(self):
        log_prob_p = 0
        for linear in self.linears:
            log_prob_p += linear.log_prob_p()
        log_prob_p += self.output_linear.log_prob_p()
        return log_prob_p

    def neg_log_likelihood_regression(self, y_pred, y_true):
        variance = torch.exp(self.log_variance)

        n = y_true.shape[0]

        log_likelihood = -0.5 * n * torch.log(2 * torch.pi * variance) - (0.5 / variance) * torch.sum((y_true - y_pred) ** 2)

        #I think it is - 
        return -log_likelihood
    
    def neg_log_likelihood_classification(self, y_pred, y_true):
        # Compute Cross-Entropy loss (negative log-likelihood)

        #one hot encode the target
        y_true = torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1])

        log_probs = torch.nn.functional.log_softmax(y_pred, dim=1)
        loss = -torch.sum(y_true * log_probs)

        # loss = torch.nn.functional.cross_entropy(y_pred, y_true, reduction='sum')

   
        return loss
    
    def train(self, train_loader, test_loader):


        dataset_size = len(train_loader.dataset)
        for epoch in range(self.config.hyper.epochs):
            train_loss = 0.0
            log_likelihood = 0.0
            logp_values = 0.0
            logq_values = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
            
                self.optimizer.zero_grad()
                self.sample()
                output = self(data) 

                # neg_log_likelihood = self.neg_log_likelihood_classification(output, target)*(dataset_size/len(data))
                neg_log_likelihood = self.neg_log_likelihood_classification(output, target)

                logp = self.log_prob_p()*len(data)/dataset_size
                logq = self.log_prob_q()*len(data)/dataset_size

                loss = neg_log_likelihood + logq - logp
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                log_likelihood += neg_log_likelihood.item()
                logp_values += logp.item()
                logq_values += logq.item()
            

            #validation loss
            val_loss = 0.0
            accuracy = 0.0

            with torch.no_grad():
                for batch_idx, (val_data, val_target) in enumerate(test_loader):
                    val_data, val_target = val_data.to(self.device), val_target.to(self.device)
                    val_output = self(val_data)
                    
                    neg_log_likelihood = self.neg_log_likelihood_classification(val_output, val_target)
                    logp = self.log_prob_p()*len(val_data)/len(test_loader.dataset)
                    logq = self.log_prob_q()*len(val_data)/len(test_loader.dataset)
                    val_loss = neg_log_likelihood + logq - logp



                    #calculate accuracy
                    _, predicted = torch.max(val_output, -1)
                    correct = (predicted == val_target).sum().item()
                    accuracy_batch = correct / len(val_target)
                    # print(f'Validation accuracy: {accuracy_batch}')
                    accuracy += accuracy_batch

            accuracy = accuracy / len(test_loader)

            #logging
            print(f'Epoch: {epoch+1} / {self.config.hyper.epochs}\tTrain Loss: {train_loss}\tValidation Loss: {val_loss}\ Negative log Likelihood: {log_likelihood}\tLogp: {logp_values}\tLogq: {logq_values}\tAccuracy: {accuracy}')
            wandb.log({"training_loss": train_loss, "val_loss": val_loss, "neg_log_likelihood": log_likelihood, "logp": logp_values, "logq": logq_values, "val_accuracy": accuracy})

       
            self.scheduler.step()  

        print('Finished Training')
        
  
    def save_model(self, directory='models', filename='FFNN_BNN.pt'):
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            print(f'Directory {directory} does not exist. please try again.')
            return
        torch.save(self.state_dict(), os.path.join(directory, filename))

    @classmethod
    def load_model(cls, directory='models', filename='FFNN_BNN.pt', *args, **kwargs):
        directory = os.path.join(os.getcwd(), directory)
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(os.path.join(directory, filename)))
        return model
    

    


# BatchEnsemble linear layer
class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        # Shared weight matrix
        self.shared_weight = Parameter(torch.randn(out_features, in_features))

        # Rank-1 factors for each ensemble member
        self.rank1_u = Parameter(torch.randn(ensemble_size, out_features))
        self.rank1_v = Parameter(torch.randn(ensemble_size, in_features))

        # Shared bias
        self.shared_bias = Parameter(torch.zeros(out_features))

    
    def vectorized_forward(self, input):

        if input.dim() == 2:
            # # Input is from the first layer (2D tensor)
            # batch_size = input.size(0)
            # Reshape input to have dimensions: [ensemble_size, batch_size, in_features]
            X = input.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        elif input.dim() == 3:
            # Input is from a subsequent layer (3D tensor)
            # batch_size = input.size(1)
            # Ensure input has dimensions: [ensemble_size, batch_size, in_features]
            X = input

        # Expand S for element-wise multiplication
        S = self.rank1_v.unsqueeze(1)  # Shape: [ensemble_size, 1, in_features]
        #repaet so it has same dimension as X ( not necessary i think but just so it matches the dimensions in the paper)
        S = S.repeat(1, X.size(1), 1) # Shape: [ensemble_size, batch_size, in_features]

        # Expand R for element-wise multiplication
        R = self.rank1_u.unsqueeze(1) # Shape: [ensemble_size, 1, out_features]
        #repeat so it has same dimension as intermediate_output ( not necessary i think but just so it matches the dimensions in the paper)
        R = R.repeat(1, X.size(1), 1) # Shape: [ensemble_size, batch_size, out_features]

        shared_weight = self.shared_weight.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        intermediate_output = (X * S).matmul(shared_weight.transpose(2, 1))

        # Perform element-wise multiplication with R
        ensemble_output = intermediate_output * R

        # Add the shared bias to each ensemble member's output
        shared_bias = self.shared_bias.unsqueeze(0).unsqueeze(1).expand(-1, X.size(1), -1)

        # print(f'shared bias: {shared_bias.shape}')

        ensemble_output += shared_bias

        return ensemble_output
    

# Ensemble network
class BatchEnsemble_FFNN(torch.nn.Module):
    # def __init__(self, input_size, hidden_sizes, output_size, ensemble_size):
    def __init__(self, config):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_sizes = config.model.hidden_sizes
        self.input_size = config.model.input_size
        self.output_size = config.model.output_size

        self.linears = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        self.ensemble_size = config.model.ensemble_size
        

        # Create linear layers
        prev_size = self.input_size
        for size in self.hidden_sizes:
            linear = EnsembleLinear(prev_size, size, self.ensemble_size)
            self.linears.append(linear)
            prev_size = size

        self.output_linear = EnsembleLinear(prev_size, self.output_size, self.ensemble_size)
        self.log_variance = Parameter(torch.tensor(0.))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.hyper.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.hyper.step_size, gamma=config.hyper.gamma)
        self.to(self.device)

    def forward(self, x):
        for linear in self.linears:
            x = linear.vectorized_forward(x)
            x = self.relu(x)
        x = self.output_linear.vectorized_forward(x)
        return x


    def log_likelihood(self, y_pred, y_true):

        variance = torch.exp(self.log_variance)
        n = y_true.size(0)
        log_likelihood = -0.5 * n * torch.log(2 * torch.pi * variance) - (0.5 / variance) * torch.sum((y_true - y_pred) ** 2)
    
        # Sum the log likelihoods across ensemble members
        total_log_likelihood = torch.sum(log_likelihood, dim=0)  # Sum along the ensemble dimension

        return total_log_likelihood
    
    def neg_log_likelihood_ensemble_member(self, y_pred, y_true):
             
            variance = torch.exp(self.log_variance)
            n = y_true.size(0)
            log_likelihood = -0.5 * n * torch.log(2 * torch.pi * variance) - (0.5 / variance) * torch.sum((y_true - y_pred) ** 2)
    
            return -log_likelihood
    
    def neg_log_likelihood_categorical(self, y_pred, y_true):

        #cross entropy loss taking into account that we are doing ensembles
        losses = [torch.nn.functional.cross_entropy(y_pred[i], y_true, reduction='sum') for i in range(self.ensemble_size)]

        loss = sum(losses)

        return loss
    
    def train(self, train_loader, test_loader):
            
            # losses = []
            # for epoch in range(num_epochs):
            #     for batch_idx, (data, target) in enumerate(train_loader):
            #         self.optimizer.zero_grad()
            #         output = self(data) 
            #         neg_log_likelihood = self.neg_log_likelihood_categorical(output, target)
            #         loss = neg_log_likelihood
            #         loss.backward()
            #         self.optimizer.step()
            #         losses.append(loss.item())
                 
    
                        
            #     self.scheduler.step()  

        for epoch in range(self.config.hyper.epochs):

            train_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self(data) 
                loss = self.neg_log_likelihood_categorical(output, target)/len(data)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            
            val_accuracy = 0.0
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (val_data, val_target) in enumerate(test_loader):
                    val_data, val_target = val_data.to(self.device), val_target.to(self.device)
                    val_output = self(val_data)
                    loss = self.neg_log_likelihood_categorical(val_output, val_target)/len(val_data)
                    val_loss += loss.item()

                    #accuracy
                    _, predicted = torch.max(val_output, -1)
                    correct = (predicted == val_target).sum().item()
                    val_accuracy += correct / len(val_target)


            val_accuracy = val_accuracy / (len(test_loader) * self.ensemble_size)
            
            print(f'Epoch: {epoch+1} / {self.config.hyper.epochs}\tTrain Loss: {train_loss}\tValidation Loss: {val_loss}\tValidation Accuracy: {val_accuracy}')
            wandb.log({"training_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})
            


    def save_model(self, directory='models', filename='FFNN_BatchEnsemble.pt'):
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.state_dict(), os.path.join(directory, filename))

    @classmethod
    def load_model(cls, directory='models', filename='FFNN_BatchEnsemble.pt', *args, **kwargs):
        directory = os.path.join(os.getcwd(), directory)
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(os.path.join(directory, filename)))
        return model


# Dense BNN layer with rank-1 factors
class Dense_rank1(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size, prior_mu, prior_sigma):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.log_scaler = 1.0

        self.epsilon = 1e-6


        # Shared weight matrix
        self.shared_weight = Parameter(torch.randn((out_features, in_features)))
        self.prior_mu_weight = torch.zeros_like(self.shared_weight)+prior_mu
        self.prior_sigma_weight = torch.zeros_like(self.shared_weight)+prior_sigma

        # Shared bias
        self.shared_bias = Parameter(torch.zeros(out_features))
        self.prior_mu_bias = torch.zeros_like(self.shared_bias)+prior_mu
        self.prior_sigma_bias = torch.zeros_like(self.shared_bias)+prior_sigma



        # Rank-1 factors for each ensemble member
        self.rank1_u = torch.randn(ensemble_size, out_features)
        self.rank1_v = torch.randn(ensemble_size, in_features)

        self.rank1_u_mean =  Parameter(self.rank1_u)
        self.rank1_u_log_std =  Parameter(torch.zeros_like(self.rank1_u)-5.*self.log_scaler)
        
        self.rank1_v_mean =  Parameter(self.rank1_v)
        self.rank1_v_log_std =  Parameter(torch.zeros_like(self.rank1_v)-5.*self.log_scaler)

        # Prior parameters
        self.prior_mu_u = torch.zeros_like(self.rank1_u_mean)+prior_mu
        self.prior_sigma_u = torch.zeros_like(self.rank1_u_mean)+prior_sigma

        self.prior_mu_v = torch.zeros_like(self.rank1_v_mean)+prior_mu
        self.prior_sigma_v = torch.zeros_like(self.rank1_v_mean)+prior_sigma
        

    @property
    def rank1_u_std(self):
        return torch.exp(self.log_scaler*self.rank1_u_log_std)

    @property
    def rank1_v_std(self):
        return torch.exp(self.log_scaler*self.rank1_v_log_std)

    def vectorized_forward(self, input):

        if input.dim() == 2:
            # # Input is from the first layer (2D tensor)
            # batch_size = input.size(0)
            # Reshape input to have dimensions: [ensemble_size, batch_size, in_features]
            X = input.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        elif input.dim() == 3:
            # Input is from a subsequent layer (3D tensor)
            # batch_size = input.size(1)
            # Ensure input has dimensions: [ensemble_size, batch_size, in_features]
            X = input

        # Expand S for element-wise multiplication
        S = self.rank1_v.unsqueeze(1)  # Shape: [ensemble_size, 1, in_features]
        #repaet so it has same dimension as X 
        S = S.repeat(1, X.size(1), 1) # Shape: [ensemble_size, batch_size, in_features]

        # Expand R for element-wise multiplication
        R = self.rank1_u.unsqueeze(1) # Shape: [ensemble_size, 1, out_features]
        #repeat so it has same dimension as intermediate_output
        R = R.repeat(1, X.size(1), 1) # Shape: [ensemble_size, batch_size, out_features]

        shared_weight = self.shared_weight.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        # Perform element-wise multiplication with S and then matrix multiplication with the shared weight
        # intermediate_output = (X * S).matmul(self.shared_weight.t())
        intermediate_output = (X * S).matmul(shared_weight.transpose(2, 1))

        # Perform element-wise multiplication with R
        ensemble_output = intermediate_output * R
        # Add the shared bias to each ensemble member's output
        shared_bias = self.shared_bias.unsqueeze(0).unsqueeze(1).expand(-1, X.size(1), -1)

        # print(f'shared bias: {shared_bias.shape}')

        ensemble_output += shared_bias
   
        return ensemble_output
    
    def distributions(self):
        # weight_dist = Normal(self.weight_mean, self.weight_std+self.epsilon)
        # bias_dist = Normal(self.bias_mean, self.bias_std+self.epsilon)

        u_dist = Normal(self.rank1_u_mean, self.rank1_u_std+self.epsilon)
        v_dist = Normal(self.rank1_v_mean, self.rank1_v_std+self.epsilon)

        return u_dist, v_dist
    
    def sample(self):
    
        self.rank1_u = self.distributions()[0].rsample()
        self.rank1_v = self.distributions()[1].rsample()
        return 
    
    def kl_divergence_u(self):
        u_dist = self.distributions()[0]

        log_q = u_dist.log_prob(self.rank1_u).sum()
        log_p =  Normal(self.prior_mu_u, self.prior_sigma_u).log_prob(self.rank1_u).sum()
        kl = -log_p + log_q
        return kl

 
    
    def kl_divergence_v(self):
        v_dist = self.distributions()[1]

        log_q = v_dist.log_prob(self.rank1_v).sum()
        log_p =  Normal(self.prior_mu_v, self.prior_sigma_v).log_prob(self.rank1_v).sum()
        kl = -log_p + log_q
        return kl


    def log_prob_w(self):

        log_prob_w = Normal(self.prior_mu_weight, self.prior_sigma_weight).log_prob(self.shared_weight).sum() + Normal(self.prior_mu_bias, self.prior_sigma_bias).log_prob(self.shared_bias).sum()
        return log_prob_w
    

class BNN_rank1(torch.nn.Module):

    # def __init__(self, input_size, hidden_sizes, output_size, ensemble_size, mu_prior=0.0, sigma_prior=1.0):
    def __init__(self, config):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_sizes = config.model.hidden_sizes
        self.input_size = config.model.input_size
        self.output_size = config.model.output_size

        self.linears = torch.nn.ModuleList()
        self.relu = torch.nn.ReLU()
        self.ensemble_size = config.model.ensemble_size
        self.mu_prior = config.hyper.mu_prior
        self.sigma_prior = config.hyper.sigma_prior


        # Create linear layers
        prev_size = self.input_size
        for size in self.hidden_sizes:
            linear = Dense_rank1(prev_size, size, self.ensemble_size, self.mu_prior, self.sigma_prior)
            self.linears.append(linear)
            prev_size = size

        self.output_linear = Dense_rank1(prev_size, self.output_size, self.ensemble_size, self.mu_prior, self.sigma_prior)

        self.log_variance = Parameter(torch.tensor(0.))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.hyper.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.hyper.step_size, gamma=config.hyper.gamma)
        self.to(self.device)

    def forward(self, x):
        for linear in self.linears:
            x = linear.vectorized_forward(x)
            x = self.relu(x)
        x = self.output_linear.vectorized_forward(x)
        return x

    def sample(self):
        for linear in self.linears:
            linear.sample()
        self.output_linear.sample()
        return
    
    def kl_divergence_u(self):
        kl_divergence_u = 0
        for linear in self.linears:
            kl_divergence_u += linear.kl_divergence_u()
        kl_divergence_u += self.output_linear.kl_divergence_u()

        return kl_divergence_u
    
    def kl_divergence_v(self):
        kl_divergence_v = 0
        for linear in self.linears:
            kl_divergence_v += linear.kl_divergence_v()
        kl_divergence_v += self.output_linear.kl_divergence_v()

        return kl_divergence_v
        
        
    def log_prob_w(self):
        log_prob_w = 0
        for linear in self.linears:
            log_prob_w += linear.log_prob_w()
        log_prob_w += self.output_linear.log_prob_w()

        return log_prob_w


    def log_likelihood(self, y_pred, y_true):

        variance = torch.exp(self.log_variance)
        n = y_true.size(0)
        log_likelihood = -0.5 * n * torch.log(2 * torch.pi * variance) - (0.5 / variance) * torch.sum((y_true - y_pred) ** 2)
    
        # Sum the log likelihoods across ensemble members
        total_log_likelihood = torch.sum(log_likelihood, dim=0)  # Sum along the ensemble dimension

        return total_log_likelihood
    
    def neg_log_likelihood_ensemble_member(self, y_pred, y_true):
             
            variance = torch.exp(self.log_variance)
            n = y_true.size(0)
            log_likelihood = -0.5 * n * torch.log(2 * torch.pi * variance) - (0.5 / variance) * torch.sum((y_true - y_pred) ** 2)
    
            return -log_likelihood
    
    def neg_log_likelihood_categorical(self, y_pred, y_true):

        # y_true = y_true.view(y_pred.shape[0], -1)
        # y_true = y_true.view(-1)
        # y_pred = y_pred.view(y_true.size(0), -1)

       
        #cross entropy loss taking into account that we are doing ensembles
        losses = [torch.nn.functional.cross_entropy(y_pred[i], y_true, reduction='sum') for i in range(self.ensemble_size)]

        loss = sum(losses)

        # Sum the log likelihoods across ensemble members
        # total_log_likelihood = torch.sum(loss, dim=0)

        return loss
    
    def train(self, train_loader, num_epochs, log_interval=30):
                
                # losses, neg_log_likelihoods, kl_divergences = [], [], []
    
                # for epoch in range(num_epochs):
                #     for batch_idx, (data, target) in enumerate(train_loader):
                    
                #         self.optimizer.zero_grad()

                #         self.sample()

                #         output = self(data) 
        
                #         neg_log_likelihood = self.neg_log_likelihood_categorical(output, target)
                #         kl_divergence_u = self.kl_divergence_u()
                #         kl_divergence_v = self.kl_divergence_v()
                #         log_prob_w = self.log_prob_w()

    
                #         loss = neg_log_likelihood + kl_divergence_u + kl_divergence_v - log_prob_w
                #         loss.backward()
                #         self.optimizer.step()
                        
                #         if batch_idx % log_interval == 0:
                #             print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                #                 f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                #         losses.append(loss.item())
                    
        
                            
                #     self.scheduler.step()  
        for epoch in range(self.config.hyper.epochs):
            
            train_loss = 0.0
            log_likelihood = 0.0
            kl_u = 0.0
            kl_v = 0.0
            log_p_w = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                self.sample()
                output = self(data) 
                neg_log_likelihood = self.neg_log_likelihood_categorical(output, target)
                kl_divergence_u = self.kl_divergence_u()*len(data)/len(train_loader.dataset)
                kl_divergence_v = self.kl_divergence_v()*len(data)/len(train_loader.dataset)
                log_prob_w = self.log_prob_w()*len(data)/len(train_loader.dataset)

                loss = neg_log_likelihood + kl_divergence_u + kl_divergence_v - log_prob_w
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                log_likelihood += neg_log_likelihood.item()
                kl_u += kl_divergence_u.item()
                kl_v += kl_divergence_v.item()
                log_p_w += log_prob_w.item()
        
            val_loss = 0.0
            accuracy = 0.0
            with torch.no_grad():
                for batch_idx, (val_data, val_target) in enumerate(test_loader):
                    val_data, val_target = val_data.to(self.device), val_target.to(self.device)
                    val_output = self(val_data)

                    val_negative_log_likelihood = self.neg_log_likelihood_categorical(val_output, val_target)
                    val_kl_divergence_u = self.kl_divergence_u()*len(val_data)/len(test_loader.dataset)
                    val_kl_divergence_v = self.kl_divergence_v()*len(val_data)/len(test_loader.dataset)
                    val_log_prob_w = self.log_prob_w()*len(val_data)/len(test_loader.dataset)

                    loss = val_negative_log_likelihood + val_kl_divergence_u + val_kl_divergence_v - val_log_prob_w
                    val_loss += loss.item()

                
                    #accuracy
                    _, predicted = torch.max(val_output, -1)
                    correct = (predicted == val_target).sum().item()
                    accuracy += correct / len(val_target)

             # also divide by ensemble size
            accuracy = accuracy / (len(test_loader) * self.ensemble_size)  
            #logging
            print(f'Epoch: {epoch+1} / {self.config.hyper.epochs}\tTrain Loss: {train_loss}\tValidation Loss: {val_loss} \tValidation Accuracy: {accuracy}')
            wandb.log({"training_loss": train_loss, "neg_log_likelihood": log_likelihood, "kl_divergence_u": kl_u, "kl_divergence_v": kl_v, "log_prob_w": log_p_w, "val_loss": val_loss, "val_accuracy": accuracy})
                
    
                    
            self.scheduler.step()


          
    
    def save_model(self, directory='models', filename='FFNN_BNN_rank1.pt'):
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            print(f'Directory {directory} does not exist. please try again.')
            return
        torch.save(self.state_dict(), os.path.join(directory, filename))

    @classmethod
    def load_model(cls, directory='models', filename='FFNN_BNN_rank1.pt', *args, **kwargs):
        directory = os.path.join(os.getcwd(), directory)
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(os.path.join(directory, filename)))
        return model
        





# CNN architectures


class ConvBlock_simple(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout_prob=0.5, pooling=True):
        super(ConvBlock_simple, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout2d(p=dropout_prob)
        self.pooling = pooling
        if self.pooling:
            self.pool = torch.nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        if self.pooling:
            x = self.pool(x)
        
        return x
    


class CNN_simple(torch.nn.Module):
    def __init__(self, config):
        super(CNN_simple, self).__init__()
        self.config = config
        self.conv_layers = config.model.conv_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_size = config.model.image_size
        self.conv_blocks = torch.nn.ModuleList([ConvBlock_simple(*layer) for layer in self.conv_layers])
        self.final_out_channels, self.final_image_size = self.calculate_final_layer_details(self.conv_layers)
        self.linear = torch.nn.Linear(self.final_out_channels * self.final_image_size * self.final_image_size, 1024)
        self.fc = torch.nn.Linear(1024, config.model.num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.hyper.lr, weight_decay=config.hyper.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.hyper.step_size, gamma=config.hyper.gamma)
        self.to(self.device)

    def calculate_final_layer_details(self, conv_layers):
        image_size = self.image_size
        out_channels = 0

        for layer in conv_layers:
            out_channels = layer[1]  # The number of output channels is the second element of the layer tuple
            if layer[-1]:  # Check if pooling is applied in the layer
                image_size = image_size // 2  # Each pooling layer reduces the image size by a factor of 2

        return out_channels, image_size

    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        

        x = x.view(x.size(0), -1)
 
        x = self.linear(x)
        x = torch.nn.functional.relu(x)
        x = self.fc(x)
        return x


    
    
    def train_custom(self, train_loader, test_loader):
        # losses = []
        # ttest_losses = []

    
        #split the data into training and validation
        # train_size = int(0.95 * len(train_loader.dataset))

        # train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, len(train_loader.dataset) - train_size])

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
        # #test on the whole validation set
        # test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

        

        for epoch in range(self.config.hyper.epochs):
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self(data) 
                
                loss = torch.nn.functional.cross_entropy(output, target, reduction='sum')
                
                loss.backward()
                self.optimizer.step()
        
                train_loss += loss.item() * len(target)

                # #accuracy
                # _, predicted = torch.max(output, 1)
                # correct = (predicted == target).sum().item()
                # accuracy = correct / len(target)
                # print(f'Training accuracy: {accuracy}')



                # if batch_idx % self.config.hyper.log_interval == 0:
                #     print(f'Train Epoch: {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                #         f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                    # print(f'Validation loss: {val_loss.item()}')
                
            #validation loss
            val_loss = 0.0
            accuracy = 0.0
            with torch.no_grad():
                for batch_idx, (val_data, val_target) in enumerate(test_loader):
                    val_data, val_target = val_data.to(self.device), val_target.to(self.device)
                    val_output = self(val_data)
                    val_loss = torch.nn.functional.cross_entropy(val_output, val_target, reduction='sum')
                    val_loss += val_loss.item() * len(val_data)

                    #calculate accuracy
                    _, predicted = torch.max(val_output, 1)
                    correct = (predicted == val_target).sum().item()
                    accuracy_batch = correct / len(val_target)
                    # print(f'Validation accuracy: {accuracy_batch}')
                    accuracy += accuracy_batch

            
            accuracy = accuracy / len(test_loader)

            
                
            

            avg_train_loss = train_loss / len(train_loader.dataset)
            avg_val_loss = val_loss / len(test_loader.dataset)
            print(f'Epoch: {epoch+1}\tTrain Loss: {avg_train_loss}\tValidation Loss: {avg_val_loss}')
            print(f'Validation accuracy: {accuracy}')
            # Logging
            wandb.log({"training_loss": avg_train_loss, "val_loss": avg_val_loss, "val_accuracy": accuracy})
        
            


                
            self.scheduler.step()  
        
        print('Finished Training')
        
        # return losses, val_losses
    
    def save_model(self, directory='models', filename='SimpleCNN.pt'):
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            print(f'Directory {directory} does not exist. please try again.')
            return
        torch.save(self.state_dict(), os.path.join(directory, filename))

    @classmethod
    def load_model(cls, directory='models', filename='SimpleCNN.pt', *args, **kwargs):
        directory = os.path.join(os.getcwd(), directory)
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(os.path.join(directory, filename)))
        return model
    
    
#Deep ensemble CNN
class CNN_DeepEnsemble(torch.nn.Module):
    # def __init__(self, num_models, conv_layers, num_classes=10, image_size=28):
    def __init__(self, config):
        super(CNN_DeepEnsemble, self).__init__()
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_models = config.model.num_models
        self.conv_layers = config.model.conv_layers
        self.num_classes = config.model.num_classes

        self.models = [CNN_simple(config) for _ in range(self.num_models)]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=config.hyper.lr, weight_decay=config.hyper.weight_decay) for model in self.models]
        self.schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.hyper.step_size, gamma=config.hyper.gamma) for optimizer in self.optimizers]
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.to(self.device)

    
    # def train(self, train_loader, num_epochs, log_interval=1):
    def custom_train(self, train_loader, test_loader):

        # losses = [[] for _ in range(self.num_models)]

        for epoch in range(self.config.hyper.epochs):
            losses = [[] for _ in range(self.num_models)]
            train_loss = 0.0

            for model, optimizer in zip(self.models, self.optimizers):
                for batch, (x, y) in enumerate(train_loader):
                    optimizer.zero_grad()
                    # loss = self.criterion(model(x), y)
                    loss = torch.nn.functional.cross_entropy(model(x), y, reduction='sum')
                    loss.backward()
                    optimizer.step()

                    # losses[self.models.index(model)].append(loss.item())
                    train_loss += loss.item()

                
                train_loss = train_loss / len(train_loader.dataset)

                losses[self.models.index(model)] = train_loss

            #validation loss
            val_loss = 0.0
            accuracies = [[] for _ in range(self.num_models)]
            
            with torch.no_grad():
                for model in self.models:
                    accuracy = 0.0
                    for batch_idx, (val_data, val_target) in enumerate(test_loader):
                        val_data, val_target = val_data.to(self.device), val_target.to(self.device)
                        val_output = model(val_data)
                        loss = torch.nn.functional.cross_entropy(val_output, val_target, reduction='mean')
                        val_loss += loss.item()

                        #accuracy
                        _, predicted = torch.max(val_output, -1)
                        correct = (predicted == val_target).sum().item()
                        accuracy += correct / len(val_target)

                    accuracy = accuracy / len(test_loader)

                    accuracies[self.models.index(model)] = accuracy

            val_loss = val_loss / len(test_loader)

            print(f'Epoch: {epoch+1} / {self.config.hyper.epochs}\tTrain Loss: {train_loss}\tValidation Loss: {val_loss}\tValidation Accuracy: {accuracies}')

            wandb.log({"training_loss": train_loss, "val_loss": val_loss, "val_accuracy": accuracies})

            
            #scheduler step
            for scheduler in self.schedulers:
                scheduler.step()

        return losses
    
    def save_model(self, directory='models'):
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            print(f'Directory {directory} does not exist. please try again.')
            return

        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), os.path.join(directory, f'DeepEnsembleCNN_model_{i}.pt'))

    def load_models(self, directory='models'):
        directory = os.path.join(os.getcwd(), directory)
        for i, model in enumerate(self.models):
            model.load_state_dict(torch.load(os.path.join(directory, f'DeepEnsembleCNN_model_{i}.pt')))

    @classmethod
    def load_model(cls, num_models, conv_layers, directory='models'):
        ensemble = cls(num_models, conv_layers)
        ensemble.load_models(directory)
        return ensemble

       

    

# Bayesian convolutional layer 
class BayesianConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, prior_mu=0.0, prior_sigma=1000.):
        super(BayesianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.epsilon = 1e-6
        self.log_scaler = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


        # Define size for weight and bias
        self.weight = torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).normal_(0, 1)

        # self.weight_mean = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).normal_(0, 1))
        self.weight_mean = Parameter(self.weight)
        # self.weight_log_std = Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).normal_(-5, 1))
        self.weight_log_std = Parameter(torch.zeros_like(self.weight) - 5*self.log_scaler)

        # self.bias_mean = Parameter(torch.Tensor(out_channels).normal_(0, 1))
        self.bias = torch.Tensor(out_channels).normal_(0, 1)
        self.bias_mean = Parameter(self.bias)
        # self.bias_log_std = Parameter(torch.Tensor(out_channels).normal_(-5, 1))
        self.bias_log_std = Parameter(torch.zeros_like(self.bias) - 5*self.log_scaler)

        # Prior parameters
        self.prior_mean = prior_mu 
        self.prior_sigma = prior_sigma 

        self.prior_mu_weight = (torch.zeros_like(self.weight)+prior_mu).to(self.device)
        self.prior_sigma_weight = (torch.zeros_like(self.weight)+prior_sigma).to(self.device)

        self.prior_mu_bias = (torch.zeros_like(self.bias)+prior_mu).to(self.device)
        self.prior_sigma_bias = (torch.zeros_like(self.bias)+prior_sigma).to(self.device)

        
        self.to(self.device)



    def forward(self, input):
        return torch.nn.functional.conv2d(input, self.weight, self.bias, self.stride, self.padding)
    
    @property
    def weight_std(self):
        return torch.exp(self.log_scaler*self.weight_log_std)
    
    @property
    def bias_std(self):
        return torch.exp(self.log_scaler*self.bias_log_std)
    
    

    def distributions(self):
        weight_dist = Normal(self.weight_mean, self.weight_std+self.epsilon)
        bias_dist = Normal(self.bias_mean, self.bias_std+self.epsilon)
        return (weight_dist, bias_dist)
    
    def sample(self):
        self.weight = self.distributions()[0].rsample()
        self.bias = self.distributions()[1].rsample()
        return
    
    def log_prob_q(self):

        log_prop_q = self.distributions()[0].log_prob(self.weight).sum() + self.distributions()[1].log_prob(self.bias).sum()
        return log_prop_q
    
    def log_prob_p(self):
        log_prop_p = Normal(self.prior_mu_weight, self.prior_sigma_weight).log_prob(self.weight).sum() + Normal(self.prior_mu_bias, self.prior_sigma_bias).log_prob(self.bias).sum()
        return log_prop_p
    
    
    def kl_divergence(self):
        kl = self.log_prob_q() - self.log_prob_p()
        return kl
    

# Convolutional block with Bayesian convolutional layer
class ConvBlock_simpleBNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, pooling = True, prior_sigma=10):
        super(ConvBlock_simpleBNN, self).__init__()
        self.pooling = pooling
        self.conv = BayesianConv2d(in_channels, out_channels, kernel_size, stride, padding, prior_sigma=prior_sigma)
        self.pool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        if self.pooling:
            x = self.pool(x)
        return x
    
# CNN with Bayesian convolutional layers
class ConvolutionalBNN(torch.nn.Module):
    def __init__(self, config):
        super(ConvolutionalBNN, self).__init__()
        self.config = config
        self.image_size = config.model.image_size
        self.conv_layers = config.model.conv_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conv_blocks = torch.nn.ModuleList([ConvBlock_simpleBNN(*layer, prior_sigma=self.config.hyper.sigma_prior) for layer in self.conv_layers])
        final_out_channels, final_image_size = self.calculate_final_layer_details(self.conv_layers)
        self.linear = torch.nn.Linear(final_out_channels * final_image_size * final_image_size, 1024)
        self.fc = torch.nn.Linear(1024, config.model.num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.hyper.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.hyper.step_size, gamma=config.hyper.gamma)
        self.to(self.device)
        


    def calculate_final_layer_details(self, conv_layers):
        image_size = self.image_size
        out_channels = 0

        for layer in conv_layers:
            out_channels = layer[1]  # The number of output channels is the second element of the layer tuple
            if layer[-1]:  # Check if pooling is applied in the layer
                image_size = image_size // 2  # Each pooling layer reduces the image size by a factor of 2

        return out_channels, image_size
    
    def forward(self, x):
        for block in self.conv_blocks:
            x = block(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = torch.nn.functional.relu(x)
        x = self.fc(x)
        return x
    
    def sample(self):
        for block in self.conv_blocks:
            block.conv.sample()
        return
    
    def logq(self):
        logq = 0
        for block in self.conv_blocks:
            logq += block.conv.log_prob_q()
        return logq
    
    def logp(self):
        logp = 0
        for block in self.conv_blocks:
            logp += block.conv.log_prob_p()
        return logp
    
    def kl_divergence(self):
        kl = 0
        for block in self.conv_blocks:
            kl += block.conv.kl_divergence_sample()
        return kl
    

    def neg_log_likelihood_categorical(self, y_pred, y_true):
        return torch.nn.functional.cross_entropy(y_pred, y_true, reduction='sum')
        # y_true = torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1])

        # log_probs = torch.nn.functional.log_softmax(y_pred, dim=1)
        # loss = -torch.sum(y_true * log_probs)
        


    
    def train_custom(self, train_loader, test_loader):

        #split the data into training and validation
        # train_size = int(0.95 * len(train_loader.dataset))

        # train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, len(train_loader.dataset) - train_size])

        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
        # #test on the whole validation set
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)



        for epoch in range(self.config.hyper.epochs):
            

            train_loss = 0.0
            log_likelihood = 0.0
            logp_values = 0.0
            logq_values = 0.0
            



            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                self.sample()
                output = self(data) 
                neg_log_likelihood = self.neg_log_likelihood_categorical(output, target)
                
                logp = self.logp()*len(data)/len(train_loader.dataset)
                logq = self.logq()*len(data)/len(train_loader.dataset)
                

                loss = neg_log_likelihood + logq - logp

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                log_likelihood += neg_log_likelihood.item()
                logp_values += logp.item()
                logq_values += logq.item()


            #validation loss
            val_loss = 0.0
            accuracy = 0.0
            log_likelihood_val = 0.0

            with torch.no_grad():
                for batch_idx, (val_data, val_target) in enumerate(test_loader):
                    val_data, val_target = val_data.to(self.device), val_target.to(self.device)
                    val_output = self(val_data)
                    
                    neg_log_likelihood = self.neg_log_likelihood_categorical(val_output, val_target)
                    log_likelihood_val += neg_log_likelihood.item()

                    logp = self.logp()*len(val_data)/len(test_loader.dataset)
                    logq = self.logq()*len(val_data)/len(test_loader.dataset)
                    val_loss = neg_log_likelihood + logq - logp



                    #calculate accuracy
                    _, predicted = torch.max(val_output, -1)
                    correct = (predicted == val_target).sum().item()
                    accuracy_batch = correct / len(val_target)
                    # print(f'Validation accuracy: {accuracy_batch}')
                    accuracy += accuracy_batch


            accuracy = accuracy / len(test_loader)
            log_likelihood_val = log_likelihood_val / len(test_loader.dataset)

            #logging
            print(f'Epoch: {epoch+1} / {self.config.hyper.epochs}\tTrain Loss: {train_loss}\tValidation Loss: {val_loss}\ Negative log Likelihood: {log_likelihood}\tLogp: {logp_values}\tLogq: {logq_values}\tAccuracy: {accuracy}')
            wandb.log({"training_loss": train_loss, "val_loss": val_loss, "neg_log_likelihood": log_likelihood, "neg_log_likelihood_val": log_likelihood_val, "logp": logp_values, "logq": logq_values, "val_accuracy": accuracy}) 

             
            # scheduler step
            self.scheduler.step()
        
        # #calculate neg_log_likelihood on val set
        # with torch.no_grad():
        #     val_loss = 0.0
        #     for val_data, val_target in test_loader:
        #         val_data, val_target = val_data.to(self.device), val_target.to(self.device)
        #         val_output = self(val_data)
        #         neg_log_likelihood = self.neg_log_likelihood_categorical(val_output, val_target)
        #         val_loss += neg_log_likelihood.item()

            
        #     #take the average 
        #     val_loss = val_loss / len(test_loader.dataset)

        # #log to wandb
        # # wandb.log({"val_neg_log_likelihood": val_loss})

        # print(f'Negative log likelihood on validation set: {val_loss}')

        print('Finished Training')
    
    def save_model(self, directory='models', filename='CNN_BNN.pt'):
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            print(f'Directory {directory} does not exist. please try again.')
            return
        torch.save(self.state_dict(), os.path.join(directory, filename))

    @classmethod
    def load_model(cls, directory='models', filename='CNN_BNN.pt', *args, **kwargs):
        directory = os.path.join(os.getcwd(), directory)
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(os.path.join(directory, filename)))
        return model


    
# Convolutional layer with BatchEnsemble
class Conv2dBatchEnsemble(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ensemble_size, stride=1, padding=1, bias=True):
        super(Conv2dBatchEnsemble, self).__init__()
        self.ensemble_size = ensemble_size
        self.stride = stride
        self.padding = padding

        # Shared convolutional filters
        self.shared_weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).normal_(0, 1.0))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels)) if bias else None

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

        # Rank-1 factors for each ensemble member
        self.rank1_r = torch.nn.Parameter(torch.Tensor(ensemble_size, 1, out_channels, 1, 1))
        self.rank1_s = torch.nn.Parameter(torch.Tensor(ensemble_size, 1, in_channels, 1, 1))

        # Initialize rank-1 factors
        torch.nn.init.normal_(self.rank1_r, 0, 0.1)
        torch.nn.init.normal_(self.rank1_s, 0, 0.1)
        
    def forward(self, x):
        in_channels = self.shared_weight.size(1)
        H = x.size(-2)
        W = x.size(-1)
        batch_size = x.size(-4)

        # Expand input X for the ensemble dimension
        X = x.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1, 1) if x.dim() == 4 else x   # Ensure correct shape: [ensemble_size, batch_size, in_channels, H, W] (check if the input is from previous layer or not)
        S = self.rank1_s
        R = self.rank1_r

        shared_weight = self.shared_weight # Shape: [out_channels, in_channels, kernel_size, kernel_size]
        shared_bias = self.bias  # Shape: [out_channels]

        # Element-wise multiplication of the input with S
        X_S = X * S  # Resulting shape: [ensemble_size, batch_size, in_channels, H, W]

        # Reshape X_S for convolution
        X_S_reshaped = X_S.view(self.ensemble_size * batch_size, in_channels, H, W)  # Shape: [ensemble_size * batch_size, in_channels, H, W]

        # # Apply shared weights to the result
        W_X_S = torch.nn.functional.conv2d(X_S_reshaped, self.shared_weight, None, self.stride, self.padding)

        # Reshape to separate ensemble and batch dimensions
        W_X_S = W_X_S.view(self.ensemble_size, batch_size, *W_X_S.shape[1:])  # Shape: [ensemble_size, batch_size, out_channels, H', W']

        # # Element-wise multiplication with R and add bias
        output = W_X_S * R  # Final shape: [ensemble_size, batch_size, out_channels, H', W']

        if self.bias is not None:
            output += shared_bias.view(1, -1, 1, 1)

        return output
    

# Convolutional block with BatchEnsemble
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ensemble_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = Conv2dBatchEnsemble(in_channels, out_channels, kernel_size, ensemble_size, stride, padding, bias=True)
        self.pool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size(-4)
        ensemble_size = self.conv.ensemble_size
        x = self.conv(x)
        x = x.view(ensemble_size*batch_size, x.size(-3), x.size(-2), x.size(-1)) # combine ensemble and batch size for pooling
        x = self.pool(x)
        x = self.relu(x)
        x = x.view(ensemble_size, batch_size, x.size(-3), x.size(-2), x.size(-1)) # separate ensemble and batch size
        # print(f'After reshaping shape: {x.shape}')
        return x

# CNN with BatchEnsemble
class BatchEnsemble_CNN(torch.nn.Module):
    def __init__(self, config):
        super(BatchEnsemble_CNN, self).__init__()
        self.config = config
        self.image_size = config.model.image_size
        self.conv_layers = config.model.conv_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"    

        self.conv_blocks = torch.nn.ModuleList([ConvBlock(*layer) for layer in self.conv_layers])
        self.ensemble_size = self.conv_layers[0][3]
        final_out_channels, final_image_size = self.calculate_final_layer_details(self.conv_layers)
        self.linear = torch.nn.Linear(final_out_channels * final_image_size * final_image_size, 32)
        self.fc = torch.nn.Linear(32, config.model.num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.hyper.lr, weight_decay=config.hyper.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.hyper.step_size, gamma=config.hyper.gamma)
        self.to(self.device)



    def forward(self, x):
        batch_size = x.size(0)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            

        x = x.view(self.ensemble_size, batch_size, -1)
        x = self.linear(x)
        x = self.fc(x)
        return x

    def calculate_final_layer_details(self, conv_layers):
        image_size = self.image_size
        out_channels = conv_layers[-1][1]
        for layer in conv_layers:
            image_size //= 2  # MaxPool2d layer

        return out_channels, image_size
    
    def neg_log_likelihood_categorical(self, y_pred, y_true):

        #cross entropy loss taking into account that we are doing ensembles
        losses = [torch.nn.functional.cross_entropy(y_pred[i], y_true, reduction='sum') for i in range(self.ensemble_size)]
        loss = sum(losses)

        return loss
    
    def train_custom(self, train_loader, test_loader):

        for epoch in range(self.config.hyper.epochs):
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self(data) 
                loss = self.neg_log_likelihood_categorical(output, target)/len(data)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            
            val_accuracy = 0.0
            val_loss = 0.0
            with torch.no_grad():
                for batch_idx, (val_data, val_target) in enumerate(test_loader):
                    val_data, val_target = val_data.to(self.device), val_target.to(self.device)
                    val_output = self(val_data)
                    loss = self.neg_log_likelihood_categorical(val_output, val_target)/len(val_data)
                    val_loss += loss.item()

                    #accuracy
                    _, predicted = torch.max(val_output, -1)
                    correct = (predicted == val_target).sum().item()
                    val_accuracy += correct / len(val_target)


            val_accuracy = val_accuracy / (len(test_loader) * self.ensemble_size)


            
            #logging
            print(f'Epoch: {epoch+1} / {self.config.hyper.epochs}\tTrain Loss: {train_loss}\tValidation Loss: {val_loss} \tValidation Accuracy: {val_accuracy}')
            wandb.log({"training_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})

        
            self.scheduler.step()

        print('Finished Training')
    
    def save_model(self, directory='models', filename='BatchEnsemble_CNN.pt'):
        directory = os.path.join(os.getcwd(), directory)
        print(f'Directory: {directory}')

        if not os.path.exists(directory):
            print(f'Directory {directory} does not exist. please try again.')
            return
        torch.save(self.state_dict(), os.path.join(directory, filename))

    @classmethod
    def load_model(cls, directory='models', filename='BatchEnsemble_CNN.pt', *args, **kwargs):
        directory = os.path.join(os.getcwd(), directory)
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(os.path.join(directory, filename)))
        return model
    


class Conv2d_Rank1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ensemble_size, stride=1, padding=1, prior_mu=0, prior_sigma=1, bias=True):
        super(Conv2d_Rank1, self).__init__()
        self.ensemble_size = ensemble_size
        self.stride = stride
        self.padding = padding
        self.log_scaler = 1.0
        self.epsilon = 1e-6
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Shared convolutional filters
        # self.shared_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.shared_weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).normal_(0, 1.0))
        self.prior_mu_weight = (torch.zeros_like(self.shared_weight)+prior_mu).to(self.device)
        self.prior_sigma_weight = (torch.zeros_like(self.shared_weight)+prior_sigma).to(self.device)

        self.bias = torch.nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.prior_mu_bias = (torch.zeros_like(self.bias)+prior_mu).to(self.device)
        self.prior_sigma_bias = (torch.zeros_like(self.bias)+prior_sigma).to(self.device)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

        # Rank-1 factors for each ensemble member
        self.rank1_r = torch.randn(ensemble_size, 1, out_channels, 1, 1).to(self.device)
        self.rank1_s = torch.randn(ensemble_size, 1, in_channels, 1, 1).to(self.device)

        self.rank1_r_mean = Parameter(self.rank1_r)
        self.rank1_r_log_std = Parameter(torch.zeros_like(self.rank1_r)-5.*self.log_scaler)

        self.rank1_s_mean = Parameter(self.rank1_s)
        self.rank1_s_log_std = Parameter(torch.zeros_like(self.rank1_s)-5.*self.log_scaler)

        # Prior parameters
        self.prior_mu_r = (torch.zeros_like(self.rank1_r_mean)+prior_mu).to(self.device)
        self.prior_sigma_r = (torch.zeros_like(self.rank1_r_mean)+prior_sigma).to(self.device)

        self.prior_mu_s = (torch.zeros_like(self.rank1_s_mean)+prior_mu).to(self.device)
        self.prior_sigma_s = (torch.zeros_like(self.rank1_s_mean)+prior_sigma).to(self.device)
        

    @property
    def rank1_r_std(self):
        return torch.exp(self.log_scaler*self.rank1_r_log_std)

    @property
    def rank1_s_std(self):
        return torch.exp(self.log_scaler*self.rank1_s_log_std)
    
    def distributions(self):
        u_dist = Normal(self.rank1_r_mean, self.rank1_r_std+self.epsilon)
        v_dist = Normal(self.rank1_s_mean, self.rank1_s_std+self.epsilon)

        return u_dist, v_dist
    
    def sample(self):
    
        self.rank1_r = self.distributions()[0].rsample()
        self.rank1_s = self.distributions()[1].rsample()
        return 
    
    def kl_divergence_u(self):
        r_dist = self.distributions()[0]

        log_q = r_dist.log_prob(self.rank1_r).sum()
        log_p =  Normal(self.prior_mu_r, self.prior_sigma_r).log_prob(self.rank1_r).sum()
        kl = -log_p + log_q
        return kl

 
    
    def kl_divergence_v(self):
        s_dist = self.distributions()[1]

        log_q = s_dist.log_prob(self.rank1_s).sum()
        log_p =  Normal(self.prior_mu_s, self.prior_sigma_s).log_prob(self.rank1_s).sum()
        kl = -log_p + log_q
        return kl


    def log_prob_w(self):

        log_prob_w = Normal(self.prior_mu_weight, self.prior_sigma_weight).log_prob(self.shared_weight).sum()
        if self.bias is not None:
            log_prob_w += Normal(self.prior_mu_bias, self.prior_sigma_bias).log_prob(self.bias).sum()
        return log_prob_w

    
        
    def forward(self, x):
        in_channels = self.shared_weight.size(1)
        H = x.size(-2)
        W = x.size(-1)
        batch_size = x.size(-4)

        # Expand input X for the ensemble dimension
        X = x.unsqueeze(0).repeat(self.ensemble_size, 1, 1, 1, 1) if x.dim() == 4 else x   # Ensure correct shape: [ensemble_size, batch_size, in_channels, H, W] (check if the input is from previous layer or not)
        S = self.rank1_s
        R = self.rank1_r

        shared_weight = self.shared_weight # Shape: [out_channels, in_channels, kernel_size, kernel_size]
        shared_bias = self.bias  # Shape: [out_channels]

        # Element-wise multiplication of the input with S
        X_S = X * S  # Resulting shape: [ensemble_size, batch_size, in_channels, H, W]

        # Reshape X_S for convolution
        X_S_reshaped = X_S.view(self.ensemble_size * batch_size, in_channels, H, W)  # Shape: [ensemble_size * batch_size, in_channels, H, W]

        # # Apply shared weights to the result
        W_X_S = torch.nn.functional.conv2d(X_S_reshaped, self.shared_weight, None, self.stride, self.padding)

        # Reshape to separate ensemble and batch dimensions
        W_X_S = W_X_S.view(self.ensemble_size, batch_size, *W_X_S.shape[1:])  # Shape: [ensemble_size, batch_size, out_channels, H', W']

        # # Element-wise multiplication with R and add bias
        output = W_X_S * R  # Final shape: [ensemble_size, batch_size, out_channels, H', W']
        if self.bias is not None:
            output += shared_bias.view(1, -1, 1, 1)

        return output
    

class ConvBlock_rank1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ensemble_size, stride=1, padding=1, pooling=True, prior_mu=0, prior_sigma=1000):
        super(ConvBlock_rank1, self).__init__()
        self.conv = Conv2d_Rank1(in_channels, out_channels, kernel_size, ensemble_size, stride, padding, prior_mu, prior_sigma, bias=True)
        self.pool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.pooling = pooling

    def forward(self, x):
        batch_size = x.size(-4)
        ensemble_size = self.conv.ensemble_size
        x = self.conv(x)
        x = self.relu(x)
        # print(f'Before reshaping shape: {x.shape}')

        # x = x.view(ensemble_size*batch_size, x.size(-3), x.size(-2), x.size(-1)) # combine ensemble and batch size for pooling
        # x = self.pool(x)
        # x = self.relu(x)
        if self.pooling:
            x = x.view(ensemble_size*batch_size, x.size(-3), x.size(-2), x.size(-1)) # combine ensemble and batch size for pooling
            x = self.pool(x)
            x = x.view(ensemble_size, batch_size, x.size(-3), x.size(-2), x.size(-1))

        # x = x.view(ensemble_size, batch_size, x.size(-3), x.size(-2), x.size(-1))
        # print(f'After reshaping shape: {x.shape}')
        return x
    

class Simple_rank1_CNN(torch.nn.Module):
    def __init__(self, config):
        super(Simple_rank1_CNN, self).__init__()
        self.config = config
        self.image_size = config.model.image_size
        self.conv_layers = config.model.conv_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conv_blocks = torch.nn.ModuleList([ConvBlock_rank1(*layer, prior_sigma=config.hyper.sigma_prior) for layer in self.conv_layers])
        self.ensemble_size = self.conv_layers[0][3]
        final_out_channels, final_image_size = self.calculate_final_layer_details(self.conv_layers)
        self.linear = torch.nn.Linear(final_out_channels * final_image_size * final_image_size, 32)
        self.fc = torch.nn.Linear(32, config.model.num_classes)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.hyper.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.hyper.step_size, gamma=config.hyper.gamma)
        self.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
            
        x = x.view(self.ensemble_size, batch_size, -1)
        x = self.linear(x)
        x = self.fc(x)
        return x

    def calculate_final_layer_details(self, conv_layers):
        image_size = self.image_size
        out_channels = conv_layers[-1][1]
        for layer in conv_layers:
            if layer[6]: # Check if pooling is applied in the layer
                image_size //= 2 # MaxPool2d layer

            # image_size //= 2  # MaxPool2d layer
        return out_channels, image_size
    

    def sample(self):
        for conv_block in self.conv_blocks:
            conv_block.conv.sample()
        return
    
    def kl_divergence_u(self):
        kl_divergence_u = 0
        for conv_block in self.conv_blocks:
            kl_divergence_u += conv_block.conv.kl_divergence_u()
        return kl_divergence_u
    
    def kl_divergence_v(self):
        kl_divergence_v = 0
        for conv_block in self.conv_blocks:
            kl_divergence_v += conv_block.conv.kl_divergence_v()
        return kl_divergence_v
        
        
    def log_prob_w(self):
        log_prob_w = 0
        for conv_block in self.conv_blocks:
            log_prob_w += conv_block.conv.log_prob_w() 
        return log_prob_w

    def neg_log_likelihood_categorical(self, y_pred, y_true):
        #cross entropy loss taking into account that we are doing ensembles
        losses = [torch.nn.functional.cross_entropy(y_pred[i], y_true, reduction='sum') for i in range(self.ensemble_size)]
        loss = sum(losses)
        return loss
   
    def train_custom(self, train_loader, test_loader):

        for epoch in range(self.config.hyper.epochs):
            
            train_loss = 0.0
            log_likelihood = 0.0
            kl_u = 0.0
            kl_v = 0.0
            log_p_w = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                self.sample()
                output = self(data) 
                neg_log_likelihood = self.neg_log_likelihood_categorical(output, target)
                kl_divergence_u = self.kl_divergence_u()*len(data)/len(train_loader.dataset)
                kl_divergence_v = self.kl_divergence_v()*len(data)/len(train_loader.dataset)
                log_prob_w = self.log_prob_w()*len(data)/len(train_loader.dataset)

                loss = neg_log_likelihood + kl_divergence_u + kl_divergence_v - log_prob_w
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                log_likelihood += neg_log_likelihood.item()
                kl_u += kl_divergence_u.item()
                kl_v += kl_divergence_v.item()
                log_p_w += log_prob_w.item()
        
            val_loss = 0.0
            accuracy = 0.0
            neg_log_likelihood_val = 0.0
            with torch.no_grad():
                for batch_idx, (val_data, val_target) in enumerate(test_loader):
                    val_data, val_target = val_data.to(self.device), val_target.to(self.device)
                    val_output = self(val_data)

                    val_negative_log_likelihood = self.neg_log_likelihood_categorical(val_output, val_target)
                    val_kl_divergence_u = self.kl_divergence_u()*len(val_data)/len(test_loader.dataset)
                    val_kl_divergence_v = self.kl_divergence_v()*len(val_data)/len(test_loader.dataset)
                    val_log_prob_w = self.log_prob_w()*len(val_data)/len(test_loader.dataset)

                    loss = val_negative_log_likelihood + val_kl_divergence_u + val_kl_divergence_v - val_log_prob_w
                    val_loss += loss.item()
                    neg_log_likelihood_val += val_negative_log_likelihood.item()

                
                    #accuracy
                    _, predicted = torch.max(val_output, -1)
                    correct = (predicted == val_target).sum().item()
                    accuracy += correct / len(val_target)

             # also divide by ensemble size
            accuracy = accuracy / (len(test_loader) * self.ensemble_size)  
            neg_log_likelihood_val = neg_log_likelihood_val / len(test_loader.dataset)

            #logging
            print(f'Epoch: {epoch+1} / {self.config.hyper.epochs}\tTrain Loss: {train_loss}\tValidation Loss: {val_loss} \tValidation Accuracy: {accuracy}')
            wandb.log({"training_loss": train_loss, "neg_log_likelihood": log_likelihood, "neg_log_likelihood_val": neg_log_likelihood_val, "kl_divergence_u": kl_u, "kl_divergence_v": kl_v, "log_prob_w": log_p_w, "val_loss": val_loss, "val_accuracy": accuracy})
                
    
                    
            self.scheduler.step()  
        
 
    
    def save_model(self, directory='models', filename='Simple_rank1_CNN.pt'):
        directory = os.path.join(os.getcwd(), directory)
        if not os.path.exists(directory):
            print(f'Directory {directory} does not exist. please try again.')
            return
        torch.save(self.state_dict(), os.path.join(directory, filename))

    @classmethod
    def load_model(cls, directory='models', filename='Simple_rank1_CNN.pt', *args, **kwargs):
        directory = os.path.join(os.getcwd(), directory)
        model = cls(*args, **kwargs)
        model.load_state_dict(torch.load(os.path.join(directory, filename)))
        return model
    
