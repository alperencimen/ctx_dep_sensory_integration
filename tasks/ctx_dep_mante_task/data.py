from tasks.base import Task
import numpy as np
import matplotlib.pyplot as plt

class CtxDepManteTask(Task):
    def __init__(self, root: str = "", version: str = "vanilla", duration: float = 500, delta_t: float = 2,
                           num_trials: int = 100, seed: int = 0, motion_coh: float = None, color_coh: float = None,
                           coherences: list = None, scale_coherences: float = 0.1, scale_context: float = 0.2,
                           scale_output: float = 0.2, fixation_duration: float = 50, fixation_std: float = 0.2, 
                           input_start_time: float = 100, input_end_time: float = 400, initial_silence: float = 50,
                           output_delay: float = 50, add_noise: bool = False, noise_scale: float = 0.02,
                           *args, **kwargs):

        self.seed = seed
        np.random.seed(self.seed)
        
        if coherences is None:
            self.coherences = [-4, -2, -1, 1, 2, 4]
        else:
            self.coherences = coherences
            
            
        #Assigning class variables
        self.root = root
        self.version = version
        self.duration = duration
        self.delta_t = delta_t
        self.num_trials = num_trials
        self.input_start_time = input_start_time
        self.input_end_time = input_end_time
        self.initial_silence = initial_silence
        self.output_delay = output_delay
        self.add_noise = add_noise
        self.noise_scale = noise_scale
        self.scale_coherences = scale_coherences
        self.scale_context = scale_context
        self.motion_coh = motion_coh
        self.color_coh = color_coh
        self.fixation_duration = fixation_duration
        self.fixation_std = fixation_std
        self.scale_output = scale_output
        
        
        #Available versions can be seen here.
        self.available_versions = {"vanilla"}
        
        """Input dimension is 4. This is because in this task, there are two stimuli, color and motion. And each corresponding can be activated in a given trial which constitutes third and fourth dimensions."""
        self.input_dims = 4
        self.output_dims = 1
        
        # Initiate the task
        super().__init__(root, version, duration, delta_t, num_trials)
        if self.root !='':
            np.savez(self.root, inputs = self.data[0],outputs = self.data[1]);
        
    def _generate_dataset(self):
        self._discretize_input()
        self._check_input_validity()
        return self.gen_batch(self.num_trials)
    
    def get_input_output_dims(self):
        return self.input_dims, self.output_dims
    
    def _discretize_input(self):
        # Discretize the inputs
        self.T = int(self.duration // self.delta_t)
        self.input_start_time_discrete = int(self.input_start_time // self.delta_t)
        self.input_end_time_discrete = int(self.input_end_time // self.delta_t)
        self.initial_silence_discrete = int(self.initial_silence // self.delta_t)
        self.output_delay_discrete = int(self.output_delay // self.delta_t)
        self.fixation_duration_discrete = int(self.fixation_duration // self.delta_t)
        
    def _check_input_validity(self):
        # Make sure the input is correct
        assert (self.version in self.available_versions), "Undefined version."
        assert (self.num_trials >= 1), "Number of trials must be non negative and greater or equal than 1."
        assert (self.seed >= 0), "Seed must be non-negative"
        assert (self.delta_t > 0), "Delta t must be non negative!"
        assert (self.input_end_time > self.input_start_time > 0), "Input start time must be earlier than input end time and they must be non-negative!"
        assert (self.initial_silence + self.input_start_time < self.input_end_time), "There is no chance to create any input. Try to increase input_end time or decrease input_start time and/or initial silence."
        assert (self.duration >= self.output_delay + self.input_end_time), "Input end time must be earlier than input_end time. Also, the summation of input_end_time + output_delay must be less than or equal to duration."
        assert (self.initial_silence > 0 and self.output_delay > 0), "input_duration and output_delay must be non-negative and greater than 0."
        assert (self.output_delay < self.duration - self.input_end_time), "Output delay time must be in the window of duration - input_end_time."
        assert all(isinstance(x, (int, float)) for x in self.coherences), f"Coherence list contains non-numeric elements! Given: {self.coherences}"
        
    def __getitem__(self, index: int):
        return self.data[index]
    
    def gen_input_output(self):
        inputs = np.zeros((self.T, self.input_dims))
        outputs = np.zeros((self.T, self.output_dims))
        
        
        #Context decision is made.
        ctx_decision = np.random.choice([1, 2])
        
        #Motion Coherence selection
        if self.motion_coh == None:#if none, chose randomly
            motion_coh = self.coherences[np.random.randint(0, len(self.coherences)-1)]
        else:
            motion_coh = self.motion_coh
            
        #Color Coherence  selection
        if self.color_coh == None:#if none, chose randomly
            color_coh = self.coherences[np.random.randint(0, len(self.coherences)-1)]
        else:
            color_coh = self.color_coh
            
        input_start = self.input_start_time_discrete + self.initial_silence_discrete
        inputs[input_start:self.input_end_time_discrete, 0] = motion_coh * self.scale_coherences #Motion coherence in dimension 0.
        inputs[input_start:self.input_end_time_discrete, 1] = color_coh * self.scale_context #Color coherence in dimension 1.

        
        #Context is selected.
        if ctx_decision == 1:# Motion coherence is selected as the decision.
            inputs[input_start + self.fixation_duration_discrete:self.input_end_time_discrete, 2] = 1 * self.scale_context
            outputs[self.input_end_time_discrete + self.output_delay_discrete: ,0] = 1*self.scale_output if (motion_coh * self.scale_context) > 0 else -1*self.scale_output
           
        else: #ctx_decision == 2. Color coherence is selected as the decision.
            inputs[input_start + self.fixation_duration_discrete:self.input_end_time_discrete, 3] = 1 * self.scale_context
            outputs[self.input_end_time_discrete + self.output_delay_discrete: ,0] = 1*self.scale_output if (color_coh * self.scale_context) > 0 else -1*self.scale_output
            
        
        if self.add_noise:
            noise = np.random.normal(0, self.noise_scale, self.T)
            inputs[:, 0] +=noise #Adding noise to the motion coherence
            inputs[:, 1] +=noise #Adding noise to the color coherence
            
        return inputs, outputs
        
    
    def gen_batch(self, batch_size):
        inputs = np.zeros((batch_size, self.T, self.input_dims))
        outputs = np.zeros((batch_size, self.T, self.output_dims))
 
        for batch in range(batch_size):
            inputs[batch], outputs[batch] = self.gen_input_output()
        return inputs, outputs
    
    
    def visualize_task(self, figsize=(20, 15), show_legend: bool = True):
        inputs, outputs = self.gen_input_output()
        
        time = np.arange(0, self.duration, self.delta_t)

        # Convert input start and end times to discrete time points
        input_start_time = self.input_start_time
        input_end_time = self.input_end_time
        
        fig, axs = plt.subplots(5, 1, figsize=figsize, sharex=True)
        
        # Plotting Motion
        axs[0].plot(time, inputs[:, 0], label="Input 1", color='r')
        axs[0].set_title('Motion Coherence')
        axs[0].set_ylabel('Coherence Strength')
        axs[0].axvline(input_start_time, color='m', linestyle='--', label='Input Start/End Time')
        axs[0].axvline(input_end_time, color='m', linestyle='--')
        if show_legend:
            axs[0].legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize='small')
        axs[0].grid(True)
            
    
    def visualize_rnn_output(self, model, loss=None, label_loss=None):
        pass
    
    
# Visualize the task if data.py is called
if __name__ == '__main__':
    np.random.seed(None)
    seed = np.random.randint(0,100)
    task = CtxDepManteTask(seed=seed, version = "vanilla",add_noise = False)
    task.visualize_task()
        
        
         
    