from tasks.base import Task
import numpy as np
import matplotlib.pyplot as plt

class CtxDepManteTask(Task):
    def __init__(self, root: str = "", version: str = "vanilla", duration: float = 500, delta_t: float = 2,
                           num_trials: int = 100, seed: int = 0, motion_coh: float = None, color_coh: float = None,
                           coherences: list = None, scale_coherences: float = 0.1, scale_context: float = 0.2,
                           scale_output: float = 0.2, fixation_duration: float = 50, varying_fixation: bool = False,
                           fixation_std: float = 10, loosen_time: float = 12, loosen_after: float = 20,
                           input_start_time: float = 100, input_end_time: float = 400, initial_silence: float = 50, 
                           output_delay: float = 50, input_duration: float = 80, add_noise: bool = False,
                           noise_scale: float = 0.03, *args, **kwargs):

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
        self.input_duration = input_duration
        self.varying_fixation = varying_fixation
        self.loosen_time = loosen_time
        self.loosen_after = loosen_after
        
        #Available versions can be seen here.
        self.available_versions = {"vanilla","random_input","loosen_coherence"}
        
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
        self.input_duration_discrete = int(self.input_duration // self.delta_t)
        self.loosen_after_discrete = int(self.loosen_after // self.delta_t)
        self.loosen_time_discrete = int(self.loosen_time // self.delta_t)
        
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
        assert (self.fixation_std >= 0), f"fixation_std cannot be negative. Given: {self.fixation_std}"
        
        if (self.version == "random_input"):
            assert (self.input_duration >= self.fixation_duration), f"Input duration must be greater or equal to fixatation duration. Given Input Duration: {self.input_duration} and Given Fixation Duration: {self.fixation_duration}"
            assert (self.input_duration >= 0 and self.input_duration < self.input_end_time - (self.input_start_time + self.initial_silence)), f"Input duration must be chosen such that it is suitable for input generation. It must be non-negative and smaller than {self.input_end_time - (self.input_start_time + self.initial_silence)}. However, given {self.input_duration}"
            assert (self.input_duration >= self.delta_t), "Input duration must be greater than delta_t value."
        
        if (self.version == "loosen_coherence"):
            assert (self.loosen_time > 0), f"'loosen_time' must be non-negative. Given: {self.loosen_time}"
            assert (self.loosen_after >= 0), f"loosen_after must be greater than or equal to 0. Given: {self.loosen_after}"
        
        
    def __getitem__(self, index: int):
        return self.data[index]
    
    def gen_input_output(self):
        inputs = np.zeros((self.T, self.input_dims))
        outputs = np.zeros((self.T, self.output_dims))
         
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
            
        #Setting the fixation duration parameter.
        if self.varying_fixation: #If True
            fixation_duration = int(np.random.normal(self.fixation_duration_discrete, self.fixation_std))
        else:
            fixation_duration = self.fixation_duration_discrete
            
        #Context decision is made. If 1 is chosen--> motion. If 2 is chosen--> Color is selected as the context.
        ctx_decision = np.random.choice([1, 2])
        
        if (self.version in {"vanilla", "loosen_coherence"}):
            #Context is created.
            if ctx_decision == 1:# Motion coherence is selected as the decision.
                inputs[self.input_start_time_discrete + self.initial_silence_discrete + fixation_duration:self.input_end_time_discrete, 2] = 1 * self.scale_context
                outputs[self.input_end_time_discrete + self.output_delay_discrete: ,0] = 1*self.scale_output if (motion_coh * self.scale_context) > 0 else -1*self.scale_output
            else: #ctx_decision == 2. Color coherence is selected as the decision.
                inputs[self.input_start_time_discrete + self.initial_silence_discrete + fixation_duration:self.input_end_time_discrete, 3] = 1 * self.scale_context
                outputs[self.input_end_time_discrete + self.output_delay_discrete: ,0] = 1*self.scale_output if (color_coh * self.scale_context) > 0 else -1*self.scale_output
                
        if self.version == "vanilla":
            inputs, outputs, ctx_decision, input_start, fixation_duration = self.gen_vanilla_input_output(inputs, outputs, ctx_decision, motion_coh, color_coh, fixation_duration)
        elif self.version == "random_input":
            inputs, outputs, ctx_decision, input_start, fixation_duration = self.gen_random_input_output(inputs, outputs, ctx_decision, motion_coh, color_coh, fixation_duration)
        elif self.version == "loosen_coherence":
            inputs, outputs, ctx_decision, input_start, fixation_duration = self.gen_loosen_coherence_output(inputs, outputs, ctx_decision, motion_coh, color_coh, fixation_duration)
            
        #Adding noise to the coherence dimensions.
        if self.add_noise:
            noise = np.random.normal(0, self.noise_scale, self.T)
            inputs[:, 0] +=noise #Adding noise to the motion coherence
            inputs[:, 1] +=noise #Adding noise to the color coherence
            
        return inputs, outputs, ctx_decision, input_start, fixation_duration
    
    def gen_vanilla_input_output(self, inputs, outputs, ctx_decision, motion_coh, color_coh, fixation_duration):
        input_start = self.input_start_time_discrete + self.initial_silence_discrete
        
        inputs[input_start:self.input_end_time_discrete, 0] = motion_coh * self.scale_coherences #Motion coherence in dimension 0.
        inputs[input_start:self.input_end_time_discrete, 1] = color_coh * self.scale_coherences #Color coherence in dimension 1.

        return inputs, outputs, ctx_decision, input_start, fixation_duration
    
    def gen_random_input_output(self, inputs, outputs, ctx_decision, motion_coh, color_coh, fixation_duration):
        possible_input_start_point = self.input_start_time_discrete + self.initial_silence_discrete
        possible_input_end_point = self.input_end_time_discrete - self.input_duration_discrete
        
        locs = np.arange(possible_input_start_point, possible_input_end_point+1)
        input_start = np.random.choice(locs) #Choosing a random input start point among all possible locations.
        
        #Context is selected.
        if ctx_decision == 1:# Motion coherence is selected as the decision.
            inputs[input_start + fixation_duration:self.input_end_time_discrete, 2] = 1 * self.scale_context
            outputs[self.input_end_time_discrete + self.output_delay_discrete: ,0] = 1*self.scale_output if (motion_coh * self.scale_context) > 0 else -1*self.scale_output
        else: #ctx_decision == 2. Color coherence is selected as the decision.
            inputs[input_start + fixation_duration:self.input_end_time_discrete, 3] = 1 * self.scale_context
            outputs[self.input_end_time_discrete + self.output_delay_discrete: ,0] = 1*self.scale_output if (color_coh * self.scale_context) > 0 else -1*self.scale_output
            
        inputs[input_start:input_start + self.input_duration_discrete, 0] = motion_coh * self.scale_coherences #Motion coherence in dimension 0.
        inputs[input_start:input_start + self.input_duration_discrete, 1] = color_coh * self.scale_coherences #Color coherence in dimension 1.

        return inputs, outputs, ctx_decision, input_start, fixation_duration

    def gen_loosen_coherence_output(self, inputs, outputs, ctx_decision, motion_coh, color_coh, fixation_duration):     
        input_start = self.input_start_time_discrete + self.initial_silence_discrete
        
        #Maintain coherence for "loosen_after" time period.
        inputs[input_start:input_start + self.loosen_after_discrete, 0] = motion_coh * self.scale_coherences #Motion coherence in dimension 0.
        inputs[input_start:input_start + self.loosen_after_discrete, 1] = color_coh * self.scale_coherences #Color coherence in dimension 1.

        #Creating a time array
        x = np.linspace(0,(self.input_end_time_discrete-input_start - self.loosen_after_discrete)//self.delta_t,self.input_end_time_discrete-input_start - self.loosen_after_discrete)
        #Decay period after "loosen_after" time period ends.
        inputs[input_start + self.loosen_after_discrete:self.input_end_time_discrete, 0] = np.exp(-x/self.loosen_time_discrete)*(motion_coh * self.scale_coherences) #Motion coherence in dimension 0.
        inputs[input_start + self.loosen_after_discrete:self.input_end_time_discrete, 1] = np.exp(-x/self.loosen_time_discrete)*(color_coh * self.scale_coherences) #Color coherence in dimension 1.

        return inputs, outputs, ctx_decision, input_start, fixation_duration
        
    def gen_batch(self, batch_size):
        inputs = np.zeros((batch_size, self.T, self.input_dims))
        outputs = np.zeros((batch_size, self.T, self.output_dims))
        ctx_decision_list = []
        input_start_list = []
        fixation_duration_list = []
        
        for batch in range(batch_size):
            inputs[batch], outputs[batch], ctx_decision, input_start, fixation_duration = self.gen_input_output()
            ctx_decision_list.append(ctx_decision)
            input_start_list.append(input_start)
            fixation_duration_list.append(fixation_duration)
            
        return inputs, outputs, ctx_decision_list, input_start_list, fixation_duration_list
    
    def visualize_task(self, figsize=(16, 18), show_legend: bool = True):
        inputs, outputs, ctx_decisions, input_start, fixation_duration = self.gen_input_output()
        
        time = np.arange(0, self.duration, self.delta_t)

        # Convert input start and end times to discrete time points
        input_start_time = self.input_start_time
        input_end_time = self.input_end_time
        
        fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plotting Motion
        axs[0].plot(time, inputs[:, 0], label="Motion Context", color='r',lw=3)
        axs[0].set_title('Motion Coherence')
        axs[0].set_ylabel('Coherence Strength')
        axs[0].axvline(input_start_time, color='m', linestyle='--', label='Input Start/End Time')
        axs[0].axvline(input_end_time, color='m', linestyle='--')

        axs[0].axvspan(input_start*self.delta_t,
                       input_start*self.delta_t + self.delta_t*fixation_duration,
                       color='green', alpha=0.3, label="Fixation period")
        if (self.version == "loosen_coherence"):
            axs[0].axvline(self.loosen_after + self.input_start_time + self.initial_silence , color='k', linestyle='dotted',label="Start loosing the coherence")
        if show_legend:
            axs[0].legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize='small')
        axs[0].grid(False)
        
        # Plotting Color Coherence
        axs[1].plot(time, inputs[:, 1], label='Color Context', color='b',lw=3)
        axs[1].set_title('Color Coherence')
        axs[1].set_ylabel('Coherence Strength')
        axs[1].axvline(input_start_time, color='m', linestyle='--', label='Input Start/End Time')
        axs[1].axvline(input_end_time, color='m', linestyle='--')
       
        axs[1].axvspan(input_start*self.delta_t,
                       input_start*self.delta_t + self.delta_t*fixation_duration,
                       color='green', alpha=0.3, label="Fixation period")
        if (self.version == "loosen_coherence"):
            axs[1].axvline(self.loosen_after + self.input_start_time + self.initial_silence , color='k', linestyle='dotted',label="Start loosing the coherence")
        if show_legend:
            axs[1].legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize='small')
        axs[1].grid(False)
        
        # Plotting Decision on axsis 2 (motion)
        axs[2].plot(time, inputs[:, 2], label='Decision on Motion', color="red",lw=3)
        axs[2].plot(time, inputs[:, 3], label='Decision on Color', color="blue",lw=3)
        axs[2].set_title('Context (Motion: Red Line | Color: Blue Line)')
        axs[2].set_ylabel('Context Strength')
        axs[2].axvline(input_start_time, color='m', linestyle='--', label='Input Start/End Time')
        axs[2].axvline(input_end_time, color='m', linestyle='--')
        if show_legend:
            axs[2].legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize='small')
        axs[2].grid(False)
        
        # Plotting Output (actual decision made)
        if ctx_decisions == 1:#Motion was selected
            decision_color = "red"
            decision_label = "Motion is selected as decision. (Red Color, first axsis)"
        else:#Color was selected
            decision_color = "blue"
            decision_label = "Color is selected as decision. (Blue Color, second axsis)"
              
        axs[3].plot(time, outputs[:, 0], label=decision_label, color=decision_color,lw=3)
        axs[3].set_title('Decision Made')
        axs[3].set_ylabel('Decision Strength')
        axs[3].axvline(self.output_delay+self.input_end_time , color='k', linestyle='--', label='Output Start Time (Input End + Output Delay)')
        if show_legend:
            axs[3].legend(loc='lower left', bbox_to_anchor=(0, 0), fontsize='small')
        axs[3].grid(False)
        
        # Ensure value 0 is fixed as the center on all subplots
        for ax in axs:
            ymin, ymax = ax.get_ylim()
            limit = max(abs(ymin), abs(ymax))
            ax.set_ylim(-limit, limit)
            ax.axhline(0, color='black', linewidth=0.3)
                
        plt.suptitle(f"Context Dependent Mante Task | Version: {self.version}", fontsize=16)      
        plt.savefig(f"mante_task_{self.version}.pdf")
        plt.tight_layout()
            
    def visualize_rnn_output(self, model, loss=None, label_loss=None):
        # Generate a batch of data with one trial.
        inputs, expected_outputs, ctx_decision, input_start, fixation_duration = self.data
        
        if isinstance(input_start, (list, np.ndarray)):
            input_start = input_start[0]
        if isinstance(fixation_duration, (list, np.ndarray)):
            fixation_duration = fixation_duration[0]
        if isinstance(ctx_decision, (list, np.ndarray)):
            ctx_decision = ctx_decision[0]
        
        predicted_outputs, _ = model.run_rnn(inputs, device=model.device)
        
        #plotting for the first trial
        inputs = np.squeeze(inputs[0])
        expected_outputs = expected_outputs[0]
        predicted_output = predicted_outputs[0, :, :]

        time = np.linspace(0, self.duration, inputs.shape[0])
 
        fig, axs = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
        
        # Motion coherence
        axs[0].plot(time, inputs[:, 0], label="Motion Coherence", color='r', lw=3)
        axs[0].set_title('Motion Coherence')
        axs[0].set_ylabel('Coherence Strength')
        axs[0].axvline(self.input_start_time, color='m', linestyle='--', label='Input Start/End')
        axs[0].axvline(self.input_end_time, color='m', linestyle='--')
        axs[0].axvspan(input_start * self.delta_t,
                       input_start * self.delta_t + self.delta_t * fixation_duration,
                       color='green', alpha=0.3, label="Fixation period")
        axs[0].legend(loc='lower left', fontsize='small')
        axs[0].grid(False)
        
        # Color coherence
        axs[1].plot(time, inputs[:, 1], label="Color Coherence", color='b', lw=3)
        axs[1].set_title('Color Coherence')
        axs[1].set_ylabel('Coherence Strength')
        axs[1].axvline(self.input_start_time, color='m', linestyle='--', label='Input Start/End')
        axs[1].axvline(self.input_end_time, color='m', linestyle='--')
        axs[1].axvspan(input_start * self.delta_t,
                       input_start * self.delta_t + self.delta_t * fixation_duration,
                       color='green', alpha=0.3, label="Fixation period")
        axs[1].legend(loc='lower left', fontsize='small')
        axs[1].grid(False)
        
        # Contexts 
        axs[2].plot(time, inputs[:, 2], label="Context: Motion", color='r', lw=3)
        axs[2].plot(time, inputs[:, 3], label="Context: Color", color='b', lw=3)
        axs[2].set_title('Context Signals')
        axs[2].set_ylabel('Context Strength')
        axs[2].axvline(self.input_start_time, color='m', linestyle='--', label='Input Start/End')
        axs[2].axvline(self.input_end_time, color='m', linestyle='--')
        axs[2].legend(loc='lower left', fontsize='small')
        axs[2].grid(False)
        
        # predicted and groundtruth
        # Choose a decision color based on the context decision.
        if ctx_decision == 1:
            decision_color = "red"
            decision_label = "Predicted Output (Motion)"
        else:
            decision_color = "blue"
            decision_label = "Predicted Output (Color)"
        
        axs[3].plot(time, predicted_output[:, 0], label=decision_label, color=decision_color, lw=3)
        axs[3].plot(time, expected_outputs[:, 0], label="Groundtruth Output", color='g', linestyle='--', lw=3)
        axs[3].set_title('Output (Groundtruth vs Predicted)')
        axs[3].set_ylabel('Decision Strength')
        axs[3].axvline(self.output_delay + self.input_end_time, color='k', linestyle='--', label='Output Start Time')
        axs[3].legend(loc='lower left', fontsize='small')
        axs[3].set_xlabel("Time (ms)")
        axs[3].grid(False)
        
        # Center the y-axis at zero in all subplots.
        for ax in axs:
            ymin, ymax = ax.get_ylim()
            limit = max(abs(ymin), abs(ymax))
            ax.set_ylim(-limit, limit)
            ax.axhline(0, color='black', linewidth=0.3)
        
        plt.suptitle(f"Context Dependent Mante Task with RNN Output | Version: {self.version}", fontsize=16)
        plt.tight_layout()
        # plt.show()


# Visualize the task if data.py is called
if __name__ == '__main__':
    np.random.seed(None)
    seed = np.random.randint(0,100)
    task = CtxDepManteTask(seed=seed, version = "vanilla",add_noise = False)
    task.visualize_task()
        
        
         
    