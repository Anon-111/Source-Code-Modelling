import math
import numpy as np

class Inverse_Square_Root_Learning_rate():
  def __init__(self, learning_rate,
               warmup_steps):
    self.learning_rate = learning_rate
    self.warmup_steps = warmup_steps
    self.iteration = 0
    
  def __call__(self):
    self.iteration += 1
    return self.learning_rate * (1 / math.sqrt(max(self.iteration,
                                                   self.warmup_steps)))
                                                   
  def plot(self, iterations = 100):
    import matplotlib.pyplot as plt
    lr = []
    for i in range(iterations):
      lr.append(self.__call__())
    plt.plot(lr)
    plt.savefig('inverse-learning-rate.png')
    plt.close()

class Polynomial_Decay_Learning_Rate():
  def __init__(self, learning_rate,
               max_epochs,
               min_learning_rate = 1e-4,
               power = 0.5):
    self.learning_rate = learning_rate
    self.max_epochs = max_epochs
    self.min_learning_rate = min_learning_rate
    self.power = power
    
    self.iteration = -1
    
  def __call__(self):
    self.iteration += 1
    return (self.learning_rate - self.min_learning_rate) * (1 - (self.iteration / self.max_epochs)) ** self.power + self.min_learning_rate
    
  def plot(self, iterations = 100):
    import matplotlib.pyplot as plt
    lr = []
    for i in range(iterations):
      lr.append(self.__call__())
    plt.plot(lr)
    plt.savefig('polynomial-decay-learning-rate.png')
    plt.close()
    
class Cosine_Learning_Rate():
  def __init__(self, learning_rate,
               decay_steps,
               min_learning_rate,
               alpha = 0.0):
    self.learning_rate = learning_rate
    self.decay_steps = decay_steps
    self.min_learning_rate = min_learning_rate
    self.alpha = alpha
    
    self.iteration = 0
    
  def __call__(self):
    global_step = min(self.iteration,
                      self.decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / self.decay_steps))
    decayed = (1 - self.alpha) * cosine_decay + self.alpha
    self.iteration += 1
    return (self.learning_rate - self.min_learning_rate) * decayed + self.min_learning_rate
    
  def plot(self, iterations = 100):
    import matplotlib.pyplot as plt
    lr = []
    for i in range(iterations):
      lr.append(self.__call__())
    plt.plot(lr)
    plt.savefig('cosine-learning-rate.png')
    plt.close()
    
class Cyclic_Learning_Rate():
  def __init__(self, learning_rate=0.01,
               max_lr=0.1,
               step_size=20.,
               gamma=0.999,
               mode='exp_range',
               warmup_steps = 100,
               stable_steps = 100):
    # code is based heavily on https://github.com/mhmoodlan/cyclic-learning-rate
    # cyclic learning rate was introduced in https://arxiv.org/pdf/1506.01186.pdf
    self.learning_rate = learning_rate
    # the learning rate of the model. Will obviously be altered in real-time by the algorithm
    self.max_lr = max_lr
    # the maximum possible learning-rate
    self.step_size = step_size
    # the step_size for the cycling. step_size per half-cycle. Recommended 2-8 cycles per epoch
    self.gamma = gamma
    # decay hyperparameter for exp
    self.mode = mode
    # the mode used for analysis
    # 'triangular', 'triangular2', 'exp_range'
    self.warmup_steps = warmup_steps
    # the number of linear warmup steps, as defined in https://arxiv.org/pdf/1706.03762.pdf
    self.stable_steps = stable_steps
    # the number of stable steps of the learning rate
    
    self.cyclic_step = 0
    
    if self.warmup_steps != 0:
      self.cyclic_step = -1 * self.warmup_steps
    else:
      self.cyclic_step = 0
    
  def __call__(self):
    self.cyclic_step += 1
    
    cyclic_step = self.cyclic_step
    if cyclic_step < self.warmup_steps:
      cyclic_step -= self.warmup_steps
    
    if cyclic_step <= 0:
      return (self.max_lr / self.warmup_steps) * (self.warmup_steps + cyclic_step)
    elif(cyclic_step - (self.warmup_steps + self.stable_steps) < 0):
      return self.max_lr
      
    cyclic_step -= (self.warmup_steps + self.stable_steps)
      
    # the step is increased each __call__
    # because class is not private, the step can be changed from outside the class if need be
    cycle = math.floor(1 + cyclic_step / (2 * self.step_size))
    
    x = abs(1 + (cyclic_step / self.step_size) - 2 * cycle)
    
    clr = max(0.0, 1 - x) * (self.max_lr - self.learning_rate)
    
    if self.mode == 'triangular2':
      clr = clr / (2 ** (cycle - 1))
      
    if self.mode == 'exp_range':
      clr = clr * (self.gamma ** cyclic_step)
    
    return clr + self.learning_rate
    
  def plot(self, iterations = 100):
    import matplotlib.pyplot as plt
    lr = []
    for i in range(iterations):
      lr.append(self.__call__())
    plt.plot(lr)
    plt.savefig('cyclic-learning-rate-{}.png'.format(self.mode))
    plt.close()
    
class Linear_Learning_Rate():
  def __init__(self, learning_rate,
               final_learning_rate,
               decay_steps,
               warmup_steps = 0,
               stable_steps = 0):
    self.learning_rate = learning_rate
    self.final_learning_rate = final_learning_rate
    self.decay_steps = decay_steps
    self.warmup_steps = warmup_steps
    self.stable_steps = stable_steps
    
    self.steps = 0
    
  def __call__(self):
    self.steps += 1
    if self.steps <= self.warmup_steps:
      return (self.learning_rate * self.steps) / (self.warmup_steps)
    elif self.steps <= self.warmup_steps + self.stable_steps:
      return self.learning_rate
    elif self.steps <= (self.warmup_steps + self.stable_steps + self.decay_steps):
      steps = self.decay_steps - self.steps + self.warmup_steps + self.stable_steps
      return self.final_learning_rate + ((self.learning_rate - self.final_learning_rate) * steps) / (self.decay_steps)
    else:
      return self.final_learning_rate
  
  def plot(self, iterations = 100):
    import matplotlib.pyplot as plt
    lr = []
    for i in range(iterations):
      lr.append(self.__call__())
    plt.plot(lr)
    plt.savefig('linear-learning-rate.png')
    plt.close()
  
class Learning_Rate():
  def __init__(self, initial_learning_rate,
               stable_learning_rate,
               final_learning_rate,
               warmup_rate,
               decay_rate,
               warmup_steps,
               decay_steps,
               step_size = 1000):
    self.initial_learning_rate = initial_learning_rate
    # the initial learning rate, at the start of the warmup steps
    # if the warmup steps is set to 0 or warmup_rate == 'constant', initial_learning_rate == stable_learning_rate
    self.stable_learning_rate = stable_learning_rate
    # the learning rate at the end of warmup steps, and the beginning of the decay steps
    self.final_learning_rate = final_learning_rate
    # the final learning rate
    # if decay_rate == 'constant', stable_learning_rate == final_learning_rate
    self.warmup_rate = warmup_rate
    # warmup refers to where the learning rate inceases over iterations
    # 'constant', 'linear-warmup'
    self.decay_rate = decay_rate
    # learning rate decreases over iterations
    # 'constant' 'linear-decay' 'polynomial' 'cosine' 'cyclic-triangular' 'cyclic-triangular2' 'cyclic-exp'
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps
    self.step_size = step_size
    # only used for cyclic learning rates
    
    self.iteration = 0
    
    if self.warmup_rate == 'constant':
      self.warmup_learning_rate = self.initial_learning_rate
      assert self.initial_learning_rate == self.stable_learning_rate
      # if the learning rate is constant, then warmup_learning_rate is set to a float, not a class from learning_rate.py
    elif self.warmup_rate == 'linear-warmup':
      self.warmup_learning_rate = Linear_Learning_Rate(learning_rate = self.stable_learning_rate - self.initial_learning_rate,
                                                       final_learning_rate = 0,
                                                       decay_steps = 0,
                                                       warmup_steps = self.warmup_steps,
                                                       stable_steps = 0)
      
    if self.decay_rate == 'constant':
      self.decay_learning_rate = self.stable_learning_rate
      assert self.stable_learning_rate == self.final_learning_rate
    elif self.decay_rate == 'linear-decay':
      self.decay_learning_rate = Linear_Learning_Rate(learning_rate = self.stable_learning_rate,
                                                      final_learning_rate = self.final_learning_rate,
                                                      decay_steps = self.decay_steps,
                                                      warmup_steps = 0,
                                                      stable_steps = 0)
    elif self.decay_rate == 'polynomial':
      self.decay_learning_rate = Polynomial_Decay_Learning_Rate(learning_rate = self.stable_learning_rate,
                                                                max_epochs = self.decay_steps,
                                                                min_learning_rate = self.final_learning_rate)
    elif self.decay_rate == 'cosine':
      self.decay_learning_rate = Cosine_Learning_Rate(learning_rate = self.stable_learning_rate,
                                                      decay_steps = self.decay_steps,
                                                      min_learning_rate = self.final_learning_rate)
    else:
      if self.decay_rate == 'cyclic-triangular':
        mode = 'triangular'
      elif self.decay_rate == 'cyclic-triangular2':
        mode = 'triangular2'
      elif self.decay_rate == 'cyclic-exp':
        mode = 'exp_range'
      self.decay_learning_rate = Cyclic_Learning_Rate(learning_rate = self.stable_learning_rate,
                                                      step_size = self.step_size,
                                                      mode = mode,
                                                      warmup_steps = 0,
                                                      stable_steps = 100)
                                                      
  def call(self):
    self.iteration += 1
    if self.iteration < self.warmup_steps:
      if self.warmup_rate == 'constant':
        return self.warmup_learning_rate
      else:
        return self.warmup_learning_rate()
    else:
      if self.decay_rate == 'constant':
        return self.decay_learning_rate
      else:
        return self.decay_learning_rate()

if __name__ == '__main__':
  learning_rate = Inverse_Square_Root_Learning_rate(learning_rate = 1.0,
                                                    warmup_steps = 1e4)
  learning_rate.plot(iterations = int(2.5e4))
  exit()
  learning_rate = Cosine_Learning_Rate(learning_rate = 1e-2,
                                       decay_steps = 100,
                                       min_learning_rate = 1e-5)
  learning_rate.plot(iterations = 100)
  learning_rate = Polynomial_Decay_Learning_Rate(learning_rate = 1e-2,
                                                 max_epochs = 100)
  learning_rate.plot(iterations = 100)
  learning_rate = Linear_Learning_Rate(1e-2,
                                       1e-4,
                                       25,
                                       25,
                                       25)
  learning_rate.plot(iterations = 100)
  learning_rate = Cyclic_Learning_Rate(warmup_steps = 0,
                                       stable_steps = 0,
                                       mode='exp_range')
  learning_rate.plot(iterations = 1000)
  learning_rate = Cyclic_Learning_Rate(warmup_steps = 0,
                                       stable_steps = 0,
                                       mode='triangular2')
  learning_rate.plot(iterations = 1000)
  learning_rate = Cyclic_Learning_Rate(warmup_steps = 0,
                                       stable_steps = 0,
                                       mode = 'triangular')
  learning_rate.plot(iterations = 1000)