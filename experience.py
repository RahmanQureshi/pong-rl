class Experience:


    def __init__(self, state, action, result_state, reward, terminal):
        self.state = state
        self.action = action
        self.result_state = result_state
        self.reward = reward
        # true or false
        # indicates whether the state and action ended in a terminal state
        # for deep q learning, this means the result_state is not used.
        self.terminal = terminal
