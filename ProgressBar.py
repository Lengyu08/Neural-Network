class ProgressBar:
    def __init__(self, total_steps, bar_len=60):
        self.total_steps = total_steps
        self.bar_len = bar_len

    def update(self, current_step):
        filled_len = int(self.bar_len * current_step / self.total_steps)
        bar = '█' * filled_len + ' ' * (self.bar_len - filled_len)
        percentage = round(100.0 * current_step / float(self.total_steps), 1)
        status = "\033[32mComplete\033[0m" if current_step == self.total_steps else "\033[31mProgress\033[0m"
        print('\r[{}] {}% {}'.format(bar, percentage, status), end='')
    