
import numpy as np
import collections 
from .element import Element

SOUND_SPEED = 340.0
 
MIC_DISTANCE_4 = 0.081
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)


def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]
 
    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)
 
 
    if all(np.abs(R)) == True:
        cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    else:
        cc = np.fft.irfft(R, n =(interp * n))
 
    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)
 
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
 
    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift
 
    tau = shift / float(interp * fs)
    
    return tau, cc

class DOA(Element):
    def __init__(self, rate=16000, chunks=10):
        super(DOA, self).__init__()

        self.queue = collections.deque(maxlen=chunks)
        self.sample_rate = rate

        self.pair = [[0, 2], [1, 3]]

    def put(self, data):
        self.queue.append(data)

        super(DOA, self).put(data)

    def get_direction(self):
        tau = [0, 0]
        theta = [0, 0]

        buf = b''.join(self.queue)
        buf = np.fromstring(buf, dtype='int16')
        for i, v in enumerate(self.pair):
            tau[i], _ = gcc_phat(buf[v[0]::4], buf[v[1]::4], fs=self.sample_rate, max_tau=MAX_TDOA_4, interp=1)
            theta[i] = np.arcsin(tau[i] / MAX_TDOA_4) * 180 / np.pi

        if np.abs(theta[0]) < np.abs(theta[1]):
            if theta[1] > 0:
                best_guess = (theta[0] + 360) % 360
            else:
                best_guess = (180 - theta[0])
        else:
            if theta[0] < 0:
                best_guess = (theta[1] + 360) % 360
            else:
                best_guess = (180 - theta[1])

            best_guess = (best_guess + 270) % 360

        best_guess = (-best_guess + 120) % 360

        return best_guess



def main():
    src = Source(rate=16000, channels=4, frames_size=320)
    ch1 = ChannelPicker(channels=4, pick=1)
    doa = DOA(rate=16000)
    src.link(ch1)
    src.link(doa)
    src.recursive_start()

    while True:
        try:
            time.sleep(1)
            position, amplitute = doa.get_direction()
            print(position)
            time.sleep(3)
        except KeyboardInterrupt:
            break

    src.recursive_stop()
    time.sleep(1)


if __name__ == '__main__':
    main()
