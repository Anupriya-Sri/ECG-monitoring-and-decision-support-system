# Class for data loading
from scipy import io
import pickle
class Loader:
    def __init__(self, path):
        self.data_path = path
    
    def load_one_rec(self, filename):

        one_record = io.loadmat(f'{self.data_path}/{filename}.mat')
        ecg_one_rec = one_record['ECG'][0,0][2].T[:,1] # Replace the 1 with lead number for extracting features of other leads
        
        return ecg_one_rec
    
    def load_batch(self, rec_names, batch_size = 32):
        
        num_batches = len(rec_names) // batch_size 
        for batch in range(num_batches):
            data = []
            
            for rec in rec_names[batch_size*batch : batch_size * (batch+1)]:
                data.append(self.load_one_rec(rec))

            yield data