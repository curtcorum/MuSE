import shutil
import os
import torch
import pickle 


class saveHelper(object):

    def __init__(self, logdir='logs',start='Starting'):
        root = logdir
        if os.path.isdir(root):
            print(logdir+"Exists")
            #shutil.rmtree(root,ignore_errors=True)
        else:
            os.makedirs(root)
        
        if os.path.isdir(root):
            print("Directory Exist; removing it")
            shutil.rmtree(root,ignore_errors=True)
        else:
            os.makedirs(root)
            
        os.makedirs(root+'/models')
        os.makedirs(root+'/samples')
        
        self.path = root+'/models/'
        self.summary = root+'/summary.txt'
        self.stats = root+'/stats.pkl'

        
        with open(self.summary, 'w') as f:
                f.write(start+"\n")
            
    def saveModel(self,net,epoch):
        file_name = 'net'+str(epoch)+".pt"
        torch.save(net.state_dict(),self.path+file_name)
        
    def write(self,text,scalar,epoch):
        with open(self.summary, 'a') as f:
            f.write('ep= \t'+str(epoch)+'\t'+text+'\t'+str(scalar)+'\n')
            
    def writeStat(self,dict):
        with open(self.stats, 'wb') as file:
            pickle.dump(dict, file)
            
    
            
            

    def readModel(self,net,epoch):
        
        file_name = 'net'+str(epoch)+".pt"
        net.load_state_dict(torch.load(self.path+file_name))
        net.eval()
        return net