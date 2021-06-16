import os
import pathlib
import collections

import numpy as np
import torch

import torch.utils.data
import cv2  # pytype: disable=attribute-error

class EchoSet(torch.utils.data.Dataset):
    def __init__(self, 
                 root,
                 split="train",
                 min_spacing=16,
                 max_length=128,
                 fixed_length=128,
                 pad=8,
                 random_clip=False,
                 dataset_mode='repeat',
                 SDmode = 'reg' # reg, cla
                 ):

        self.folder       = pathlib.Path(root)
        self.split        = split
        self.max_length   = max_length
        self.min_length   = min_spacing
        self.fixed_length = fixed_length
        self.padding      = pad
        self.mode         = dataset_mode # repeat, sample, full
        self.random_clip  = random_clip
        self.attenuation  = 3 # Exponent to smooth the labels, choose odd numbers, not too big
        self.SDmode       = SDmode #reg, cla

        self.fnames       = []
        self.outcome      = []
        self.ejection     = []
        self.fps          = []

        if not os.path.exists(root):
            raise ValueError("Path does not exist: "+root)

        with open(self.folder / "FileList.csv") as f:
            self.header   = f.readline().strip().split(",")
            filenameIndex = self.header.index("FileName")
            splitIndex    = self.header.index("Split")
            efIndex       = self.header.index("EF")
            fpsIndex      = self.header.index("FPS")
            for line in f:
                lineSplit = line.strip().split(',')
                
                # Get name of the video file
                fileName = os.path.splitext(lineSplit[filenameIndex])[0]+".avi"
                # Get subset category (train, val, test)
                fileMode = lineSplit[splitIndex].lower()
                
                # Get EF
                ef = lineSplit[efIndex]
                
                fps = lineSplit[fpsIndex]
                
                # Keep only entries where the video exist and "mode" corresponds to what is asked
                if split in ["all", fileMode] and os.path.exists(self.folder / "Videos" / fileName):
                    self.fnames.append(fileName)
                    self.outcome.append(lineSplit)
                    self.ejection.append(float(ef))
                    self.fps.append(int(fps))
                    

        self.frames = collections.defaultdict(list)
        self.trace = collections.defaultdict(_defaultdict_of_lists)

        # Volume and frames metadata - not used in UVT
        with open(self.folder / "VolumeTracings.csv") as f:
            header = f.readline().strip().split(",")
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            # Read lines one by one and store processed data
            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(',')
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                frame = int(frame)
                filename = os.path.splitext(filename)[0]
                
                # New frame index for the given filename
                if frame not in self.trace[filename]:
                    self.frames[filename].append(frame)
                
                # Add volume lines to trace
                self.trace[filename][frame].append((x1, y1, x2, y2))
        
        # Transform into numpy array
        for filename in self.frames:
            for frame in self.frames[filename]:
                self.trace[filename][frame] = np.array(self.trace[filename][frame])
        
        # Reject all files which do not have both ED and ES frames
        keep = [(len(self.frames[os.path.splitext(f)[0]]) >= 2) and (abs(self.frames[os.path.splitext(f)[0]][0] - self.frames[os.path.splitext(f)[0]][-1]) > self.min_length) for f in self.fnames]

        # Prepare for getitem
        self.fnames = [f for (f, k) in zip(self.fnames, keep) if k]
        self.outcome = [f for (f, k) in zip(self.outcome, keep) if k]
        self.ejection = [f for (f, k) in zip(self.ejection, keep) if k]
        self.fps = [f for (f, k) in zip(self.fps, keep) if k]
            
    def __getitem__(self, index):
        
        if self.mode == 'repeat':
            
            path = os.path.join(self.folder, "Videos", self.fnames[index])

            # Load video into np.array
            video = loadvideo(path).astype(np.float32)
            key = os.path.splitext(self.fnames[index])[0]

            # Scale pixel values from 0-255 to 0-1
            video /= 255.0

            video = np.moveaxis(video, 0, 1)
            
            samp_size = abs(self.frames[key][0]-self.frames[key][-1])
            if samp_size > self.fixed_length//2:
                video = video[::2,:,:,:]
                large_key = int(self.frames[key][-1]//2)
                small_key = int(self.frames[key][0]//2)
            else:
                large_key = self.frames[key][-1]
                small_key = self.frames[key][0]
                
            # Frames, Channel, Height, Width
            f, c, h, w = video.shape
            
            first_poi = min(small_key, large_key)
            last_poi  = max(small_key, large_key)
            
            label     = np.zeros(f)
            if self.SDmode == 'cla':
                label[small_key] = 1 # End systole (small)
                label[large_key] = 2 # End diastole (large)
            elif self.SDmode == 'reg':
                label[small_key] = -1 # End systole (small)
                label[large_key] = 1 # End diastole (large)
                dist = abs(small_key-large_key)
                sign = -int((small_key-large_key)/dist)
                center = min(small_key,large_key)+dist//2
                label[small_key+sign:large_key:sign] = sign*np.power((np.arange(small_key+sign,large_key,sign)-center)/dist*2,self.attenuation)

            nlabel = []
            nvideo = [] 
            repeat = 0
            while len(nvideo) < self.fixed_length+1:
                nvideo.append(video[first_poi])
                nvideo.extend(video[first_poi+1:last_poi])
                nvideo.append(video[last_poi])
                nvideo.extend(video[last_poi-1:first_poi:-1])
                
                nlabel.append(label[first_poi])
                nlabel.extend(label[first_poi+1:last_poi])
                nlabel.append(label[last_poi])
                nlabel.extend(label[last_poi-1:first_poi:-1])
                
                repeat += 1
                
            nvideo = np.stack(nvideo)
            nlabel = np.stack(nlabel)
            
            start_index = np.random.randint(nvideo.shape[0]-self.fixed_length)
            
            nvideo = nvideo[start_index:start_index+self.fixed_length]
            nlabel = nlabel[start_index:start_index+self.fixed_length]
            ejection = self.ejection[index]
            filename = os.path.splitext(self.fnames[index])[0]
            
            if self.padding is not None:
                p = self.padding
                nvideo = np.pad(nvideo, ((0,0),(0,0),(p,p),(p,p)), mode='constant', constant_values=0)
            
            
            return filename, nvideo, nlabel, ejection, repeat, self.fps[index]
        
        elif self.mode == 'full':
            path = os.path.join(self.folder, "Videos", self.fnames[index])
            # Load video into np.array
            video = loadvideo(path).astype(np.float32)
            key = os.path.splitext(self.fnames[index])[0]
            
            # Scale pixel values from 0-255 to 0-1
            video /= 255.0

            # Channel, Frames, Height, Width
            c, f, h, w = video.shape           
            
            # Add padding for easier VAE training
            if self.padding is not None:
                p = self.padding
                video = np.pad(video, ((0,0),(0,0),(p,p),(p,p)), mode='constant', constant_values=0)
            
            filename    = filename = os.path.splitext(self.fnames[index])[0]
            video = np.moveaxis(video, 0, 1)
            label     = np.zeros(f)
            if self.SDmode == 'cla':
                label[self.frames[key][0]] = 1 # End systole (small)
                label[self.frames[key][-1]] = 2# End diastole (large)
            elif self.SDmode == 'reg':
                label[self.frames[key][0]] = -1 # End systole (small)
                label[self.frames[key][-1]] = 1# End diastole (large)
            ejection    = self.ejection[index]
            repeat      = 0
            fps         = self.fps[index]
            
            return filename, video, label, ejection, repeat, fps
        
        elif self.mode == 'sample':
            path = os.path.join(self.folder, "Videos", self.fnames[index])

            # Load video into np.array
            video = loadvideo(path).astype(np.float32)
            key = os.path.splitext(self.fnames[index])[0]
            
            # Scale pixel values from 0-255 to 0-1
            video /= 255.0

            video = np.moveaxis(video, 0, 1)
            
            samp_size = abs(self.frames[key][0]-self.frames[key][-1])
            if samp_size > self.fixed_length:
                video = video[::2,:,:,:]
                large_key = int(self.frames[key][-1]//2)
                small_key = int(self.frames[key][0]//2)
            else:
                large_key = self.frames[key][-1]
                small_key = self.frames[key][0]
                
            # Frames, Channel, Height, Width
            f, c, h, w = video.shape
            
            first_poi = min(small_key, large_key)
            last_poi  = max(small_key, large_key)
            dist = abs(small_key-large_key)
            
            
            if self.SDmode == 'cla':
                label     = np.zeros(f)
                label[small_key] = 1 # End systole (small)
                label[large_key] = 2 # End diastole (large)
            elif self.SDmode == 'reg':
                label     = np.ones(f)/2
                label[small_key] = 0 # will become -1, End systole (small)
                label[large_key] = 1 # End diastole (large)
                sign = int((small_key-large_key)/dist)
                if sign > 0: # large -> small    
                    label[max(0,large_key-dist)+1:large_key]  = 1-np.power((np.abs(np.arange(max(0,large_key-dist)+1,large_key)-large_key)/dist),3)
                    label[large_key+1:small_key]              = np.power((np.abs(np.arange(large_key+1,small_key)-small_key)/dist),3)
                    label[small_key+1:min(f, small_key+dist)] = 1-np.power((np.abs(np.arange(small_key+1,min(f, small_key+dist))-min(f, small_key+dist))/dist),3)
                else:  # small -> large
                    label[max(0,small_key-dist)+1:small_key]  = np.power((np.abs(np.arange(max(0,small_key-dist)+1,small_key)-small_key)/dist),3)
                    label[small_key+1:large_key]              = np.power((np.abs(np.arange(small_key+1,large_key)-small_key)/dist),1/3)
                    label[large_key+1:min(f, large_key+dist)] = np.power((np.abs(np.arange(large_key+1,min(f, large_key+dist))-min(f, large_key+dist))/dist),3)
                label = label*2-1
                
            
            divider     = np.random.random_sample()*5+2
            start_index = first_poi - dist//divider
            start_index = int(max(0, start_index)//2*2)    
            
            divider     = np.random.random_sample()*5+2
            end_index   = last_poi +1 + dist//divider #+1 to INCLUDE the frame
            end_index   = int(min(f, end_index)//2*2)
            
            end_index   = int(min(f, end_index)//2*2)
            step = int( np.ceil((end_index-start_index)/ self.max_length) )
            
            video = video[start_index:end_index:step, :, :, :]
            label = label[start_index:end_index]
            label = torch.nn.functional.max_pool1d(torch.tensor(label[None,None,:]), step).squeeze().numpy()
            window_width = video.shape[0]
            # Add blank frames to avoid confusing the network with unlabeled ED and ES frames
            missing_frames = self.fixed_length - window_width
            if missing_frames > 0:
                missing_frames_before = np.random.randint(missing_frames)
                missing_frames_after  = missing_frames - missing_frames_before
                video = np.concatenate((np.zeros((missing_frames_before, c, h, w)), video, np.zeros((missing_frames_after, c, h, w))), axis=0)
                label = np.concatenate((np.zeros(missing_frames_before), label, np.zeros(missing_frames_after))) # NOT NECESSARY !
            else:
                missing_frames_before = 0
                missing_frames_after  = missing_frames - missing_frames_before
            attention = np.zeros_like(label)
            attention[missing_frames_before:missing_frames_before+window_width] = 1
                
            ejection = self.ejection[index]
            filename = os.path.splitext(self.fnames[index])[0]
            repeat      = attention
            fps         = self.fps[index]
            
            if self.padding is not None:
                p = self.padding
                video = np.pad(video, ((0,0),(0,0),(p,p),(p,p)), mode='constant', constant_values=0)
            
            if video.shape[0] != 128 or label.shape[0] != 128:
                raise ValueError('WTF??', self.fixed_length, window_width, video.shape[0], label.shape[0])
            
            return filename, video, label, ejection, repeat, fps
        
        else:
            raise ValueError(self.mode, "is not a proper mode, choose: 'sample', 'full', 'repeat'")
            
    def __len__(self):
        return len(self.fnames)


def _defaultdict_of_lists():
    """Returns a defaultdict of lists.

    This is used to avoid issues with Windows (if this function is anonymous,
    the Echo dataset cannot be used in a dataloader).
    """

    return collections.defaultdict(list)


def loadvideo(filename: str):
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_width, frame_height, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count] = frame

    v = v.transpose((3, 0, 1, 2))

    assert v.size > 0

    return v
