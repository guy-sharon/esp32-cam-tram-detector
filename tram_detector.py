import inspect
import contextlib
import tinytuya
from tkinter import Tk
from tkinter import simpledialog
import socket
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import urllib, sys, traceback
from datetime import datetime, timedelta
import matplotlib.dates
import psutil
import winsound
import time
import scipy.ndimage as ndi
import telepot
from moviepy.editor import ImageSequenceClip
from threading import Thread
import subprocess
from PIL import Image, ImageSequence

#%%
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout
    
#%%
def fprint(*args):
    txt = str(datetime.now()) + ": "
    for i in range(len(args)):
        txt += str(args[i]) + " "
    print(txt)
    with open('log.txt', 'a') as f:
        f.write(txt + "\n")
        
def plot_past():
    
    if len(Detector.concat_imgs) == 0 or len(Detector.scores) == 0:
        return
    
    if len(Detector.concat_imgs) <= Detector.ind:
        Detector.ind = len(Detector.concat_imgs) - 1
    
    concat_img = Detector.concat_imgs[Detector.ind]
    cv.imshow("capture", concat_img)
    
def on_change(value):
    Detector.ind = int(Detector.buffer_size * value / 10000)
    
    plot_past()
    
def find(arr):
    return [i for i, x in enumerate(arr) if x]
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def cor2(im1,im2):
    im1 = (im1 - np.mean(im1)) / np.std(im1)
    im2 = (im2 - np.mean(im2)) / np.std(im2)
    return np.sum(im1 * im2) / im1.shape[0] / im1.shape[1]

def imshift(im,shift):
    im = np.hstack((im[:,shift[0]:],im[:,:shift[0]]))
    im = np.vstack((im[shift[1]:,:],im[:shift[1],:]))
    return im

def massDensity(im, dists):
    im = cv.normalize(im,None,0,10,cv.NORM_MINMAX)
    im_exp = np.exp(im)
    im_exp = im_exp / np.sum(im_exp)
    cy, cx = ndi.center_of_mass(im_exp)
    im_exp = imshift(im_exp,(int(cx),int(cy)))
    I = np.sum(dists * im_exp)
    return cx,cy,I   
    
def MovieMoveDir(imgs):
    prev_blur = None
    mv_dir = np.array((0,0))
    cx1=cy1=I1=0
    
    dists = np.zeros_like(imgs[0],dtype=float)
    for i in range(-int(dists.shape[0]/2),int(dists.shape[0]/2)):
        for j in range(-int(dists.shape[1]/2),int(dists.shape[1]/2)):
            dists[i,j] = i**2 + j**2;
            
    for i in range(1,len(imgs)):
        im1 = imgs[i].astype(float)
        im2 = imgs[i-1].astype(float)
        diff = cv.absdiff(im1, im2)
        blur = cv.GaussianBlur(diff, (5, 5), 0)
        
        cx2,cy2,I2 = massDensity(blur, dists)
        # blur[int(cy2),int(cx2)] = 255
        # plt.imshow(blur)
        # plt.show()
        if np.any(prev_blur != None):
            I = I1 + I2
            new_mv_dir = np.array(((cx2-cx1)/I,(cy2-cy1)/I))
            mv_dir = mv_dir + new_mv_dir
            
        prev_blur = blur.copy()
        cx1 = cx2; cy1 = cy2; I1 = I2;
        
    return mv_dir

def DeclareDetection():    
    if len(Detector.tram_images) < 10 or len(Detector.tram_images) > 500:
        fprint(f"incorrect number of tram_images ({len(Detector.tram_images)})")
        return None
    
    if (Detector.last_declaration_time != None and (datetime.now()-Detector.last_declaration_time).total_seconds() < Detector.train_declaration_cooldown_sec):
        fprint(f"detection rejected, last was {(datetime.now()-Detector.last_declaration_time).total_seconds()}s ago")
        return None
      
    Detector.last_declaration_time = datetime.now()
    for i in range(2):
        winsound.PlaySound("SystemHand", winsound.SND_ASYNC)
        time.sleep(0.1)
         
    Thread(target=SendTelegramNotification, args=(Detector.tram_images,)).start()
        
    return 1

def CorrectGain():
    if (not Detector.DoGainCorrection or Detector.score > Detector.bg_noise): return
    if (datetime.now() - Detector.LastGainCorrectionTime).total_seconds() < Detector.GainCorrectionIntervalSec:
        return
    
    Detector.LastGainCorrectionTime = datetime.now()
    gain_correction = -(Detector.Brightness - Detector.GainDesiredBrightness) / Detector.GainBrightnessInterval
    fprint("brightness = ",round(Detector.Brightness,2),". gain_correction = ",round(gain_correction,2))
    if np.abs(gain_correction) >= 1:
        AdjustSensitivity(gain_correction)    

def GetParam(param_name):
    for i in range(len(Detector.trackbarNames)):
        if Detector.trackbarNames[i] == param_name:
            break   
    
    min_val     = Detector.trackbarValues[i][0]
    max_val     = Detector.trackbarValues[i][1]
    curr_val    = Detector.trackbarCurrValues[i]
    return min_val, max_val, curr_val, i

def AdjustSensitivity(step):
    fprint('adjusting sensitivity...')
    if step == 0: 
        return
    
    if step > 0:
        param_names = ["aec_value","agc_gain"]
    else:
        param_names = ["agc_gain","aec_value"]
    
    fp_min, fp_max, fp_curr, fp_ind = GetParam(param_names[0])
    sp_min, sp_max, sp_curr, sp_ind = GetParam(param_names[1])
    inds = [fp_ind, sp_ind]
    curr_vals = [fp_curr, sp_curr]
    
    agc_step = np.sign(step)
    if np.abs(step) > 4:
        aec_limit_den = 1
        step *= 5
    else:
        aec_limit_den = 30
        
    if param_names[0] == "aec_value":
        aec_step = int(max(-fp_curr/aec_limit_den-1,min(fp_curr/aec_limit_den+1,10*step)))
        steps = [aec_step, agc_step]
        start_cnt = [Detector.start_cnt_th - 20, 0]
    else:
        aec_step = int(max(-sp_curr/aec_limit_den-1,min(sp_curr/aec_limit_den+1,10*step)))
        steps = [agc_step, aec_step]
        start_cnt = [0, Detector.start_cnt_th - 20]
        
    new_fp = max(fp_min,min(fp_max,fp_curr + steps[0]))
    new_sp = max(sp_min,min(sp_max,sp_curr + steps[1]))
    new_vals = [new_fp, new_sp]
    
    if fp_curr == new_fp:
        if sp_curr == new_sp:
            fprint('Failed to adjust sensitivity... both parameters are at their limits...')
            Detector.LastGainCorrectionTime = datetime.now() + timedelta(seconds=30)
            return
        i = 1 # secondery
    else:
        i = 0 # primary
        
    param              = param_names[i]
    Detector.start_cnt = start_cnt[i]
        
    fprint('adjusting ' + param + "... start_cnt = ",Detector.start_cnt)
    Detector.trackbarCurrValues[inds[i]] = new_vals[i]
    ManualSetParam(param,new_vals[i])
    fprint(param + ' set ', curr_vals[i], ' -> ', new_vals[i])
    Detector.prev_im = None
        
def SetManualGainModeAndZeroGain():
    for i in range(len(Detector.trackbarNames)):
        if Detector.trackbarNames[i] == "gain_ctrl":
            break
        
    header = 'prms'
    bytes_data = np.zeros(22,dtype=int) + 128
    bytes_data[i] = 0;
    
    cmd = header.encode() + bytes(list(bytes_data + 127))
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
    clientSocket.settimeout(1)
    clientSocket.connect(("10.100.102.2",8088));
    clientSocket.settimeout(None)
    clientSocket.send(cmd)
    clientSocket.close()
    
    for i in range(len(Detector.trackbarNames)):
        if Detector.trackbarNames[i] == "agc_gain":
            break
    header = 'prms'
    bytes_data = np.zeros(22,dtype=int) + 128
    bytes_data[i] = 0;
    cmd = header.encode() + bytes(list(bytes_data + 127))
    time.sleep(0.1)
    clientSocket.send(cmd)
    clientSocket.close()
    
def Detector(original_image):
    Detector.last_recieve_datetime = datetime.now()
    
    real_im_border = [int(Detector.im_border[0]*original_image.shape[0]),int(Detector.im_border[1]*original_image.shape[0]),int(Detector.im_border[2]*original_image.shape[1]),int(Detector.im_border[3]*original_image.shape[1])]
    im = cv.cvtColor(original_image[real_im_border[0]:real_im_border[1],real_im_border[2]:real_im_border[3]], cv.COLOR_BGR2GRAY)
    
        
    if psutil.virtual_memory()[2] > 90:
        fprint('out of memory')
        raise KeyboardInterrupt 
        
        
        
    dt = (datetime.now()-Detector.t).microseconds/1e6
    Detector.t = datetime.now()
    if (dt != 0):
        Detector.fps = 0.9*Detector.fps + 0.1/dt
    # fprint(str(psutil.virtual_memory()[2])+"%", str(round(Detector.fps,2))+"fps")
    
    
    
    plot_past()
    if np.all(Detector.scores != None) and Detector.ind < len(Detector.all_scores_times) - 1:
        if cv.waitKey(1) == 27: raise KeyboardInterrupt 
    Detector.ind += Detector.dind
    
    
    
    if np.any(Detector.prev_im != None):
        diff = cv.absdiff(Detector.prev_im, im)
        blur = cv.GaussianBlur(diff, (5, 5), 0)
        new_score = np.mean(blur)
        if np.all(Detector.score == None):
            Detector.score = new_score
        else:    
            Detector.score = 0.3*Detector.score + 0.7*new_score

        if np.all(Detector.scores != None) and len(Detector.scores) == Detector.buffer_size:
            Detector.scores[0:-1] = Detector.scores[1:]
            Detector.scores[-1] = Detector.score
            Detector.all_scores_times[0:-1] = Detector.all_scores_times[1:]
            Detector.all_scores_times[-1] = datetime.now()
        else:
            if np.any(Detector.scores == None):
                Detector.scores = np.array([Detector.score])
            else:
                Detector.scores = np.vstack([Detector.scores, Detector.score])
            Detector.all_scores_times = np.append(Detector.all_scores_times,datetime.now())                       
        Detector.start_cnt += 1
        
        # brightness
        hsv_image = cv.cvtColor(original_image,cv.COLOR_BGR2HSV)
        brightness = np.mean(hsv_image[:,:,2])
        Detector.Brightness = 0.95 * Detector.Brightness + 0.05 * brightness
        
        #
        if Detector.start_cnt < Detector.start_cnt_th:
            Detector.bg_noise = Detector.score
        else:
            _,_, curr_val, _ = GetParam("agc_gain")
            eff_sigmoid_scaling = max(1,Detector.SigmoidScaling - curr_val / 4)
            alpha = sigmoid(eff_sigmoid_scaling*(Detector.bg_noise - Detector.score))
            Detector.bg_noise = (1 - alpha) * Detector.bg_noise + alpha * Detector.score
        
        # correct agc or aec using the brightness of the image
        CorrectGain()
        
        
        ## score
        if Detector.score > Detector.bg_noise:
            if Detector.signal_above_noise_start_time == None:
                Detector.signal_above_noise_start_time = datetime.now()
        else:
            Detector.signal_above_noise_start_time = None
            if Detector.has_detected_started: ## DETECTION!
                Detector.has_detected_ended = True
        
        # 2
        if Detector.has_detected_ended:
            Detector.signal_above_noise_start_time = None
            Detector.has_detected_started = False
            Detector.has_detected_ended = False
            det_val = DeclareDetection()
        else:    
            det_val = None
            
        # 1
        if Detector.signal_above_noise_start_time != None and \
           ((datetime.now() - Detector.signal_above_noise_start_time).total_seconds() > Detector.train_detection_duration_sec) and \
           Detector.has_detected_started == False:
            Detector.has_detected_started = True
          
        
        
            
        
        if Detector.signal_above_noise_start_time != None or Detector.has_detected_started:
            Detector.tram_images.append(original_image.copy())
        else:
            Detector.tram_images = []
            
            
            
        if np.all(Detector.bg_noises != None) and len(Detector.bg_noises) == Detector.buffer_size:
            Detector.bg_noises[0:-1] = Detector.bg_noises[1:]
            Detector.bg_noises[-1] = Detector.bg_noise
            Detector.detection[0:-1] = Detector.detection[1:]
            Detector.detection[-1] = det_val
        else:
            if np.any(Detector.bg_noises == None):
                Detector.bg_noises = np.array([Detector.bg_noise])
                Detector.detection = None
            else:
                Detector.detection = np.append(Detector.detection,det_val)
                Detector.bg_noises = np.vstack([Detector.bg_noises, Detector.bg_noise])
            
            
            
            
        if Detector.plot_cnt == Detector.plotDecimation - 1:
            if Detector.LastManualCommandDisplayCnt >= Detector.LastManualCommandDisplayTh:
                Detector.LastManualCommandDisplayCnt = 0
                Detector.LastManualCommand = ""
            else:
                Detector.LastManualCommandDisplayCnt += 1
                    
            Detector.plot_cnt = 0
            plt.clf()
            if len(Detector.all_scores_times) > 1:
                plt.plot_date(matplotlib.dates.date2num(Detector.all_scores_times),Detector.scores,marker=None,ls='-')
                plt.plot_date(matplotlib.dates.date2num(Detector.all_scores_times),Detector.bg_noises,marker=None,ls='--')
                plt.plot_date(matplotlib.dates.date2num(Detector.all_scores_times[Detector.detection==-1]),Detector.scores[Detector.detection==-1],markersize=20,marker=">",ls=' ',color='black')
                plt.plot_date(matplotlib.dates.date2num(Detector.all_scores_times[Detector.detection==1]),Detector.scores[Detector.detection==1],markersize=20,marker="<",ls=' ',color='black')
            plt.ylim(bottom=0)
            Detector.fig.canvas.draw()
            graph_img = np.frombuffer(Detector.fig.canvas.tostring_rgb(), dtype=np.uint8)
            graph_img = graph_img.reshape(Detector.fig.canvas.get_width_height()[::-1] + (3,))
            size_diff = graph_img.shape[1] - original_image.shape[1]
            n = int(size_diff/2)

            padded_im = cv.rectangle(original_image, real_im_border[3::-2], real_im_border[2::-2], (10,10,255), 1)
            if n > 0:
                padded_im = np.pad(padded_im, [(0, 0), (n, size_diff-n), (0,0)], mode='constant', constant_values=0)
            else:
                graph_img = np.pad(graph_img, [(0, 0), (-n,-size_diff+n), (0,0)], mode='constant', constant_values=0)
            network_usage = Detector.jpg_len * Detector.fps
            padded_im = cv.putText(padded_im,"fps = " + str(round(Detector.fps,2)),(30,30),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255, 255),1)
            padded_im = cv.putText(padded_im,Detector.trackbarNames[Detector.currTb],(30,60),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255, 255),1)
            padded_im = cv.putText(padded_im,str(round(network_usage/1024/1024,2)) + " MB/s",(30,90),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255, 255),1)
            padded_im = cv.putText(padded_im,Detector.LastManualCommand,(30,120),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255, 255),1)
            padded_im = cv.putText(padded_im,datetime.now().strftime("%d%m%Y %H:%M:%S"),(30,150),cv.FONT_HERSHEY_SIMPLEX,0.5,(255, 255, 255, 255),1)
            
            debug_val = str(Detector.has_detected_started*1) + str(Detector.has_detected_ended*1) + " "
            if Detector.signal_above_noise_start_time != None: debug_val += str(round((datetime.now()-Detector.signal_above_noise_start_time).total_seconds(),2)) + " s"
            padded_im = cv.putText(padded_im,debug_val,(padded_im.shape[1]-210,30),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255, 255),1)
            padded_im = cv.putText(padded_im,str(Detector.jpg_len) + " B",(padded_im.shape[1]-210,60),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255, 255),1)
            padded_im = cv.putText(padded_im,"is_start = " + str(1*(Detector.start_cnt < Detector.start_cnt_th)),(padded_im.shape[1]-210,90),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255, 255),1)
            padded_im = cv.putText(padded_im,"BR = " + str(round(Detector.Brightness,2)),(padded_im.shape[1]-210,120),cv.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255, 255),1)
            
            concat_img = np.concatenate((padded_im,graph_img))
            if len(Detector.concat_imgs) == Detector.buffer_size:
                Detector.concat_imgs[0:-1] = Detector.concat_imgs[1:]
                Detector.concat_imgs[-1] = concat_img
            else:
                Detector.concat_imgs.append(concat_img)
                
            if Detector.record:
                if not ('rec' in Detector.__dict__.keys() and Detector.rec.isOpened()):
                    Detector.rec_fname = datetime.now().strftime("recordings\%Y_%m_%d__%H%M%S.mp4")
                    Detector.rec = cv.VideoWriter(Detector.rec_fname,cv.VideoWriter_fourcc(*'DIVX'), 20, concat_img.shape[1::-1])
                Detector.rec.write(concat_img)
                
        else:
            Detector.plot_cnt += 1
            
        
        
    Detector.prev_im = im
    
    if Detector.doLivePlot:
        if Detector.live_plot_cnt == 1:
            Detector.live_plot_cnt = 0
            key = cv.waitKeyEx(1)
            vals = Detector.trackbarValues[Detector.currTb]
            max_val = vals[1] - vals[0]
            if key == 43:
                Detector.cut_factor = max(0,Detector.cut_factor - 0.1)
                SetCutFactor(Detector.cut_factor)
            if key == 45:
                Detector.cut_factor = min(1,Detector.cut_factor + 0.1)
                SetCutFactor(Detector.cut_factor)
            if key == 27:
                raise KeyboardInterrupt 
            if key == 2621440: # down
                Detector.currTb = (Detector.currTb - 1) % len(Detector.trackbarNames)
                SetTrackbar()
            if key == 2490368: # up
                Detector.currTb = (Detector.currTb + 1) % len(Detector.trackbarNames)
                SetTrackbar()
            if key == 2424832: # left
                # v = cv.getTrackbarPos('param', 'capture')
                Detector.trackbarCurrValues[Detector.currTb] = max(min(Detector.trackbarCurrValues[Detector.currTb] - 1,max_val),0)
                cv.setTrackbarPos('param', 'capture', Detector.trackbarCurrValues[Detector.currTb])
            if key == 2555904: # right
                # v = cv.getTrackbarPos('param', 'capture')
                Detector.trackbarCurrValues[Detector.currTb] = max(min(Detector.trackbarCurrValues[Detector.currTb] + 1,max_val),0)
                cv.setTrackbarPos('param', 'capture', Detector.trackbarCurrValues[Detector.currTb])
            if key == 13: # enter
                ChangeCamConfig()
        else: Detector.live_plot_cnt += 1


def TelegramBotHandleResponses():
    try:
        if Detector.HandlerRunning:
            return
        Detector.HandlerRunning = True
        
        chat_id = None
        response = Detector.Bot.getUpdates()
        if len(response) == 0:
            Detector.HandlerRunning = False
            return
        
        update_id = response[0]['update_id']
        Detector.Bot.getUpdates(update_id + 1)
        chat_id = str(response[0]['message']['chat']['id'])
        if not 'text' in list(response[0]['message'].keys()): return
        text = response[0]['message']['text']
        fprint("received message '" + text + "' from chat_id = " + str(chat_id))

        if text == 'help':
            Detector.Bot.sendMessage(chat_id, inspect.getsource(TelegramBotHandleResponses))            

        if text == 'exit':
            subprocess.call("TASKKILL /F /IM pythonw.exe", shell=True)

        #
        if text in ("stop","abort","halt"):
            Detector.RunStatus[chat_id] = False
            Detector.Bot.sendMessage(chat_id, "stopping...")
            
        if text in ("start","resume","begin","continue"):
            Detector.RunStatus[chat_id] = True
            Detector.Bot.sendMessage(chat_id, "resuming...")
            Detector.start_cnt                      = 0
            Detector.signal_above_noise_start_time  = None
            Detector.has_detected_started           = False
            Detector.has_detected_ended             = False
        
        if "status" in text:
            SendStatus(chat_id, text)
        
        if 'debug' in text:
            if 'enable' in text:
                chat_ind = int(text.split(' ')[1])
                Detector.RunStatus[chat_id] = True
                Detector.Bot.sendMessage(chat_id, f'enabling {chat_ind}')
            if 'disbale' in text:
                chat_ind = int(text.split(' ')[1])
                Detector.RunStatus[chat_id] = False
                Detector.Bot.sendMessage(chat_id, f'disbaling {chat_ind}')
                
    except Exception as e:
        try:
            fprint(e)
            Detector.Bot.sendMessage(chat_id, "error: " + str(e))
        except:
            pass
    
    Detector.HandlerRunning = False

def SendStatus(chat_id, text):
    # direction = "right"
    # if "left" in text:
    #     direction = "left"
    
    # dates = []
    # with open('events.txt', 'r') as f:
    #     lines = f.readlines()
    # for line in lines:
    #     if direction in line:
    #         dates.append(datetime.strptime(line.split("__")[0], "%Y-%m-%d_%H:%M:%S"))
        
    Detector.Bot.sendMessage(chat_id, "alive")

def SendTelegramNotification(tram_images):
    
    video_fname             = "tmp.gif"      
    compressed_video_fname  = "compressed.gif"
    
    tram_imgs = np.array(tram_images)
    tmp_img = cv.cvtColor(tram_imgs[0], cv.COLOR_BGR2GRAY)
    imgs = np.zeros((len(tram_images),tmp_img.shape[0],tmp_img.shape[1]))

    for i in range(len(tram_imgs)):
        imgs[i,:,:] = cv.cvtColor(tram_imgs[i], cv.COLOR_BGR2GRAY)
    mv_dir = MovieMoveDir(imgs)
    
    clip = ImageSequenceClip(list(tram_images), fps=25)
    clip.write_gif(video_fname, fps=25)
        
    # compress gif
    optimizeVideo(video_fname, compressed_video_fname)
    
    if mv_dir[0] < 0: direction = "left"
    else: direction = "right"
    
    string = datetime.now().strftime("%H:%M:%S") + ': Tram Detected Going ' + direction
    fprint(string)
    
    with open('events.txt', 'a') as f:
        f.write(datetime.now().strftime("%Y-%m-%d_%H:%M:%S__") + direction)
        f.write("\n")
    
    for chat_id in list(Detector.RunStatus.keys()):
        if Detector.RunStatus[chat_id]:
            Detector.Bot.sendVideo(chat_id,  video = open(compressed_video_fname, 'rb'), caption = string)

def optimizeVideo(fname,compressed_fname):
    vid = Image.open(fname)
    frames = [f.copy() for f in ImageSequence.Iterator(vid)]
    new_vid = []
    for frame in frames:
        new_vid.append(frame)
    new_vid[1].save(compressed_fname,save_all=True,append_images=new_vid[1:],loop=0)
    subprocess.run(['gifsicle-1.92\gifsicle.exe','--colors=64','--lossy=50','-O2',fname,'-o',compressed_fname])

def SendCamConfig(jpeg_quality,frame_size):
    Detector.start_cnt = 0
    
    header = 'prmscam'
    bytes_data = np.zeros(19,dtype=int) + 128
    if frame_size == None:
        bytes_data[0] = -1
    else:
        bytes_data[0] = frame_size
    bytes_data[1] = jpeg_quality
    cmd = header.encode() + bytes(list(bytes_data + 127))
    for i in range(3):
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
        clientSocket.connect((Detector.IP,8088));
        clientSocket.send(cmd)
        clientSocket.close()

def SetCutFactor(val):    
    Detector.start_cnt = 0
    
    header = 'prmscut'
    Detector.cut_factor = val
    bytes_data = np.zeros(19,dtype=int) + 128
    bytes_data[0] = min(255,max(0,round(val * 255)))
    cmd = header.encode() + bytes(list(bytes_data))
    for i in range(3):
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
        clientSocket.connect((Detector.IP,8088));
        clientSocket.send(cmd)
        clientSocket.close()
        
    
def ChangeCamConfig():    
    ws = Tk()       
    jpeg_quality = simpledialog.askinteger("Input", "jpeg_quality?",parent=ws,minvalue=10, maxvalue=63)
    frame_size = simpledialog.askinteger("Input", "frame_size?",parent=ws,minvalue=0, maxvalue=13)
    ws.destroy()
    SendCamConfig(jpeg_quality,frame_size)
    InitDetector()
    
def GetCMDBytes(param_name, value):
    for i in range(len(Detector.trackbarNames)):
        if Detector.trackbarNames[i] == param_name:
            break
        
    bytes_data = np.zeros(22,dtype=int) + 255
    header = 'prms'
    if Detector.trackbarNames[i] == 'aec_value':
        total = value
        byte1 = int(total / 256)
        byte2 = total - byte1 * 256
        bytes_data[i]   = byte1;
        bytes_data[i+1] = byte2;
    else:
        bytes_data[i] = value + Detector.trackbarOffset + 127;
    
    try:
        return header.encode() + bytes(list(bytes_data))
    except:
        fprint("ERROR in 'GetCMDBytes':")
        fprint("bytes_data = ",bytes_data)
        fprint("param = ",Detector.trackbarNames[i])
        fprint("value = ",value)
        
        
def ManualSetParam(param_name, value):    
    for i in range(len(Detector.trackbarNames)):
        if Detector.trackbarNames[i] == param_name:
            Detector.trackbarCurrValues[i] = value
            break
        
    cmd = GetCMDBytes(param_name, value)
    Detector.LastManualCommand = param_name + '=' + str(value)
    Detector.LastManualCommandDisplayCnt = 0
    for i in range(3):
        clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
        clientSocket.connect((Detector.IP,8088));
        clientSocket.send(cmd)
        clientSocket.close()
        
def update_prms(value):
    if Detector.trackbarActive == False:
        return
    
    fprint("Updated param '" + Detector.trackbarNames[Detector.currTb],'to',value)
    Detector.trackbarCurrValues[Detector.currTb] = value
    cmd = GetCMDBytes(Detector.trackbarNames[Detector.currTb], value)
    Detector.start_cnt = 0
    clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM);
    clientSocket.connect((Detector.IP,8088));
    clientSocket.send(cmd)
    clientSocket.close()
    
def SetTrackbar():
    Detector.trackbarActive = False
    vals = Detector.trackbarValues[Detector.currTb]
    Detector.trackbarOffset = vals[0]
    cv.setTrackbarMin('param', 'capture', 0)
    cv.setTrackbarMax('param', 'capture', vals[1] - vals[0])
    cv.setTrackbarPos('param', 'capture', Detector.trackbarCurrValues[Detector.currTb])
    Detector.trackbarActive = True

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  return result

def Smart_Plug_On():
    Detector.smart_plug.turn_on()

def Smart_Plug_Off():
    Detector.smart_plug.turn_off()
    
def isAlive():
    fprint("no response for", (datetime.now() - Detector.last_recieve_datetime).total_seconds(), 'seconds')
    return (datetime.now() - Detector.last_recieve_datetime).total_seconds() < 5

def ResetSmartPlug():
    fprint("RESTARTING SMART PLUG")
    Smart_Plug_Off()
    time.sleep(1)
    Smart_Plug_On()
    time.sleep(3)
    InitConfig()
    Detector.last_recieve_datetime = datetime.now() + timedelta(seconds=10)
    
#%%
def InitDetector():
    
    ################################## Configuration ##################################
    
    Detector.train_detection_duration_sec   = 5
    Detector.train_declaration_cooldown_sec = 15
    Detector.buffer_size                    = 1000
    Detector.doLivePlot                     = True
    Detector.plotDecimation                 = 1
    Detector.record                         = True
    
    Detector.all_scores_times               = np.array([])
    Detector.score                          = None
    Detector.scores                         = None
    Detector.bg_noise                       = None
    Detector.bg_noises                      = None
    Detector.signal_above_noise_start_time  = None
    Detector.detection                      = np.array([])
    Detector.has_detected_started           = False
    Detector.has_detected_ended             = False
    Detector.fps                            = 0
    Detector.ind                            = 0
    Detector.prev_im                        = None
    Detector.live_plot_cnt                  = 0
    Detector.plot_cnt                       = 0
    Detector.t                              = datetime.now()
    Detector.concat_imgs                    = []
    Detector.dind                           = 1
    plt.plot_date([],[])
    Detector.fig                            = plt.figure(figsize=(11,3))
    Detector.last_declaration_time          = None
    Detector.ang                            = 0
    Detector.tram_images                    = np.array([])
    Detector.start_cnt                      = 0
    Detector.start_cnt_th                   = 40
    Detector.SigmoidScaling                 = 15
    
    Detector.trackbarNames                  = ("Brightness","Contrast","Saturation","Effect","Whitebal","awb_gain","wb_mode","exposure_ctrl","aec2","ae_level","gain_ctrl","agc_gain","aec_value")
    Detector.trackbarValues                 = ((-2,2),(-2,2),(-2,2),(0,6),(0,1),(0,1),(0,4),(0,1),(0,1),(-2,-2),(0,1),(0,15),(0,1200))
    Detector.trackbarCurrValues             = np.zeros((len(Detector.trackbarNames),),dtype=int)
    Detector.currTb                         = 0
    Detector.trackbarOffset                 = 0
    Detector.trackbarActive                 = True
    
    Detector.DoGainCorrection               = True
    Detector.GainDesiredBrightness          = 130
    Detector.GainBrightnessInterval         = 20
    Detector.GainCorrectionIntervalSec      = 5
    Detector.LastGainCorrectionTime         = datetime.now()
    Detector.Gain                           = 0
    Detector.Brightness                     = 0
    
    Detector.LastManualCommand              = ""
    Detector.LastManualCommandDisplayCnt    = 0
    Detector.LastManualCommandDisplayTh     = 20
    
    apiToken                                = ''
    Detector.Bot                            = telepot.Bot(apiToken)
    Detector.BotResponseRunning             = False
    Detector.RunStatus                      = {'':True, '':False}
    Detector.HandlerRunning                 = False
    
    Detector.cut_factor                     = 0
    Detector.IP                             = ""
    
    Detector.last_recieve_datetime          = datetime.now()
    Detector.smart_plug_id                  = ''
    Detector.smart_plug_ip                  = ''
    Detector.smart_plug_key                 = ''
    Detector.smart_plug                     = tinytuya.OutletDevice(dev_id=Detector.smart_plug_id,address='Auto',local_key=Detector.smart_plug_key,version=3.3)    
    
    for i in range(10): cv.waitKey(1)
    cv.destroyAllWindows()
    cv.namedWindow("capture", cv.WINDOW_NORMAL)
    cv.resizeWindow("capture", 1800, 800)
    cv.createTrackbar('slider', "capture", 0, 10000, on_change)
    cv.createTrackbar('param', "capture", 0, 1, update_prms)

def InitConfig():
    fprint('Initializing hardware configuration...')
    SendCamConfig(15,9)
    SetCutFactor(0.55)
    Detector.trackbarCurrValues = np.zeros((len(Detector.trackbarNames),),dtype=int)
    ManualSetParam("Contrast", 2)
    ManualSetParam("Effect", 2)
    ManualSetParam("exposure_ctrl",0)
    ManualSetParam("aec_value",12)

#%%    
fprint("")
fprint("NEW SESSION")
wait_to_date = datetime(2023,4,20,18,30)
while (datetime.now() < wait_to_date):
    print("\014")
    fprint('waiting for ',(wait_to_date - datetime.now()).total_seconds(),' sec')
    time.sleep(1)

og_border           = [0, 1, 0, 1]
og_border           = [0.05, 0.33, 0.03, 0.25]
Detector.im_border  = [0.05, 0.95, 0.05, 0.95]
InitDetector()
ResetSmartPlug()
SetTrackbar()

stream = None
while stream == None or stream.isclosed():
    try:
        if (not isAlive()): ResetSmartPlug()
        with urllib.request.urlopen("http://" + Detector.IP + "/mjpeg/1", timeout=1) as stream:
            stream_bytes = b''
            while not stream.isclosed():
                Thread(target=TelegramBotHandleResponses).start()    
                if np.any(list(Detector.RunStatus.values())) == False:
                    continue
                
                stream_bytes += stream.read(1024)
                a = stream_bytes.find(b'\xff\xd8')
                b = stream_bytes.find(b'\xff\xd9', a)
                if a != -1 and b != -1:
                    jpg = stream_bytes[a:b+2]
                    Detector.jpg_len = len(jpg)
                    stream_bytes = stream_bytes[b+2:]
                    with nostdout():
                        original_image = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv.IMREAD_COLOR)                
                    real_og_border = [int(og_border[0]*original_image.shape[0]),int(og_border[1]*original_image.shape[0]),int(og_border[2]*original_image.shape[1]),int(og_border[3]*original_image.shape[1])]
                    Detector(original_image[real_og_border[0]:real_og_border[1],real_og_border[2]:real_og_border[3]])
                    
    except KeyboardInterrupt:
      cv.destroyAllWindows()
      if Detector.record:
          Detector.rec.release()
      sys.exit()
    except Exception as e:
        traceback.print_exc()
        # raise KeyboardInterrupt
        fprint(str(e))
        # pass
       
       