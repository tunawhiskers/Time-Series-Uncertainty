import numpy as np

from noise_gen import var_mean_bl_mean_fit, var_mean_bl_var_fit







def test_bl_mean():
    dt = 0.001
    f_c = 5.0
    B = 10.0
    
    tlens = np.linspace(0.1, 60, 10)
    means1 = []
    means2 = []
    means_a = []
    stds1 = []
    stds2 = []
    
    for tlen in tlens:
    
        np.random.seed(int(time.time()))
        
        ns = 100
        
        std1 = []
        std2 = []
        mean1 = []
        mean2=[]
        std_snoise = []
        for i in range(ns):
            
            
            noise = band_limited_noise(f_c, B, max(tlens)*10, dt, 5)
            
            
            snoise = noise[0:int(tlen/dt)]
            
            
            std_snoise.append(np.var(snoise))

            var_anal = var_mean_bl(len(snoise)*dt, B)
            var_mean = var_mean_bl_mean_fit(snoise, False)/np.var(noise) 
            var_mean2 = var_mean_bl_var_fit(snoise, False)/np.var(noise) 
         
            
            std1.append((var_mean - var_anal)**2)
            std2.append((var_mean2 - var_anal)**2)
            
            mean1.append(var_mean)
            mean2.append(var_mean2)
    
        means_a.append(var_anal)
        means1.append(np.mean(mean1))
        means2.append(np.mean(mean2))
        
        stds1.append(np.std(mean1))
        stds2.append(np.std(mean2))
    tlens = tlens*0.1
    print(tlens)
    print(means1)
    plt.errorbar(x = tlens, y = means1, yerr = stds1, label = "1", capsize = 2)
    plt.errorbar(x = tlens, y = means2, yerr = stds2, label = "2", capsize = 2)
    plt.plot(tlens, means_a, label = "anal")
    plt.grid()
    plt.legend()
    plt.show()
    plt.loglog(tlens, stds1, label = "1")
    plt.loglog(tlens, stds2, label = "2")
    plt.legend()
    plt.grid()
    plt.show()



    #df = pd.DataFrame(mean2)
    #ax = df.plot.hist(bins=20, alpha=0.5)
    

test_bl_mean()
    #rint(nb, std, rem, noise.size - rem)
#if(batches[0].size != batches[-1].size):
#    n_add = batches[0].size - batches[-1].size    
#    new_avg = np.average(batches[-1])
#    new_ar = np.array([new_avg]*n_add)
#    batches[-1] = np.concatenate((batches[-1], new_ar))
#    for b in batches:
#        print(b.shape)
#np.stack(batches)
#batched_avgs = np.average(batches,1)
#print(batched_avgs)
#for n in range(3, int(len(noise)/10)):
#    print(n)
#    bavg = [np.average(b) for b in np.array_split(noise, n)]
#   np.average(np.array_split(noise,3))



#plt.plot(fft_freqs[fft_freqs>0], (fft*np.conj(fft))[fft_freqs>0])
#plt.xlim(0,f_c+B)
#plt.grid()
#plt.show()

#plt.plot(noise)
#scaled = np.int16(noise/np.max(np.abs(noise)) * 32767)*2
#a = scipy.io.wavfile.write('C:\\Users\\pnorm\\Desktop\\test.wav', 44100, scaled)
