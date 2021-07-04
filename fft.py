import numpy as np
import matplotlib.pyplot as plt
import collections
# =============================================================================

#Szybka transformacja Fouriera (FFT)
def fourier(data):
    x = np.asarray(data, dtype=float)
    N = len(x)
    n = np.arange(N)  
    k = n.reshape((N,1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def inverse_fourier(data):
    x = np.asarray(data, dtype=float)
    N = len(x)
    n = np.arange(N)  
    k = n.reshape((N,1))
    M = np.exp(2j * np.pi * k * n / N)
    return 1 / N * np.dot(M, x)

def plot(x, title='', xlabel='', ylabel=''):
    plt.plot(x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.clf()
    
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = fourier(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


# =============================================================================    
def main():
    
###___Wybierz zestaw_____
# =============================================================================    
    zestaw = 17186%20; 
    print ("Zestaw ",zestaw) #Zestaw 6 
    print('***')
    
    
###___Wgraj dane______
# =============================================================================
#       N = 1024 - iloć próbek
#       f = 1024/1 - Częstotliwoć próbkowania
#       T = 1/1024 - Okres próbkowania      
#       data        - Im
#       numer probki  - Re 
# =============================================================================
    file = open("data\setf"+str(zestaw)+".csv")
    data = file.read() #string
    data=np.asarray(data.split('\n')[:-1], dtype=float)
    # plot(data,'DATA', 'Time = 1s', 'Amplitude')
    
###___Wykonaj transformacje______________________________
# =============================================================================      
    fft = np.abs(fourier(data))
    print("DFT implementation correct?: ", np.allclose(fourier(data), np.fft.fft(data)))
    print("Inverse Fourier correct?: ", np.allclose(data, np.fft.ifft(fourier(data))))   
    # plot(res, 'FFT', 'Frequency', 'Amplitude')  
     
    t_n = 1 # czas probkowania w sek
    N = 1024 # ilosc probek
    T = t_n / N # okres probkowania
    f_s = 1/T # czestotliwosc probkowania
     
    f_values, fft_values = get_fft_values(data, T, N, f_s)
    print(f_values)
    # plt.plot(f_values, fft_values, linestyle='-', color='blue')
    # plt.xlabel('Frequency [Hz]', fontsize=16)
    # plt.ylabel('Amplitude', fontsize=16)
    # plt.title("Frequency domain of the signal", fontsize=16)
    # plt.show() 

    # print("frequencies in every time point: ", abs(fft))
    
###___Szumy i sygnał glowny___________________________________________________________
# =============================================================================   
    threshold = np.max(fft)/1.7
    
    main_signal = fft.copy()
    main_signal[main_signal < threshold] = 0
    main_signal = inverse_fourier(main_signal)
    
    noise = data - main_signal
    
    # plot(main_signal, 'MAIN_SIGNAL', 'Time', 'Intensity') 
    # plot(noise, 'NOISE', 'Time', 'Intensity') 
    
    
###___Sygnały składowe________________________________________________
# =============================================================================  
    from math import ceil
    # from detecta import detect_peaks
    # print("peaks:", detect_peaks(fft_values, threshold = 1, show=True))  
    
    peaks = np.where(fft>threshold)[0]
    signals = []
    ampl_freq = []
    half_peaks = ceil(len(peaks)/2)
    
    for peak in peaks:
        signal = fft.copy()
        signal[signal != signal[peak]] = 0
        center_of_mass = sum(np.abs(signal))/len(signal)
        ampl_freq.append(center_of_mass) 
        signal = inverse_fourier(signal)         
        signals.append(signal)
        plot(signal, 'SIGNAL of freq: '+str(peak), 'Time','Intensity') 
    
    # for peak in peaks:
    #     signal = fft.copy()
    #     signal[signal != signal[peak]] = 0
    #     center_of_mass = sum(np.abs(signal))/len(signal)
    #     ampl_freq.append(center_of_mass) 
    #     signal = inverse_fourier(signal)         
    #     signals.append(signal)
    #     plot(signal, 'SIGNAL of freq: '+str(peak), 'Time','Intensity')   
    
    # print ("Signals: ", signals)   
    # plot(sum(signals), "SUMED_SIGNALS",'Time','Amplitude')
    # plot(sum(signals)+noise, "SUMED_SIGNALS_+_NOISE",'Time','Amplitude')
    
    # SPRAWDZENIE
    print("Summes back to data ?: ", np.allclose(data, sum(abs(signals)) + noise))
    print("Summes back to main signal ?: ", np.allclose(main_signal, sum(signals)))
    print('----------------------------------------------------------------------')


###___Podaj wzory_________________________________________________
# =============================================================================    
    print("___PARAMETRY SYGNAŁU WEJŚCIOWEGO_____")
    ampl = np.abs(data)*2
    freq = f_s #np.count_nonzero(data == 0)
    powrMain = np.sum(ampl**2)
    print('amplituda max: ', np.max(ampl))
    print('częstotliwosc: ', freq)
    print('moc: ',powrMain)
    print()
    
    print("___PARAMETRY SYGNAŁÓW SKŁADOWYCH_____")
    for i in range(half_peaks) :
        ampl = np.abs(signals[i].imag)*2
        freq = peaks[i]
        powr = np.sum(ampl**2)
        print('Sygnał ', i+1, '===')
        print('amplituda max: ', np.max(ampl))
        # print('Amplituda freq i time taka sama?', ampl_freq[i]*2, np.max(ampl))
        print('częstotliwosc: ', freq)
        print('moc: ', powr/powrMain*100, '%')
        print()
        
    print("___PARAMETRY SZUMU_____")
    ampl = np.abs(noise.imag)*2
    powr = np.sum(ampl**2)
    print('amplituda max: ', np.max(ampl))
    print('moc: ', powr/powrMain*100, '%')
    print()
    
    
     
if __name__ == '__main__':
    main()
