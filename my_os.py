from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import os
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks


def open_xlsx(fname):
    """
    the function read the info file of the experiment
    :param fname: file name
    :return:
    pandas data frame
    """
    data = pd.read_excel(fname)
    data = data.set_index('Parameter')
    return data


def open_csv(file_name):
    """
    the function read the waveform file *.csv:
    current and voltage divider
    :param file_name: file name
    :return:
    dictionary of original values [V]
    {
        'time': Rogowski time [us],
        'Rogowski': Rogowski signal,
        'Trig_out': XXRapid camera trig-out signal,
        'Systron': Systron generator trig signal,
        'Tektronix': Tektronix voltage divider signal,
    }
    in original units
    """
    waveform = pd.read_csv(file_name)
    plt.plot(1.0e6 * waveform['s'], waveform['Volts'], label='Current')
    plt.plot(1.0e6 * waveform['s.1'], waveform['Volts.1'], label='4Quick trig')
    plt.plot(1.0e6 * waveform['s.2'], waveform['Volts.2'], label='Main trig')
    plt.plot(1.0e6 * waveform['s.3'], waveform['Volts.3'], label='Tektronix')
    plt.xlabel('t, us')
    plt.legend()
    plt.title('original data')
    plt.grid()
    plt.savefig('Report/original.png')
    plt.show()
    ret = {
        'time': 1.0e6 * waveform['s'],
        'Rogowski': waveform['Volts'],
        'Trig_out': waveform['Volts.1'],
        'Systron': waveform['Volts.2'],
        'Tektronix': waveform['Volts.3'],
    }

    return ret


def open_folder():
    """
    The function loads the data of SC experiment from file dialog;
    the experiment folder includes:
    '*_info.xlsx' file with scalar data of experiment;
    'SC*DDMMYY.csv' file with waveforms;
    :return:
    dictionary:
    {
    'waveforms': dictionary of the original waveforms data,
    'info':dataframe of SC experiment
    }
    """
    folder_name = filedialog.askdirectory(
        #initialdir='./example'
        initialdir='C:/Users/User/OneDrive - Technion/UEWE/Foils/Butterfly_cloud/Butterfly_MAGEN_2024/Processing'
    )
    os.chdir(folder_name)
    os.makedirs('Report', exist_ok=True)
    files_data = dict()
    for file_name in os.listdir():
        if (file_name.split('.')[-1] == 'csv') & (file_name[0] == 's'):
            files_data['waveform'] = open_csv(file_name)
            continue
        if file_name.split('.')[-1] == 'xlsx':
            files_data['info'] = open_xlsx(file_name)
            continue
    pass
    return files_data
