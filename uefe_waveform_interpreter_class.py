from my_os import *
import pandas as pd
from scipy.optimize import curve_fit


class uefe_waveform_interpreter_class:
    """
    the class processes the uefe experiment waveform file shot_*.csv;
    the file consists of Rogowski values,
    Tektronix values,
    synchro pulses,
    also it uses clues from file info.xlsx;
    see example folder;
    """

    def __init__(self):
        self.data_dict = open_folder()
        self.log_file = open('Report/report_text.txt', 'w')
        self.curdir = os.curdir
        self.sort_data_dict()
        self.smoothed_report()
        self.physical_values_report()
        try:
            self.OriginLab_report()
        except Exception as ex:
            print(ex)
        self.log_file.close()

    def OriginLab_report(self):
        import originpro as op
        op.new()
        path = os.getcwd()
        save_name = path + '\\SC_OriginLab_report.opju'
        b = op.save(save_name)
        waveform_sheet = op.new_sheet(lname='Waveform')
        current_sheet = op.new_sheet(lname='Current')
        trig_out_sheet = op.new_sheet(lname='Trig_out')
        Systron_sheet = op.new_sheet(lname='Systron')
        Tektronix_sheet = op.new_sheet(lname='Tektronix')

        current_sheet.from_df(self.df_Current)
        trig_out_sheet.from_df(self.df_Trig_out)
        Systron_sheet.from_df(self.df_Systron)
        Tektronix_sheet.from_df(self.df_Tektronix)

        # waveform_graph = op.new_graph(template='Waveform_butterfly', lname='Waveform')
        try:
            waveform_graph = op.new_graph(template='SC_interpreter', lname='Waveform')
        except:
            waveform_graph = op.new_graph(template='4Ys_YY-YY', lname='Waveform')
        waveform_graph[0].add_plot(current_sheet, coly=2, colx=0, type='line')
        waveform_graph[0].set_xlim(begin=0, end=self.df_Current['us'].max())
        waveform_graph[0].set_ylim(begin=-np.abs(self.df_Current['kA']).max(), end=np.abs(self.df_Current['kA']).max())

        waveform_graph[1].add_plot(Tektronix_sheet, coly=2, colx=0, type='line')
        # waveform_graph[1].set_xlim(begin=0, end=self.wf_time_power.max())
        waveform_graph[1].set_ylim(begin=-np.abs(self.df_Tektronix['kV']).max(),
                                   end=np.abs(self.df_Tektronix['kV']).max())

        waveform_graph[2].add_plot(trig_out_sheet, coly=1, colx=0, type='line')
        waveform_graph[2].set_xlim(begin=0, end=self.df_Current['us'].max())
        waveform_graph[2].set_ylim(begin=-np.abs(self.df_Trig_out['V']).max(), end=np.abs(self.df_Trig_out['V']).max())

        waveform_graph[3].add_plot(Systron_sheet, coly=1, colx=0, type='line')
        waveform_graph[3].set_xlim(begin=0, end=self.df_Current['us'].max())
        waveform_graph[3].set_ylim(begin=-np.abs(self.df_Systron['V']).max(), end=np.abs(self.df_Systron['V']).max())

        op.save()
        op.exit()

    def physical_values_report(self):
        """
        Saves calculated values
        :return:
        """
        self.df_Current['kA'] = self.df_Current['V'] * self.Rogovski_ampl*1.0e-3
        self.df_Tektronix['kV'] = self.df_Tektronix['V'] * self.Tektronix_VD * 1.0e-3
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        ax.set(
            xlabel='Time, us',
            ylabel='Current, kA',
            title='Physical data',
            ylim=[-np.abs(self.df_Current['kA']).max(), np.abs(self.df_Current['kA']).max()]
        )
        ax1.set(
            ylabel='Voltage, kV',
            ylim=[-np.abs(self.df_Tektronix['kV']).max(), np.abs(self.df_Tektronix['kV']).max()]
        )
        ax1.plot(self.df_Tektronix['us'], self.df_Tektronix['kV'])
        ax.plot(self.df_Current['us'], self.df_Current['kA'], 'r')

        plt.savefig('Report/physical_values.png')
        plt.show()
        self.df_Current.to_excel('Report/Current(kA).xlsx')
        self.df_Tektronix.to_excel('Report/Voltage(kV).xlsx')

    def find_coefficients(self):
        pic_index = find_peaks(-self.df_Current['V'].values, prominence=1, distance=100)[0]
        pic_time = self.df_Current['us'].values[pic_index]
        pic_volt = self.df_Current['V'].values[pic_index]
        noise_ind = np.argwhere(self.df_Current['us'].values < 0)
        noise = self.df_Current['V'].values[noise_ind].max()
        current_start_ind = np.argwhere(np.abs(self.df_Current['V'].values) > noise).min()
        self.current_start_time = self.df_Current['us'].values[current_start_ind]
        current_start_volt = self.df_Current['V'].values[current_start_ind]

        def my_exp(x, a, b):
            return -a * np.exp(-x / b)

        opt, err = curve_fit(my_exp, pic_time, pic_volt)
        time_to_approx = np.arange(pic_time[0], pic_time[-1], np.gradient(self.df_Current['us'].values).mean())
        Rogowski_to_approx = my_exp(time_to_approx, opt[0], opt[1])
        plt.plot(time_to_approx, Rogowski_to_approx, label='decay')
        plt.plot(pic_time, pic_volt, 'o', label='picks')
        plt.plot(self.current_start_time, current_start_volt, 'o', label='current start')
        plt.plot(self.df_Current['us'], self.df_Current['V'], label='Rogowski')
        plt.xlabel('t, us')
        plt.ylabel('signal, V')
        plt.legend()
        plt.grid()
        plt.savefig('Report/picks.png')
        plt.show()
        self.period = np.gradient(pic_time).mean()
        self.log_file.write(f'Period is {self.period:3.2e} us\n')
        self.period *= 1.0e-6
        self.I_sc = 2.0 * np.pi * self.Capacity * self.U_0 / self.period * 1.0e-3  # kA
        self.log_file.write(f'Short circuit current is {self.I_sc:3.2e} kA\n')
        self.L_sc = (1 / self.Capacity) * (self.period / 2.0 / np.pi) ** 2
        self.log_file.write(f'Short circuit inductance is {self.L_sc:3.2e} H\n')
        self.Rogovski = -self.I_sc / pic_volt[0]
        self.log_file.write(f'Rogowski coefficient is {self.Rogovski:3.2e} kA/V\n')
        self.rise_time = pic_time.min() - self.current_start_time
        self.log_file.write(f'Rise time is {self.rise_time:3.2e} us\n')
        self.decay_time = opt[1]
        self.log_file.write(f'Decay time is {self.decay_time:3.2e} us\n')
        self.resistance = 2 * self.L_sc / self.decay_time / 1.0e-6  # Ohm
        self.log_file.write(f'Resistance is {self.resistance:3.2e} Ohm\n')

    def sort_data_dict(self):
        self.n_conv = self.data_dict['info']['Value']['Rogovski_conv']
        self.Rogovski_ampl = self.data_dict['info']['Value']['Rogovski_ampl']
        self.Tektronix_VD = self.data_dict['info']['Value']['Tektronix_VD']
        self.df_Current = pd.DataFrame({
            'us': self.data_dict['waveform']['time'],
            'V': self.data_dict['waveform']['Rogowski']
        })
        self.df_Current = self.df_Current.rolling(self.n_conv, min_periods=1).mean()
        self.df_Systron = pd.DataFrame({
            'us': self.data_dict['waveform']['time'],
            'V': self.data_dict['waveform']['Systron']
        })
        self.df_Tektronix = pd.DataFrame({
            'us': self.data_dict['waveform']['time'],
            'V': self.data_dict['waveform']['Tektronix']
        })
        self.df_Tektronix = self.df_Tektronix.rolling(self.n_conv, min_periods=1).mean()
        self.df_Trig_out = pd.DataFrame({
            'us': self.data_dict['waveform']['time'],
            'V': self.data_dict['waveform']['Trig_out']
        })

    def smoothed_report(self):
        plt.title('smoothed')
        plt.plot(self.df_Current['us'], self.df_Current['V'], label='Rogowski')
        plt.plot(self.df_Trig_out['us'], self.df_Trig_out['V'], label='4Quick trig')
        plt.plot(self.df_Systron['us'], self.df_Systron['V'], label='Main trig')
        plt.plot(self.df_Tektronix['us'], self.df_Tektronix['V'], label='Tektronix')
        plt.xlabel('t, us')
        plt.ylabel('signal, V')
        plt.legend()
        plt.grid()
        plt.savefig('Report/smoothed.png')
        plt.show()
