import matplotlib.pyplot as plt
import numpy as np

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
        self.I_dot_report()
        self.U_res_report()
        self.R_report()
        self.log_file.write(f'Mass is {self.mass:3.2e} mg;\n')
        self.Power_report()
        self.E_report()
        try:
            self.OriginLab_report()
        except Exception as ex:
            print(ex)
        self.log_file.close()

    def E_report(self):
        E_array = self.df_Power['MW'].values * np.gradient(self.df_Power['us'].values)
        for i in range(1, E_array.size):
            E_array[i] += E_array[i - 1]
        self.df_Energy = pd.DataFrame(
            {
                'us': self.df_U_res['us'].values,
                'J': E_array,
                'J/mg': E_array / self.mass
            }
        )
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        ax.set(
            xlabel='Time, us',
            ylabel='Voltage, kV',
            title='Energy',
            ylim=[-self.Tektronix_ylim, self.Tektronix_ylim]
        )
        self.Energy_ylim = np.abs(self.df_Energy['J'].loc[self.df_Energy['us'] > 0]).max()
        ax1.set(
            ylabel='Energy, J',
            ylim=[-self.Energy_ylim, self.Energy_ylim]
        )
        ax1.grid()
        ax.plot(self.df_Tektronix['us'], self.df_Tektronix['kV'])
        Energy_plot, = ax1.plot(self.df_Energy['us'], self.df_Energy['J'], 'r')
        plt.tight_layout()
        plt.draw()
        plt.savefig('Report/Energy.png')
        ax1.set(
            ylabel='Energy/mass, J/mg',
            ylim=[-self.Energy_ylim / self.mass, self.Energy_ylim / self.mass]
        )
        Energy_plot.set_data(self.df_Energy['us'], self.df_Energy['J/mg'])
        plt.tight_layout()
        plt.savefig('Report/Energy_per_mass.png')
        plt.draw()
        plt.show()
        self.df_Energy.to_excel('Report/Energy(J).xlsx')
        self.log_file.writelines([
            f'Energy limit is {self.Energy_ylim:3.2e} J;\n',
            f'Energy per mass limit is {self.Energy_ylim / self.mass:3.2e} J/mg;\n'
        ])

    def Power_report(self):
        self.df_Power = pd.DataFrame({
            'us': self.df_U_res['us'].values,
            'MW': self.df_U_res['kV'] * self.I_function(self.df_U_res['us'].values),
            'MW/mg': self.df_U_res['kV'] * self.I_function(self.df_U_res['us'].values) / self.mass
        })
        fig, ax = plt.subplots()
        ax1 = ax.twinx()

        ax.set(
            xlabel='Time, us',
            ylabel='Voltage, kV',
            title='Power',
            ylim=[-self.Tektronix_ylim, self.Tektronix_ylim]
        )
        self.Power_ylim = np.abs(self.df_Power['MW'].loc[self.df_Power['us'] > 0]).max()
        ax1.set(
            ylabel='Power, MW',
            ylim=[-self.Power_ylim, self.Power_ylim]
        )
        ax1.grid()
        ax.plot(self.df_Tektronix['us'], self.df_Tektronix['kV'])
        power_plot, = ax1.plot(self.df_Power['us'], self.df_Power['MW'], 'r')
        plt.tight_layout()
        plt.draw()
        plt.savefig('Report/Power.png')
        ax1.set(
            ylabel='Power/mass, MW/mg',
            ylim=[-self.Power_ylim / self.mass, self.Power_ylim / self.mass]
        )
        power_plot.set_data(self.df_Power['us'], self.df_Power['MW/mg'])
        plt.savefig('Report/Power_per_mass.png')
        plt.tight_layout()
        plt.draw()
        plt.show()
        self.df_Power.to_excel('Report/Power(MW).xlsx')
        self.log_file.writelines([
            f'Power limit is {self.Power_ylim:3.2e} MW;\n',
            f'Power per mass limit is {self.Power_ylim / self.mass:3.2e} MW/mg;\n'
        ])

    def R_report(self):
        self.df_Resistance = pd.DataFrame({
            'us': self.df_U_res['us'].values,
            'Ohm': self.df_U_res['kV'] / self.I_function(self.df_U_res['us'].values)
        })
        self.df_Resistance['Ohm'] = np.where(self.I_function(self.df_U_res['us'].values) == 0, 0,
                                             self.df_Resistance['Ohm'].values)
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        ax.set(
            xlabel='Time, us',
            ylabel='Voltage, kV',
            title='Resistance',
            ylim=[-self.Tektronix_ylim, self.Tektronix_ylim]
        )
        self.R_ylim = np.abs(self.df_Resistance['Ohm'].loc[self.df_Resistance['us'] > 0]).max()
        ax1.set(
            ylabel='R, Ohm',
            ylim=[-self.R_ylim, self.R_ylim]
        )
        ax.plot(self.df_Tektronix['us'], self.df_Tektronix['kV'])
        ax1.plot(self.df_Resistance['us'], self.df_Resistance['Ohm'], 'r')
        plt.tight_layout()
        ax1.grid()
        plt.savefig('Report/Resistance.png')
        plt.show()
        self.df_Resistance.to_excel('Report/R(Ohm).xlsx')
        self.log_file.write(f'Resistance limit is {self.R_ylim} Ohm;\n')

    def I_dot_function(self, time):
        ret = np.interp(time, self.df_I_dot['us'], self.df_I_dot['kA/us'])
        return ret

    def Tektronix_function(self, time):
        ret = np.interp(time, self.df_Tektronix['us'], self.df_Tektronix['kV'])
        return ret

    def I_function(self, time):
        noise = np.abs(self.df_Current['kA'].loc[self.df_Current['us'] < 0]).max()
        clear_I = np.where(np.abs(self.df_Current['kA'].values) < noise, 0, self.df_Current['kA'].values)
        ret = np.interp(time, self.df_Current['us'].values, clear_I)
        return ret

    def U_res_report(self):
        df_I_dot_to_plot = self.df_I_dot.loc[self.df_I_dot['us'] > 0]
        I_dot_max_index = np.abs(df_I_dot_to_plot['kA/us']).values.argmax()
        I_dot_max_time = df_I_dot_to_plot['us'].values[I_dot_max_index]
        Tektronix_I_dot_max = self.Tektronix_function(I_dot_max_time)
        L = Tektronix_I_dot_max / self.I_dot_ylim
        self.df_U_res = pd.DataFrame({
            'us': self.df_Tektronix['us'].values,
            'kV': self.df_Tektronix['kV'].values - L * self.I_dot_function(self.df_Tektronix['us'].values)
        })
        fig, ax = plt.subplots()
        ax1 = ax.twinx()

        ax.set(
            xlabel='Time, us',
            ylabel='Voltage, kV',
            title='U res comparison',
            ylim=[-self.Tektronix_ylim, self.Tektronix_ylim]
        )
        self.U_res_ylim = np.abs(self.df_U_res['kV'].loc[self.df_U_res['us'] > 0]).max()
        ax1.set(
            ylabel='U_res, kV',
            ylim=[-self.U_res_ylim, self.U_res_ylim]
        )
        ax.plot(self.df_Tektronix['us'], self.df_Tektronix['kV'])
        ax1.plot(self.df_U_res['us'], self.df_U_res['kV'], 'r')
        plt.tight_layout()
        ax1.grid()
        plt.savefig('Report/U_res_comparison.png')
        plt.show()
        self.log_file.writelines([
            f'Inductance is {L:3.2e} uH;\n',
            f'U_res limit is {self.U_res_ylim:3.2e} kV;\n'
        ])

    def I_dot_report(self):
        self.df_I_dot = pd.DataFrame({
            'us': self.df_Current['us'].values,
            'kA/us': np.gradient(self.df_Current['kA'].values) / np.gradient(self.df_Current['us'].values)
        })
        self.df_I_dot = self.df_I_dot.rolling(self.n_conv, min_periods=1).mean()
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        ax.set(
            xlabel='Time, us',
            ylabel='Voltage, kV',
            title='I dot comparison',
            ylim=[-self.Tektronix_ylim, self.Tektronix_ylim]
        )
        self.I_dot_ylim = np.abs(self.df_I_dot['kA/us'].loc[self.df_I_dot['us'] > 0]).max()
        ax1.set(
            ylabel='I_dot, kA/us',
            ylim=[-self.I_dot_ylim, self.I_dot_ylim]
        )
        ax.plot(self.df_Tektronix['us'], self.df_Tektronix['kV'])
        ax1.plot(self.df_I_dot['us'], self.df_I_dot['kA/us'], 'r')
        plt.tight_layout()
        ax1.grid()
        plt.savefig('Report/I_dot_comparison.png')
        plt.show()

        self.df_I_dot.to_excel('Report/I_dot(kA_us).xlsx')
        self.log_file.write(f'I_dot limit is {self.I_dot_ylim} kA/us;\n')

    def OriginLab_report(self):
        import originpro as op
        op.new()
        path = os.getcwd()
        save_name = path + '\\UEFE_waveform_OriginLab_report.opju'
        b = op.save(save_name)
        current_sheet = op.new_sheet(lname='Current')
        trig_out_sheet = op.new_sheet(lname='Trig_out')
        Systron_sheet = op.new_sheet(lname='Systron')
        Tektronix_sheet = op.new_sheet(lname='Tektronix')
        I_dot_sheet = op.new_sheet(lname='I_dot')
        U_res_sheet = op.new_sheet(lname='U_res')
        R_sheet = op.new_sheet(lname='R')
        Power_sheet = op.new_sheet(lname='Power')
        Energy_sheet = op.new_sheet(lname='Power')

        current_sheet.from_df(self.df_Current)
        trig_out_sheet.from_df(self.df_Trig_out)
        Systron_sheet.from_df(self.df_Systron)
        Tektronix_sheet.from_df(self.df_Tektronix)
        I_dot_sheet.from_df(self.df_I_dot)
        U_res_sheet.from_df(self.df_U_res)
        R_sheet.from_df(self.df_Resistance)
        Power_sheet.from_df(self.df_Power)
        Energy_sheet.from_df(self.df_Energy)

        # waveform_graph = op.new_graph(template='Waveform_butterfly', lname='Waveform')
        try:
            waveform_graph = op.new_graph(template='SC_interpreter', lname='Waveform')
        except:
            waveform_graph = op.new_graph(template='4Ys_YY-YY', lname='Waveform')
        waveform_graph[0].add_plot(current_sheet, coly=2, colx=0, type='line')
        waveform_graph[0].set_xlim(begin=0, end=self.df_Current['us'].max())
        waveform_graph[0].set_ylim(begin=-self.Current_ylim, end=self.Current_ylim)

        waveform_graph[1].add_plot(Tektronix_sheet, coly=2, colx=0, type='line')
        waveform_graph[1].set_xlim(begin=0, end=self.df_Tektronix['us'].max())
        waveform_graph[1].set_ylim(begin=-self.Tektronix_ylim, end=self.Tektronix_ylim)

        waveform_graph[2].add_plot(trig_out_sheet, coly=1, colx=0, type='line')
        waveform_graph[2].set_xlim(begin=0, end=self.df_Current['us'].max())
        waveform_graph[2].set_ylim(begin=-np.abs(self.df_Trig_out['V']).max(), end=np.abs(self.df_Trig_out['V']).max())

        waveform_graph[3].add_plot(Systron_sheet, coly=1, colx=0, type='line')
        waveform_graph[3].set_xlim(begin=0, end=self.df_Current['us'].max())
        waveform_graph[3].set_ylim(begin=-np.abs(self.df_Systron['V']).max(), end=np.abs(self.df_Systron['V']).max())

        I_dot_graph = op.new_graph(template='UEFE_U_res', lname='I_dot')
        I_dot_graph[0].add_plot(current_sheet, coly=2, colx=0, type='line')
        I_dot_graph[0].set_xlim(begin=0, end=self.df_Current['us'].max())
        I_dot_graph[0].set_ylim(begin=-self.Current_ylim, end=self.Current_ylim)

        I_dot_graph[1].add_plot(Tektronix_sheet, coly=2, colx=0, type='line')
        I_dot_graph[1].set_xlim(begin=0, end=self.df_Tektronix['us'].max())
        I_dot_graph[1].set_ylim(begin=-self.Tektronix_ylim, end=self.Tektronix_ylim)

        I_dot_graph[2].add_plot(trig_out_sheet, coly=1, colx=0, type='line')
        I_dot_graph[2].set_xlim(begin=0, end=self.df_Current['us'].max())
        I_dot_graph[2].set_ylim(begin=-np.abs(self.df_Trig_out['V']).max(), end=np.abs(self.df_Trig_out['V']).max())

        I_dot_graph[3].add_plot(I_dot_sheet, coly=1, colx=0, type='line')
        I_dot_graph[3].set_xlim(begin=0, end=self.df_I_dot['us'].max())
        I_dot_graph[3].set_ylim(begin=-self.I_dot_ylim, end=self.I_dot_ylim)

        U_res_graph = op.new_graph(template='UEFE_U_res', lname='U_res')
        U_res_graph[0].add_plot(current_sheet, coly=2, colx=0, type='line')
        U_res_graph[0].set_xlim(begin=0, end=self.df_Current['us'].max())
        U_res_graph[0].set_ylim(begin=-self.Current_ylim, end=self.Current_ylim)

        U_res_graph[1].add_plot(Tektronix_sheet, coly=2, colx=0, type='line')
        U_res_graph[1].set_xlim(begin=0, end=self.df_Tektronix['us'].max())
        U_res_graph[1].set_ylim(begin=-self.Tektronix_ylim, end=self.Tektronix_ylim)

        U_res_graph[2].add_plot(trig_out_sheet, coly=1, colx=0, type='line')
        U_res_graph[2].set_xlim(begin=0, end=self.df_Current['us'].max())
        U_res_graph[2].set_ylim(begin=-np.abs(self.df_Trig_out['V']).max(), end=np.abs(self.df_Trig_out['V']).max())

        U_res_graph[3].add_plot(U_res_sheet, coly=1, colx=0, type='line')
        U_res_graph[3].set_xlim(begin=0, end=self.df_U_res['us'].max())
        U_res_graph[3].set_ylim(begin=-self.U_res_ylim, end=self.U_res_ylim)

        R_graph = op.new_graph(template='UEFE_resistance', lname='R')
        R_graph[0].add_plot(current_sheet, coly=2, colx=0, type='line')
        R_graph[0].set_xlim(begin=0, end=self.df_Current['us'].max())
        R_graph[0].set_ylim(begin=-self.Current_ylim, end=self.Current_ylim)

        R_graph[1].add_plot(Tektronix_sheet, coly=2, colx=0, type='line')
        R_graph[1].set_xlim(begin=0, end=self.df_Tektronix['us'].max())
        R_graph[1].set_ylim(begin=-self.Tektronix_ylim, end=self.Tektronix_ylim)

        R_graph[2].add_plot(trig_out_sheet, coly=1, colx=0, type='line')
        R_graph[2].set_xlim(begin=0, end=self.df_Current['us'].max())
        R_graph[2].set_ylim(begin=-np.abs(self.df_Trig_out['V']).max(), end=np.abs(self.df_Trig_out['V']).max())

        R_graph[3].add_plot(R_sheet, coly=1, colx=0, type='line')
        R_graph[3].set_xlim(begin=0, end=self.df_U_res['us'].max())
        R_graph[3].set_ylim(begin=-self.R_ylim, end=self.R_ylim)

        Power_graph = op.new_graph(template='UEFE_power', lname='Power')
        Power_graph[0].add_plot(current_sheet, coly=2, colx=0, type='line')
        Power_graph[0].set_xlim(begin=0, end=self.df_Current['us'].max())
        Power_graph[0].set_ylim(begin=-self.Current_ylim, end=self.Current_ylim)

        Power_graph[1].add_plot(Tektronix_sheet, coly=2, colx=0, type='line')
        Power_graph[1].set_xlim(begin=0, end=self.df_Tektronix['us'].max())
        Power_graph[1].set_ylim(begin=-self.Tektronix_ylim, end=self.Tektronix_ylim)

        Power_graph[2].add_plot(trig_out_sheet, coly=1, colx=0, type='line')
        Power_graph[2].set_xlim(begin=0, end=self.df_Current['us'].max())
        Power_graph[2].set_ylim(begin=-np.abs(self.df_Trig_out['V']).max(), end=np.abs(self.df_Trig_out['V']).max())

        Power_graph[3].add_plot(Power_sheet, coly=1, colx=0, type='line')
        Power_graph[3].set_xlim(begin=0, end=self.df_U_res['us'].max())
        Power_graph[3].set_ylim(begin=-self.Power_ylim, end=self.Power_ylim)

        Energy_graph = op.new_graph(template='UEFE_Energy', lname='Energy')
        Energy_graph[0].add_plot(current_sheet, coly=2, colx=0, type='line')
        Energy_graph[0].set_xlim(begin=0, end=self.df_Current['us'].max())
        Energy_graph[0].set_ylim(begin=-self.Current_ylim, end=self.Current_ylim)

        Energy_graph[1].add_plot(Tektronix_sheet, coly=2, colx=0, type='line')
        Energy_graph[1].set_xlim(begin=0, end=self.df_Tektronix['us'].max())
        Energy_graph[1].set_ylim(begin=-self.Tektronix_ylim, end=self.Tektronix_ylim)

        Energy_graph[2].add_plot(trig_out_sheet, coly=1, colx=0, type='line')
        Energy_graph[2].set_xlim(begin=0, end=self.df_Current['us'].max())
        Energy_graph[2].set_ylim(begin=-np.abs(self.df_Trig_out['V']).max(), end=np.abs(self.df_Trig_out['V']).max())

        Energy_graph[3].add_plot(Energy_sheet, coly=1, colx=0, type='line')
        Energy_graph[3].set_xlim(begin=0, end=self.df_U_res['us'].max())
        Energy_graph[3].set_ylim(begin=-self.Energy_ylim, end=self.Energy_ylim)

        Power_per_mass_graph = op.new_graph(template='UEFE_power_per_mass', lname='Power_per_mass')
        Power_per_mass_graph[0].add_plot(current_sheet, coly=2, colx=0, type='line')
        Power_per_mass_graph[0].set_xlim(begin=0, end=self.df_Current['us'].max())
        Power_per_mass_graph[0].set_ylim(begin=-self.Current_ylim, end=self.Current_ylim)

        Power_per_mass_graph[1].add_plot(Tektronix_sheet, coly=2, colx=0, type='line')
        Power_per_mass_graph[1].set_xlim(begin=0, end=self.df_Tektronix['us'].max())
        Power_per_mass_graph[1].set_ylim(begin=-self.Tektronix_ylim, end=self.Tektronix_ylim)

        Power_per_mass_graph[2].add_plot(trig_out_sheet, coly=1, colx=0, type='line')
        Power_per_mass_graph[2].set_xlim(begin=0, end=self.df_Current['us'].max())
        Power_per_mass_graph[2].set_ylim(begin=-np.abs(self.df_Trig_out['V']).max(),
                                         end=np.abs(self.df_Trig_out['V']).max())

        Power_per_mass_graph[3].add_plot(Power_sheet, coly=2, colx=0, type='line')
        Power_per_mass_graph[3].set_xlim(begin=0, end=self.df_U_res['us'].max())
        Power_per_mass_graph[3].set_ylim(begin=-self.Power_ylim / self.mass, end=self.Power_ylim / self.mass)

        Energy_per_mass_graph = op.new_graph(template='UEFE_Energy_per_mass', lname='Energy_per_mass')
        Energy_per_mass_graph[0].add_plot(current_sheet, coly=2, colx=0, type='line')
        Energy_per_mass_graph[0].set_xlim(begin=0, end=self.df_Current['us'].max())
        Energy_per_mass_graph[0].set_ylim(begin=-self.Current_ylim, end=self.Current_ylim)

        Energy_per_mass_graph[1].add_plot(Tektronix_sheet, coly=2, colx=0, type='line')
        Energy_per_mass_graph[1].set_xlim(begin=0, end=self.df_Tektronix['us'].max())
        Energy_per_mass_graph[1].set_ylim(begin=-self.Tektronix_ylim, end=self.Tektronix_ylim)

        Energy_per_mass_graph[2].add_plot(trig_out_sheet, coly=1, colx=0, type='line')
        Energy_per_mass_graph[2].set_xlim(begin=0, end=self.df_Current['us'].max())
        Energy_per_mass_graph[2].set_ylim(begin=-np.abs(self.df_Trig_out['V']).max(),
                                          end=np.abs(self.df_Trig_out['V']).max())

        Energy_per_mass_graph[3].add_plot(Energy_sheet, coly=2, colx=0, type='line')
        Energy_per_mass_graph[3].set_xlim(begin=0, end=self.df_U_res['us'].max())
        Energy_per_mass_graph[3].set_ylim(begin=-self.Energy_ylim / self.mass, end=self.Energy_ylim / self.mass)

        op.save()
        op.exit()

    def physical_values_report(self):
        """
        Saves calculated values
        :return:
        """
        self.df_Current['kA'] = self.df_Current['V'] * self.Rogovski_ampl * 1.0e-3
        self.df_Tektronix['kV'] = self.df_Tektronix['V'] * self.Tektronix_VD * 1.0e-3
        df_Tektronix_to_plot = self.df_Tektronix.loc[self.df_Tektronix['us'] > 0]
        self.Tektronix_ylim = np.abs(df_Tektronix_to_plot['kV']).max()
        self.Current_ylim = np.abs(self.df_Current['kA']).max()
        fig, ax = plt.subplots()
        ax1 = ax.twinx()
        ax.set(
            xlabel='Time, us',
            ylabel='Current, kA',
            title='Physical data',
            ylim=[-self.Current_ylim, self.Current_ylim]
        )

        ax1.set(
            ylabel='Voltage, kV',
            ylim=[-self.Tektronix_ylim, self.Tektronix_ylim]
        )
        ax1.plot(self.df_Tektronix['us'], self.df_Tektronix['kV'])
        ax.plot(self.df_Current['us'], self.df_Current['kA'], 'r')
        plt.tight_layout()
        ax1.grid()
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
        self.h_foil = self.data_dict['info']['Value']['Thickness']
        self.waist = self.data_dict['info']['Value']['Waist']
        self.w_foil = self.data_dict['info']['Value']['Width']
        self.l_foil = self.data_dict['info']['Value']['Length']
        self.density = self.data_dict['info']['Value']['Density']
        self.mass = self.density * self.h_foil * self.l_foil * 0.5 * (self.w_foil + self.waist)  # * 1.0e-3  # mg
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
