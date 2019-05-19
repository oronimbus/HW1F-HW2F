from IPython.display import display
from bqplot import LinearScale, Axis, Lines, Figure, Hist
import pandas as pd
import ipywidgets as ipw
import numpy as np
import bqplot.pyplot as plt


class HullWhite():
    def __init__(self):
        
        #the model restricts changing step size
        self.steps = 1/52

        # Line chart and histo
        self.x_sc         = LinearScale()
        self.y_sc         = LinearScale()

        self.x_sc_2       = LinearScale()
        self.y_sc_2       = LinearScale()

        self.ax_x         = Axis(label='Weeks', scale=self.x_sc, grid_lines='dashed')
        self.ax_y         = Axis(label='Rate', scale=self.y_sc, orientation='vertical',grid_lines='dashed')
        self.ax_x_2       = Axis(label='Rate', scale=self.x_sc_2,grid_lines='none')
        self.ax_y_2       = Axis(label='Count', scale=self.y_sc_2, orientation='vertical', grid_lines='dashed')

        #HW1F Charts
        self.line1        = Lines(x=[], y=[[], []], scales={'x': self.x_sc, 'y': self.y_sc}, stroke_width=3,colors=['red','green'])
        self.line2        = Lines(x=[], y=[[]], scales={'x': self.x_sc, 'y': self.y_sc}, labels=['MC'])
        self.hist1        = Hist(sample=[], scales={'sample': self.x_sc_2, 'count': self.y_sc_2}, bins=0)
        self.fig1         = Figure(axes=[self.ax_x, self.ax_y], marks=[self.line1, self.line2], title='Hull White 1 Factor Dynamics')
        self.fig2         = Figure(axes=[self.ax_x_2, self.ax_y_2], marks=[self.hist1], title='Final Distribution of Rates')
        
        #HW2F Charts
        self.line3        = Lines(x=[], y=[[], []], scales={'x': self.x_sc, 'y': self.y_sc}, stroke_width=3,colors=['red','green'])
        self.line4        = Lines(x=[], y=[[]], scales={'x': self.x_sc, 'y': self.y_sc})
        self.hist2        = Hist(sample=[], scales={'sample': self.x_sc_2, 'count': self.y_sc_2}, bins=0)
        self.fig3         = Figure(axes=[self.ax_x, self.ax_y], marks=[self.line3, self.line4], title='Hull White 2 Factor Dynamics')
        self.fig4         = Figure(axes=[self.ax_x_2, self.ax_y_2], marks=[self.hist2], title='Final Distribution of Rates')


        # Input widgets
        self.ZC1W         = ipw.FloatText(description='1W',value=2.07,layout=ipw.Layout(width='17%', height='100%'))
        self.ZC3M         = ipw.FloatText(description='3M',value=2.29,layout=ipw.Layout(width='17%', height='100%'))
        self.ZC6M         = ipw.FloatText(description='6M',value=2.45,layout=ipw.Layout(width='17%', height='100%'))
        self.ZC1Y         = ipw.FloatText(description='1Y',value=2.7,layout=ipw.Layout(width='17%', height='100%'))
        self.ZC2Y         = ipw.FloatText(description='2Y',value=3.02,layout=ipw.Layout(width='17%', height='100%'))
        self.ZC6Y         = ipw.FloatText(description='6Y',value=4,layout=ipw.Layout(width='17%', height='100%'))
        self.ZC10Y        = ipw.FloatText(description='10Y',value=4.7,layout=ipw.Layout(width='17%', height='100%'))


        self.normalVol    = ipw.FloatSlider(value=100,min=0,max=1000,description='Normal Vol (bps)')
        self.normalVol2   = ipw.FloatSlider(value=100,min=0,max=1000,description='Normal Vol 2 (bps)')
        self.meanRev      = ipw.FloatSlider(value=1,min=0.0001,max=10, steps=0.01, description='Mean Reversion')
        self.meanRev2     = ipw.FloatSlider(value=3,min=0.0001,max=10, steps=0.01, description='Mean Reversion 2')
        self.numPaths     = ipw.IntSlider(value=50,min=1,max=500,step=1, description='Paths ')
        self.showPaths    = ipw.IntSlider(value=5,min=0,max=100,step=1, description='Display Paths ')
        self.correlation  = ipw.FloatSlider(value=.1,min=-1,max=1,steps=0.01, description='Correlation')



        # Layout with tabs
        self.tab = ipw.Tab()
        self.tab_contents = [0,1]
        self.children = [ipw.VBox([ipw.HBox([self.normalVol, self.meanRev, self.numPaths, self.showPaths]),
                              ipw.HBox([self.fig1,self.fig2])]),
                         ipw.VBox([ipw.HBox([self.normalVol, self.normalVol2, self.meanRev, self.meanRev2]),
                              ipw.HBox([self.correlation, self.numPaths, self.showPaths]),
                              ipw.HBox([self.fig3, self.fig4])])]


        self.tab.children = self.children
        self.tab.set_title(0, 'HW1F')
        self.tab.set_title(1, 'HW2F')
        
        #Observers
        self.ZC1W.observe(self.process_wrapper,'value')
        self.ZC3M.observe(self.process_wrapper,'value')
        self.ZC6M.observe(self.process_wrapper,'value')
        self.ZC1Y.observe(self.process_wrapper,'value')
        self.ZC2Y.observe(self.process_wrapper,'value')
        self.ZC6Y.observe(self.process_wrapper,'value')
        self.ZC10Y.observe(self.process_wrapper,'value')
 
        self.meanRev.observe(self.process_wrapper, 'value')
        self.meanRev2.observe(self.process_wrapper, 'value')
        self.normalVol.observe(self.process_wrapper, 'value')
        self.normalVol2.observe(self.process_wrapper, 'value')        
        self.numPaths.observe(self.process_wrapper, 'value')
        self.showPaths.observe(self.process_wrapper, 'value')
        self.correlation.observe(self.process_wrapper, 'value')
        self.paths = self.numPaths.value          


        self.form = ipw.VBox([ipw.HBox([self.ZC1W, self.ZC3M, self.ZC6M, self.ZC1Y, self.ZC2Y, self.ZC6Y, self.ZC10Y],layout={'width':'80%'}), self.tab])                

        display(self.form)
        
        
    def process_wrapper(self, click=None):
            
        #curve stuff
        curveZC = self.curveInit()
        zc = curveZC['ZC rates']    
        fwd = curveZC['FWD rates']
            
        #HW1F stuff
        self.line1.y = [zc, fwd]
        self.line1.x = curveZC['ZC rates'].index
        
        MC_r = self.HW1F()
        self.line2.y = MC_r.head(self.showPaths.value)
        self.line2.x = MC_r.T.index

        self.hist1.sample = MC_r.iloc[:,-1]
        self.hist1.bins   = int(self.paths/3)
            
            
        #HW2F stuff
        self.line3.y = [zc, fwd]
        self.line3.y = curveZC['ZC rates'].index
        
        MC_r2 = self.HW2F()
        self.line4.y = MC_r2.head(self.showPaths.value)
        self.line4.x = MC_r2.T.index

        self.hist2.sample = MC_r2.iloc[:,-1]
        self.hist2.bins   = int(self.paths/3)



    def curveInit(self, click=None):
        '''
        Linearly interpolates between zero coupon prices given. Calculates forward rates as (e^rT/e^rt -1)*steps
        '''

        years = np.arange(self.steps, 10 + self.steps, self.steps)

        # creates a list mapping the ZC rates to time to maturity
        ZCs = [[1/52, self.ZC1W.value],
                [.25, self.ZC3M.value],
                [.5, self.ZC6M.value],
                [1, self.ZC1Y.value],
                [2, self.ZC2Y.value],
                [6, self.ZC6Y.value],
                [10, self.ZC10Y.value]]

        curveZC = np.full([len(years)], np.nan)
        curveZC[0], curveZC[-1] = ZCs[0][1], ZCs[-1][1]

        for i in range(1,len(ZCs)-1):
            k = np.where(years == ZCs[i][0])
            curveZC[k] = ZCs[i][1]

        curveZC = pd.DataFrame({'ZC rates': curveZC}).interpolate(method='linear')
        curveZC['dt'] = years

        fwd = np.zeros(len(years))

        curveZC['FWD rates'] = fwd

        curveZC['dt_s'] = curveZC['dt'].shift(1)
        curveZC['ZC_s'] = curveZC['ZC rates'].shift(1)
        curveZC['FWD rates'] = ((np.exp(curveZC['ZC rates']/100 * curveZC['dt']) / np.exp(curveZC['ZC_s']/100 * curveZC['dt_s'])) - 1) / self.steps * 100
        curveZC['FWD rates'][0] = ZCs[0][1]

        return curveZC

        
    def HW1F(self, click=None):
        '''  
        The HW1F model follows below dynamics:
        dr(t) = [theta(t) - a*r(t)]dt + sigma*dW(t) and
        theta(t) = Ft(0,t) + a*F(0,t) + sigma^2/2a*(1-exp(-2at)),
        where theta(t) is the drift term that calibrates the model to the term structure of interest rates,
        F(0,t) is the instantaneous zero forward from time 0 to maturity T,
        Ft(0,t) is the first derivative of F(0,t) with respect to T. This is calculated as the slope at (y2-y1)/(x2-x1).
        Finally, a is the mean reversion speed, sigma is constant vol and dW refers to Geometric Brownian Motion.
        The functions runs a Monte Carlo Simulation for the amount of paths given and plots it. 
        For further info on the model see: http://www.thetaris.com/wiki/Hull-White_model 
        '''

        a     = self.meanRev.value
        sigma = self.normalVol.value/100
        data  = self.curveInit()

        data['t'] = pd.DataFrame(np.arange(0, 10 + self.steps, self.steps)) #t0,t1,t2,t3...
        data['dT'] = data['dt'] - data['t'] #time interval between periods
        data['Ft_0_t'] = (data['FWD rates'].shift(1) - data['FWD rates'])/(data['dt'].shift(1) - data['dt'])
        data['alpha_t'] = np.power(sigma,2)/(2*a)*(1-np.exp(-2*a*data['t']))

        length_T = len(data['dT']) 

        # brownian motion of shape (steps, paths) which we use np.einsum for
        random  = np.random.standard_normal(size=(length_T, self.paths))
        dt_sqrt = np.sqrt(data['dT'])
        dWt     = np.einsum('i,ij->ij', dt_sqrt, random)

        # initialises final outputs from our HW1F model 'r' as well as change 'dr(t)'
        MC_r    = np.zeros((length_T, self.paths))    
        MC_r[0] = self.ZC1W.value    
        MC_drt  = np.zeros((length_T, self.paths))

        #first calculates theta, then the change in r, and finally r(t)
        for i in range(1,length_T):
            q_t = data.loc[i,'Ft_0_t'] + data.loc[i,'FWD rates'] * a + data.loc[i,'alpha_t'] - a * MC_r[i-1]
            MC_drt[i] = q_t * data.loc[i,'dT'] + sigma * dWt[i] 
            MC_r[i]   = MC_r[i-1] + MC_drt[i]

        MC_r = pd.DataFrame(MC_r).T
                
        return MC_r


    def HW2F(self, click=None):
        '''
        The HW2F is an extension of the 1-Factor model and follows these dynamics:
        dr(t) = [theta(t) + u(t) - ar(t)]dt + sigma1(t)*dZ1(t),
        theta(t) = Ft(0,t) + a*F(0,t) + sigma^2/2a*(1-exp(-2at)),
        du = -bud(t)dt + sigma2(t)*dZ2(t),
        with u(0)=0, and dZ1(t)*dZ2(t) = pdt
        a and b refer to the individual mean reversion speeds, p measures the correlation between the two GBM's.
        u refers to the dynamics of a stochastic process called Ornstein-Uhlenbeck.
        For more info see: http://www.thetaris.com/wiki/Two_factor_Hull-White_model
        '''

        a      = self.meanRev.value
        b      = self.meanRev2.value
        sigma1 = self.normalVol.value/100
        sigma2 = self.normalVol2.value/100
        data   = self.curveInit()
        p      = self.correlation.value

        # set up different time scales
        data['t']  = pd.DataFrame(np.arange(0, 10 + self.steps, self.steps))
        data['dT'] = data['dt'] - data['t']
        length_T   = len(data['dT'])

        #deterministic components in theta(t)
        data['Ft_0_t'] = (data['FWD rates'].shift(1) - data['FWD rates'])/(data['dt'].shift(1) - data['dt'])
        data['alpha_t'] = np.power(sigma1,2)/(2*a)*(1-np.exp(-2*a*data['t']))


        # creates the two brownian motions using the correlaation coefficient
        random   = np.random.standard_normal(size=(length_T, self.paths))
        dt_sqrt  = np.sqrt(data['dT'])
        dZ1t     = np.einsum('i,ij->ij', dt_sqrt, random)
        pdt      = np.array(p*data['dT'])[:, np.newaxis]
        dZ2t     = pdt / dZ1t

        # initialises final outputs from our HW2F model 'r' as well as change 'dr(t)'
        MC_r2    = np.zeros((length_T, self.paths))    
        MC_r2[0] = self.ZC1W.value    
        MC_dr2t  = np.zeros((length_T, self.paths))


        #final HW2F dynamics, where first the ornstein-uhlenbeck process du(t) is initialised
        ut       = np.zeros((length_T, self.paths))
        
        for t in range(1,length_T):
            ut[t]      = ut[t-1] + (-b*ut[t-1] * data.loc[t,'dT'] + sigma2 * dZ2t[t])
            theta      = data.loc[t,'Ft_0_t'] + data.loc[t,'FWD rates'] * a + data.loc[t,'alpha_t'] 
            q_t        = theta + ut[t] - a * MC_r2[t-1]
            MC_dr2t[t] = q_t * data.loc[t, 'dT'] + sigma1 * dZ1t[t]
            MC_r2[t]   = MC_r2[t-1] + MC_dr2t[t]

        MC_r2 = pd.DataFrame(MC_r2).T

            
        return MC_r2
    
    


