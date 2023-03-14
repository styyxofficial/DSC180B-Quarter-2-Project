import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def create_raster(sl, spikes, channels):
    return

def plot_trajectories2L(z, choices, accuracy, bin_size):
    first = True
    first2= True
    fig = go.Figure()
    
    for i in range(len(z)):
        if ((choices[i]==1) & (accuracy[i]==1)):
            if first:
                fig.add_trace(go.Scatter3d(x=np.arange(-100, 1000, bin_size), y=z[i][:, 0], z=z[i][:, 1],
                        mode='lines', line={'color':'blue', 'width':1}, legendgroup='right', name='Wheel Turned Right', showlegend=True))
                first = False
            else:
                fig.add_trace(go.Scatter3d(x=np.arange(-100, 1000, bin_size), y=z[i][:, 0], z=z[i][:, 1],
                            mode='lines', line={'color':'blue', 'width':1}, legendgroup='right', showlegend=False))
            
        elif ((choices[i]==-1) & (accuracy[i]==1)):
            if first2:
                fig.add_trace(go.Scatter3d(x=np.arange(-100, 1000, bin_size), y=z[i][:, 0], z=z[i][:, 1],
                        mode='lines', line={'color':'red', 'width':1}, legendgroup='left', name='Wheel Turned Left', showlegend=True))
                first2 = False
            else:
                fig.add_trace(go.Scatter3d(x=np.arange(-100, 1000, bin_size), y=z[i][:, 0], z=z[i][:, 1],
                            mode='lines', line={'color':'red', 'width':1}, legendgroup='left', showlegend=False))
        
        """
        Code below is if you wish to plot the trajectories when the mouse was incorrect
        """
        # elif ((training_trials[i]['choice']==1) & (training_trials[i]['feedbackType']==-1)):
        #     if first:
        #         fig = px.line_3d(x=np.arange(-0.1, 1, 0.05), y=z_test[i][:, 0], z=z_test[i][:, 1], color=["green"]*time_bins, width=800, height=800)
        #         first = False
        #     else:
        #         fig.add_scatter3d(x=np.arange(-0.1, 1, 0.05), y=z_test[i][:, 0], z=z_test[i][:, 1], mode='lines', line={'color':"green"}, opacity=0.5)
            
        # elif ((training_trials[i]['choice']==-1) & (training_trials[i]['feedbackType']==-1)):
        #     if first:
        #         fig = px.line_3d(x=np.arange(-0.1, 1, 0.05), y=z_test[i][:, 0], z=z_test[i][:, 1], color=["black"]*time_bins, width=800, height=800)
        #         first = False
        #     else:
        #         fig.add_scatter3d(x=np.arange(-0.1, 1, 0.05), y=z_test[i][:, 0], z=z_test[i][:, 1], mode='lines', line={'color':"black"}, opacity=1)
    fig.update_layout(scene = dict(
                    xaxis_title='Time (ms)',
                    yaxis_title='Latent Variable 1',
                    zaxis_title='Latent Variable 2'),
                    width=1000, height=1000, title='Latent Variables over Time'
                    )
    
    fig.show()