import numpy as np
import GPyOpt
import plotly.graph_objects as go

VS    = input('Input volume fraction of S component [%] : ')
VS    = float(VS)
VGoss = input('Input volume fraction of Goss component [%] : ')
VGoss = float(VGoss)


def setInputData(data3d):

    name = 'Input'
    color = '#0000ff'

    dim_ex_normalize = VS / 50.
    ind_3d = np.where(np.isclose(X_data[:, 1], dim_ex_normalize, atol=1./50.))
    data3d = data3d[ind_3d]

    dim_ex_normalize = VGoss / 50.
    ind_3d = np.where(np.isclose(data3d[:, 2], dim_ex_normalize, atol=1./50.))
    data3d = data3d[ind_3d]

    x = data3d[:, 0] * 50.
    y = data3d[:, 4] * 50.
    z = data3d[:, 3] * 50.

    trace_input = go.Scatter3d(x=x, y=y, z=z,
                               mode='markers',
                               name=name,
                               marker = dict(color=color, size=5),
                               showlegend=False,
                               hovertext=data3d[:,-1],
                               )

    return trace_input

def setMean(XX, YY, ZZ, m):
    trace_m = go.Volume(x=XX.flatten(),
                        y=YY.flatten(),
                        z=ZZ.flatten(),
                        value=m.flatten(),
                        isomin=-0.3,  # m.min(),
                        isomax=10,  # m.max(),
                        opacity=0.2,
                        surface_count=25,
                        name='Mean',
                        colorscale='RdBu',
                        colorbar=dict(len=0.7, x=0.9, tickvals=([-0.3] + list(range(0, 12, 2))),
                                      tickfont=dict(family='Times New Roman', size=42, color='black')),
                        reversescale=True,
                        )

    return trace_m


def setStDev(XX, YY, ZZ, v):
    trace_v = go.Volume(x=XX.flatten(),
                        y=YY.flatten(),
                        z=ZZ.flatten(),
                        value=v.flatten(),
                        isomin=0,  # v.min(),
                        isomax=1.2,  # v.max(),
                        opacity=0.2,
                        surface_count=25,
                        name='StDev',
                        colorscale='RdBu',
                        colorbar=dict(len=0.7, x=0.9, tickfont=dict(family='Times New Roman', size=42, color='black')),
                        reversescale=True,
                        )

    return trace_v


def setAcqu(XX, YY, ZZ, acqu):
    trace_ac = go.Volume(x=XX.flatten(),
                         y=YY.flatten(),
                         z=ZZ.flatten(),
                         value=acqu.flatten(),
                         isomin=0,  # acqu.min(),
                         isomax=1,  # acqu.max(),
                         opacity=0.2,
                         surface_count=25,
                         name='Acquisition',
                         colorscale='RdBu',
                         colorbar=dict(len=0.7, x=0.9, tickfont=dict(family='Times New Roman', size=42, color='black')),
                         reversescale=True,
                         )

    return trace_ac

if __name__ == "__main__":
    input_data = np.loadtxt('ev_all_3d.dat', skiprows=1)

    X_data = input_data[:, 2:] / 50.
    Y_data = input_data[:, 1].reshape(-1, 1)

    bounds = [{'name': 'Cube',   'type': 'continuous', 'domain': (0, 1)},
              {'name': 'S',      'type': 'continuous', 'domain': (0, 1)},
              {'name': 'Goss',   'type': 'continuous', 'domain': (0, 1)},
              {'name': 'Brass',  'type': 'continuous', 'domain': (0, 1)},
              {'name': 'Copper', 'type': 'continuous', 'domain': (0, 1)}]

    constraint = [{'name': 'constr', 'constraint': 'x[:,0] + x[:,1] + x[:,2] + x[:,3] + x[:,4] - 2.'}]

    np.random.seed(123)
    BOpt = GPyOpt.methods.BayesianOptimization(f=None,
                                               domain=bounds,
                                               acquisition_type='EI',
                                               exact_feval=False,
                                               normalize_Y=True,
                                               X=X_data,
                                               Y=Y_data,
                                               acquisition_jitter=0.01,
                                               constraints=constraint)

    BOpt.suggest_next_locations()
    resol = 20
    xyz = [np.linspace(0, 1, resol),
           np.array([VS]) / 50.,
           np.array([VGoss]) / 50.,
           np.linspace(0, 1, resol),
           np.linspace(0, 1, resol)]

    pp = [np.meshgrid(*tuple(xyz))[i] for i in range(5)]
    xyz_shape = xyz[0].shape[0] * xyz[1].shape[0] * xyz[2].shape[0] * xyz[3].shape[0] * xyz[4].shape[0]
    ppos = [pp[i].reshape(xyz_shape, 1) for i in range(5)]
    pos = np.hstack((ppos))

    model = BOpt.model
    acqu = BOpt.acquisition.acquisition_function(pos)
    acqu_normalized = -(acqu - acqu.max())/(acqu.max() - acqu.min())
    m, v = model.predict(pos)

    std = Y_data.std()
    if std > 0:
        m *=  std
    m += Y_data.mean()
    v *= std

    reshape = [xyz[0].shape[0], xyz[3].shape[0], xyz[4].shape[0]]

    m = m.reshape(reshape).transpose(2, 0, 1)
    v = v.reshape(reshape).transpose(2, 0, 1)
    acqu = acqu_normalized.reshape(reshape).transpose(2, 0, 1)

    xx = np.linspace(0, 50, resol)
    yy = np.linspace(0, 50, resol)
    zz = np.linspace(0, 50, resol)
    XX, YY, ZZ = np.meshgrid(xx,yy,zz)

    trace_input = setInputData(X_data)
    trace_m = setMean(XX, YY, ZZ, m)
    trace_v = setStDev(XX, YY, ZZ, v)
    trace_ac = setAcqu(XX, YY, ZZ, acqu)

    trace_graph = [trace_m, trace_v, trace_ac]
    trace_title = ['Mean function', 'SD function', 'Acquisition function']

    for i in range(3):
        fig = go.Figure(data=[trace_input, trace_graph[i]])
        # Default parameters which are used when `layout.scene.camera` is not provided
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0.1, y=0, z=-0.2),
            eye=dict(x=-1.5, y=-1.5, z=0.9)
        )
        fig.update_layout(scene = dict(xaxis = dict(title=dict(text='Cube', font=dict(family='Times New Roman', size=42, color='black')),
                                                    range = [0,50],
                                                    dtick=10,
                                                    tickfont=dict(family='Times New Roman', size=22, color='black')),
                                       yaxis = dict(title=dict(text='Copper', font=dict(family='Times New Roman', size=42, color='black')),
                                                    range = [0,50],
                                                    dtick=10,
                                                    tickfont=dict(family='Times New Roman', size=22, color='black')),
                                       zaxis = dict(title=dict(text='Brass', font=dict(family='Times New Roman', size=42, color='black')),
                                                    range = [0,50],
                                                    tickmode = 'array',
                                                    tickvals = [0, 10, 20, 30, 40, 50],
                                                    ticktext = ['', '10', '20', '30', '40', '50'],
                                                    tickfont=dict(family='Times New Roman', size=22, color='black'))),
                          scene_aspectmode='cube',
                          margin=dict(r=20, l=10, b=10, t=10),
                          scene_camera=camera,
                          title=dict(text=trace_title[i],
                                     font=dict(family='Times New Roman', size=50, color='black'),
                                     x=0.1, y=0.99,
                                     )
                           )
        fig.show()