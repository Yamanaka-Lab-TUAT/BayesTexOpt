import os
import GPyOpt
import numpy as np
from objective import ani_inplane


def obj_func(x):
    ret = []
    for _x in x:
        tex_info = np.empty((2, 6))
        rand_vol = 100. - float(_x[0]) - float(_x[1]) -\
            float(_x[2]) - float(_x[3]) - float(_x[4])
        tex_info[0, :] = np.array([float(_x[0]),
                                   float(_x[1]),
                                   float(_x[2]),
                                   float(_x[3]),
                                   float(_x[4]),
                                   rand_vol])
        tex_info[1, :] = np.array([15.,
                                   15.,
                                   15.,
                                   15.,
                                   15.,
                                   0.])
        result = ani_inplane(tex_info, ave_num=10)
        ret.append(result)
    return np.array(ret)


if __name__ == "__main__":
    save_data_dir = 'Opt_result/'

    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    use_initial_file = True  # if 'initial_data.csv' exists, set use_initial_file = True
    if not use_initial_file:
        input_ini = np.array([[0., 0., 0., 0., 0.],
                              [3.8, 16.7, 8.2, 20.3, 10.5],
                              [1.3, 4.7, 28.5, 16.8, 3.7],
                              [2.06, 39.8, 4.5, 20.2, 20.5],
                              [22.1, 16., 18.5, 1.9, 4.],
                              [50., 0., 0., 0., 0.],
                              [0., 50., 0., 0., 0.],
                              [0., 0., 50., 0., 0.],
                              [0., 0., 0., 50., 0.],
                              [0., 0., 0., 0., 50.]])
        output_ini = obj_func(input_ini).reshape(input_ini.shape[0], 1)
        print(output_ini)
        np.savetxt('initial_data.csv', np.append(input_ini, output_ini, axis=1), delimiter=',')
    else:
        initial_data = np.loadtxt('initial_data.csv', delimiter=',')
        input_ini = initial_data[:, :-1]
        output_ini = initial_data[:, -1].reshape(-1, 1)

    # Bayesian Optimization
    bounds = [{'name': 'Cube_vol', 'type': 'continuous', 'domain': (0., 50.)},
              {'name': 'S_vol', 'type': 'continuous', 'domain': (0., 50.)},
              {'name': 'Goss_vol', 'type': 'continuous', 'domain': (0., 50.)},
              {'name': 'Brass_vol', 'type': 'continuous', 'domain': (0., 50.)},
              {'name': 'Copper_vol', 'type': 'continuous', 'domain': (0., 50.)}]

    # Define constraints of variables
    constraints = [{'name': 'constr_1',
                    'constraint': 'x[:,0] + x[:,1] + x[:,2] + x[:,3] + x[:,4] - 100.'}]

    # Create the feasible region od the problem
    feasible_region = GPyOpt.Design_space(space=bounds, constraints=constraints)

    # --- CHOOSE the objective
    objective = GPyOpt.core.task.SingleObjective(obj_func)

    # --- CHOOSE the Gaussian Process modeling
    GPmodel = GPyOpt.models.GPModel(exact_feval=True,
                                    optimize_restarts=10,
                                    verbose=False,
                                    noise_var=None)

    # --- CHOOSE the acquisition optimizer (default: the BFGS optimizer of the acquisition)
    aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

    # --- CHOOSE the Expected Improvement acquisition function
    acquation_param = 0.01
    acquisition = GPyOpt.acquisitions.AcquisitionEI(GPmodel,
                                                    feasible_region,
                                                    optimizer=aquisition_optimizer,
                                                    jitter=acquation_param)

    # --- CHOOSE a collection method
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

    # Create BO object
    bo = GPyOpt.methods.ModularBayesianOptimization(GPmodel,
                                                    feasible_region,
                                                    objective,
                                                    acquisition,
                                                    evaluator,
                                                    X_init=input_ini,
                                                    Y_init=output_ini)

    # Bayesian optimization
    max_iter = 100
    bo.run_optimization(max_iter=max_iter, verbosity=True,
                        evaluations_file=save_data_dir + 'ev_all.dat',
                        models_file=save_data_dir + 'model.dat')

    print("optimized parameters: {0}".format(bo.x_opt))
    print("optimized loss: {0}".format(bo.fx_opt))
