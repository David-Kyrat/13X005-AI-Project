import plot_util as pu
from naive_bayes import get_distrib_parameters, normal_pdf
import numpy as np


def plot_compairaison():
    """
    Plot for each class, the probability distribution associated with each feature X_k.
    """
    from main import FEAT, LABELS
    parameters = get_distrib_parameters(FEAT, LABELS)
    for y in parameters.keys():
        mu_0, std_0 = parameters[y][0]
        mu_1, std_1 = parameters[y][1]
        mu_2, std_2 = parameters[y][2]
        mu_3, std_3 = parameters[y][3]
        x_plot_0 = np.linspace(mu_0-10, mu_0+10, 1000)
        x_plot_1 = np.linspace(mu_0-10, mu_1+10, 1000)
        x_plot_2 = np.linspace(mu_2-10, mu_2+10, 1000)
        x_plot_3 = np.linspace(mu_3-10, mu_3+10, 1000)
        y_plot_0 = normal_pdf(mu_0,std_0)(x_plot_0)
        y_plot_1 = normal_pdf(mu_1,std_1)(x_plot_1)
        y_plot_2 = normal_pdf(mu_2,std_2)(x_plot_2)
        y_plot_3 = normal_pdf(mu_3,std_3)(x_plot_3)
        pu.plot_vs(plot_x=x_plot_0, plot_x2=x_plot_1, plot_f1=y_plot_0, plot_f2=y_plot_1, plot_x3=x_plot_2, plot_x4=x_plot_3, 
                   plot_f3=y_plot_2, plot_f4=y_plot_3, f1label='X_0',f2label='X_1',f3label='X_2',f4label='X_3',
                   title=f"Courbe des distribution de probabilité sachant Y={y}",xlabel="X_k=x",ylabel=f"p(X_k=x|y={y})",  filename=f"src/res/comp_normal_law_Y_{y}")



