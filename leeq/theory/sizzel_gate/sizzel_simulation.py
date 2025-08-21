import datetime
import multiprocessing
import pickle

import matplotlib.colors as colors
import numpy as np
import qutip as qp
from matplotlib import cm
from matplotlib import pyplot as plt
from tqdm.contrib.concurrent import process_map

N = 4

alpha_1 = 182.91980658920147
alpha_2 = 184.13358901547146
J = 5.758916466558136

freq_q1 = 4258.789014451565
freq_q2 = 4008.852189211678  # freq_q1 - 70  # 9241

global freq_delta
freq_delta = freq_q2 - freq_q1  # -70

points_x = 2001
points_y = 401


a = qp.tensor(qp.destroy(N), qp.identity(N))
ad = a.dag()
b = qp.tensor(qp.identity(N), qp.destroy(N))
bd = b.dag()


def ret_Hnonlin(alpha1, alpha2, J):
    # Hnonlin = -chi*(ad + a)**2 * (bd + b)**2
    Hnonlin = -alpha1 / 2 * bd ** 2 * b ** 2 - alpha2 / 2 * ad ** 2 * a ** 2 - J * (ad * b + bd * a)
    return Hnonlin


def ret_Hdrive(eps):
    return 0.5 * eps * (ad + a) + 0.5 * eps * (bd + b)


def ret_Hdetune(drive_delta, freq_delta):
    return -drive_delta * ad * a + (-drive_delta - freq_delta) * bd * b


def find_shifts(H):
    N = H.dims[0][0]

    psi00 = qp.tensor(qp.fock(N, 0), qp.fock(N, 0))
    psi10 = qp.tensor(qp.fock(N, 1), qp.fock(N, 0))
    psi01 = qp.tensor(qp.fock(N, 0), qp.fock(N, 1))
    psi11 = qp.tensor(qp.fock(N, 1), qp.fock(N, 1))
    psi20 = qp.tensor(qp.fock(N, 2), qp.fock(N, 0))
    psi02 = qp.tensor(qp.fock(N, 0), qp.fock(N, 2))
    psi21 = qp.tensor(qp.fock(N, 2), qp.fock(N, 1))
    psi12 = qp.tensor(qp.fock(N, 1), qp.fock(N, 2))
    psi22 = qp.tensor(qp.fock(N, 2), qp.fock(N, 2))

    Elevels, Estates = H.eigenstates()
    E00 = Elevels[np.argmax([np.abs(x.overlap(psi00)) for x in Estates])]
    E10 = Elevels[np.argmax([np.abs(x.overlap(psi10)) for x in Estates])]
    E01 = Elevels[np.argmax([np.abs(x.overlap(psi01)) for x in Estates])]
    E11 = Elevels[np.argmax([np.abs(x.overlap(psi11)) for x in Estates])]
    E20 = Elevels[np.argmax([np.abs(x.overlap(psi20)) for x in Estates])]
    E02 = Elevels[np.argmax([np.abs(x.overlap(psi02)) for x in Estates])]
    E21 = Elevels[np.argmax([np.abs(x.overlap(psi21)) for x in Estates])]
    E12 = Elevels[np.argmax([np.abs(x.overlap(psi12)) for x in Estates])]
    E22 = Elevels[np.argmax([np.abs(x.overlap(psi22)) for x in Estates])]

    delta_11 = E11 + E00 - E10 - E01
    delta_12 = E12 + E00 - E10 - E02

    delta_21 = E21 + E00 - E20 - E01
    delta_22 = E22 + E00 - E20 - E02

    return delta_11, delta_12, delta_21, delta_22


def ret_zz(alpha1, alpha2, J, eps, delta, freq_delta):
    H = ret_Hnonlin(2 * np.pi * alpha1, 2 * np.pi * alpha2, 2 * np.pi * J) + ret_Hdrive(2 * np.pi * eps) + ret_Hdetune(
        2 * np.pi * delta,
        2 * np.pi * freq_delta)
    return np.asarray(find_shifts(H)) / (2 * np.pi)


def solve_zz(x):
    eps, delta_MHz = x

    return ret_zz(alpha_1, alpha_2, J, eps, delta_MHz, freq_delta)


k_search_range = 1


def solve_t(delta_values):
    m = np.asarray([
        [delta_values[0], delta_values[1], delta_values[3], delta_values[2]],
        [delta_values[1], delta_values[0], delta_values[2], delta_values[3]],
        [delta_values[2], delta_values[3], delta_values[1], delta_values[0]],
        [delta_values[3], delta_values[2], delta_values[0], delta_values[1]],
    ]) * np.pi * 2

    k_value_optimal = None
    t_result_optimal = -1e4
    try:
        m_inv = np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return np.sum(t_result_optimal), k_value_optimal, t_result_optimal

    phases = np.asarray([
        1, 2, 2, 1
    ]) * np.pi * 2 / 3

    k_value_optimal = None
    t_result_optimal = -1e6

    for i in range(-k_search_range, k_search_range + 1):
        for j in range(-k_search_range, k_search_range + 1):
            for k in range(-k_search_range, k_search_range + 1):
                for l in range(-k_search_range, k_search_range + 1):
                    k_values = np.asarray([i, j, k, l])
                    t_result = m_inv @ phases + np.pi * 2 * k_values
                    if (t_result >= 0).all():
                        if k_value_optimal is None or np.sum(t_result) < np.sum(t_result_optimal):
                            k_value_optimal = k_values
                            t_result_optimal = t_result

    return np.sum(t_result_optimal), k_value_optimal, t_result_optimal


def main(dump_name):
    # pool = multiprocessing.Pool(6)
    # print(ret_zz(-200, 3, 60, -40, -70))

    Nx = points_x
    Ny = points_y

    eps_list = np.linspace(0, 20, Nx)
    delta_list = np.linspace(-500, 150, Ny)

    zz_list = process_map(solve_zz, [(eps, delta_MHz) for eps in eps_list for delta_MHz in delta_list], chunksize=10)

    zz_list = np.array(zz_list)

    zz_list = zz_list.reshape([Nx, Ny, 4])

    data = {
        'alpha_1': alpha_1,
        'alpha_2': alpha_2,
        'J': J,
        'freq_delta': freq_delta,
        'N': N,
        'zz_list': zz_list,
        'delta_list': delta_list,
        'eps_list': eps_list,
        'time': datetime.datetime.now()
    }

    with open(dump_name, 'wb') as f:
        pickle.dump(data, f)


def solve_all_t_values(file, plot=False):
    assert isinstance(file, str)

    with open(file, 'rb') as f:
        data = pickle.load(f)

    zz_list = data['zz_list']
    delta_list = data['delta_list']
    eps_list = data['eps_list']
    simulation_time = data['time']

    def solve_t_wrapper(x):
        i, j = x
        delta_values = zz_list[i, j, :]
        return solve_t(delta_values)

    X, Y = np.meshgrid(delta_list, eps_list)

    t_map = process_map(solve_t, [zz_list[i, j, :] for i in range(len(eps_list)) for j in range(len(delta_list))],
                        chunksize=1000)

    data['t_map'] = t_map

    with open(file, 'wb') as f:
        pickle.dump(data, f)

    if plot:
        t_map_plot = np.asarray([x[0] for x in t_map]).reshape(zz_list.shape[:2])

        plt.figure()
        plt.title(r"Time required to implement qutrit CZ")
        plt.pcolor(X, Y, t_map_plot, cmap="RdBu", norm=colors.SymLogNorm(linthresh=0.0003, vmin=-1e6, vmax=1e6))
        plt.colorbar()
        plt.xlabel(r"$Drive~detuning (MHz)$")
        plt.ylabel('Drive~strength (MHz)')
        plt.show()


def print_specific_configuration(drive_strength, detuning):
    zz = solve_zz((drive_strength, detuning))
    ts = solve_t(zz)

    print(zz)
    print(ts)


def plot_data(data):
    if isinstance(data, str):
        with open(data, 'rb') as f:
            data = pickle.load(f)

    zz_list = data['zz_list']
    delta_list = data['delta_list']
    eps_list = data['eps_list']
    simulation_time = data['time']

    X, Y = np.meshgrid(delta_list, eps_list)

    names = [
        r'$\delta_{11}$',
        r'$\delta_{12}$',
        r'$\delta_{21}$',
        r'$\delta_{22}$'
    ]

    print(data.keys())

    print(data['alpha'])

    if alpha_1 not in data:
        data['alpha_1'] = data['alpha']
        data['alpha_2'] = data['alpha']

        names = [
            x + rf'$\alpha={data["alpha"]}$[MHz] $J={data["J"]}$[MHz] $\Delta={data["freq_delta"]}$[MHz] $N={data["N"]}$'
            for x in names]

    else:

        names = [
            x + rf'$\alpha={data["alpha_1"]},{data["alpha_2"]}$[MHz] $J={data["J"]}$[MHz] $\Delta={data["freq_delta"]}$[MHz] $N={data["N"]}$'
            for x in names]

    for i, name in enumerate(names):
        plt.figure()
        plt.title(name)
        # norm = colors.SymLogNorm(linthresh=0.0003, vmin=0, vmax=10)
        # plt.pcolor(X, Y, -zz_list[:, :, i], cmap="RdBu",norm = colors.SymLogNorm(linthresh=0.0003))
        plt.pcolor(X, Y, -zz_list[:, :, i], cmap="RdBu", norm=colors.SymLogNorm(linthresh=0.0003, vmin=-20, vmax=20))
        plt.colorbar()
        plt.xlabel(r"$Drive~detuning (MHz)$")
        plt.ylabel('Drive~strength (MHz)')

    print(simulation_time)
    plt.show()


def plot_time(file):
    if isinstance(file, str):
        with open(file, 'rb') as f:
            data = pickle.load(f)

    zz_list = data['zz_list']
    delta_list = data['delta_list']
    eps_list = data['eps_list']
    simulation_time = data['time']
    t_map = data['t_map']

    zz = solve_zz((0, 0))
    ts = solve_t(zz)
    t0 = ts[0]

    X, Y = np.meshgrid(delta_list, eps_list)

    t_map_plot = np.asarray([x[0] for x in t_map]).reshape(zz_list.shape[:2])

    t_map_plot_diff = t_map_plot - t0

    t_map_plot[t_map_plot_diff > -0.5 * t0] = -1e4

    if alpha_1 not in data:
        data['alpha_1'] = data['alpha']
        data['alpha_2'] = data['alpha']

    fig, ax = plt.subplots()
    # ax.set_title(
    #    r"Time required to implement qutrit CZ" + rf'$\alpha={data["alpha_1"]},{data["alpha_2"]}$[MHz] $J={data["J"]}$[MHz] $\Delta={data["freq_delta"]}$[MHz] $N={data["N"]}$')
    norm = colors.SymLogNorm(linthresh=0.0003, vmin=-100, vmax=100)

    norm = colors.LogNorm(vmin=1, vmax=np.max(t_map_plot))
    # norm = colors.PowerNorm(0.5)
    cmap = 'viridis_r'
    # cmap = 'RdBu'
    #
    ax.pcolor(X, Y, t_map_plot, norm=norm, cmap=cmap)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

    # cs = ax.contour(X, Y, np.abs(t_map_plot), levels=[1], colors=['Orange'])
    # ax.clabel(cs, inline=True, fontsize=8)
    ax.patch.set_facecolor('black')

    ax.set_xlabel(r"$Drive~detuning (MHz)$")
    ax.set_ylabel('Drive~strength (MHz)')
    plt.show()


def print_exact_point(filename, x, y):
    if isinstance(filename, str):
        with open(filename, 'rb') as f:
            data = pickle.load(f)

    zz_list = data['zz_list']
    delta_list = data['delta_list']
    eps_list = data['eps_list']
    simulation_time = data['time']
    t_map = data['t_map']

    X, Y = np.meshgrid(delta_list, eps_list)

    x_index = np.argmin(np.abs(X - x))
    y_index = np.argmin(np.abs(Y - y))

    t_map_print = np.asarray([x[0] for x in t_map]).reshape(zz_list.shape[:2])

    # print(f"[{x_index}:{y_index}]zz:{zz_list[:, x_index, y_index]},t:{t_map_print]}")


if __name__ == '__main__':
    name_70 = 'detuning_70MHz.pickle'
    name_100 = 'detuning_100MHz.pickle'
    name_300 = 'detuning_300MHz.pickle'
    name_500 = 'detuning_500MHz.pickle'

    name_real = 'Giulio_A2_B2.pickle'
    name_A2_B2_corrected = 'Giulio_A2_B2_corrected_detailed.pickle'
    name_A2_B2 = 'Giulio_A2_B2_corrected.pickle'
    # main(name_real)
    # solve_all_t_values(name_real)

    # freq_delta = -70

    # freq_delta = -100
    # main(name_100)

    # freq_delta = -300
    # main(name_300)

    # freq_delta = -500
    # main(name_500)
    # solve_all_t_values(name_100)
    # solve_all_t_values(name_300)
    # solve_all_t_values(name_500)
    # solve_all_t_values(name_70)
    # print_specific_configuration(drive_strength=20, detuning=-350)

    # main(name_A2_B2_corrected)

    # solve_all_t_values(name_A2_B2_corrected)
    # plot_data(name_70)
    plot_time(name_70)
    # print(ret_zz(alpha, J, 10, -50, freq_delta))
