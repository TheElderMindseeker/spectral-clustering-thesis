import csv
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def per_plot():
    plot_dist = 9
    func_dist = 1.35
    colormap = mpl.colormaps['Set2']

    data = defaultdict(dict)
    funcs, plots = set(), set()
    with open('results/performance.csv') as data_file:
        reader = csv.reader(data_file)
        next(reader)  # Skip header.
        for func, plot, *_, r_extr, r_match in reader:
            plot = int(plot)
            funcs.add(func)
            plots.add(plot)
            data[func][plot] = (float(r_extr), float(r_match))
    funcs = sorted(funcs)
    plots = sorted(plots)

    plt.figure(figsize=(9, 7), dpi=300)

    x = []
    height = []
    for func_idx, func in enumerate(funcs):
        for plot_idx, plot in enumerate(plots):
            x.append(plot_dist * plot_idx + func_dist * func_idx)
            height.append(100 * data[func][plot][0])
    plt.bar(x, height, width=1, color=colormap(7), label='Extraction Rate')

    for func_idx, func in enumerate(funcs):
        x = []
        height = []
        colors = []
        for plot_idx, plot in enumerate(plots):
            x.append(plot_dist * plot_idx + func_dist * func_idx)
            height.append(100 * data[func][plot][1])
            colors.append(colormap(func_idx))
        plt.bar(x, height, width=1, color=colors, label=func.capitalize())

    plt.hlines(
        100,
        -1,
        14 * plot_dist,
        linestyles='dashed',
        label='Perfect Rate',
        color=colormap(6),
    )
    values = plot_dist * np.arange(len(plots)) + (len(funcs) - 1) * func_dist / 2
    plt.xticks(values, plots)
    plt.xlim((-2, 14 * plot_dist))
    plt.xlabel('Plots by index in dataset')
    plt.ylabel('Extraction and matching rates [%]')
    plt.ylim((0, 140))
    plt.legend(fontsize='11')
    plt.show()


def per_forest_type():
    forest_dist = 6
    func_dist = 1.35
    colormap = mpl.colormaps['Set2']

    data = defaultdict(lambda: defaultdict(list))
    funcs, forests = set(), set()
    with open('results/performance.csv') as data_file:
        reader = csv.reader(data_file)
        next(reader)  # Skip header.
        for func, _, n_trees, forest, _, r_extr, r_match in reader:
            funcs.add(func)
            forests.add(forest)
            data[func][forest].append((float(r_extr), float(r_match), int(n_trees)))
    funcs = sorted(funcs)
    forests = sorted(forests)

    for func in funcs:
        for forest in forests:
            total_extr, total_match, total_trees = 0, 0, 0
            for r_extr, r_match, n_trees in data[func][forest]:
                total_extr += r_extr * n_trees
                total_match += r_match * n_trees
                total_trees += n_trees
            data[func][forest] = (total_extr / total_trees, total_match / total_trees)

    plt.figure(figsize=(9, 6), dpi=300)

    x = []
    height = []
    for func_idx, func in enumerate(funcs):
        for forest_idx, forest in enumerate(forests):
            x.append(forest_dist * forest_idx + func_dist * func_idx)
            height.append(100 * data[func][forest][0])
    plt.bar(x, height, width=1, color=colormap(7), label='Extraction Rate')

    for func_idx, func in enumerate(funcs):
        x = []
        height = []
        colors = []
        for forest_idx, forest in enumerate(forests):
            x.append(forest_dist * forest_idx + func_dist * func_idx)
            height.append(100 * data[func][forest][1])
            colors.append(colormap(func_idx))
        plt.bar(x, height, width=1, color=colors, label=func)

    plt.hlines(
        100,
        -1,
        len(forests) * forest_dist,
        linestyles='dashed',
        label='Perfect Rate',
        color=colormap(6),
    )
    values = forest_dist * np.arange(len(forests)) + (len(funcs) - 1) * func_dist / 2
    plt.xticks(values, forests)
    plt.xlim((-1, len(forests) * forest_dist - 1))
    plt.xlabel('Forest Type')
    plt.ylabel('Extraction and matching rates [%]')
    plt.ylim((0, 120))
    plt.legend(fontsize='11')
    plt.show()


def per_density_class():
    forest_dist = 7.5
    func_dist = 1.35
    colormap = mpl.colormaps['Set2']
    thr1, thr2 = 300, 1000
    density_thresholds = [thr1, thr2]
    density_classes = (
        f'Low [0-{thr1}]',
        f'Medium [{thr1}-{thr2}]',
        f'High [{thr2}+]',
    )

    data = defaultdict(lambda: defaultdict(list))
    funcs = set()
    with open('results/performance.csv') as data_file:
        reader = csv.reader(data_file)
        next(reader)  # Skip header.
        for func, _, n_trees, _, density, r_extr, r_match in reader:
            density = float(density)
            funcs.add(func)
            d = 0
            while d < len(density_thresholds) and density > density_thresholds[d]:
                d += 1
            density_class = density_classes[d]
            data[func][density_class].append(
                (float(r_extr), float(r_match), int(n_trees))
            )
    funcs = sorted(funcs)

    for func in funcs:
        for density_class in density_classes:
            total_extr, total_match, total_trees = 0, 0, 0
            for r_extr, r_match, n_trees in data[func][density_class]:
                total_extr += r_extr * n_trees
                total_match += r_match * n_trees
                total_trees += n_trees
            if total_trees == 0:
                continue
            data[func][density_class] = (
                total_extr / total_trees,
                total_match / total_trees,
            )

    plt.figure(figsize=(9, 6), dpi=300)

    x = []
    height = []
    for func_idx, func in enumerate(funcs):
        for forest_idx, density_class in enumerate(density_classes):
            if len(data[func][density_class]) == 0:
                continue
            x.append(forest_dist * forest_idx + func_dist * func_idx)
            height.append(100 * data[func][density_class][0])
    plt.bar(x, height, width=1, color=colormap(7), label='Extraction Rate')

    for func_idx, func in enumerate(funcs):
        x = []
        height = []
        colors = []
        for forest_idx, density_class in enumerate(density_classes):
            if len(data[func][density_class]) == 0:
                continue
            x.append(forest_dist * forest_idx + func_dist * func_idx)
            height.append(100 * data[func][density_class][1])
            colors.append(colormap(func_idx))
        plt.bar(x, height, width=1, color=colors, label=func)

    plt.hlines(
        100,
        -1,
        len(density_classes) * forest_dist,
        linestyles='dashed',
        label='Perfect Rate',
        color=colormap(6),
    )
    values = forest_dist * np.arange(len(density_classes)) + (len(funcs) - 1) * func_dist / 2
    plt.xticks(values, density_classes)
    plt.xlim((-1, len(density_classes) * forest_dist - 1))
    plt.xlabel('Stem Density Class')
    plt.ylabel('Extraction and matching rate [%]')
    plt.ylim((0, 120))
    plt.legend(fontsize='11')
    plt.show()


def height_accuracy(similarity_func):
    colormap = mpl.colormaps['Set2']

    data = []
    with open('results/match.csv') as data_file:
        reader = csv.reader(data_file)
        next(reader)  # Skip header.
        for sim_func, r_height, d_height, *_ in reader:
            if sim_func == similarity_func:
                data.append((float(r_height), float(d_height)))
    data = np.array(data)

    X = np.reshape(data[:, 0], (data.shape[0], 1))
    y_true = data[:, 1]
    regression = LinearRegression().fit(X, y_true)
    reg_x = np.array((5, 41))
    reg_y = regression.predict(reg_x[:, np.newaxis])
    y_pred = regression.predict(X)
    r2_score = regression.score(X, y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (np.max(y_true) - np.min(y_true))

    plt.figure(figsize=(4, 4), dpi=300)

    x = np.array((5, 41))
    y = np.array((5, 41))
    plt.title(f'Height Estimation Accuracy ({similarity_func.capitalize()})')
    plt.text(5, 39, f'R2 Score: {r2_score:.4f}')
    plt.text(5, 36, f'NRMSD: {100 * nrmse:.2f}%')
    plt.plot(x, y, '--', color=colormap(3), label='Ideal Accuracy')
    plt.plot(reg_x, reg_y, color=colormap(1), label='Linear Regression')
    plt.scatter(data[:, 0], data[:, 1], s=4, color=colormap(2), label='Matched Tree')
    plt.legend(loc='lower right', fontsize='11')
    plt.xlim((3, 43))
    plt.xlabel('Reference tree height [m]')
    plt.ylim((3, 43))
    plt.ylabel('Estimated tree height [m]')
    plt.show()


def matching_per_height():
    colormap = mpl.colormaps['Set2']
    levels = ('[0-15]', '[15-20]', '[20-23]', '[23-25]', '[25-30]', '[30+]')
    thresholds = (15, 20, 23, 25, 30)

    unmatched = defaultdict(lambda: {l: 0 for l in levels})
    matched = defaultdict(lambda: {l: 0 for l in levels})
    funcs = set()

    with open('results/nomatch.csv') as data_file:
        reader = csv.reader(data_file)
        next(reader)  # Skip header.
        for sim_func, height in reader:
            funcs.add(sim_func)
            height = float(height)
            level = 0
            while level < len(thresholds) and height > thresholds[level]:
                level += 1
            unmatched[sim_func][levels[level]] += 1
    funcs.remove('quadratic')
    funcs.remove('arctangent')
    funcs = sorted(funcs)

    with open('results/match.csv') as data_file:
        reader = csv.reader(data_file)
        next(reader)  # Skip header.
        for sim_func, height, *_ in reader:
            height = float(height)
            level = 0
            while level < len(thresholds) and height > thresholds[level]:
                level += 1
            matched[sim_func][levels[level]] += 1

    plt.figure(figsize=(9, 6), dpi=300)

    places = (10, 20, 30, 40, 50, 60)
    for func_idx, func in enumerate(funcs):
        m_rates = [
            100 * matched[func][level] / (matched[func][level] + unmatched[func][level])
            for level in levels
        ]
        if func == 'adaptive':
            z_order = 10
            line_width = 3.5
        else:
            z_order = 1
            line_width = 2
        plt.plot(
            places,
            m_rates,
            zorder=z_order,
            linewidth=line_width,
            color=colormap(func_idx),
            label=func.capitalize(),
        )

    plt.xlabel('Tree Height Layer [m]')
    plt.xticks(places, levels)
    plt.xlim((5, 65))
    plt.ylabel('Matching rate [%]')
    plt.ylim((0, 120))
    plt.legend(loc='upper left', fontsize='15')
    plt.grid()
    plt.show()


def total_match_rate():
    colormap = mpl.colormaps['Set2']
    bar_dist = 2

    data = defaultdict(list)
    funcs = set()
    with open('results/performance.csv') as data_file:
        reader = csv.reader(data_file)
        next(reader)  # Skip header.
        for func, _, n_trees, _, _, r_extr, r_match in reader:
            funcs.add(func)
            data[func].append(
                (float(r_extr), float(r_match), int(n_trees))
            )

    for func in funcs:
        total_extr, total_match, total_trees = 0, 0, 0
        for r_extr, r_match, n_trees in data[func]:
            total_extr += r_extr * n_trees
            total_match += r_match * n_trees
            total_trees += n_trees
        if total_trees == 0:
            continue
        data[func] = (
            100 * total_extr / total_trees,
            100 * total_match / total_trees,
        )

    funcs = sorted(funcs)

    plt.figure(figsize=(8, 6), dpi=300)

    x_positions = [bar_dist * idx for idx in range(len(funcs))]
    extr_height = [data[func][0] for func in funcs]
    match_height = [data[func][1] for func in funcs]
    plt.bar(
        x_positions,
        extr_height,
        width=1,
        color=colormap(7),
        label='Extraction rate',
        zorder=1,
    )
    plt.bar(
        x_positions,
        match_height,
        width=1,
        color=colormap(4),
        label='Matching rate',
        zorder=10,
    )
    plt.bar(
        x_positions[2],
        match_height[2],
        width=1,
        color=colormap(1),
        zorder=20,
    )

    plt.hlines(
        100,
        -2,
        bar_dist * len(funcs),
        linestyles='dashed',
        color=colormap(6),
    )
    plt.xlim((-bar_dist/2, bar_dist * len(funcs) - bar_dist/2))
    plt.xticks(
        x_positions,
        [func.capitalize() for func in funcs],
    )
    plt.ylabel('Extraction and matching rates [%]')
    plt.ylim((0, 160))
    plt.legend(fontsize="15")
    plt.show()
