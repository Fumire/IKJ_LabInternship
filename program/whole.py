import multiprocessing
import os
import time
import csv
import gzip
import scipy.io
import numpy
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy
import pprint
import sklearn.cluster


def get_matrix(filename):
    return pandas.DataFrame(scipy.io.mmread(filename).toarray())


def get_feature_ids(filename):
    return [row[0] for row in csv.reader(gzip.open(filename, mode="rt"), delimiter="\t")]


def get_gene_name(filename):
    return [row[1] for row in csv.reader(gzip.open(filename, mode="rt"), delimiter="\t")]


def get_feature_type(filename):
    return [row[2] for row in csv.reader(gzip.open(filename, mode="rt"), delimiter="\t")]


def get_barcodes(filename):
    return [row[0] for row in csv.reader(gzip.open(filename, mode="rt"), delimiter="\t")]


data = dict()


def get_all(ID):
    if ID in data:
        return data[ID]
    dirname = "/BiO/Live/jwlee230/181113_spermatogenesis/result/" + ID + "/outs/filtered_feature_bc_matrix"
    matrix_dir = os.path.join(dirname, "matrix.mtx.gz")
    features_path = os.path.join(dirname, "features.tsv.gz")
    barcodes_path = os.path.join(dirname, "barcodes.tsv.gz")

    data[ID] = {"matrix": get_matrix(matrix_dir), "feature_ids": get_feature_ids(features_path), "gene_name": get_gene_name(features_path), "feature_type": get_feature_type(features_path), "barcodes": get_barcodes(barcodes_path)}

    return data[ID]


now = time.strftime("%m%d%H%M%S")
figure_directory = "/BiO/Live/jwlee230/181113_spermatogenesis/figures/"
IDs = ["NS_SW1", "NS_SW2", "NS_SW3", "NS_SW4"]


def get_whole_data():
    whole_projection = pandas.read_csv("/BiO/Live/jwlee230/181113_spermatogenesis/result/aggr/outs/analysis/tsne/2_components/projection.csv", header=0)

    whole_projection["std_TSNE-1"] = scipy.stats.zscore(whole_projection["TSNE-1"])
    whole_projection["std_TSNE-2"] = scipy.stats.zscore(whole_projection["TSNE-2"])

    return whole_projection


def draw_all():
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    projection = get_whole_data()

    plt.figure()
    plt.scatter(projection["std_TSNE-1"], projection["std_TSNE-2"], alpha=0.6)

    plt.grid(True)
    plt.title("Total")
    plt.xlabel("Standardized TSNE-1")
    plt.ylabel("Standardized TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "total" + "_" + now + ".png")
    plt.close()


def get_real_barcodes(ID):
    projection = pandas.read_csv("/BiO/Live/jwlee230/181113_spermatogenesis/result/" + ID + "/outs/analysis/tsne/2_components/projection.csv", header=0)

    return [barcode[:-1] + ID[-1] for barcode in projection["Barcode"]]


def get_data_from_id(ID):
    projection = get_whole_data()
    return projection[numpy.isin(projection["Barcode"], get_real_barcodes(ID))]


def draw_all_with_color():
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    plt.figure()
    for ID in IDs:
        projection = get_data_from_id(ID)
        plt.scatter(projection["std_TSNE-1"], projection["std_TSNE-2"], alpha=0.6, label=ID)

    plt.grid(True)
    plt.title("Total")
    plt.xlabel("Standardized TSNE-1")
    plt.ylabel("Standardized TSNE-2")
    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "total_" + now + ".png")
    plt.close()


def draw_tSNE(ID):
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    whole_projection = get_whole_data()

    wanted = whole_projection[numpy.isin(whole_projection["Barcode"], get_real_barcodes(ID))]
    unwanted = whole_projection[numpy.invert(numpy.isin(whole_projection["Barcode"], get_real_barcodes(ID)))]

    plt.figure()
    plt.scatter(unwanted["std_TSNE-1"], unwanted["std_TSNE-2"], c="tab:gray", alpha=0.6)
    plt.scatter(wanted["std_TSNE-1"], wanted["std_TSNE-2"], c="tab:blue", alpha=1)

    plt.grid(True)
    plt.title(ID)
    plt.xlabel("Standardized TSNE-1")
    plt.ylabel("Standardized TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + ID + "_" + now + ".png")
    plt.close()


def make_cluster_dict(cells):
    given = dict()
    for i in range(max(cells) + 1):
        given[i] = list(filter(lambda x: cells[x] == i, list(range(len(cells)))))
    return given


def clustering_Kmeans_with_num(ID, num_groups):
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    projection = get_data_from_id(ID)

    kmeans = sklearn.cluster.KMeans(n_clusters=num_groups, random_state=0, n_jobs=-1).fit(numpy.array([_ for _ in zip(projection["std_TSNE-1"], projection["std_TSNE-2"])]))

    projection["group"] = kmeans.fit_predict([_ for _ in zip(projection["std_TSNE-1"], projection["std_TSNE-2"])])

    plt.figure()
    plt.scatter(projection["std_TSNE-1"], projection["std_TSNE-2"], c=projection["group"])
    plt.scatter([elem[0] for elem in kmeans.cluster_centers_], [elem[1] for elem in kmeans.cluster_centers_], c="k", marker="X", s=500)
    for i, loc in enumerate(kmeans.cluster_centers_):
        plt.text(loc[0] + 0.05, loc[1], str(i), fontsize=30, bbox=dict(color='white', alpha=0.8))

    plt.grid(True)
    plt.title("KMeans: " + str(num_groups))
    plt.xlabel("Standardized TSNE-1")
    plt.ylabel("Standardized TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "KMeans_" + ID + "_" + str(num_groups) + "_" + now + ".png")
    plt.close()

    return (make_cluster_dict(projection["group"]), kmeans.cluster_centers_)


def clustering_Kmeans(ID, num=10):
    return [clustering_Kmeans_with_num(ID, i) for i in range(2, num + 1)]


def gene_in_cells(ID, cell_numbers=None):
    all_data = get_all(ID)
    all_data["matrix"].index = all_data["gene_name"]

    if cell_numbers is None:
        return all_data["matrix"]

    data = all_data["matrix"].copy()

    data.drop(all_data["matrix"].columns[list(filter(lambda x: x not in cell_numbers, list(range(all_data["matrix"].shape[1]))))], axis=1, inplace=True)

    return data


def gene_sum_in_cells(ID, cell_numbers=None, num_gene=None):
    data = gene_in_cells(ID, cell_numbers).sum(axis=1).sort_values(ascending=False)
    data = data[data > 0]

    return data if num_gene is None else data[:num_gene]


def gene_mean_in_cells(ID, cell_numbers=None, num_gene=None, text=True):
    data = gene_in_cells(ID, cell_numbers).mean(axis=1).sort_values(ascending=False)
    data = data[data > 0]

    return data if num_gene is None else data[:num_gene]


def check_valid_function(cluster_function):
    allowed_functions = [clustering_Kmeans_with_num]
    if cluster_function not in allowed_functions:
        print("cluster_function must be in", allowed_functions)
        return False
    else:
        return True


def stacked_bar_gene_sum(ID, cluster_function, num_groups=10, num_gene=5):
    if not check_valid_function(cluster_function):
        return
    cluster_group, cluster_centers = cluster_function(ID, num_groups)

    gene_list = numpy.swapaxes([list(gene_sum_in_cells(ID, cluster_group[i], num_gene)) for i in cluster_group], 0, 1)
    gene_name = numpy.swapaxes([list(gene_sum_in_cells(ID, cluster_group[i], num_gene).index) for i in cluster_group], 0, 1)

    pprint.pprint(gene_list)
    pprint.pprint(gene_name)

    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    plt.figure()
    plt.bar(numpy.arange(num_groups), gene_list[0], 0.35)
    for i in range(1, num_gene):
        plt.bar(numpy.arange(num_groups), gene_list[i], 0.35, bottom=numpy.sum(numpy.array([gene_list[j] for j in range(i)]), axis=0))

    gene_tick = numpy.amax(numpy.sum(gene_list, axis=0)) / 5 / num_gene
    for i in range(num_groups):
        for j in range(num_gene):
            plt.text(i + 0.05, (j + 1) * gene_tick, gene_name[j][i], fontsize=10, bbox=dict(color="white", alpha=0.3))

    plt.grid(True)
    plt.title("Stacked Bar " + ID + " with " + str(num_gene) + " Gene")
    plt.xlabel("Group")
    plt.ylabel("# of Gene")
    plt.xticks(numpy.arange(num_groups), list(range(num_groups)))

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "StackedBar_" + ID + "_" + str(num_groups) + "_" + str(num_gene) + "_" + now + ".png")
    plt.close()


def heatmap_sum(ID, cluster_function, num_groups=10, num_gene=None):
    if not check_valid_function(cluster_function):
        return

    cluster_group, cluster_centers = cluster_function(ID, num_groups)

    gene_list = [gene_sum_in_cells(ID, cluster_group[i], num_gene) for i in cluster_group]
    gene_name = [gene_sum_in_cells(ID, cluster_group[i], num_gene).index for i in cluster_group]

    pprint.pprint(gene_list)
    pprint.pprint(gene_name)


if __name__ == "__main__":
    stacked_bar_gene_sum("NS_SW1", clustering_Kmeans_with_num)
    # heatmap_sum("NS_SW1", clustering_Kmeans_with_num)
