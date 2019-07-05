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
import sklearn.cluster


def get_matrix(filename):
    return scipy.io.mmread(filename)


def get_feature_ids(filename):
    return [row[0] for row in csv.reader(gzip.open(filename, mode="rt"), delimiter="\t")]


def get_gene_name(filename):
    return [row[1] for row in csv.reader(gzip.open(filename, mode="rt"), delimiter="\t")]


def get_feature_type(filename):
    return [row[2] for row in csv.reader(gzip.open(filename, mode="rt"), delimiter="\t")]


def get_barcodes(filename):
    return [row[0] for row in csv.reader(gzip.open(filename, mode="rt"), delimiter="\t")]


def get_all(dirname):
    matrix_dir = os.path.join(dirname, "matrix.mtx.gz")
    features_path = os.path.join(dirname, "features.tsv.gz")
    barcodes_path = os.path.join(dirname, "barcodes.tsv.gz")

    return {"matrix": get_matrix(matrix_dir), "feature_ids": get_feature_ids(features_path), "gene_name": get_gene_name(features_path), "feature_type": get_feature_type(features_path), "barcodes": get_barcodes(barcodes_path)}


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


def clustering_Kmeans_with_num(ID, num_groups):
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    projection = get_data_from_id(ID)

    kmeans = sklearn.cluster.KMeans(n_clusters=num_groups, random_state=0, n_jobs=-1).fit(numpy.array([_ for _ in zip(projection["std_TSNE-1"], projection["std_TSNE-2"])]))

    projection["group"] = kmeans.fit_predict([_ for _ in zip(projection["std_TSNE-1"], projection["std_TSNE-2"])])

    plt.figure()
    plt.scatter(projection["std_TSNE-1"], projection["std_TSNE-2"], c=projection["group"])
    plt.scatter([elem[0] for elem in kmeans.cluster_centers_], [elem[1] for elem in kmeans.cluster_centers_], c="k", marker="X", s=500)

    plt.grid(True)
    plt.title("KMeans: " + str(num_groups))
    plt.xlabel("Standardized TSNE-1")
    plt.ylabel("Standardized TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "KMeans_" + ID + "_" + str(num_groups) + "_" + now + ".png")
    plt.close()

    return projection


def clustering_Kmeans(ID, num=10):
    return [clustering_Kmeans_with_num(ID, i) for i in range(2, num + 1)]


if __name__ == "__main__":
    clustering_Kmeans_with_num("NS_SW1", 10)
