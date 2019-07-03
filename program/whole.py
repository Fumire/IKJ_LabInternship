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


def Total():
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    projections = list()
    for ID in IDs:
        projections.append(pandas.read_csv("/BiO/Live/jwlee230/181113_spermatogenesis/result/" + ID + "_reanalyze/outs/analysis/tsne/2_components/projection.csv", header=0))

    plt.figure()
    for data, color in zip(projections, ["C0", "C1", "C2", "C3"]):
        plt.scatter(data["TSNE-1"], data["TSNE-2"], c=color, alpha=0.6)

    plt.grid(True)
    plt.title("Total")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "total" + "_" + now + ".png")
    plt.close()


def tSNE_normal(ID):
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    projection = pandas.read_csv("/BiO/Live/jwlee230/181113_spermatogenesis/result/" + ID + "_reanalyze/outs/analysis/tsne/2_components/projection.csv", header=0)

    plt.figure()
    plt.scatter(projection["TSNE-1"], projection["TSNE-2"])

    plt.grid(True)
    plt.title(ID)
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + ID + "_" + now + ".png")
    plt.close()


def clustering_Kmeans_with_num(ID, num_groups):
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    projection = pandas.read_csv("/BiO/Live/jwlee230/181113_spermatogenesis/result/" + ID + "_reanalyze/outs/analysis/tsne/2_components/projection.csv", header=0)

    projection["std_TSNE-1"] = scipy.stats.zscore(projection["TSNE-1"])
    projection["std_TSNE-2"] = scipy.stats.zscore(projection["TSNE-2"])

    kmeans = sklearn.cluster.KMeans(n_clusters=num_groups, random_state=0, n_jobs=-1).fit(numpy.array([_ for _ in zip(projection["std_TSNE-1"], projection["std_TSNE-2"])]))

    color = kmeans.fit_predict([_ for _ in zip(projection["std_TSNE-1"], projection["std_TSNE-2"])])

    plt.figure()
    plt.scatter(projection["std_TSNE-1"], projection["std_TSNE-2"], c=color)
    plt.scatter([elem[0] for elem in kmeans.cluster_centers_], [elem[1] for elem in kmeans.cluster_centers_], c="k", marker="X", s=500)

    plt.grid(True)
    plt.title("KMeans: " + str(num_groups))
    plt.xlabel("Standard TSNE-1")
    plt.ylabel("Standard TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "KMeans_" + ID + "_" + str(num_groups) + "_" + now + ".png")
    plt.close()


def clustering_Kmeans(ID):
    for i in range(2, 11):
        clustering_Kmeans_with_num(ID, i)


def clustering_mean_shift_with_num(ID, num_groups):
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    projection = pandas.read_csv("/BiO/Live/jwlee230/181113_spermatogenesis/result/" + ID + "_reanalyze/outs/analysis/tsne/2_components/projection.csv", header=0)

    projection["std_TSNE-1"] = scipy.stats.zscore(projection["TSNE-1"])
    projection["std_TSNE-2"] = scipy.stats.zscore(projection["TSNE-2"])

    ms = sklearn.cluster.MeanShift(n_jobs=-1)
    color = ms.fit_predict(numpy.array([_ for _ in zip(projection["std_TSNE-1"], projection["std_TSNE-2"])]))

    plt.figure()
    plt.scatter(projection["std_TSNE-1"], projection["std_TSNE-2"], c=color)
    plt.scatter([elem[0] for elem in ms.cluster_centers_], [elem[1] for elem in ms.cluster_centers_], c="k", marker="X", s=500)

    plt.grid(True)
    plt.title("KMeans: " + str(num_groups))
    plt.xlabel("Standard TSNE-1")
    plt.ylabel("Standard TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "MeanShift_" + ID + "_" + str(num_groups) + "_" + now + ".png")
    plt.close()


if __name__ == "__main__":
    clustering_mean_shift_with_num("NS_SW1", 3)
