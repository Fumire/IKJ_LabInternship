import multiprocessing
import hashlib
import os
import pickle
import os
import time
import csv
import gzip
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
    dirname = "/home/jwlee/Spermatogenesis/result/" + ID + "/outs/filtered_feature_bc_matrix"
    matrix_dir = os.path.join(dirname, "matrix.mtx.gz")
    features_path = os.path.join(dirname, "features.tsv.gz")
    barcodes_path = os.path.join(dirname, "barcodes.tsv.gz")

    data[ID] = {"matrix": get_matrix(matrix_dir), "feature_ids": get_feature_ids(features_path), "gene_name": get_gene_name(features_path), "feature_type": get_feature_type(features_path), "barcodes": get_barcodes(barcodes_path)}

    return data[ID]


now = time.strftime("%m%d%H%M%S")
figure_directory = "/home/jwlee/Spermatogenesis/figures/"
IDs = ["NS_SW1", "NS_SW2", "NS_SW3", "NS_SW4"]


def select_highly_variable_genes(raw_data, show=True, datum_point=95):
    data = pandas.DataFrame.from_dict({"means": raw_data.mean(axis=1).to_numpy(), "cvs": numpy.true_divide(raw_data.var(axis=1).to_numpy(), raw_data.mean(axis=1))})

    data = data.loc[(data["cvs"] > 0) & (data["means"] > 0)]

    selected = data.loc[(data["cvs"] >= numpy.percentile(data["cvs"], datum_point)) & (data["means"] >= numpy.percentile(data["means"], datum_point))]
    unselected = data.loc[(data["cvs"] < numpy.percentile(data["cvs"], datum_point)) | (data["means"] < numpy.percentile(data["means"], datum_point))]

    raw_data = raw_data.iloc[selected.index]
    print("Gene & Cell:", raw_data.shape)

    if show:
        mpl.use("Agg")
        mpl.rcParams.update({"font.size": 30})

        plt.figure()
        plt.scatter(numpy.log(selected["means"]), numpy.log(selected["cvs"]), c="blue", alpha=0.6, label="Selected")
        plt.scatter(numpy.log(unselected["means"]), numpy.log(unselected["cvs"]), c="red", alpha=0.6, label="Unselected")

        plt.grid(True)
        plt.title(str(selected.shape[0]) + " Genes: " + str(100 - datum_point) + "%")
        plt.xlabel("log(means)")
        plt.ylabel("log(CV)")
        plt.legend()

        fig = plt.gcf()
        fig.set_size_inches(24, 18)
        fig.savefig(figure_directory + "HighlyVariableGene_" + now + ".png")
        plt.close()

    return raw_data


def get_whole_data(genes=None):
    def make_md5(data):
        if data is None:
            return hashlib.md5("".encode("utf-8")).hexdigest()
        else:
            return hashlib.md5(str(sorted(genes)).encode("utf-8")).hexdigest()

    if os.path.exists(make_md5(genes) + ".data"):
        with open(make_md5(genes) + ".data", "rb") as f:
            return pickle.load(f)

    if genes is not None and "ref" in genes:
        data = get_matrix("/home/jwlee/Spermatogenesis/result/ref/outs/filtered_feature_bc_matrix/matrix.mtx.gz")
        print(data)

        data = sklearn.decomposition.PCA(random_state=0, n_components=data.shape[1]).fit_transform(numpy.swapaxes(data.values, 0, 1))
        print("PCA data:", data)
        print("Cell & Gene-like:", len(data), len(data[0]))

        data = numpy.swapaxes(sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(data), 0, 1)

        projection = dict()
        projection["Barcode"] = get_barcodes("/home/jwlee/Spermatogenesis/result/ref/outs/filtered_feature_bc_matrix/barcodes.tsv.gz")
        projection["std_TSNE-1"] = scipy.stats.zscore(data[0])
        projection["std_TSNE-2"] = scipy.stats.zscore(data[1])

        with open(make_md5(genes) + ".data", "wb") as f:
            pickle.dump(projection, f)

        return projection

    data = get_matrix("/home/jwlee/Spermatogenesis/result/aggr/outs/filtered_feature_bc_matrix/matrix.mtx.gz")

    if genes is None:
        data = select_highly_variable_genes(data)
    else:
        data["gene"] = get_gene_name("/home/jwlee/Spermatogenesis/result/aggr/outs/filtered_feature_bc_matrix/features.tsv.gz")
        data = data[data["gene"].isin(genes)]
        del data["gene"]

    data = sklearn.decomposition.PCA(random_state=0, n_components="mle").fit_transform(numpy.swapaxes(data.values, 0, 1))
    print("PCA data: ", data)
    print("Cell & Gene-like:", len(data), len(data[0]))

    data = numpy.swapaxes(sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(data), 1, 0)

    projection = dict()
    projection["Barcode"] = numpy.array(get_barcodes("/home/jwlee/Spermatogenesis/result/aggr/outs/filtered_feature_bc_matrix/barcodes.tsv.gz"))
    projection["std_TSNE-1"] = scipy.stats.zscore(data[0])
    projection["std_TSNE-2"] = scipy.stats.zscore(data[1])

    projection = pandas.DataFrame.from_dict(projection)

    whole_data[make_md5(genes)] = projection

    return projection


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
    projection = pandas.read_csv("/home/jwlee/Spermatogenesis/result/" + ID + "/outs/analysis/tsne/2_components/projection.csv", header=0)

    return [barcode[:-1] + ID[-1] for barcode in projection["Barcode"]]


def get_data_from_id(ID, genes=None):
    if ID == "ref":
        return get_whole_data(genes=["ref"])

    projection = get_whole_data(genes)
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
    cells = cells.tolist()
    given = dict()
    for i in range(max(cells) + 1):
        given[i] = list(filter(lambda x: cells[x] == i, list(range(len(cells)))))
    return given


def clustering_Spectral_with_num(ID, num_groups):
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    projection = get_data_from_id(ID)

    projection["group"] = sklearn.cluster.SpectralClustering(n_clusters=num_groups, random_state=0, n_jobs=-1).fit_predict(projection[["std_TSNE-1", "std_TSNE-2"]].values)

    group = make_cluster_dict(projection["group"])
    data = [group[i] for i in group]
    cluster_centers = [numpy.mean([projection.loc[d, "std_TSNE-1"], projection.loc[d, "std_TSNE-2"]], axis=1) for d in data]

    plt.figure()
    plt.scatter(projection["std_TSNE-1"], projection["std_TSNE-2"], c=projection["group"])
    plt.scatter([elem[0] for elem in cluster_centers], [elem[1] for elem in cluster_centers], c="k", marker="X")
    for i, loc in enumerate(cluster_centers):
        plt.text(loc[0] + 0.05, loc[1], str(i), fontsize=30, bbox=dict(color="white", alpha=0.8))

    plt.grid(True)
    plt.title("Spectral: " + str(num_groups))
    plt.xlabel("Standardized TSNE-1")
    plt.ylabel("Standardized TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "Spectral_" + ID + "_" + str(num_groups) + "_" + now + ".png")
    plt.close()

    return (group, cluster_centers)


def clustering_Kmeans_with_num(ID, num_groups):
    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    projection = get_data_from_id(ID)

    kmeans = sklearn.cluster.KMeans(n_clusters=num_groups, random_state=0, n_jobs=-1).fit(projection[["std_TSNE-1", "std_TSNE-2"]].values)

    projection["group"] = kmeans.fit_predict(projection[["std_TSNE-1", "std_TSNE-2"]].values)

    plt.figure()
    plt.scatter(projection["std_TSNE-1"], projection["std_TSNE-2"], c=projection["group"])
    plt.scatter([elem[0] for elem in kmeans.cluster_centers_], [elem[1] for elem in kmeans.cluster_centers_], c="k", marker="X", s=500)
    for i, loc in enumerate(kmeans.cluster_centers_):
        plt.text(loc[0] + 0.05, loc[1], str(i), fontsize=30, bbox=dict(color="white", alpha=0.8))

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


def gene_mean_in_cells(ID, cell_numbers=None, num_gene=100, text=True):
    data = gene_in_cells(ID, cell_numbers).mean(axis=1).sort_values(ascending=False)
    data = data[data > 0]

    return data if num_gene is None else data[:num_gene]


def check_valid_function(cluster_function):
    allowed_functions = [clustering_Kmeans_with_num, clustering_Spectral_with_num]
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
    plt.ylabel("# of Genes")
    plt.xticks(numpy.arange(num_groups), list(range(num_groups)))

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "StackedBar_" + ID + "_" + str(num_groups) + "_" + str(num_gene) + "_" + now + ".png")
    plt.close()


def stacked_bar_gene_mean(ID, cluster_function, num_groups=10, num_gene=5):
    if not check_valid_function(cluster_function):
        return
    cluster_group, cluster_centers = cluster_function(ID, num_groups)

    gene_list = numpy.swapaxes([list(gene_mean_in_cells(ID, cluster_group[i], num_gene)) for i in cluster_group], 0, 1)
    gene_name = numpy.swapaxes([list(gene_mean_in_cells(ID, cluster_group[i], num_gene).index) for i in cluster_group], 0, 1)

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
    plt.title("Stacked Bar " + ID + " with " + str(num_gene) + " Genes")
    plt.xlabel("Group")
    plt.ylabel("# of Gene")
    plt.xticks(numpy.arange(num_groups), list(range(num_groups)))

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "StackedBar_" + ID + "_" + str(num_groups) + "_" + str(num_gene) + "_" + now + ".png")
    plt.close()


def sort_index(gene_list):
    group_order = [tuple(list(scipy.stats.rankdata(data)) + [i]) for i, data in enumerate(gene_list)]

    group_order.sort()

    group_order = [list(elem)[-1] for elem in group_order]
    answer = [[i for i in gene_list[j]] for j in group_order]

    return (group_order, answer)


def heatmap_sum_top(ID, cluster_function, num_groups=10, num_gene=None, show_text=True):
    if not check_valid_function(cluster_function):
        return

    cluster_group, cluster_centers = cluster_function(ID, num_groups)

    gene_name = list(gene_sum_in_cells(ID).index)
    if num_gene is not None:
        gene_name = gene_name[:num_gene]
    gene_name = sorted(gene_name)

    group_order, gene_list = sort_index([gene_sum_in_cells(ID, cluster_group[i]) for i in cluster_group])
    for i, data in enumerate(gene_list):
        data.drop(labels=list(filter(lambda x: x not in gene_name, list(data.index))), inplace=True)
        data.sort_index(inplace=True)
        gene_list[i] = scipy.stats.zscore(data.tolist())

    pprint.pprint(gene_name)
    pprint.pprint(gene_list)

    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    plt.figure()
    plt.imshow(gene_list)

    plt.title("HeatMap _ " + ID + "_" + str(num_gene) + " Genes")
    plt.xlabel("Genes")
    plt.ylabel("Groups")
    plt.xticks(numpy.arange(len(gene_name)), gene_name, fontsize=10, rotation=90)
    plt.yticks(numpy.arange(len(group_order)), group_order, fontsize=10)

    threshold = numpy.amax([numpy.amax(i) for i in gene_list]) / 2
    for i in range(len(gene_name)):
        for j in range(num_groups):
            if show_text:
                plt.text(j, i, str(gene_list[i][j]), color="white" if gene_list[i][j] < threshold else 'black', fontsize=10)

    fig = plt.gcf()
    fig.set_size_inches(max(24, len(gene_name) * 0.5), 18)
    fig.savefig(figure_directory + "HeatMap_" + ID + "_" + str(num_groups) + "_" + str(len(gene_name)) + "_" + now + ".png")
    plt.close()

    return (cluster_group, group_order, cluster_centers)


def heatmap_mean_top(ID, cluster_function, num_groups=10, num_gene=None, show_text=True):
    if not check_valid_function(cluster_function):
        return

    cluster_group, cluster_centers = cluster_function(ID, num_groups)

    gene_name = list(gene_mean_in_cells(ID).index)
    if num_gene is not None:
        gene_name = gene_name[:num_gene]
    gene_name = sorted(gene_name)

    gene_list = [gene_mean_in_cells(ID, cluster_group[i]).sort_index() for i in cluster_group]
    for i, data in enumerate(gene_list):
        data = data.add(pandas.Series(0, index=gene_name), fill_value=0)
        data.drop(labels=list(filter(lambda x: x not in gene_name, list(data.index))), inplace=True)
        data.sort_index(inplace=True)
        gene_list[i] = scipy.stats.zscore(data.tolist())

    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    plt.figure()
    plt.imshow(gene_list)

    plt.title("HeatMap_" + ID + "_" + str(num_gene) + " Genes")
    plt.xlabel("Genes")
    plt.ylabel("Groups")
    plt.xticks(numpy.arange(len(gene_name)), gene_name, fontsize=10, rotation=90)
    plt.yticks(numpy.arange(num_groups), list(range(num_groups)), fontsize=10)

    threshold = numpy.amax([numpy.amax(i) for i in gene_list]) / 2
    for i in range(len(gene_name)):
        for j in range(num_groups):
            if show_text:
                plt.text(j, i, str(gene_list[i][j]), color="white" if gene_list[i][j] < threshold else 'black', fontsize=10)

    fig = plt.gcf()
    fig.set_size_inches(max(24, len(gene_name) * 0.5), 18)
    fig.savefig(figure_directory + "HeatMap_" + ID + "_" + str(num_groups) + "_" + str(len(gene_name)) + "_" + now + ".png")
    plt.close()

    return (cluster_group, list(range(num_groups)), cluster_centers)


def find_marker_gene(ID, cluster_function, num_groups=10):
    if not check_valid_function(cluster_function):
        return

    cluster_group, cluster_centers = cluster_function(ID, num_groups)

    whole_cells = gene_in_cells(ID)
    marker_gene = list()
    for i in cluster_group:
        selected_gene = list()
        data = gene_in_cells(ID, cell_numbers=cluster_group[i])
        for row in list(data.index):
            value = scipy.stats.ttest_ind(list(data.loc[row]), list(whole_cells.loc[row]))
            if value[0] > 0 and value[1] < 0.05:
                selected_gene.append((value[1], row))
        selected_gene.sort()
        print(selected_gene[:10])
        marker_gene.append(tuple(selected_gene[i][1] for i in range(3)))

    return marker_gene


gene_1 = ["Grfa1", "Zbtb16", "Nanos3", "Nanos2", ",Sohlh1", "Neurog3", "Piwil4", "Lin28a", "Utf1", "Kit", "Uchl1", "Dmrt1", "Sohlh2", "Dazl", "Stra8", "Scml2", "Rpa2", "Rad51", "Rhox13", "Dmc1", "Melob", "Sycp1", "Sycp3", "Ccnb1ip1", "Hormad1", "Piwil2", "Piwil1", "Atr", "Mybl1", "Dyx1c1", "Msh3", "Ccnb1", "Spo11", "Ldha", "Ldhc", "Cetn4", "Tekt1", "Acr", "Ssxb1", "Ssxb2", "Acrv1", "Catsper3", "Catsper1", "Saxo1", "Hsfy2", "Txndc8", "Tnp1", "Tnp2", "Tmod4", "Gapdhs", "Car2", "Prm2", "Prm1", "Prm3", "Pgk2", "Wt1", "Sox9", "Cyp11a1", "Nr5a1", "Star", "Hsd3b1", "Clu", "Cyp17a1", "Gata4", "Acta2"]
gene_2 = ["Id4", "Gfra1", "Zbtb16", "Stra8", "Rhox13", "Sycp3", "Dmc1", "Piwil1", "Pgk2", "Acr", "Gapdhs", "Prm1"]


def heatmap_given_genes(ID, cluster_function, gene_name=gene_1, num_groups=10):
    if not check_valid_function(cluster_function):
        return

    cluster_group, cluster_centers = cluster_function(ID, num_groups)

    raw_gene_list = [gene_mean_in_cells(ID, cluster_group[i]) for i in cluster_group]
    gene_list = [[None for i in gene_name] for j in raw_gene_list]
    for i, data in enumerate(raw_gene_list):
        for j, gene in enumerate(gene_name):
            gene_list[i][j] = float(data.loc[gene]) if (gene in list(data.index)) else 0.0

    for i, data in enumerate(gene_list):
        if numpy.unique(data).size > 1:
            gene_list[i] = scipy.stats.zscore(data)
        else:
            gene_list[i] = [0 for gene in data]

    group_order, gene_list = sort_index(gene_list)
    pprint.pprint(gene_list)

    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    plt.figure()
    plt.imshow(gene_list)

    plt.title("Heatmap_" + ID + "_" + str(len(gene_name)) + " Genes")
    plt.xlabel("Genes")
    plt.ylabel("Groups")
    plt.xticks(numpy.arange(len(gene_name)), gene_name, fontsize=10, rotation=90)
    plt.yticks(numpy.arange(num_groups), group_order, fontsize=10)

    fig = plt.gcf()
    fig.set_size_inches(max(24, 0.5 * len(gene_name)), max(18, 0.2 * num_groups))
    fig.savefig(figure_directory + "HeatMap_" + ID + "_" + str(num_groups) + "_" + str(len(gene_name)) + "_" + now + ".png")
    plt.close()

    return (cluster_group, group_order, cluster_centers)


def pseudotime(ID, cluster_function, num_groups=100, select_gene=True):
    if not check_valid_function:
        return

    if select_gene:
        cluster_group, group_order, cluster_centers = heatmap_given_genes(ID, cluster_function, num_groups=num_groups)
    else:
        cluster_group, group_order, cluster_centers = heatmap_mean_top(ID, cluster_function, show_text=False)
    projection = get_data_from_id(ID)

    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    plt.figure()
    for i in cluster_group:
        plt.scatter(projection["std_TSNE-1"].iloc[cluster_group[i]], projection["std_TSNE-2"].iloc[cluster_group[i]], c=["C" + str(i % 10) for _ in range(projection["std_TSNE-1"].iloc[cluster_group[i]].size)])
    for i in range(1, len(cluster_centers)):
        plt.arrow(cluster_centers[group_order[i - 1]][0], cluster_centers[group_order[i - 1]][1], 0.8 * (cluster_centers[group_order[i]][0] - cluster_centers[group_order[i - 1]][0]), 0.8 * (cluster_centers[group_order[i]][1] - cluster_centers[group_order[i - 1]][1]), width=0.05, edgecolor=None, linestyle=":")

    plt.grid(True)
    plt.title("Ordering Groups")
    plt.xlabel("Standardized TSNE-1")
    plt.ylabel("Standardized TSNE-2")

    fig = plt.gcf()
    fig.set_size_inches(24, 18)
    fig.savefig(figure_directory + "Arrow_" + ID + "_" + str(num_groups) + "_" + now + ".png")
    plt.close()


def bar_given_genes(ID, cluster_function, gene_name=gene_1, num_groups=10):
    if not check_valid_function(cluster_function):
        return

    cluster_group, cluster_centers = cluster_function(ID, num_groups)

    raw_gene_list = [gene_mean_in_cells(ID, cluster_group[i]) for i in cluster_group]
    gene_list = [[None for i in gene_name] for j in cluster_group]
    for i, data in enumerate(raw_gene_list):
        for j, gene in enumerate(gene_name):
            gene_list[i][j] = float(data.loc[gene]) if (gene in list(data.index)) else 0.0

    for i, data in enumerate(gene_list):
        if numpy.unique(data).size > 1:
            gene_list[i] = scipy.stats.zscore(data)
        else:
            gene_list[i] = [0 for _ in data]

    mpl.use("Agg")
    mpl.rcParams.update({"font.size": 30})

    plt.figure()
    fig, ax = plt.subplots(num_groups)

    for i, data in enumerate(gene_list):
        for j, high in enumerate(data):
            ax[i].bar(j, high, color="C" + str(j % 10), edgecolor="k", label=gene_name[j])

    plt.setp(ax, xticks=list(range(len(gene_name))), xticklabels=gene_name)

    fig = plt.gcf()
    fig.set_size_inches(max(24, 2.5 * len(gene_name)), max(18, 4 * num_groups))
    fig.savefig(figure_directory + "Bar_graph_" + ID + "_" + str(num_groups) + "_" + str(len(gene_name)) + "_" + now + ".png")
    plt.close()


def get_common_genes(ID, cluster_function, num_groups=10):
    if not check_valid_function:
        return

    cluster_group, cluster_centers = cluster_function(ID, num_groups)

    gene_list = [list(gene_mean_in_cells(ID, cluster_group[i]).index) for i in cluster_group]

    common_gene = set(gene_list[0])
    for gene in gene_list[1:]:
        if not common_gene:
            return common_gene
        common_gene = common_gene & set(gene)

    pprint.pprint(common_gene)
    print(len(common_gene))

    return common_gene


def scatter_given_genes(ID, genes=gene_1):
    def change_scale(gene_expression):
        minimum, maximum = min(gene_expression), max(gene_expression)

        return list(map(lambda x: (x - minimum) / (maximum - minimum), gene_expression))

    data_1 = get_data_from_id(ID, genes)
    data_2 = get_all(ID)

    for gene in genes:
        try:
            number_gene = data_2["gene_name"].index(gene)
        except ValueError:
            print(gene, "is not here")
            continue

        gene_expression = change_scale(data_2["matrix"].iloc[number_gene].values)

        mpl.use("Agg")
        mpl.rcParams.update({"font.size": 30})

        plt.figure()
        for x, y, alpha in zip(data_1["std_TSNE-1"], data_1["std_TSNE-2"], gene_expression):
            plt.scatter(x, y, c='k', alpha=0.1)
            plt.scatter(x, y, c='b', alpha=alpha)

        plt.grid(True)
        plt.title(ID + "_" + gene)
        plt.xlabel("Standardized TSNE-1")
        plt.ylabel("Standardized TSNE-2")

        fig = plt.gcf()
        fig.set_size_inches(24, 18)
        fig.savefig(figure_directory + "Scatter_" + ID + "_" + gene + "_" + now + ".png")
        plt.close()

        print(gene, "Done!!")


def get_whole_data_3d(genes=None):
    def make_md5(data):
        if data is None:
            return hashlib.md5("3d".encode("utf-8")).hexdigest()
        else:
            return hashlib.md5(("3d" + str(sorted(data))).encode("utf-8")).hexdigest()

    if os.path.exists(make_md5(genes) + ".data"):
        with open(make_md5(genes) + ".data", "rb") as f:
            return pickle.load(f)

    data = get_matrix("/home/jwlee/Spermatogenesis/result/aggr/outs/filtered_feature_bc_matrix/matrix.mtx.gz")

    if genes is None:
        data = select_highly_variable_genes(data)
    else:
        data["gene"] = get_gene_name("/home/jwlee/Spermatogenesis/result/aggr/outs/filtered_feature_bc_matrix/features.tsv.gz")
        data = data[data["gene"].isin(genes)]
        del data["gene"]

    data = sklearn.decomposition.PCA(random_state=0, n_components="mle").fit_transform(numpy.swapaxes(data.values, 0, 1))
    print("PCA data: ", data)
    print("Cell & Gene-like:", len(data), len(data[0]))

    data = numpy.swapaxes(sklearn.manifold.TSNE(n_components=3, random_state=0).fit_transform(data), 0, 1)

    projection = dict()
    projection["Barcode"] = numpy.array(get_barcodes("/home/jwlee/Spermatogenesis/result/aggr/outs/filtered_feature_bc_matrix/barcodes.tsv.gz"))
    projection["std_TSNE-1"] = scipy.stats.zscore(data[0])
    projection["std_TSNE-2"] = scipy.stats.zscore(data[1])
    projection["std_TSNE-3"] = scipy.stats.zscore(data[2])

    projection = pandas.DataFrame.from_dict(projection)

    with open(make_md5(genes) + ".data", "wb") as f:
        pickle.dump(projection, f)

    return projection


if __name__ == "__main__":
    get_whole_data_3d()
    for _ in range(5):
        print("\a")
