# IKJ_LAB_INTERNSHIP

This is what I have done in IKJ lab internship.

## Getting Started

This project needs some FASTQ files.

## Makefile.whole

## Whole.py

### get_matrix(filename)
Read matrix file in to Pandas dataframe.
#### Paramteters
1. filename: Path for file
#### Returns
1. Matrix via Pandas dataframe.

### get_feature_ids(filename)
Get feature id
#### Paramteters
1. filename: Path for file
#### Returns
1. List for feature id

### get_gene_name(filename)
Get gene names
#### Paramteters
1. filename: Path for file
#### Returns
1. List for gene name

### get_feature_type(filename)
Get feature type such as Gene Expression
#### Paramteters
1. filename: Path for file
#### Returns
1. List of feature type

### get_barcodes(filename)
Get barcode for each cell
#### Paramteters
1. filename: Path for file
#### Returns
1. List for barcodes of each cells

### get_all(filename)
Get all data according to ID
#### Paramteters
1. filename: Path for file
#### Returns
1. Dictionary for data
- matrix: get_matrix()
- features_ids: get_feature_ids()
- gene_name: get_gene_name()
- feature_type: get_feature_type()
- barcodes: get_barcodes()

### select_highly_variable_genes(raw_data, show=True, datum_point=95)
Select highly variable genes
#### Paramteters
1. Raw data: List of List. Gene expression data
2. show: Boolean. If True, draw plot for gene expression
3. datum_point: float. x means, select genes which is higher than x% genes.
#### Returns
1. List of List. Highly variable genes.

### get_whole_data(genes=None)
Get aggregated data and do PCA & TSNE dimesion reduction.
#### Paramteters
1. genes: List or String.
- List: selected gene list
- string: "ref". use refences data or not.
#### Returns
1. Pandas dataframe: projection data

### draw_all()
Draw scatter for TSNE data

### get_real_barcodes(ID)
Get barcodes and concatenate numbers
#### Paramteters
1. ID: String. Name.
#### Returns
1. List. Changed barcodes.

### get_data_from_id(ID)
Get data from ID
#### Paramteters
1. ID: "ref" or sample name.
#### Returns
1. List.
- if ID is "ref", reference data
- Else, the data which have given ID in aggregated data

### draw_all_with_color()
Draw all data in one plot with different color

### draw_tSNE(ID)
Draw data according to ID in plot.
#### Paramteters
1. ID

### make_cluster_dict(cells)
Make cluster data into dictionary
#### Paramteters
1. cells: List. Each elements mean group number
#### Returns
1. Dictionary. Each values mean cell number which have same group.
