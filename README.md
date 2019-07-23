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
1.1. matrix: get_matrix()
1.2. get_feature_ids()
1.3. get_gene_name()
1.4. get_feature_type()
1.5. get_barcodes()

### select_highly_variable_genes(raw_data, show=True, datum_point=95)
Select highly variable genes
#### Paramteters
1. Raw data: List of List. Gene expression data
2. show: Boolean. If True, draw plot for gene expression
3. datum_point: float. x means, select genes which is higher than x% genes.
#### Returns
1. List of List. Highly variable genes.
