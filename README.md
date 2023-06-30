# PCDAL: A Perturbation Consistency-Driven Active Learning Approach for Medical Image Segmentation and Classification 

To use the our active learning method, run the related infer**.py

require:
--imgs_infer_path: the img dir of the img which require to secelt for labeled
--weight_path: the model weight acquired by training through labeled data

output:
a table names result.csv, which sort Sort based on the value of consistency loss.
Default: From maximum to minimum based on losses

Paper in arxiv:
https://arxiv.org/abs/2306.16918

