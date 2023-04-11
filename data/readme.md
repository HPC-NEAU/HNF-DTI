## Data

### Benchmark Datasets

The gold standard public benchmark dataset obtained from http://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/, which was built by Yamanishi.

The datasets have four categories of subsets Ion channels (IC), G-protein-coupled receptors (GPCR), Enzymes, and Nuclear receptors (NR).

The drugbank_approved dataset  is employed for verifying the performance of the models.

###  Cross-domain Label Reversal Dataset

The cross-domain label reversal dataset specifically to improve the generalization performance of the proposed method.

The cross-domain label reversal dataset is built in these manners:

Firstly, we collect the DTIs from the above benchmark datasets randomly while each DTI should be contained in the two classes. 

Secondly, the DTIs in the training set appear only in one class of samples, while the opposite class of samples is only in the test set. 
