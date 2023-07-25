# Efficiently predicting high resolution mass spectra using graph neural networks

This repository is supporting material for our [ICML 2023 paper](https://proceedings.mlr.press/v202/murphy23a/murphy23a.pdf). We provide code to reproduce our results here, public-domain datasets we used + processing notebooks, and trained model weights for GrAFF-MS pluis our NEIMS implementation. Dependencies are listed in `requirements.txt`.

To just run GrAFF-MS using our trained weights:

`python run-graff-ms.py lightning_logs/graff-ms/version_0/checkpoints/epoch=96-step=27257.ckpt example_input.tsv example_output.msp --min_probability 0.001`

Each line of `example_input.tsv` yields an individual spectrum in `example_output.msp`, and should be four tab-delimited fields: a unique identifier for the spectrum, a SMILES string, a precursor type (`[M+H]+` or `[M-H]-`) and a normalized collision energy (see [Thermo Scientific NCE  technical note](https://tools.thermofisher.com/content/sfs/brochures/PSB104-Normalized-Collision-Energy-Technology-EN.pdf)). The `--min_probability` flag is an intensity cutoff to keep the output files small.

We do not include NIST-20 or files derived from it (apart from our train-test splits and model weights) for licensing reasons. NIST-20 can be purchased from [GC Image]( https://www.gcimage.com/nist.html) ("MS/MS only" option) and extracted into a file `hr_msms_nist.MSP` using the [Library Conversion Tool](https://chemdata.nist.gov/mass-spc/ms-search/Library_conversion_tool.html) for MS Windows, making sure "2008 MS Search compatible" is turned off.

Once `hr_msms_nist.MSP` has been generated, and placed in `./data/nist-20/`, it must be parsed into a dataframe:

`python preprocess-nist.py data/nist-20/hr_msms_nist.MSP data/nist-20/inchikeys.pkl`

To then train GrAFF-MS:

`python train-graff-ms.py data/nist-20/hr_msms_nist.pkl`

And to train the NEIMS baseline:

`python train-neims.py data/nist-20/hr_msms_nist.pkl`

To reproduce our evaluations, first run the preprocessing notebooks in the `./data/` subfolders:

* `casmi-16/casmi-16.ipynb`
* `chembl/chembl.ipynb`
* `gnps/gnps.ipynb`
* `cfm-id/cfm-id.ipynb`

Then run `evaluate-models.ipynb`.

Note: this codebase is provided for the purpose of reproducing the results shown in our ICML manuscript. If you are a user of mass spectrometry interested in a spectrum prediction model with a more polished interface, consider the excellent work of our colleagues in the Coley group: https://github.com/samgoldman97/ms-pred
