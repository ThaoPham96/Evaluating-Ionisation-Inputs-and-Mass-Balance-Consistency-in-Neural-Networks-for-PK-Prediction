# Component notices

This repository combines author software, author-generated results, and data derived from external resources. These components do not share a blanket license.

* The existing author software license is preserved at `reproducibility/revision_work/analysis/LICENSE` (MIT, copyright 2026 Thao Pham and the PIPK-Net authors). It does not license third-party data, commercial calculated properties, pretrained weights, or upstream packages. Later campaign files retain their original notices.
* The modified Chemprop component retains its original `LICENSE` and `NOTICE` in `reproducibility/revision_work/analysis/vendor/chemprop_physio/`.
* Runtime packages and the public PyTorch base retain their upstream licenses and notices in the image. The recipe installs packages rather than redistributing their source under the author code license. `runtime_recipe/notice_inventory.py` can enumerate installed distribution notices.
* The e-Drug3D-derived historical PK extraction is distributed with acknowledgement to e-Drug3D, IPMC/CNRS. Clause 6 of the [IPMC terms](https://chemoinfo.ipmc.cnrs.fr/Terms_of_Use.html) permits acknowledged copying, modification and distribution, subject to its terms and express exceptions; commercial use requires the stated license. No unrestricted commercial license is assigned here.
* ChEMBL-derived fields retain attribution to the [ChEMBL database](https://www.ebi.ac.uk/chembl/), its [CC BY-SA 3.0 terms and calculator-property qualifications](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions). The historical extraction release and original calculator settings were not recorded. The transformations here comprise identity reconciliation, masks, cohort membership, and feature encoding; exact archived descriptor values are preserved.
* Class assignments combine cited literature, permanent-charge evidence, structural assessments and archived descriptors. The source ledger records these distinctions; they are not all ChEMBL or ChemAxon calculations.

See [data provenance and terms](docs/data-terms.md) for the source-specific evidence. Citation of this study does not replace the upstream acknowledgements or terms.
