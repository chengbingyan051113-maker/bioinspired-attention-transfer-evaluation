# Local input schema

Each of cfp_train.csv, cfp_validation.csv, uwf_train.csv and uwf_validation.csv
requires sample_id, image_path, grade (integer 0–4), lesion_NAME and
lesion_NAME_mask for MA, HE, IH, VB_IRMA, NV, VH and RD.
Masks are binary; labels are binary wherever mask=1.
Use complete local paths, unique image IDs and disjoint image partitions.
These image checks cannot establish patient or source independence.
CFP and UWF are separate unpaired cohorts.
